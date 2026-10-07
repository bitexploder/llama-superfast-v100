#pragma once

// Chunked gated-delta-net prefill on the Volta tensor cores (LLAMA_GDN_CHUNKED, default on).
//
// The token-serial kernel runs S_t = g_t S_{t-1} + k_t u_t^T with u_t = beta_t (v_t - g_t S_{t-1}^T k_t) and
// o_t = scale S_t^T q_t one token at a time. Over a chunk of C = 64 tokens with b_t the cumulative log decay
// inside the chunk (Gamma_t = exp(b_t)) and S_0 the state entering the chunk, the same recurrence is
//   (I + A) U = diag(beta) V - diag(beta Gamma) K S_0,   A[t][s] = beta_t (k_t . k_s) exp(b_t - b_s) for s < t
// so with T = (I + A)^-1, W = T diag(beta Gamma) K and U0 = T diag(beta) V:
//   U   = U0 - W S_0
//   O   = diag(Gamma) Q S_0 + P U,               P[t][s] = (q_t . k_s) exp(b_t - b_s) for s <= t
//   S_C = Gamma_C S_0 + K'^T U,                   K'[s]   = exp(b_C - b_s) k_s
// (Yang et al., Gated Delta Networks; FLA's chunk_gated_delta_rule). Two kernels:
//   gdn_chunk_prep:  one block per (chunk, head, sequence), all chunks at once: the decays, K K^T and Q K^T on
//                    the tensor cores, T by forward substitution in fp32, W and U0 on the tensor cores. Writes W,
//                    diag(Gamma) Q, P and K'^T in mma A-fragment order (fp16) and U0 in accumulator order (fp32).
//   gdn_chunk_state: one block per (32 value columns, head, sequence), the chunks in order. The value columns of
//                    the recurrence are independent, so the block keeps its 128 x 32 slice of the state in fp32
//                    registers; each chunk is W S_0 and Q S_0, then P U and K'^T U.
// Operands are fp16 (mma.m8n8k4, fp32 accumulation); the decays, the solve and the state are fp32. The summation
// order differs from the token-serial kernel, so the outputs agree to fp16 operand precision, not bitwise.
//
// mma.m8n8k4: each quadpair qp = (lane >> 2) & 3 computes its own 8x8x4 product. With sr = lane & 3,
// hi = (lane >> 4) & 1 and r = sr + 4 hi, a lane holds A[r][0..3] (.row) or A[4 hi + 0..3][sr] (.col), and
// B[0..3][r] (.col) or B[sr][4 hi + 0..3] (.row); C element e sits at row (e & 2) | 4 hi | (lane & 1), column
// (e & 1) | (lane & 2) | (e & 4). A warp tile here is 8 rows x 32 columns with A shared by the quadpairs; with a
// .row B, quadpair qp's local column c is tile column 16 (c >> 2) + 4 qp + (c & 3) (gdnc_bcol), which keeps the
// half-warp's sixteen 8-byte B loads in distinct banks at the pitches below.

#include <cuda_fp16.h>
#include <cstdint>

#define GDN_CHUNK      64   // tokens per chunk
#define GDN_CHUNK_D    128  // S_v (the state is D x D per head)
#define GDN_CHUNK_NCOL 32   // value columns per gdn_chunk_state block

// scratch per (sequence, head, chunk)
#define GDN_CHUNK_REC_H (GDN_CHUNK * GDN_CHUNK_D * 3 + GDN_CHUNK * GDN_CHUNK) // halves: W, Qg, K'^T, P
#define GDN_CHUNK_REC_F (GDN_CHUNK * GDN_CHUNK_D)                            // floats: U0
#define GDN_CHUNK_OFF_W  0
#define GDN_CHUNK_OFF_QG (GDN_CHUNK * GDN_CHUNK_D)
#define GDN_CHUNK_OFF_KT (GDN_CHUNK * GDN_CHUNK_D * 2)
#define GDN_CHUNK_OFF_P  (GDN_CHUNK * GDN_CHUNK_D * 3)

template <bool acol, bool brow>
static __device__ __forceinline__ void gdnc_mma(float (&c)[8], const uint32_t a0, const uint32_t a1,
                                                const uint32_t b0, const uint32_t b1) {
#define GDNC_MMA(L) asm("mma.sync.aligned.m8n8k4." L ".f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, " \
        "{%0,%1,%2,%3,%4,%5,%6,%7};"                                                                                 \
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])           \
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1))
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(GGML_USE_HIP)
    if constexpr (!acol && !brow) { GDNC_MMA("row.col"); }
    if constexpr (!acol &&  brow) { GDNC_MMA("row.row"); }
    if constexpr ( acol &&  brow) { GDNC_MMA("col.row"); }
    if constexpr ( acol && !brow) { GDNC_MMA("col.col"); }
#endif
#undef GDNC_MMA
}

static __device__ __forceinline__ int gdnc_crow(const int lane, const int e) {
    return (e & 2) | ((lane & 16) >> 2) | (lane & 1);
}
static __device__ __forceinline__ int gdnc_ccol(const int lane, const int e) {
    return (e & 1) | (lane & 2) | (e & 4);
}
// tile column of quadpair qp's local column c for a .row B
static __device__ __forceinline__ int gdnc_bcol(const int qp, const int c) {
    return 16 * (c >> 2) + 4 * qp + (c & 3);
}

// .row A-fragment order of an M x Kd matrix: element (m, k) at ((m/8) (Kd/8) + k/8) 64 + (m%8) 8 + k%8, so a lane's
// two k-steps are 16 contiguous bytes
template <int Kd>
static __device__ __forceinline__ int gdnc_apos(const int m, const int k) {
    return ((m >> 3) * (Kd / 8) + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7);
}
// .col A-fragment order of an M x Kd matrix: the 16 bytes of lane (sr, hi) in block (m/8, k/8) hold
// A[8 (m/8) + 4 hi + 0..3][8 (k/8) + sr] then A[..][8 (k/8) + 4 + sr]
template <int Kd>
static __device__ __forceinline__ int gdnc_acolpos(const int m, const int k) {
    return (((m >> 3) * (Kd / 8) + (k >> 3)) * 8 + ((m >> 2) & 1) * 4 + (k & 3)) * 8 + ((k >> 2) & 1) * 4 + (m & 3);
}

struct gdn_chunk_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    const float * state_in;  // [n_seqs][H][D][D], element (i, j) of S at j * D + i
    float *       state_out; // same layout
    float *       dst;       // [n_seqs][n_tokens][H][D]
    __half *      rec_h;     // [n_seqs][H][n_chunks][GDN_CHUNK_REC_H]
    float *       rec_f;     // [n_seqs][H][n_chunks][GDN_CHUNK_REC_F]
    float *       rec_g;     // [n_seqs][H][n_chunks] Gamma_C
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    int H, neqk1, rq3, n_tokens, n_c, n_chunks;
    float scale;
};

// gdn_chunk_prep shared memory (pitches in elements; the regions are reused as the kernel proceeds)
#define GDNC_KS  (GDN_CHUNK_D + 4)    // Ks[s][i], a .col B:  16 rows per half-warp, pitch 66 words
#define GDNC_BR  (GDN_CHUNK_D + 16)   // Kb, Vs [s][n], a .row B: pitch 72 words
#define GDNC_TS  (GDN_CHUNK + 8)      // T[t][s], a .row A
#define GDNC_AS  (GDN_CHUNK + 4)      // A[t][s], fp32
#define GDNC_R0  0                                        // Ks, then Kb
#define GDNC_R1  (GDNC_R0 + GDN_CHUNK * GDNC_BR * 2)      // Qs, then A, then Vs, then the W staging
#define GDNC_R2  (GDNC_R1 + GDN_CHUNK * GDNC_BR * 2)      // the P staging, then T
#define GDNC_R3  (GDNC_R2 + GDN_CHUNK * GDNC_TS * 2)      // b, beta, Gamma
#define GDNC_SM_PREP (GDNC_R3 + 3 * GDN_CHUNK * 4)
static_assert(GDN_CHUNK * GDNC_AS * 4 <= GDN_CHUNK * GDNC_BR * 2, "A fits region 1");

template <bool stub_solve>
__global__ void __launch_bounds__(256, 2) gdn_chunk_prep(const gdn_chunk_args p) {
    constexpr int C = GDN_CHUNK, D = GDN_CHUNK_D;
    extern __shared__ __align__(16) unsigned char gdnc_smem[];
    __half * Ks = (__half *) (gdnc_smem + GDNC_R0);
    __half * Kb = (__half *) (gdnc_smem + GDNC_R0);
    __half * Qs = (__half *) (gdnc_smem + GDNC_R1);
    float  * As = (float  *) (gdnc_smem + GDNC_R1);
    __half * Vs = (__half *) (gdnc_smem + GDNC_R1);
    __half * Ws = (__half *) (gdnc_smem + GDNC_R1);
    __half * Ps = (__half *) (gdnc_smem + GDNC_R2);
    __half * Ts = (__half *) (gdnc_smem + GDNC_R2);
    float  * bs = (float  *) (gdnc_smem + GDNC_R3);
    float  * be = bs + C;
    float  * ga = be + C;

    // the value heads that read one q/k head (h = rep neqk1 + iq1) are adjacent blocks, so q and k come from L2
    const int rqk = p.H / p.neqk1;
    const int h = (blockIdx.x % rqk) * p.neqk1 + blockIdx.x / rqk, chunk = blockIdx.y, seq = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int t0 = chunk * C;
    const int valid = min(C, p.n_c - t0);

    const int iq1 = h % p.neqk1, iq3 = seq / p.rq3;
    const float * qb = p.q + iq3 * p.sq3 + (int64_t) t0 * p.sq2 + iq1 * p.sq1;
    const float * kb = p.k + iq3 * p.sq3 + (int64_t) t0 * p.sq2 + iq1 * p.sq1;
    const float * vb = p.v + seq * p.sv3 + (int64_t) t0 * p.sv2 + h * p.sv1;
    const int64_t gb = seq * p.sb3 + (int64_t) t0 * p.sb2 + h * p.sb1;

    const int64_t rec = ((int64_t) seq * p.H + h) * p.n_chunks + chunk;
    __half * rW = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_W;
    __half * rQ = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_QG;
    __half * rK = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_KT;
    __half * rP = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_P;
    float  * rU = p.rec_f + rec * GDN_CHUNK_REC_F;

    // v is read after the solve: start it towards L2 now (one 128-byte line per thread)
    if (tid < 4 * valid) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(vb + (int64_t) (tid >> 2) * p.sv2 + (tid & 3) * 32));
    }

    // decays: inclusive scan of g over the chunk (padding: g = 0, beta = 0)
    if (w == 0) {
        float g0 = lane      < valid ? p.g[gb + (int64_t) lane        * p.sb2] : 0.0f;
        float g1 = lane + 32 < valid ? p.g[gb + (int64_t) (lane + 32) * p.sb2] : 0.0f;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float x0 = __shfl_up_sync(0xffffffff, g0, o);
            const float x1 = __shfl_up_sync(0xffffffff, g1, o);
            if (lane >= o) { g0 += x0; g1 += x1; }
        }
        g1 += __shfl_sync(0xffffffff, g0, 31);
        bs[lane] = g0; bs[lane + 32] = g1;
        ga[lane] = expf(g0); ga[lane + 32] = expf(g1);
        be[lane]      = lane      < valid ? p.beta[gb + (int64_t) lane        * p.sb2] : 0.0f;
        be[lane + 32] = lane + 32 < valid ? p.beta[gb + (int64_t) (lane + 32) * p.sb2] : 0.0f;
    }
    __syncthreads();
    const float bC = bs[C - 1];
    if (tid == 0) {
        p.rec_g[rec] = expf(bC);
    }

    // q: Qs, and diag(Gamma) Q as a .row A (rows t, k = i), one 16-byte segment per thread and step
#pragma unroll 2
    for (int it = 0; it < C * D / 8 / 256; it++) {
        const int sg = it * 256 + tid;
        const int t = ((sg >> 7) << 3) | (sg & 7), i8 = ((sg >> 3) & 15) * 8;
        float4 x0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), x1 = x0;
        if (t < valid) {
            x0 = *(const float4 *) (qb + (int64_t) t * p.sq2 + i8);
            x1 = *(const float4 *) (qb + (int64_t) t * p.sq2 + i8 + 4);
        }
        __half2 * qs = (__half2 *) (Qs + t * GDNC_KS + i8);
        qs[0] = __floats2half2_rn(x0.x, x0.y); qs[1] = __floats2half2_rn(x0.z, x0.w);
        qs[2] = __floats2half2_rn(x1.x, x1.y); qs[3] = __floats2half2_rn(x1.z, x1.w);
        const float gt = ga[t];
        uint4 o;
        __half2 * oh = (__half2 *) &o;
        oh[0] = __floats2half2_rn(gt * x0.x, gt * x0.y); oh[1] = __floats2half2_rn(gt * x0.z, gt * x0.w);
        oh[2] = __floats2half2_rn(gt * x1.x, gt * x1.y); oh[3] = __floats2half2_rn(gt * x1.z, gt * x1.w);
        *(uint4 *) (rQ + sg * 8) = o;
    }
    // k: Ks, and K'^T = (exp(b_C - b_s) k_s)^T as a .col A (rows i, k = s), one 16-byte segment per thread and step:
    // segment sg = ((i/8) 8 + s/8) 8 + 4 hi + sr holds k[s0 + sr][i0 + 0..3] and k[s0 + 4 + sr][i0 + 0..3]
#pragma unroll 2
    for (int it = 0; it < C * D / 8 / 256; it++) {
        const int sg = it * 256 + tid;
        const int sr = sg & 3, hi = (sg >> 2) & 1, s8 = (sg >> 3) & 7, i8 = sg >> 6;
        const int s = 8 * s8 + sr, i0 = 8 * i8 + 4 * hi;
        float4 x0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), x1 = x0;
        if (s < valid) {
            x0 = *(const float4 *) (kb + (int64_t) s * p.sq2 + i0);
        }
        if (s + 4 < valid) {
            x1 = *(const float4 *) (kb + (int64_t) (s + 4) * p.sq2 + i0);
        }
        __half2 * k0 = (__half2 *) (Ks + s * GDNC_KS + i0);
        __half2 * k1 = (__half2 *) (Ks + (s + 4) * GDNC_KS + i0);
        k0[0] = __floats2half2_rn(x0.x, x0.y); k0[1] = __floats2half2_rn(x0.z, x0.w);
        k1[0] = __floats2half2_rn(x1.x, x1.y); k1[1] = __floats2half2_rn(x1.z, x1.w);
        const float f0 = expf(bC - bs[s]), f1 = expf(bC - bs[s + 4]);
        uint4 o;
        __half2 * oh = (__half2 *) &o;
        oh[0] = __floats2half2_rn(f0 * x0.x, f0 * x0.y); oh[1] = __floats2half2_rn(f0 * x0.z, f0 * x0.w);
        oh[2] = __floats2half2_rn(f1 * x1.x, f1 * x1.y); oh[3] = __floats2half2_rn(f1 * x1.z, f1 * x1.w);
        *(uint4 *) (rK + sg * 8) = o;
    }
    __syncthreads();

    const int sr = lane & 3, hi = (lane >> 4) & 1;
    const int r  = sr | (hi << 2);
    const int qp = (lane >> 2) & 3;

    // K K^T and Q K^T: warp w owns rows t in [8w, 8w + 8) and all 64 columns s as two halves, s = 32 hf + 4 n + qp
    // for quadpair column n (a .col B from Ks, sixteen distinct rows per half-warp)
    {
        float kk[2][8] = {}, qk[2][8] = {};
#pragma unroll 8
        for (int ks = 0; ks < D / 4; ks++) {
            const uint2 ak = *(const uint2 *) (Ks + (8 * w + r) * GDNC_KS + 4 * ks);
            const uint2 aq = *(const uint2 *) (Qs + (8 * w + r) * GDNC_KS + 4 * ks);
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const uint2 bk = *(const uint2 *) (Ks + (32 * hf + 4 * r + qp) * GDNC_KS + 4 * ks);
                gdnc_mma<false, false>(kk[hf], ak.x, ak.y, bk.x, bk.y);
                gdnc_mma<false, false>(qk[hf], aq.x, aq.y, bk.x, bk.y);
            }
        }
        __syncthreads(); // Qs becomes A, and the P staging is free
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int t = 8 * w + gdnc_crow(lane, e);
                const int s = 32 * hf + 4 * gdnc_ccol(lane, e) + qp;
                const float dec = s <= t ? expf(bs[t] - bs[s]) : 0.0f;
                As[t * GDNC_AS + s] = s < t ? be[t] * dec * kk[hf][e] : 0.0f;
                Ps[gdnc_apos<C>(t, s)] = __float2half_rn(dec * qk[hf][e]);
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < C * C / 8; idx += 256) {
        *(uint4 *) (rP + idx * 8) = *(const uint4 *) (Ps + idx * 8);
    }
    __syncthreads(); // the P staging becomes T

    // T = (I + A)^-1 by forward substitution, one column per thread of warps 0 and 1; meanwhile warps 2..7 fill
    // Kb = beta Gamma k (k again, from L2) and hold beta v in registers for Vs, which replaces A afterwards
    constexpr int NV = (C * D / 4 + 191) / 192; // float4 of v per thread of warps 2..7
    float4 vr[NV];
    if (tid < C) {
        const int c = tid;
        if constexpr (stub_solve) {
#pragma unroll
            for (int t = 0; t < C; t++) {
                Ts[t * GDNC_TS + c] = __float2half_rn(t == c ? 1.0f : 0.0f);
            }
        } else {
            float x[C];
#pragma unroll
            for (int t = 0; t < C; t++) {
                float a0 = t == c ? 1.0f : 0.0f, a1 = 0.0f;
#pragma unroll
                for (int s4 = 0; s4 < t; s4 += 4) {
                    const float4 a = *(const float4 *) (As + t * GDNC_AS + s4);
                    a0 -= a.x * x[s4];
                    if (s4 + 1 < t) a1 -= a.y * x[s4 + 1];
                    if (s4 + 2 < t) a0 -= a.z * x[s4 + 2];
                    if (s4 + 3 < t) a1 -= a.w * x[s4 + 3];
                }
                x[t] = a0 + a1;
                Ts[t * GDNC_TS + c] = __float2half_rn(x[t]);
            }
        }
    } else {
        const int u = tid - C;
#pragma unroll
        for (int m = 0; m < NV; m++) {
            const int idx = m * 192 + u;
            const int s = idx >> 5, j4 = (idx & 31) * 4;
            vr[m] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (idx < C * D / 4 && s < valid) {
                const float4 x = *(const float4 *) (vb + (int64_t) s * p.sv2 + j4);
                const float  f = be[s];
                vr[m] = make_float4(f * x.x, f * x.y, f * x.z, f * x.w);
            }
        }
        for (int idx = u; idx < C * D / 4; idx += 192) {
            const int s = idx >> 5, i4 = (idx & 31) * 4;
            float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (s < valid) {
                x = *(const float4 *) (kb + (int64_t) s * p.sq2 + i4);
            }
            const float f = be[s] * ga[s];
            __half2 * d = (__half2 *) (Kb + s * GDNC_BR + i4);
            d[0] = __floats2half2_rn(f * x.x, f * x.y);
            d[1] = __floats2half2_rn(f * x.z, f * x.w);
        }
    }
    __syncthreads(); // A is dead
    if (tid >= C) {
        const int u = tid - C;
#pragma unroll
        for (int m = 0; m < NV; m++) {
            const int idx = m * 192 + u;
            if (idx < C * D / 4) {
                const int s = idx >> 5, j4 = (idx & 31) * 4;
                __half2 * d = (__half2 *) (Vs + s * GDNC_BR + j4);
                d[0] = __floats2half2_rn(vr[m].x, vr[m].y);
                d[1] = __floats2half2_rn(vr[m].z, vr[m].w);
            }
        }
    }
    __syncthreads();

    // W = T Kb and U0 = T Vs (.row B): warp w owns rows [8w, 8w + 8), four column quarters
    {
        float wa[4][8] = {}, ua[4][8] = {};
        const int nks = 2 * (w + 1); // T is lower triangular
#pragma unroll 2
        for (int ks = 0; ks < C / 4; ks++) {
            if (ks < nks) {
                const uint2 a = *(const uint2 *) (Ts + (8 * w + r) * GDNC_TS + 4 * ks);
#pragma unroll
                for (int qq = 0; qq < 4; qq++) {
                    const int n = 32 * qq + 16 * hi + 4 * qp;
                    const uint2 bk = *(const uint2 *) (Kb + (4 * ks + sr) * GDNC_BR + n);
                    const uint2 bv = *(const uint2 *) (Vs + (4 * ks + sr) * GDNC_BR + n);
                    gdnc_mma<false, true>(wa[qq], a.x, a.y, bk.x, bk.y);
                    gdnc_mma<false, true>(ua[qq], a.x, a.y, bv.x, bv.y);
                }
            }
        }
        // U0 straight out in gdn_chunk_state's accumulator order: [column group][row block][lane][8]
#pragma unroll
        for (int qq = 0; qq < 4; qq++) {
            float4 * d = (float4 *) (rU + ((qq * 8 + w) * 32 + lane) * 8);
            d[0] = make_float4(ua[qq][0], ua[qq][1], ua[qq][2], ua[qq][3]);
            d[1] = make_float4(ua[qq][4], ua[qq][5], ua[qq][6], ua[qq][7]);
        }
        __syncthreads(); // Vs becomes the W staging
#pragma unroll
        for (int qq = 0; qq < 4; qq++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int t = 8 * w + gdnc_crow(lane, e);
                const int i = 32 * qq + gdnc_bcol(qp, gdnc_ccol(lane, e));
                Ws[gdnc_apos<D>(t, i)] = __float2half_rn(wa[qq][e]);
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < C * D / 8; idx += 256) {
        *(uint4 *) (rW + idx * 8) = *(const uint4 *) (Ws + idx * 8);
    }
}

// gdn_chunk_state: block = 8 warps, (column group cg of 32 value columns, head, sequence)
#define GDNC_SP 48 // Ss[i][j] and Us[s][j] pitch, halves (a .row B: 24 words)

__global__ void __launch_bounds__(256, 3) gdn_chunk_state(const gdn_chunk_args p) {
    constexpr int C = GDN_CHUNK, D = GDN_CHUNK_D, NC = GDN_CHUNK_NCOL;
    __shared__ __align__(16) __half Ss[D * GDNC_SP]; // S_0 as fp16, [i][j]
    __shared__ __align__(16) __half Us[C * GDNC_SP]; // U as fp16, [s][j]

    const int cg = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int sr = lane & 3, hi = (lane >> 4) & 1;
    const int r  = sr | (hi << 2);
    const int qp = (lane >> 2) & 3;
    const int nb = 16 * hi + 4 * qp; // this lane's four B columns

    const int64_t sh = (int64_t) seq * p.H + h;
    const float * sin  = p.state_in  + sh * D * D;
    float *       sout = p.state_out + sh * D * D;

    // the state slice: rows i in [16w, 16w + 16) as two 8-row blocks, all 32 columns
    float sa[2][8];
#pragma unroll
    for (int b = 0; b < 2; b++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int i  = 16 * w + 8 * b + gdnc_crow(lane, e);
            const int jl = gdnc_bcol(qp, gdnc_ccol(lane, e));
            sa[b][e] = sin[(int64_t) (NC * cg + jl) * D + i];
            Ss[i * GDNC_SP + jl] = __float2half_rn(sa[b][e]);
        }
    }
    __syncthreads();

    for (int chunk = 0; chunk < p.n_chunks; chunk++) {
        const int64_t rec = sh * p.n_chunks + chunk;
        const __half * rW = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_W;
        const __half * rQ = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_QG;
        const __half * rK = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_KT;
        const __half * rP = p.rec_h + rec * GDN_CHUNK_REC_H + GDN_CHUNK_OFF_P;
        const float  * rU = p.rec_f + rec * GDN_CHUNK_REC_F + ((cg * 8 + w) * 32 + lane) * 8;

        // the next chunk's record towards L2: a quarter of the shared part per column group, and this group's U0
        if (chunk + 1 < p.n_chunks) {
            const char * nh = (const char *) (p.rec_h + (rec + 1) * GDN_CHUNK_REC_H);
            const char * nf = (const char *) (p.rec_f + (rec + 1) * GDN_CHUNK_REC_F + cg * 8 * 32 * 8);
            constexpr int lines_h = GDN_CHUNK_REC_H * 2 / 128 / 4; // 112
            if (tid < lines_h) {
                asm volatile("prefetch.global.L2 [%0];" :: "l"(nh + (cg * lines_h + tid) * 128));
            } else if (tid < lines_h + 64) {
                asm volatile("prefetch.global.L2 [%0];" :: "l"(nf + (tid - lines_h) * 128));
            }
        }

        // W S_0 and diag(Gamma) Q S_0: rows t in [8w, 8w + 8)
        float ua[8] = {}, oa[8] = {};
#pragma unroll 4
        for (int k2 = 0; k2 < D / 8; k2++) {
            const uint4 aw = *(const uint4 *) (rW + (w * (D / 8) + k2) * 64 + r * 8);
            const uint4 aq = *(const uint4 *) (rQ + (w * (D / 8) + k2) * 64 + r * 8);
            const uint2 b0 = *(const uint2 *) (Ss + (8 * k2 + sr)     * GDNC_SP + nb);
            const uint2 b1 = *(const uint2 *) (Ss + (8 * k2 + 4 + sr) * GDNC_SP + nb);
            gdnc_mma<false, true>(ua, aw.x, aw.y, b0.x, b0.y);
            gdnc_mma<false, true>(oa, aq.x, aq.y, b0.x, b0.y);
            gdnc_mma<false, true>(ua, aw.z, aw.w, b1.x, b1.y);
            gdnc_mma<false, true>(oa, aq.z, aq.w, b1.x, b1.y);
        }
        {
            const float4 u0 = *(const float4 *) (rU);
            const float4 u1 = *(const float4 *) (rU + 4);
            const float u[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int s  = 8 * w + gdnc_crow(lane, e);
                const int jl = gdnc_bcol(qp, gdnc_ccol(lane, e));
                Us[s * GDNC_SP + jl] = __float2half_rn(u[e] - ua[e]);
            }
        }
        const float gC = p.rec_g[rec];
        __syncthreads();

#pragma unroll
        for (int b = 0; b < 2; b++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                sa[b][e] *= gC;
            }
        }
        // P U and K'^T U
#pragma unroll 2
        for (int k2 = 0; k2 < C / 8; k2++) {
            const uint4 ak0 = *(const uint4 *) (rK + (((2 * w)     * (C / 8) + k2) * 8 + 4 * hi + sr) * 8);
            const uint4 ak1 = *(const uint4 *) (rK + (((2 * w + 1) * (C / 8) + k2) * 8 + 4 * hi + sr) * 8);
            const uint2 b0 = *(const uint2 *) (Us + (8 * k2 + sr)     * GDNC_SP + nb);
            const uint2 b1 = *(const uint2 *) (Us + (8 * k2 + 4 + sr) * GDNC_SP + nb);
            gdnc_mma<true, true>(sa[0], ak0.x, ak0.y, b0.x, b0.y);
            gdnc_mma<true, true>(sa[1], ak1.x, ak1.y, b0.x, b0.y);
            gdnc_mma<true, true>(sa[0], ak0.z, ak0.w, b1.x, b1.y);
            gdnc_mma<true, true>(sa[1], ak1.z, ak1.w, b1.x, b1.y);
            if (k2 <= w) { // P is lower triangular
                const uint4 ap = *(const uint4 *) (rP + (w * (C / 8) + k2) * 64 + r * 8);
                gdnc_mma<false, true>(oa, ap.x, ap.y, b0.x, b0.y);
                gdnc_mma<false, true>(oa, ap.z, ap.w, b1.x, b1.y);
            }
        }

        // outputs
        const int t0 = chunk * C;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int t  = t0 + 8 * w + gdnc_crow(lane, e);
            const int jl = gdnc_bcol(qp, gdnc_ccol(lane, e));
            if (t < p.n_c) {
                p.dst[(((int64_t) seq * p.n_tokens + t) * p.H + h) * D + NC * cg + jl] = oa[e] * p.scale;
            }
        }
        // the next chunk's state operand (every warp is past its reads of Ss: barrier above)
#pragma unroll
        for (int b = 0; b < 2; b++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int i  = 16 * w + 8 * b + gdnc_crow(lane, e);
                const int jl = gdnc_bcol(qp, gdnc_ccol(lane, e));
                Ss[i * GDNC_SP + jl] = __float2half_rn(sa[b][e]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int b = 0; b < 2; b++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int i  = 16 * w + 8 * b + gdnc_crow(lane, e);
            const int jl = gdnc_bcol(qp, gdnc_ccol(lane, e));
            sout[(int64_t) (NC * cg + jl) * D + i] = sa[b][e];
        }
    }
}

static size_t gdn_chunk_scratch_bytes(const int n_seqs, const int H, const int n_chunks) {
    const size_t n = (size_t) n_seqs * H * n_chunks;
    return n * (GDN_CHUNK_REC_H * sizeof(__half) + GDN_CHUNK_REC_F * sizeof(float) + sizeof(float));
}

// p.rec_* are set by the caller from one scratch allocation of gdn_chunk_scratch_bytes
template <bool stub_solve = false>
static void gdn_chunk_launch(const gdn_chunk_args & p, const int n_seqs, cudaStream_t stream) {
    gdn_chunk_prep<stub_solve><<<dim3(p.H, p.n_chunks, n_seqs), 256, GDNC_SM_PREP, stream>>>(p);
    gdn_chunk_state<<<dim3(GDN_CHUNK_D / GDN_CHUNK_NCOL, p.H, n_seqs), 256, 0, stream>>>(p);
}
