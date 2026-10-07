// the QSA prompt attention on Volta's mma.m8n8k4 tensor cores.
//
// A port of Strata-V100 (https://github.com/jmnargi/Strata-V100 at a4b679b): prompt_attn_v70_kernel<0> (fp16 KV), its
// Smem70<0>, mma884 and launch70's shared-memory set-up, from src/kernels/cuda/qsa_prompt_attn.cu (upstream PR 600).
//
//   MIT License
//
//   Copyright (c) 2026 Niko1221 and the Strata contributors
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
// The node is qsa-attn.cu's: q F32 [256, n_head, n_tok], k/v BF16 or F16 cache views [256, n_head_kv, n_cells], mask
// F32 [n_sel, n_tok], ids I32 [n_sel, n_tok], dst F32 [256, n_head, n_tok], op_params[0] the scale.
//
// One 128-thread block per (token, KV head) walks the token's n_sel slots in chunks of 32 with an online softmax (base
// 2), with the KV head's 12 query heads as 16 MMA rows (4 zero rows). Strata's kernel as it is, except:
//   - q is read through the node's strides, and scale_log2 = log2(e) * op_params[0];
//   - slot c of token tok is cell ids[tok*n_sel + c] (no page table); its K row is k + cell*nb2 + kvh*nb1, V likewise;
//   - the gather converts BF16 to FP16 (the bits as a float, clamped to +-65504, rounded); F16 rows copy as they are;
//   - the per-slot mask, times log2(e), is added to the slot's score (a -INFINITY slot gets probability 0);
//   - rows h < 12 are written to dst through its strides.
// Volta's m8n8k4 (FP16 in, FP32 accumulate): per warp four independent 8x8x4 products, one per quad-pair (QP q = lanes
// 4q..4q+3 and 4q+16..4q+19; thread j = (lane & 3) + 4 * (lane >> 4) within it). The thread holds row j of A (k 0..3 as
// two registers) and column j of B (k 0..3); C register i is row (lane & 1) + 4 * (lane >> 4) + 2 * ((i >> 1) & 1),
// column (lane & 2) + 4 * (i >> 2) + (i & 1) of its QP's 8x8 tile.
//   q.k: warp w owns dims [64w, 64w+64); QP q takes cells 8q..8q+7 of the 32-cell chunk; 16 k-steps of 4 dims, two
//        8-row blocks (12 heads + 4 zero rows), q as hi + lo halves. The warp's partial goes to S.part[w]; the softmax
//        adds the four groups in a fixed order (deterministic).
//   p.v: warp w owns dims [64w, 64w+64) again; QP q takes the 16 dims 64w+16q.. (two n-tiles of 8); k = the 32 cells
//        in steps of 4; p is split into hi + lo halves.
// K and V share one shared-memory region: K is gathered before the scores and V after the softmax, so
// the block takes 46,928 bytes and two fit in an SM's 96 KiB. The arithmetic is as before.
// No side stream, no host sync; CUDA-graph safe.
#include "qsa-attn-mma.cuh"

#include <atomic>

static constexpr int QAM_HD      = 256;          // head_dim
static constexpr int QAM_G       = 12;           // query heads per KV head
static constexpr int QAM_CH      = 32;           // cells per chunk
static constexpr int QAM_THREADS = 128;          // 4 warps: scores by cell (8 each), p.v by dimension (64 each)
static constexpr int QAM_QS      = QAM_HD + 8;   // q row stride in halves (bank-conflict-free fragment loads)

static __device__ __forceinline__ void mma884(float * c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#ifdef VOLTA_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, b0, b1);
    NO_DEVICE_CODE;   // m8n8k4 is a Volta instruction; the host only routes here on sm_70
#endif // VOLTA_MMA_AVAILABLE
}

static __device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {   // element k in the low half
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t *>(&h);
}

// two BF16 values (low first) as a half2: each widened to FP32, clamped to FP16's range, rounded to nearest
static __device__ __forceinline__ uint32_t bf16x2_to_h2(uint32_t w) {
    const float lo = fminf(fmaxf(__uint_as_float(w << 16),           -65504.0f), 65504.0f);
    const float hi = fminf(fmaxf(__uint_as_float(w & 0xffff0000u),   -65504.0f), 65504.0f);
    return pack_h2(lo, hi);
}

struct qsa_attn_mma_smem {
    static constexpr int KROW = QAM_HD + 8;
    static constexpr int VROW = QAM_HD + 8;
    static_assert(KROW == VROW, "K and V share kv");
    __half qh[16][QAM_QS];
    __half ql[16][QAM_QS];
    __half kv[QAM_CH][KROW];         // the chunk's K rows for the scores, then its V rows for p.v
    float ks[QAM_CH][4];
    float vs[QAM_CH][4];
    float part[4][16][QAM_CH + 4];   // q.k per dim group (scaled by the cell's K scale)
    float s[16][QAM_CH + 4];         // the chunk's probabilities
    float qmax[QAM_THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    float msk[QAM_CH];               // the chunk's slot masks, times log2(e)
    long long row[QAM_CH];           // the chunk's cells (-1: past n_sel)
};

template <bool is_bf16>
__launch_bounds__(QAM_THREADS)
static __global__ void qsa_attn_mma_kernel(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const float * __restrict__ mask, const int32_t * __restrict__ ids, char * __restrict__ attn,
        const int n_sel, const size_t nbq1, const size_t nbq2, const size_t nbk1, const size_t nbk2,
        const size_t nbv1, const size_t nbv2, const size_t nbd1, const size_t nbd2, const float scale_log2) {
    constexpr int HD = QAM_HD, G = QAM_G, CH = QAM_CH, THREADS = QAM_THREADS;
    extern __shared__ __align__(16) unsigned char qsa_attn_mma_smem_raw[];
    qsa_attn_mma_smem & S = *reinterpret_cast<qsa_attn_mma_smem *>(qsa_attn_mma_smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    q    += qi*nbq2 + (size_t) kvh*G*nbq1;
    attn += qi*nbd2 + (size_t) kvh*G*nbd1;
    ids  += (size_t) qi*n_sel;
    mask += (size_t) qi*n_sel;
    k    += kvh*nbk1;
    v    += kvh*nbv1;
    const int n = n_sel;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);   // quad-pair, thread within it
    const int crow = (lane & 1) + 4 * (lane >> 4);                        // C row of register 0 (+2 per register pair)
    const int ccol = lane & 2;                                           // C column of register 0 (+4 per half, +1 per reg)

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(((const float *) (q + (i / HD)*nbq1))[i % HD]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? ((const float *) (q + h*nbq1))[d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[2][2][8];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[rb][nt][i] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            float mk = -INFINITY;
            if (t < nh) {
                r  = ids[c0 + t];
                mk = mask[c0 + t] * 1.4426950408889634f;
            }
            S.row[t] = r;
            S.msk[t] = mk;
        }
        __syncthreads();   // rows ready; the previous chunk's p.v is done with kv, s
        // gather the chunk's K rows (16-byte pieces of 8 values; BF16 converted to FP16 on the way) and the K and V
        // scales (1: fp16 KV); the V rows follow the softmax, into the same kv
        constexpr int PIECES = HD * (int) sizeof(__half) / 16;   // per row
        {
            for (int i = t; i < CH * PIECES; i += THREADS) {
                const int c = i / PIECES, pc = i % PIECES;
                const long long r = S.row[c];
                uint4 kx = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    kx = __ldg(reinterpret_cast<const uint4 *>(k + r*nbk2) + pc);
                    if constexpr (is_bf16) {
                        kx = make_uint4(bf16x2_to_h2(kx.x), bf16x2_to_h2(kx.y), bf16x2_to_h2(kx.z), bf16x2_to_h2(kx.w));
                    }
                }
                *reinterpret_cast<uint4 *>(reinterpret_cast<unsigned char *>(&S.kv[c][0]) + pc * 16) = kx;
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const float a = S.row[c] >= 0 ? 1.0f : 0.0f;
                S.ks[c][g] = a;
                S.vs[c][g] = a;
            }
        }
        __syncthreads();
        // scores: warp w = dim group w (64 dims), QP q = cells 8q..8q+7. The hi and lo halves of q accumulate in
        // separate m8n8k4 chains (tg, tgl), added in FP32 at the end: Volta's tensor cores truncate as they
        // accumulate, and in one chain behind the hi products most of the lo half was lost. The same for p.v below.
        {
            float tg[2][8], tgl[2][8];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) tg[rb][i] = tgl[rb][i] = 0.0f;
            const int cell_b = qp * 8 + jj;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const int d0 = warp * 64 + ks * 4;
                const uint2 raw = *reinterpret_cast<const uint2 *>(&S.kv[cell_b][d0]);
                const uint32_t b0 = raw.x;
                const uint32_t b1 = raw.y;
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const uint2 ah = *reinterpret_cast<const uint2 *>(&S.qh[rb * 8 + jj][d0]);
                    const uint2 al = *reinterpret_cast<const uint2 *>(&S.ql[rb * 8 + jj][d0]);
                    mma884(tg[rb], ah.x, ah.y, b0, b1);
                    mma884(tgl[rb], al.x, al.y, b0, b1);
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = rb * 8 + crow + 2 * ((i >> 1) & 1);
                    const int cell = qp * 8 + ccol + 4 * (i >> 2) + (i & 1);
                    S.part[warp][row][cell] = (tg[rb][i] + tgl[rb][i]) * S.ks[cell][warp];
                }
        }
        __syncthreads();   // every warp is done reading K from kv
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j;
                const float sum4 = ((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c];
                x[j] = c < nh ? sum4 * qdown + S.msk[c] : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        // gather the chunk's V rows into kv, as the K rows above
        for (int i = t; i < CH * PIECES; i += THREADS) {
            const int c = i / PIECES, pc = i % PIECES;
            const long long r = S.row[c];
            uint4 vx = make_uint4(0, 0, 0, 0);
            if (r >= 0) {
                vx = __ldg(reinterpret_cast<const uint4 *>(v + r*nbv2) + pc);
                if constexpr (is_bf16) {
                    vx = make_uint4(bf16x2_to_h2(vx.x), bf16x2_to_h2(vx.y), bf16x2_to_h2(vx.z), bf16x2_to_h2(vx.w));
                }
            }
            *reinterpret_cast<uint4 *>(reinterpret_cast<unsigned char *>(&S.kv[c][0]) + pc * 16) = vx;
        }
        __syncthreads();   // s, alpha and V in kv ready
        // p.v: warp w owns dims [64w, 64w+64), QP q its 16 dims 64w+16q..; the V scale is folded into p relative to
        // the chunk's largest, times 2^14
        {
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, S.vs[c][warp]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[2][2][8], tmpl[2][2][8];   // the hi and lo halves of p in separate chains (see the scores)
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) tmp[rb][nt][i] = tmpl[rb][nt][i] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 4; ++ks) {
                const int cc0 = ks * 4;
                float w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) w[e] = S.vs[cc0 + e][warp] * vup;
                uint2 ah[2], al[2];
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const float4 pv = *reinterpret_cast<const float4 *>(&S.s[rb * 8 + jj][cc0]);
                    const float p0 = pv.x * w[0], p1 = pv.y * w[1], p2 = pv.z * w[2], p3 = pv.w * w[3];
                    ah[rb].x = pack_h2(p0, p1);
                    ah[rb].y = pack_h2(p2, p3);
                    const float2 f01 = __half22float2(*reinterpret_cast<const __half2 *>(&ah[rb].x));
                    const float2 f23 = __half22float2(*reinterpret_cast<const __half2 *>(&ah[rb].y));
                    al[rb].x = pack_h2(p0 - f01.x, p1 - f01.y);
                    al[rb].y = pack_h2(p2 - f23.x, p3 - f23.y);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int d = warp * 64 + qp * 16 + nt * 8 + jj;
                    const __half2 h0 = __halves2half2(S.kv[cc0][d], S.kv[cc0 + 1][d]);
                    const __half2 h1 = __halves2half2(S.kv[cc0 + 2][d], S.kv[cc0 + 3][d]);
                    const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t *>(&h1);
#pragma unroll
                    for (int rb = 0; rb < 2; ++rb) {
                        mma884(tmp[rb][nt], ah[rb].x, ah[rb].y, b0, b1);
                        mma884(tmpl[rb][nt], al[rb].x, al[rb].y, b0, b1);
                    }
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                float a_[2];
                a_[0] = S.alpha[rb * 8 + crow];
                a_[1] = S.alpha[rb * 8 + crow + 2];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        acc[rb][nt][i] = fmaf(acc[rb][nt][i], a_[(i >> 1) & 1], (tmp[rb][nt][i] + tmpl[rb][nt][i]) * vdown);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rb * 8 + crow + 2 * h;
            if (row >= G) continue;
            const float l = S.lsum[row];
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            float * out = (float *) (attn + row*nbd1);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int i = hh * 4 + h * 2;               // registers i, i+1: row h, columns ccol + 4 * hh + {0, 1}
                    const int d = warp * 64 + qp * 16 + nt * 8 + ccol + 4 * hh;
                    *reinterpret_cast<float2 *>(out + d) = make_float2(acc[rb][nt][i] * inv, acc[rb][nt][i + 1] * inv);
                }
        }
}

template <bool is_bf16>
static void qsa_attn_mma_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * ids = dst->src[5];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const float scale_log2 = 1.4426950408889634f * scale;

    const int bytes = (int) sizeof(qsa_attn_mma_smem);
    CUDA_SET_SHARED_MEMORY_LIMIT(qsa_attn_mma_kernel<is_bf16>, bytes);
    {
        // the largest shared carveout, so two blocks fit an SM; once per instantiation and device
        static bool carveout_set[GGML_CUDA_MAX_DEVICES] = { false };
        const int id = ggml_cuda_get_device();
        if (!carveout_set[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(qsa_attn_mma_kernel<is_bf16>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared));
            carveout_set[id] = true;
        }
    }
    static std::atomic<bool> residency_logged{false};
    if (!residency_logged.exchange(true)) {
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, qsa_attn_mma_kernel<is_bf16>, QAM_THREADS, bytes));
        GGML_LOG_WARN("qsa attention mma: smem %d bytes, %d blocks per SM\n", bytes, blocks);
    }

    const dim3 grid((unsigned) q->ne[2], (unsigned) k->ne[1]);
    qsa_attn_mma_kernel<is_bf16><<<grid, QAM_THREADS, bytes, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data, (const float *) m->data,
        (const int32_t *) ids->data, (char *) dst->data, (int) ids->ne[0],
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], dst->nb[1], dst->nb[2], scale_log2);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_qsa_attn_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    GGML_ASSERT(q->ne[0] == QAM_HD && q->ne[1] == k->ne[1]*QAM_G);
    if (k->type == GGML_TYPE_BF16) {
        qsa_attn_mma_launch<true>(ctx, dst);
    } else {
        qsa_attn_mma_launch<false>(ctx, dst);
    }
}
