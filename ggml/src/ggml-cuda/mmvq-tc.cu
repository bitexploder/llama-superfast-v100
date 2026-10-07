// Volta tensor-core path for the verify's multi-token dense K-quant GEMVs (toggle LLAMA_MMVQ_TC).
//
// At 3 to 8 tokens the dp4a kernels issue a dot product per (row, column) and are instruction-bound on Volta.
// Here each warp takes 32 rows, dequantizes one superblock of each row to fp16 in registers and runs the
// m8n8k4 tiles of mma.cuh with the activations as the B operand, so 8 columns cost the same as one.
//
// Numerics (not bit-identical to the dp4a path, and closer to fp64):
//  - The activations stay fp16 instead of q8_1. A prep kernel (it replaces quantize_q8_1 for these products)
//    scales each token by a power of two so its max |x| lands in [2^14, 2^15): only the exponent moves, the
//    scale is undone exactly in the epilogue. A token with a non-finite value is quantized to q8_1 instead and
//    that column is computed with the dp4a dot product (vecdotq.cuh), as the dp4a path would.
//  - Weights enter the mma as exact small integers: sc*q for Q5_K (at most 63*31 = 1953 < 2048), built from
//    the 0x6400 magic (1024 + q) with one HFMA2. The superblock scale d multiplies each superblock's fp32 partial
//    sum, and the dmin*m term is applied in fp32 from per-32 sums of the fp16 activations, as the dp4a path does.
//  - Within a superblock the k order is permuted so that one 32-bit qs word feeds one mma pair; the prep kernel
//    writes the activations in the same order.

#include "mmvq-tc.cuh"
#include "mma.cuh"
#include "vecdotq.cuh"

#include <cfloat>

using namespace ggml_cuda_mma;

#define MMVQ_TC_WARPS        4   // warps per block, each takes 32 rows at a time
#define MMVQ_TC_MIN_ROWS     65536 // see ggml_cuda_mmvq_tc_use
#define MMVQ_TC_PREP_THREADS 1024
#define MMVQ_TC_PREP_VALS    8   // so K is at most 8192

static int ggml_cuda_mmvq_tc_mask() {
    static const int mask = [] {
        const char * e = getenv("LLAMA_MMVQ_TC");
        return e == nullptr ? GGML_CUDA_MMVQ_TC_ALL : atoi(e);
    }();
    return mask;
}

static int ggml_cuda_mmvq_tc_bit(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_K: return GGML_CUDA_MMVQ_TC_Q5_K;
        default:             return 0;
    }
}

bool ggml_cuda_mmvq_tc_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int device) {
    if (!(ggml_cuda_mmvq_tc_mask() & ggml_cuda_mmvq_tc_bit(src0->type))) {
        return false;
    }
    if (!volta_mma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    if (dst->op != GGML_OP_MUL_MAT || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[1] < 3 || src1->ne[1] > 8 || src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    // One warp takes 32 rows and walks all of K, so the kernel needs many row tiles to hide its latency. Measured on
    // a V100: the 248,320-row output head gains at 3 to 8 tokens (at 2 it is 2% slower), while every
    // product up to 12,288 rows (attn_qkv, attn_gate, ssm_out, attn_output, attn_k, the shared expert) is slower.
    if (src0->ne[1] < MMVQ_TC_MIN_ROWS) {
        return false;
    }
    return ggml_is_contiguous(src0) && src0->ne[0] % QK_K == 0 && src0->ne[0] <= MMVQ_TC_PREP_THREADS*MMVQ_TC_PREP_VALS && src0->ne[1] % 32 == 0 && src1->ne[0] == src0->ne[0] &&
        src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && dst->ne[0] == src0->ne[1];
}

// permuted position (0..255) of superblock element k: the kernel's chunk k/8 is one mma pair fed by one qs word
template <ggml_type type>
static __device__ __forceinline__ int mmvq_tc_pos(const int k) {
    static_assert(type == GGML_TYPE_Q5_K, "bad type");
    // qs byte 32*il + l holds k = 64*il + l (low nibble) and 64*il + 32 + l (high nibble); a qs word holds l = 4*w..4*w+3,
    // its halves go to the tile as bytes (0, 2), (1, 3) of the low nibbles, then of the high nibbles
    const int il = k / 64, hi = (k % 64) / 32, l = k % 32, b = l % 4;
    return (il*8 + l/4)*8 + hi*4 + (((b & 1) << 1) | (b >> 1));
}

// One block per token: fp32 activations -> fp16 in the kernel's order, scaled by a power of two so max |x| is in
// [2^14, 2^15), plus the sum of each 32 fp16 values (for the dmin*m term). xscale[c] is the factor that undoes it,
// or 0 if the token holds a non-finite value; then the token is quantized to q8_1 instead, as quantize_q8_1 does.
// Each thread keeps its (at most MMVQ_TC_PREP_VALS) values in registers, so the activations are read once.
template <ggml_type type>
static __global__ void __launch_bounds__(MMVQ_TC_PREP_THREADS)
mmvq_tc_prep(const float * __restrict__ x, const int64_t stride_col_x, const int K, const int ncols,
             half * __restrict__ xh, float * __restrict__ xsum, float * __restrict__ xscale, block_q8_1 * __restrict__ q8) {
    constexpr int nwarps = MMVQ_TC_PREP_THREADS/WARP_SIZE;
    const int c = blockIdx.x, lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    x += c*stride_col_x;

    // value i of this thread is k = i*MMVQ_TC_PREP_THREADS + threadIdx.x, so each warp holds whole 32-blocks
    float v[MMVQ_TC_PREP_VALS];
    float amax = 0.0f;
    bool  bad  = false;
#pragma unroll
    for (int i = 0; i < MMVQ_TC_PREP_VALS; ++i) {
        const int k = i*MMVQ_TC_PREP_THREADS + threadIdx.x;
        v[i] = k < K ? x[k] : 0.0f;
        bad  = bad || !(fabsf(v[i]) <= FLT_MAX);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    __shared__ float s_amax[nwarps];
    __shared__ int   s_bad[nwarps];
    amax = warp_reduce_max(amax);
    bad  = __any_sync(0xFFFFFFFF, bad);
    if (lane == 0) {
        s_amax[warp] = amax;
        s_bad[warp]  = bad;
    }
    __syncthreads();
    amax = lane < nwarps ? s_amax[lane] : 0.0f;
    bad  = __any_sync(0xFFFFFFFF, lane < nwarps && s_bad[lane]);
    amax = warp_reduce_max(amax);

    if (bad) {
        block_q8_1 * y = q8 + (int64_t) c*(K/QK8_1);
#pragma unroll
        for (int i = 0; i < MMVQ_TC_PREP_VALS; ++i) {
            const int k = i*MMVQ_TC_PREP_THREADS + threadIdx.x;
            if (i*MMVQ_TC_PREP_THREADS + warp*WARP_SIZE >= K) {
                break;
            }
            const float am  = warp_reduce_max<QK8_1>(fabsf(v[i]));
            const float sum = warp_reduce_sum<QK8_1>(v[i]);
            const float d = am / 127.0f;
            y[k/QK8_1].qs[lane] = am == 0.0f ? 0 : roundf(v[i] / d);
            if (lane == 0) {
                y[k/QK8_1].ds = make_half2(d, sum);
            }
        }
        if (threadIdx.x == 0) {
            xscale[c] = 0.0f;
        }
        return;
    }

    // amax = m*2^e with m in [0.5, 1); scaling by 2^(15 - e) puts it in [2^14, 2^15)
    int e = 0;
    frexpf(amax, &e);
    const int sh = amax == 0.0f ? 0 : max(e - 15, -100);
    const float inv = __int_as_float((127 - sh) << 23); // 2^-sh, exact

    half * y = xh + (int64_t) c*K;
#pragma unroll
    for (int i = 0; i < MMVQ_TC_PREP_VALS; ++i) {
        const int k = i*MMVQ_TC_PREP_THREADS + threadIdx.x;
        if (i*MMVQ_TC_PREP_THREADS + warp*WARP_SIZE >= K) {
            break;
        }
        const half h = __float2half_rn(v[i] * inv);
        y[(k/QK_K)*QK_K + mmvq_tc_pos<type>(k % QK_K)] = h;
        const float s = warp_reduce_sum(__half2float(h));
        if (lane == 0) {
            xsum[(k/32)*ncols + c] = s;
        }
    }
    if (threadIdx.x == 0) {
        xscale[c] = __int_as_float((127 + sh) << 23); // 2^sh
    }
}

static __device__ __forceinline__ half2    mmvq_tc_u2h(const uint32_t u) { return *(const half2 *) &u; }

static __device__ __forceinline__ uint4 mmvq_tc_ldg(const uint4 * p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

// The columns whose token held a non-finite value, with the dp4a dot product of mul_mat_vec_q.
template <ggml_type type, int ncols>
static __device__ __noinline__ void mmvq_tc_fallback(
        const void * __restrict__ vx, const int64_t stride_row, const int nsb, const int row0,
        const float * __restrict__ xscale, const block_q8_1 * __restrict__ q8, float * __restrict__ dst, const int64_t stride_col_dst) {
    static_assert(type == GGML_TYPE_Q5_K, "bad type");
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    const int lane = threadIdx.x % WARP_SIZE;
    for (int c = 0; c < ncols; ++c) {
        if (xscale[c] != 0.0f) {
            continue;
        }
        const block_q8_1 * y = q8 + (int64_t) c*nsb*(QK_K/QK8_1);
        for (int i = 0; i < 32; ++i) {
            float sum = 0.0f;
            for (int kbx = lane/(qi/vdr); kbx < nsb; kbx += WARP_SIZE/(qi/vdr)) {
                const int kqs = vdr*(lane % (qi/vdr));
                sum += vec_dot_q5_K_q8_1(vx, y + kbx*(QK_K/QK8_1), (row0 + i)*stride_row + kbx, kqs);
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                dst[c*stride_col_dst + row0 + i] = sum;
            }
        }
    }
}

template <int ncols>
static __global__ void __launch_bounds__(MMVQ_TC_WARPS*WARP_SIZE)
mmvq_tc_q5_K(const block_q5_K * __restrict__ vx, const int64_t stride_row, const int nsb, const int nrows,
             const half * __restrict__ xh, const float * __restrict__ xsum, const float * __restrict__ xscale,
             const block_q8_1 * __restrict__ q8, float * __restrict__ dst, const int64_t stride_col_dst) {
#if defined(VOLTA_MMA_AVAILABLE)
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;
    constexpr int WB = sizeof(block_q5_K)/sizeof(uint4); // 11

    // each warp stages its 32 rows x one superblock with coalesced 16-byte loads, then each thread reads its row
    __shared__ uint4 wbuf[MMVQ_TC_WARPS][WARP_SIZE*WB];
    __shared__ float mcs[MMVQ_TC_WARPS][WARP_SIZE][ncols]; // each row's dmin*m term, for the epilogue

    // mma.cuh's tiles index lanes by threadIdx.x, so the warps of a block are along y
    const int t = threadIdx.x, warp = threadIdx.y;
    const int n = tile_B::get_i(0); // the activation column this thread loads for B
    const int K = nsb*QK_K;
    const half * xn = xh + (int64_t) (n < ncols ? n : 0)*K;

    bool any_bad = false;
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        any_bad = any_bad || xscale[c] == 0.0f;
    }

    uint4 * buf = wbuf[warp];
    const int ntiles = nrows/32;
    for (int it = blockIdx.x*MMVQ_TC_WARPS + warp; it < ntiles; it += gridDim.x*MMVQ_TC_WARPS) {
        tile_C D;
        float mc[ncols] = {0.0f};
        const uint4 * wt = (const uint4 *) (vx + (int64_t) it*32*stride_row);
        const int row_u4 = stride_row*WB;

        int   offs[WB];
        uint4 nx[WB];
#pragma unroll
        for (int i = 0; i < WB; ++i) {
            const int idx = i*WARP_SIZE + t;
            offs[i] = (idx/WB)*row_u4 + idx % WB;
            nx[i] = mmvq_tc_ldg(wt + offs[i]);
        }

#pragma unroll 1
        for (int sb = 0; sb < nsb; ++sb) {
#pragma unroll
            for (int i = 0; i < WB; ++i) {
                buf[i*WARP_SIZE + t] = nx[i];
            }
            __syncwarp();
            uint4 b[WB];
#pragma unroll
            for (int i = 0; i < WB; ++i) {
                b[i] = buf[t*WB + i];
            }
            __syncwarp();
            if (sb + 1 < nsb) {
#pragma unroll
                for (int i = 0; i < WB; ++i) {
                    nx[i] = mmvq_tc_ldg(wt + offs[i] + (sb + 1)*WB);
                }
            }

            const float2 dm = __half22float2(mmvq_tc_u2h(b[0].x));
            uint8_t q[12];
            memcpy(q, &b[0].y, 12);
            half2 s2[8], z2[8];
            float mj[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int sc, m;
                if (j < 4) {
                    sc = q[j] & 63;
                    m  = q[j + 4] & 63;
                } else {
                    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    m  = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
                }
                s2[j] = __half2half2(__int2half_rn(sc));
                z2[j] = __half2half2(__int2half_rn(-1024*sc));
                mj[j] = (float) m;
            }
            const float * xs = xsum + sb*8*ncols;
#pragma unroll
            for (int c = 0; c < ncols; ++c) {
                float a = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    a += mj[j] * xs[j*ncols + c];
                }
                mc[c] += dm.y * a;
            }

            // sc*q as exact integers: (1024 + q)*sc - 1024*sc in one HFMA2
            tile_C Ds;
            const uint32_t * qh = (const uint32_t *) &b[1];
            const uint32_t * qs = (const uint32_t *) &b[3];
#pragma unroll
            for (int il = 0; il < 4; ++il) {
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    const uint32_t v = qs[il*8 + w];
                    const uint32_t H = qh[w] >> (2*il);
                    const uint32_t P0 = ( v        & 0x000F000F) | ((H << 4) & 0x00100010) | 0x64006400;
                    const uint32_t P1 = ((v >>  8) & 0x000F000F) | ((H >> 4) & 0x00100010) | 0x64006400;
                    const uint32_t P2 = ((v >>  4) & 0x000F000F) | ((H << 3) & 0x00100010) | 0x64006400;
                    const uint32_t P3 = ((v >> 12) & 0x000F000F) | ((H >> 5) & 0x00100010) | 0x64006400;
                    tile_A A;
                    A.x[0] = __hfma2(mmvq_tc_u2h(P0), s2[2*il + 0], z2[2*il + 0]);
                    A.x[1] = __hfma2(mmvq_tc_u2h(P1), s2[2*il + 0], z2[2*il + 0]);
                    A.x[2] = __hfma2(mmvq_tc_u2h(P2), s2[2*il + 1], z2[2*il + 1]);
                    A.x[3] = __hfma2(mmvq_tc_u2h(P3), s2[2*il + 1], z2[2*il + 1]);
                    tile_B B;
                    if (n < ncols) {
                        *(uint4 *) B.x = __ldg((const uint4 *) (xn + (sb*32 + il*8 + w)*8));
                    }
                    mma(Ds, A, B);
                }
            }
            // d of the rows this thread holds in D
            const float d0 = __shfl_sync(0xFFFFFFFF, dm.x, tile_C::get_i(0));
            const float d2 = __shfl_sync(0xFFFFFFFF, dm.x, tile_C::get_i(2));
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                D.x[l] += (l & 2 ? d2 : d0) * Ds.x[l];
            }
        }

#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            mcs[warp][t][c] = mc[c];
        }
        __syncwarp();
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = tile_C::get_i(l), j = tile_C::get_j(l);
            if (j < ncols) {
                const float s = xscale[j];
                if (s != 0.0f) {
                    dst[j*stride_col_dst + it*32 + i] = (D.x[l] - mcs[warp][i][j]) * s;
                }
            }
        }
        __syncwarp();
        if (any_bad) {
            mmvq_tc_fallback<GGML_TYPE_Q5_K, ncols>(vx, stride_row, nsb, it*32, xscale, q8, dst, stride_col_dst);
        }
    }
#else
    GGML_UNUSED_VARS(vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, dst, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

template <int ncols>
static void mmvq_tc_launch_q5_K(const int device, const block_q5_K * vx, const int64_t stride_row, const int nsb, const int nrows,
        const half * xh, const float * xsum, const float * xscale, const block_q8_1 * q8, float * dst, const int64_t stride_col_dst,
        cudaStream_t stream) {
    // persistent blocks: as many as fit on the device, or fewer for a small matrix
    static int occ[GGML_CUDA_MAX_DEVICES] = {};
    if (occ[device] == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ[device], mmvq_tc_q5_K<ncols>, MMVQ_TC_WARPS*WARP_SIZE, 0));
        occ[device] = std::max(occ[device], 1);
    }
    const int ntiles = nrows/32;
    const int grid   = std::min((ntiles + MMVQ_TC_WARPS - 1)/MMVQ_TC_WARPS, occ[device]*ggml_cuda_info().devices[device].nsm);
    mmvq_tc_q5_K<ncols><<<grid, dim3(WARP_SIZE, MMVQ_TC_WARPS), 0, stream>>>
        (vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, dst, stride_col_dst);
}

void ggml_cuda_mul_mat_vec_q_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int     device = ctx.device;
    const int64_t K      = src0->ne[0];
    const int     nsb    = (int) (K/QK_K);
    const int     nrows  = (int) src0->ne[1];
    const int     ncols  = (int) src1->ne[1];
    cudaStream_t  stream = ctx.stream();

    // one scratch allocation: fp16 activations, per-32 sums, per-token scales, and q8_1 for a non-finite token
    const size_t sz_xh   = GGML_PAD(ncols*K*sizeof(half), 256);
    const size_t sz_xsum = GGML_PAD(ncols*(K/32)*sizeof(float), 256);
    const size_t sz_sc   = 256;
    const size_t sz_q8   = ncols*(K/QK8_1)*sizeof(block_q8_1);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool(), sz_xh + sz_xsum + sz_sc + sz_q8);
    half       * xh     = (half       *) scratch.get();
    float      * xsum   = (float      *) (scratch.get() + sz_xh);
    float      * xscale = (float      *) (scratch.get() + sz_xh + sz_xsum);
    block_q8_1 * q8     = (block_q8_1 *) (scratch.get() + sz_xh + sz_xsum + sz_sc);

    const float * x            = (const float *) src1->data;
    const int64_t stride_col_x = src1->nb[1]/sizeof(float);
    float       * y            = (float *) dst->data;
    const int64_t stride_col_y = dst->nb[1]/sizeof(float);

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_K);
    mmvq_tc_prep<GGML_TYPE_Q5_K><<<ncols, MMVQ_TC_PREP_THREADS, 0, stream>>>(x, stride_col_x, (int) K, ncols, xh, xsum, xscale, q8);
    CUDA_CHECK(cudaGetLastError());

    const block_q5_K * vx = (const block_q5_K *) src0->data;
    const int64_t stride_row = src0->nb[1]/sizeof(block_q5_K);
    switch (ncols) {
        case 2: mmvq_tc_launch_q5_K<2>(device, vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, y, stride_col_y, stream); break;
        case 3: mmvq_tc_launch_q5_K<3>(device, vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, y, stride_col_y, stream); break;
        case 4: mmvq_tc_launch_q5_K<4>(device, vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, y, stride_col_y, stream); break;
        case 5: mmvq_tc_launch_q5_K<5>(device, vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, y, stride_col_y, stream); break;
        case 6: mmvq_tc_launch_q5_K<6>(device, vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, y, stride_col_y, stream); break;
        case 7: mmvq_tc_launch_q5_K<7>(device, vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, y, stride_col_y, stream); break;
        case 8: mmvq_tc_launch_q5_K<8>(device, vx, stride_row, nsb, nrows, xh, xsum, xscale, q8, y, stride_col_y, stream); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}
