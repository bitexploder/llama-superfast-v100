#include "mmv-hc.cuh"
#include "mmvf.cuh"
#include "unary.cuh"

// mul_mat_vec_f gives every output row its own block. For the 320-wide up-projection that is one
// 4-byte load per thread followed by a block-wide reduction, about 220 GB/s on a V100, and every
// block re-reads the whole FP32 activation vector. These kernels issue all of a thread's weight
// loads before any arithmetic and read the activations once per block for several rows.
//
// They reproduce mul_mat_vec_f's arithmetic exactly: the same pairs feed the same FP32
// accumulators in the same order, and the same butterfly trees combine them, so the results are
// bit-identical to the generic kernel's. Only the way the weights are read changes.

static constexpr int HC_BLOCK_SIZE    = 256;
// mul_mat_vec_f serves BF16 up to 8 columns on these devices with a block size that depends only on the row
// length, so each column's arithmetic is the same at every width (5-token verifies had fallen
// back to it at 2.4x the time, and lost the fused epilogue)
static constexpr int HC_MAX_NCOLS_DST = 8;

// short rows (hc_lr = 320). mul_mat_vec_f uses a 160-thread block: thread t takes pair t, each of
// its 5 warps reduces 32 pairs with a butterfly, and a final butterfly runs over the 5 warp sums
// padded with zeros. Here 8 lanes share a row, and lane l holds pairs l, l+8, l+16 and l+24 of
// every 32-pair span, so the span's first two butterfly levels (xor 16, xor 8) are lane-local
// and its last three are the shuffles of warp_reduce_sum<8>.
static constexpr int HC_SHORT_K     = 320;
static constexpr int HC_SHORT_LANES = 8;
static constexpr int HC_SHORT_SPANS = HC_SHORT_K/64;

// long rows (hc_dim = 4*2560). mul_mat_vec_f uses a 256-thread block: thread t accumulates pairs
// t, t+256, ... in order, then each of its 8 warps reduces with a butterfly, and a final butterfly
// runs over the 8 warp sums padded with zeros. This kernel keeps that assignment and order.
static constexpr int HC_LONG_K      = 10240;
static constexpr int HC_LONG_PAIRS  = HC_LONG_K/2/HC_BLOCK_SIZE;

// n_embd-wide rows (2560): mul_mat_vec_f picks the same 256-thread block (5 pairs per thread), so the
// long-row kernel reproduces it too. These are the untagged BF16 products: the router, the indexer
// projections, GDN alpha/beta, the shared-expert gate and PLE.
static constexpr int HC_EMBD_K      = 2560;
static constexpr int HC_EMBD_PAIRS  = HC_EMBD_K/2/HC_BLOCK_SIZE;

static __device__ __forceinline__ float hc_bf16_lo(const uint32_t w) {
    return __uint_as_float(w << 16);
}

static __device__ __forceinline__ float hc_bf16_hi(const uint32_t w) {
    return __uint_as_float(w & 0xFFFF0000u);
}

// one BF16 pair into an accumulator, as mul_mat_vec_f's two ggml_cuda_mad calls do
static __device__ __forceinline__ float hc_mad2(const uint32_t w, const float2 y, float acc) {
    acc = fmaf(hc_bf16_lo(w), y.x, acc);
    acc = fmaf(hc_bf16_hi(w), y.y, acc);
    return acc;
}

// the SCALE and UNARY nodes' own arithmetic (scale_f32, ggml_cuda_op_silu_single, op_sigmoid), applied
// to the value the product would have stored, so the fused result is bit-identical to the chain's
template <int epi>
static __device__ __forceinline__ float hc_epilogue(float v, const ggml_cuda_hc_epilogue p) {
    if constexpr (epi == GGML_CUDA_HC_EPI_SCALE_SILU) {
        v = p.s0 * v + p.b0;
        return ggml_cuda_op_silu_single(v);
    } else if constexpr (epi == GGML_CUDA_HC_EPI_SCALE_SIGMOID_SCALE) {
        v = p.s0 * v + p.b0;
        v = 1.0f / (1.0f + expf(-v));
        return p.s1 * v + p.b1;
    } else {
        return v;
    }
}

template <int ncols_dst>
static __global__ void __launch_bounds__(HC_BLOCK_SIZE)
mul_mat_vec_hc_short(
        const nv_bfloat16 * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst) {
    constexpr int npairs         = HC_SHORT_K/2;
    constexpr int rows_per_block = HC_BLOCK_SIZE/HC_SHORT_LANES;

    __shared__ float2 y_s[ncols_dst*npairs];

    const int tid  = threadIdx.x;
    const int lane = tid % HC_SHORT_LANES;
    const int row  = blockIdx.x*rows_per_block + tid/HC_SHORT_LANES;

    // weights first, so their loads are in flight while the activations are staged.
    // w[s][q] is pair 32*s + 8*q + lane of the row.
    uint32_t w[HC_SHORT_SPANS][4] = {};
    if (row < nrows) {
        const uint32_t * x_row = (const uint32_t *) (x + row*stride_row);
#pragma unroll
        for (int s = 0; s < HC_SHORT_SPANS; ++s) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                w[s][q] = x_row[32*s + 8*q + lane];
            }
        }
    }

    for (int i = tid; i < ncols_dst*npairs; i += HC_BLOCK_SIZE) {
        const int j = i / npairs;
        y_s[i] = ((const float2 *) (y + j*stride_col_y))[i - j*npairs];
    }
    __syncthreads();

    float sum[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        float span[HC_SHORT_SPANS];
#pragma unroll
        for (int s = 0; s < HC_SHORT_SPANS; ++s) {
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                v[q] = hc_mad2(w[s][q], y_s[j*npairs + 32*s + 8*q + lane], 0.0f);
            }
            span[s] = (v[0] + v[2]) + (v[1] + v[3]);
            span[s] = warp_reduce_sum<HC_SHORT_LANES>(span[s]);
        }
        // the generic kernel's final butterfly over [span0..span4, 0 x 27]
        static_assert(HC_SHORT_SPANS == 5, "the final reduction below is written out for 5 spans");
        sum[j] = ((span[0] + span[4]) + span[2]) + (span[1] + span[3]);
    }

    if (row >= nrows) {
        return;
    }
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        if (lane == j) {
            dst[j*stride_col_dst + row] = sum[j];
        }
    }
}

// one block's rows of the long-row product; block is the block's index within its own matrix
template <int npairs, int ncols_dst, int rows_per_block, int epi>
static __device__ __forceinline__ void mul_mat_vec_hc_long_block(
        const nv_bfloat16 * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const ggml_cuda_hc_epilogue ep, const int block) {
    constexpr int nwarps = HC_BLOCK_SIZE/WARP_SIZE;
    constexpr int nsums  = rows_per_block*ncols_dst;

    __shared__ float sum_s[nwarps][nsums];

    const int tid  = threadIdx.x;
    const int row0 = block*rows_per_block;

    uint32_t w[rows_per_block][npairs] = {};
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        if (row0 + r < nrows) {
            const uint32_t * x_row = (const uint32_t *) (x + (row0 + r)*stride_row);
#pragma unroll
            for (int k = 0; k < npairs; ++k) {
                w[r][k] = x_row[k*HC_BLOCK_SIZE + tid];
            }
        }
    }

    // each activation pair is read once and used for every row of the block
    float sum[rows_per_block][ncols_dst] = {{0.0f}};
#pragma unroll
    for (int k = 0; k < npairs; ++k) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 yv = ((const float2 *) (y + j*stride_col_y))[k*HC_BLOCK_SIZE + tid];
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                sum[r][j] = hc_mad2(w[r][k], yv, sum[r][j]);
            }
        }
    }

    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float s = warp_reduce_sum(sum[r][j]);
            if (lane == 0) {
                sum_s[warp][r*ncols_dst + j] = s;
            }
        }
    }
    __syncthreads();

    if (tid < nsums) {
        float g[nwarps];
#pragma unroll
        for (int i = 0; i < nwarps; ++i) {
            g[i] = sum_s[i][tid];
        }
        // the generic kernel's final butterfly over [g0..g7, 0 x 24]
        static_assert(nwarps == 8, "the final reduction below is written out for 8 warps");
        const float s = ((g[0] + g[4]) + (g[2] + g[6])) + ((g[1] + g[5]) + (g[3] + g[7]));
        const int r = tid / ncols_dst;
        const int j = tid % ncols_dst;
        if (row0 + r < nrows) {
            dst[j*stride_col_dst + row0 + r] = hc_epilogue<epi>(s, ep);
        }
    }
}

template <int npairs, int ncols_dst, int rows_per_block, int epi>
static __global__ void __launch_bounds__(HC_BLOCK_SIZE)
mul_mat_vec_hc_long(
        const nv_bfloat16 * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const ggml_cuda_hc_epilogue ep) {
    mul_mat_vec_hc_long_block<npairs, ncols_dst, rows_per_block, epi>(
        x, y, dst, nrows, stride_row, stride_col_y, stride_col_dst, ep, blockIdx.x);
}

// Several n_embd-wide products of one shared activation in one launch: block b belongs to
// the matrix whose block range holds it and runs exactly the block that matrix's own launch would run. A row's
// result depends only on its own weights and the activations, never on the block layout, so every output is
// bit-identical to the separate launches.
struct mul_mat_vec_hc_group_args {
    const nv_bfloat16 * x[GGML_CUDA_BF16_GEMV_GROUP_MAX];
    float             * dst[GGML_CUDA_BF16_GEMV_GROUP_MAX];
    int64_t             stride_row[GGML_CUDA_BF16_GEMV_GROUP_MAX];
    int64_t             stride_col_dst[GGML_CUDA_BF16_GEMV_GROUP_MAX];
    int                 nrows[GGML_CUDA_BF16_GEMV_GROUP_MAX];
    int                 block0[GGML_CUDA_BF16_GEMV_GROUP_MAX + 1];
    int                 n;
};

template <int npairs, int ncols_dst>
static __global__ void __launch_bounds__(HC_BLOCK_SIZE)
mul_mat_vec_hc_long_group(const float * __restrict__ y, const int64_t stride_col_y, const mul_mat_vec_hc_group_args args) {
    int m = 0;
#pragma unroll
    for (int k = 1; k < GGML_CUDA_BF16_GEMV_GROUP_MAX; ++k) {
        m += k < args.n && (int) blockIdx.x >= args.block0[k];
    }
    mul_mat_vec_hc_long_block<npairs, ncols_dst, 1, GGML_CUDA_HC_EPI_NONE>(
        args.x[m], y, args.dst[m], args.nrows[m], args.stride_row[m], stride_col_y, args.stride_col_dst[m],
        ggml_cuda_hc_epilogue(), blockIdx.x - args.block0[m]);
}

template <int npairs, int ncols_dst, int epi>
static void mul_mat_vec_hc_long_cuda(
        const nv_bfloat16 * x, const float * y, float * dst, const int nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const ggml_cuda_hc_epilogue & ep, cudaStream_t stream) {
    const dim3 block_dims(HC_BLOCK_SIZE, 1, 1);

    // measured on a V100: one row per block is fastest for a single token, where the activations are
    // cheap to re-read. With several tokens they are not, so two rows share each activation read,
    // provided that still leaves at least two blocks per SM (the 320-row down-projection, not the
    // 4-row injection).
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    if (ncols_dst > 1 && nrows >= 4*nsm) {
        const dim3 block_nums((nrows + 1) / 2, 1, 1);
        mul_mat_vec_hc_long<npairs, ncols_dst, 2, epi><<<block_nums, block_dims, 0, stream>>>
            (x, y, dst, nrows, stride_row, stride_col_y, stride_col_dst, ep);
    } else {
        const dim3 block_nums(nrows, 1, 1);
        mul_mat_vec_hc_long<npairs, ncols_dst, 1, epi><<<block_nums, block_dims, 0, stream>>>
            (x, y, dst, nrows, stride_row, stride_col_y, stride_col_dst, ep);
    }
}

template <int ncols_dst>
static void mul_mat_vec_hc_cuda(
        const nv_bfloat16 * x, const float * y, float * dst, const int64_t ncols, const int nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const ggml_cuda_hc_epilogue & ep, cudaStream_t stream) {
    if (ncols == HC_SHORT_K) {
        GGML_ASSERT(ep.kind == GGML_CUDA_HC_EPI_NONE);
        constexpr int rows_per_block = HC_BLOCK_SIZE/HC_SHORT_LANES;
        const dim3 block_nums((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
        const dim3 block_dims(HC_BLOCK_SIZE, 1, 1);
        mul_mat_vec_hc_short<ncols_dst><<<block_nums, block_dims, 0, stream>>>
            (x, y, dst, nrows, stride_row, stride_col_y, stride_col_dst);
        return;
    }

    GGML_ASSERT(ncols == HC_LONG_K);
    switch (ep.kind) {
        case GGML_CUDA_HC_EPI_NONE:
            mul_mat_vec_hc_long_cuda<HC_LONG_PAIRS, ncols_dst, GGML_CUDA_HC_EPI_NONE>(x, y, dst, nrows, stride_row, stride_col_y, stride_col_dst, ep, stream);
            break;
        case GGML_CUDA_HC_EPI_SCALE_SILU:
            mul_mat_vec_hc_long_cuda<HC_LONG_PAIRS, ncols_dst, GGML_CUDA_HC_EPI_SCALE_SILU>(x, y, dst, nrows, stride_row, stride_col_y, stride_col_dst, ep, stream);
            break;
        case GGML_CUDA_HC_EPI_SCALE_SIGMOID_SCALE:
            mul_mat_vec_hc_long_cuda<HC_LONG_PAIRS, ncols_dst, GGML_CUDA_HC_EPI_SCALE_SIGMOID_SCALE>(x, y, dst, nrows, stride_row, stride_col_y, stride_col_dst, ep, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_mul_mat_vec_hc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                              const ggml_cuda_hc_epilogue * epi, ggml_tensor * out) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const ggml_cuda_hc_epilogue ep = epi ? *epi : ggml_cuda_hc_epilogue();
    if (ep.kind != GGML_CUDA_HC_EPI_NONE) {
        // the epilogue lives in the long-row kernel only, and writes out in dst's layout
        if (ne00 != HC_LONG_K || out == nullptr || out->type != GGML_TYPE_F32 || !ggml_are_same_shape(out, dst) ||
            !ggml_is_contiguous(out) || !ggml_is_contiguous(dst)) {
            return false;
        }
    }

    if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (ne00 != HC_SHORT_K && ne00 != HC_LONG_K) {
        return false;
    }
    if (ne11 < 1 || ne11 > HC_MAX_NCOLS_DST || ne01 > INT_MAX || ne02*ne03 != 1 || ne12*ne13 != 1) {
        return false;
    }
    if (nb00 != ggml_type_size(src0->type) || nb10 != sizeof(float) || nb0 != sizeof(float)) {
        return false;
    }
    // weights are read as BF16 pairs and activations as float2, as in mul_mat_vec_f
    if ((uintptr_t) src0->data % 4 != 0 || nb01 % 4 != 0 || (uintptr_t) src1->data % 8 != 0 || nb11 % 8 != 0) {
        return false;
    }

    const nv_bfloat16 * x   = (const nv_bfloat16 *) src0->data;
    const float       * y   = (const float       *) src1->data;
    float             * d   = (float             *) (ep.kind != GGML_CUDA_HC_EPI_NONE ? out : dst)->data;
    const int64_t       s01 = nb01 / nb00;
    const int64_t       s11 = nb11 / nb10;
    const int64_t       s1  = nb1  / nb0;
    cudaStream_t        stream = ctx.stream();

    switch (ne11) {
        case 1: mul_mat_vec_hc_cuda<1>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        case 2: mul_mat_vec_hc_cuda<2>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        case 3: mul_mat_vec_hc_cuda<3>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        case 4: mul_mat_vec_hc_cuda<4>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        case 5: mul_mat_vec_hc_cuda<5>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        case 6: mul_mat_vec_hc_cuda<6>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        case 7: mul_mat_vec_hc_cuda<7>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        case 8: mul_mat_vec_hc_cuda<8>(x, y, d, ne00, ne01, s01, s11, s1, ep, stream); break;
        default: GGML_ABORT("fatal error");
    }
    return true;
}

// LLAMA_BF16_GEMV=0 keeps mul_mat_vec_f for the untagged n_embd-wide BF16 products.
static bool ggml_cuda_bf16_gemv_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_BF16_GEMV");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

bool ggml_cuda_mul_mat_vec_bf16_ok(const int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    if (!ggml_cuda_bf16_gemv_enabled()) {
        return false;
    }
    if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (ne00 != HC_EMBD_K) {
        return false;
    }
    if (ne11 < 1 || ne11 > HC_MAX_NCOLS_DST || ne01 > INT_MAX || ne02*ne03 != 1 || ne12*ne13 != 1) {
        return false;
    }
    if (nb00 != ggml_type_size(src0->type) || nb10 != sizeof(float) || nb0 != sizeof(float)) {
        return false;
    }
    if ((uintptr_t) src0->data % 4 != 0 || nb01 % 4 != 0 || (uintptr_t) src1->data % 8 != 0 || nb11 % 8 != 0) {
        return false;
    }
    // only where the generic kernel would run, with its 256-thread block (warp size 32)
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_info().devices[device].warp_size != WARP_SIZE ||
        !ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        return false;
    }
    return true;
}

bool ggml_cuda_mul_mat_vec_bf16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    if (!ggml_cuda_mul_mat_vec_bf16_ok(ctx.device, src0, src1, dst)) {
        return false;
    }

    const nv_bfloat16 * x   = (const nv_bfloat16 *) src0->data;
    const float       * y   = (const float       *) src1->data;
    float             * d   = (float             *) dst->data;
    const int64_t       s01 = nb01 / nb00;
    const int64_t       s11 = nb11 / nb10;
    const int64_t       s1  = nb1  / nb0;
    const ggml_cuda_hc_epilogue ep;
    cudaStream_t        stream = ctx.stream();

    switch (ne11) {
        case 1: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 1, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        case 2: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 2, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        case 3: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 3, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        case 4: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 4, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        case 5: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 5, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        case 6: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 6, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        case 7: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 7, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        case 8: mul_mat_vec_hc_long_cuda<HC_EMBD_PAIRS, 8, GGML_CUDA_HC_EPI_NONE>(x, y, d, ne01, s01, s11, s1, ep, stream); break;
        default: GGML_ABORT("fatal error");
    }
    return true;
}

void ggml_cuda_mul_mat_vec_bf16_group(ggml_backend_cuda_context & ctx, const int n, const ggml_tensor * const * src0,
                                      const ggml_tensor * src1, ggml_tensor * const * dst) {
    GGML_ASSERT(n >= 1 && n <= GGML_CUDA_BF16_GEMV_GROUP_MAX);
    mul_mat_vec_hc_group_args args = {};
    args.n = n;
    int nblocks = 0;
    for (int k = 0; k < n; ++k) {
        GGML_ASSERT(src0[k]->ne[0] == HC_EMBD_K && dst[k]->src[1]->data == src1->data && dst[k]->ne[1] == src1->ne[1]);
        args.x[k]              = (const nv_bfloat16 *) src0[k]->data;
        args.dst[k]            = (float *) dst[k]->data;
        args.stride_row[k]     = src0[k]->nb[1] / src0[k]->nb[0];
        args.stride_col_dst[k] = dst[k]->nb[1] / dst[k]->nb[0];
        args.nrows[k]          = (int) src0[k]->ne[1];
        args.block0[k]         = nblocks;
        nblocks += args.nrows[k];
    }
    args.block0[n] = nblocks;

    const float * y   = (const float *) src1->data;
    const int64_t s11 = src1->nb[1] / src1->nb[0];
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(HC_BLOCK_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();
    switch (src1->ne[1]) {
        case 1: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 1><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        case 2: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 2><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        case 3: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 3><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        case 4: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 4><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        case 5: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 5><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        case 6: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 6><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        case 7: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 7><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        case 8: mul_mat_vec_hc_long_group<HC_EMBD_PAIRS, 8><<<block_nums, block_dims, 0, stream>>>(y, s11, args); break;
        default: GGML_ABORT("fatal error");
    }
}
