#include "norm.cuh"
#include "qpn-source.cuh"
#include "unary.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// A lone RMS_NORM -> MUL by the weight whose output is the input of products on repacked weights (qwen35's
// first layer): rms_norm_f32<block_size, true>'s loops, reduction and store expression for one weight row, the normed
// row also kept in shared memory after the reduction's slots, then its prepared input (qpn-source.cuh); row = token.
template <int block_size>
static __global__ void rms_norm_mul_qpn_f32(const float * x, float * dst, const int ncols, const int64_t stride_row, const float eps,
        const float * mul, const ggml_cuda_qpn_dst q) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    x   += row*stride_row;
    dst += (int64_t) row*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    float * s_x = s_sum + 32;
    for (int col = tid; col < ncols; col += block_size) {
        const float y = scale * x[col] * mul[col];
        dst[col] = y;
        s_x[col] = y;
    }
    __syncthreads();
    qpn_source_row_smem(s_x, ncols/QK_K, row, q);
}

// the MTP draft step's two input norms and their concat in one kernel: RMS_NORM -> MUL of the hidden row and of the
// token embedding, then CONCAT along dim 0, as rms_norm_f32<1024, true> computes each row (the same columns per thread, the same
// block reduction, the same expressions), written into its half of the concat's row. One block per (row, half)
static __global__ void __launch_bounds__(1024) rms_norm_mul2_concat_f32(
        const float * x0, const int64_t sx0, const float * w0, const float eps0,
        const float * x1, const int64_t sx1, const float * w1, const float eps1,
        float * dst, const int ncols) {
    constexpr int block_size = 1024;
    const int row  = blockIdx.x;
    const int half = blockIdx.y;
    const int tid  = threadIdx.x;
    const float * x   = half == 0 ? x0 + row*sx0 : x1 + row*sx1;
    const float * mul = half == 0 ? w0 : w1;
    const float   eps = half == 0 ? eps0 : eps1;
    dst += (int64_t) row*2*ncols + (int64_t) half*ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col] * mul[col];
    }
}

void ggml_cuda_op_rms_norm_mul2_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * norm0, const ggml_tensor * mul0,
        const ggml_tensor * norm1, const ggml_tensor * mul1, ggml_tensor * concat) {
    const ggml_tensor * x0 = norm0->src[0];
    const ggml_tensor * x1 = norm1->src[0];
    const ggml_tensor * w0 = mul0->src[0] == norm0 ? mul0->src[1] : mul0->src[0];
    const ggml_tensor * w1 = mul1->src[0] == norm1 ? mul1->src[1] : mul1->src[0];
    float eps0 = 0.0f, eps1 = 0.0f;
    memcpy(&eps0, norm0->op_params, sizeof(float));
    memcpy(&eps1, norm1->op_params, sizeof(float));
    const int ncols = (int) x0->ne[0];
    const int nrows = (int) x0->ne[1];
    const dim3 blocks(nrows, 2, 1);
    rms_norm_mul2_concat_f32<<<blocks, 1024, 32*sizeof(float), ctx.stream()>>>(
        (const float *) x0->data, (int64_t) (x0->nb[1]/sizeof(float)), (const float *) w0->data, eps0,
        (const float *) x1->data, (int64_t) (x1->nb[1]/sizeof(float)), (const float *) w1->data, eps1,
        (float *) concat->data, ncols);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    // one weight row over T <= 8 rows (16) whose output is the input of products on repacked weights
    ggml_cuda_qpn_dst q;
    const size_t smem_q = (32 + (size_t) ne00) * sizeof(float);
    if (ne02 == 1 && ne03 == 1 && ggml_is_contiguous(mul_src) && ggml_nelements(mul_src) == ne00 && ggml_is_contiguous(mul_tensor) &&
        ne00 % QK_K == 0 && smem_q <= 48*1024 && ggml_cuda_qpn_source_begin(mul_tensor, ne00, (int) ne01, stream, &q)) {
        const dim3 blocks_num(ne01, 1, 1);
        if (ne00 < 1024) {
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(256, 1, 1), smem_q, stream};
            ggml_cuda_kernel_launch(rms_norm_mul_qpn_f32<256>, launch_params, src0_d, dst_d, (int) ne00, s01, eps, mul_d, q);
        } else {
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(1024, 1, 1), smem_q, stream};
            ggml_cuda_kernel_launch(rms_norm_mul_qpn_f32<1024>, launch_params, src0_d, dst_d, (int) ne00, s01, eps, mul_d, q);
        }
        ggml_cuda_qpn_source_end(mul_tensor, ne00, stream, q);
        return;
    }

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// The hyper-connection combine (dsv4_hc_post_f32 without comb) folded into the grouped RMS norm and its
// gamma multiply that read it (LLAMA_FOLD_HC_NORM). Each block computes one [n_embd] row of
// the combine exactly as dsv4_hc_post_f32 does, stores it (the residual stream stays a graph tensor), and
// then runs rms_norm_f32<block_size, true>'s reduction and store on it, so both outputs are bit-identical.
template <int block_size>
static __global__ void hc_post_rms_norm_mul_f32(
        const float * hx, const float * hres, const float * hpost, float * hdst,
        const int64_t sx0, const int64_t sx1,
        const int64_t sr0, const int64_t sr1, const int64_t sr2,
        const int64_t sp0, const int64_t sp1,
        const int64_t sd1, const int64_t sd2,
        const int ncols, const float eps,
        const float * mul, const int64_t mul_s1, const int64_t mul_s2, const int mul_nrows, const int mul_nchannels,
        float * dst) {
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int idst  = blockIdx.x; // stream
    const int it    = blockIdx.y; // token
    const int tid   = threadIdx.x;

    float * x = hdst + idst*sd1 + it*sd2;
    mul += (idst % mul_nrows)*mul_s1 + (it % mul_nchannels)*mul_s2;
    dst += ((int64_t) it*nrows + idst)*ncols;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const int64_t i0 = col;
        float sum = hx[i0*sx0 + it*sx1] * hpost[idst*sp0 + it*sp1];
        sum += hres[i0*sr0 + idst*sr1 + it*sr2];
        x[col] = sum;

        const float xi = sum;
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col] * mul[col];
    }
}

void ggml_cuda_op_hc_post_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post, ggml_tensor * rms_norm, ggml_tensor * mul_tensor) {
    const ggml_tensor * hx    = hc_post->src[0];
    const ggml_tensor * hres  = hc_post->src[1];
    const ggml_tensor * hpost = hc_post->src[2];
    const ggml_tensor * mul_src = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int     ncols    = hc_post->ne[0];
    const int     hc       = hc_post->ne[1];
    const int     n_tokens = hc_post->ne[2];

    const dim3 blocks_num(hc, n_tokens, 1);
    const int64_t sx0 = hx->nb[0]/sizeof(float),    sx1 = hx->nb[1]/sizeof(float);
    const int64_t sr0 = hres->nb[0]/sizeof(float),  sr1 = hres->nb[1]/sizeof(float),  sr2 = hres->nb[2]/sizeof(float);
    const int64_t sp0 = hpost->nb[0]/sizeof(float), sp1 = hpost->nb[1]/sizeof(float);
    const int64_t sd1 = hc_post->nb[1]/sizeof(float), sd2 = hc_post->nb[2]/sizeof(float);
    const int64_t mul_s1 = mul_src->nb[1]/sizeof(float), mul_s2 = mul_src->nb[2]/sizeof(float);

    // the same block size as rms_norm_f32_cuda picks, so the reduction runs in the same order
    if (ncols < 1024) {
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(256, 1, 1), 32 * sizeof(float), ctx.stream()};
        ggml_cuda_kernel_launch(hc_post_rms_norm_mul_f32<256>, launch_params,
            (const float *) hx->data, (const float *) hres->data, (const float *) hpost->data, (float *) hc_post->data,
            sx0, sx1, sr0, sr1, sr2, sp0, sp1, sd1, sd2, ncols, eps,
            (const float *) mul_src->data, mul_s1, mul_s2, (int) mul_src->ne[1], (int) mul_src->ne[2], (float *) mul_tensor->data);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(1024, 1, 1), 32 * sizeof(float), ctx.stream()};
        ggml_cuda_kernel_launch(hc_post_rms_norm_mul_f32<1024>, launch_params,
            (const float *) hx->data, (const float *) hres->data, (const float *) hpost->data, (float *) hc_post->data,
            sx0, sx1, sr0, sr1, sr2, sp0, sp1, sd1, sd2, ncols, eps,
            (const float *) mul_src->data, mul_s1, mul_s2, (int) mul_src->ne[1], (int) mul_src->ne[2], (float *) mul_tensor->data);
    }
}

// The GDN output's gated norm, RMS_NORM -> MUL by the weight, then MUL by SIGMOID(z)
// (LLAMA_FOLD_NORM_GATE), in one launch at the sigmoid's place. rms_norm_f32<block_size, true>'s loops,
// reduction and store expression, then the product op_sigmoid(z) * normed as unary_gated_op_kernel forms it:
// each value is rounded as the separate kernels round it, so dst is bit-identical. With silu (qwen35's
// build_norm_gated) the gate is op_silu(z), as the fused UNARY -> MUL launch forms it.
static __device__ __forceinline__ float norm_gate_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x)); // unary.cu's op_sigmoid
}

template <int block_size, bool silu>
static __device__ __forceinline__ float norm_gate_op(float z) {
    if constexpr (silu) {
        return ggml_cuda_op_silu_single(z); // unary.cu's op_silu
    } else {
        return norm_gate_sigmoid(z);
    }
}

// With n_reg > 0 (ncols <= n_reg*block_size) each thread loads its x, weight and gate and forms the gate before the
// reduction, keeping them in registers, so nothing is loaded after it; the values and the order are the same.
template <int block_size, bool silu, int n_reg>
static __global__ void rms_norm_mul_sigmoid_gate_f32(
        const float * x, float * dst, const int ncols,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const float * mul, const float * z) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
    z   += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    [[maybe_unused]] float xr[n_reg > 0 ? n_reg : 1];
    [[maybe_unused]] float wr[n_reg > 0 ? n_reg : 1];
    [[maybe_unused]] float gr[n_reg > 0 ? n_reg : 1];

    ggml_cuda_pdl_sync();
    if constexpr (n_reg > 0) {
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                const float xi = x[col];
                tmp += xi * xi;
                xr[k] = xi;
                wr[k] = mul[col];
                gr[k] = norm_gate_op<block_size, silu>(z[col]);
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[col];
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (n_reg > 0) {
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                const float normed = scale * xr[k] * wr[k];
                dst[col] = gr[k] * normed;
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float normed = scale * x[col] * mul[col];
            dst[col] = norm_gate_op<block_size, silu>(z[col]) * normed;
        }
    }
}

// The gated norm whose output is the input of products on repacked weights (qwen35's GDN output, heads of
// 128 columns): one block of 256 threads per pair of heads (one 256-column slice) and token. Each half is the block of
// rms_norm_mul_sigmoid_gate_f32<256, silu, n_reg> for its head: its thread tid' < 128 holds column tid', and its
// reduction is block_reduce's: the warp sums, then the sum over [its 4 warp sums, 4 zeros, 24 zeros], which is what
// the 8-warp block forms (its warps 4-7 hold no column and sum to 0). The gated values also go to shared memory, and
// the block's first warp writes the slice's prepared input (qpn-source.cuh).
template <bool silu>
static __global__ void rms_norm_mul_gate_qpn_f32(
        const float * x, float * dst, const int nrows,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const float * mul, const float * z, const ggml_cuda_qpn_dst q) {
    constexpr int ncols = 128;
    ggml_cuda_pdl_lc();
    const int nchannels = gridDim.y;

    const int half_id   = threadIdx.x / ncols;
    const int tid       = threadIdx.x % ncols;
    const int row       = 2*blockIdx.x + half_id;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
    z   += ((sample*nchannels + channel)*nrows + row)*ncols;

    ggml_cuda_pdl_sync();
    const float xi = x[tid];
    float tmp = 0.0f; // partial sum for thread in warp
    tmp += xi * xi;
    const float wi = mul[tid];
    const float gi = norm_gate_op<256, silu>(z[tid]);

    // sum up partial sums: block_reduce<SUM, 256>'s two stages for this half's 4 warps
    __shared__ float s_sum[8];
    __shared__ float s_x[2*ncols];
    tmp = warp_reduce_sum(tmp);
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        s_sum[warp_id] = tmp;
    }
    __syncthreads();
    tmp = 0.0f;
    if (lane_id < 4) {
        tmp = s_sum[4*half_id + lane_id];
    }
    tmp = warp_reduce_sum(tmp);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    const float normed = scale * xi * wi;
    const float y = gi * normed;
    dst[tid] = y;
    s_x[threadIdx.x] = y;
    __syncthreads();
    if (warp_id == 0) {
        qpn_source_slice_smem(s_x, blockIdx.x, sample*nchannels + channel, q); // the token is (sequence, token)
    }
}

void ggml_cuda_op_rms_norm_mul_sigmoid_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul_tensor, const ggml_tensor * sigmoid, ggml_tensor * dst) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * z = sigmoid->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    GGML_ASSERT(x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && z->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && ggml_is_contiguous(w) && ggml_nelements(w) == x->ne[0]);
    GGML_ASSERT(ggml_is_contiguous(z) && ggml_is_contiguous(dst) && ggml_are_same_shape(z, dst) && ggml_are_same_shape(x, dst));
    const bool silu = ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SILU;
    GGML_ASSERT(silu || ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID);

    const int64_t ne00 = x->ne[0], ne01 = x->ne[1], ne02 = x->ne[2], ne03 = x->ne[3];
    const int64_t s01 = x->nb[1]/sizeof(float), s02 = x->nb[2]/sizeof(float), s03 = x->nb[3]/sizeof(float);

    // heads of 128 columns over T <= 8 tokens, the output the input of products on repacked weights (up to 16
    // tokens, of one sequence or several: [128, heads, tokens, sequences] is ne02*ne03 rows of the product's input, sequence-major)
    ggml_cuda_qpn_dst q;
    if (ne00 == 128 && ne01 % 2 == 0 && ggml_cuda_qpn_source_begin(dst, ne00*ne01, (int) (ne02*ne03), ctx.stream(), &q)) {
        const ggml_cuda_kernel_launch_params launch_params = {dim3(ne01/2, ne02, ne03), dim3(256, 1, 1), 0, ctx.stream()};
        if (silu) {
            ggml_cuda_kernel_launch(rms_norm_mul_gate_qpn_f32<true>, launch_params,
                (const float *) x->data, (float *) dst->data, (int) ne01, s01, s02, s03, eps, (const float *) w->data, (const float *) z->data, q);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_gate_qpn_f32<false>, launch_params,
                (const float *) x->data, (float *) dst->data, (int) ne01, s01, s02, s03, eps, (const float *) w->data, (const float *) z->data, q);
        }
        ggml_cuda_qpn_source_end(dst, ne00*ne01, ctx.stream(), q);
        return;
    }

    // the same block size as rms_norm_mul_f32_cuda picks, so the reduction runs in the same order
    const dim3 blocks_num(ne01, ne02, ne03);
    const auto launch = [&](auto silu_tag) {
        constexpr bool c_silu = decltype(silu_tag)::value;
        constexpr int n_reg = 4;
        if (ne00 < 1024) {
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(256, 1, 1), 32 * sizeof(float), ctx.stream()};
            ggml_cuda_kernel_launch(rms_norm_mul_sigmoid_gate_f32<256, c_silu, n_reg>, launch_params,
                (const float *) x->data, (float *) dst->data, (int) ne00, s01, s02, s03, eps, (const float *) w->data, (const float *) z->data);
        } else if (ne00 <= n_reg*1024) {
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(1024, 1, 1), 32 * sizeof(float), ctx.stream()};
            ggml_cuda_kernel_launch(rms_norm_mul_sigmoid_gate_f32<1024, c_silu, n_reg>, launch_params,
                (const float *) x->data, (float *) dst->data, (int) ne00, s01, s02, s03, eps, (const float *) w->data, (const float *) z->data);
        } else {
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(1024, 1, 1), 32 * sizeof(float), ctx.stream()};
            ggml_cuda_kernel_launch(rms_norm_mul_sigmoid_gate_f32<1024, c_silu, 0>, launch_params,
                (const float *) x->data, (float *) dst->data, (int) ne00, s01, s02, s03, eps, (const float *) w->data, (const float *) z->data);
        }
    };
    if (silu) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
}

// The residual ADD folded into the RMS norm and weight multiply that read it (LLAMA_FOLD_ADD_NORM):
// qwen35's trunk, ADD(block out, residual) -> RMS_NORM -> MUL by the norm weight. Each block forms one row of the
// sum as k_bin_bcast's op_add does and stores it (the residual stream stays a graph tensor), then runs
// rms_norm_f32<block_size, true>'s reduction and store on it, so both outputs are bit-identical. Every column is
// read and written by one thread, in both loops, so the sum may sit in place of either addend and the normed
// output in place of either addend (same layout), but not over the sum.
// With n_reg > 0 (ncols <= n_reg*block_size) each thread keeps its sums and weights in registers across the
// reduction, so nothing is loaded after it; the values and the order are the same.
// With qpn (n_reg > 0 only) the block also keeps its normed row in shared memory after the reduction's
// slots and writes the row's prepared input from it (qpn-source.cuh), one warp per 256-column slice; row = token.
template <int block_size, int n_reg, bool qpn = false>
static __global__ void add_rms_norm_mul_f32(
        const float * a, const float * b, float * sum, float * dst, const int ncols,
        const int64_t sa1, const int64_t sb1, const float eps, const float * mul, const ggml_cuda_qpn_dst q) {
    static_assert(!qpn || n_reg > 0, "the prepared input needs the registers path");
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    a   += row*sa1;
    b   += row*sb1;
    sum += (int64_t) row*ncols;
    dst += (int64_t) row*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    [[maybe_unused]] float xr[n_reg > 0 ? n_reg : 1];
    [[maybe_unused]] float wr[n_reg > 0 ? n_reg : 1];

    ggml_cuda_pdl_sync();
    if constexpr (n_reg > 0) {
        // every load before any store: the sum may sit in place of an addend, so a store would hold back the loads
        // after it (each column is read and written by this thread only)
        float ar[n_reg], br[n_reg];
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                ar[k] = a[col];
                br[k] = b[col];
                wr[k] = mul[col];
            }
        }
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                const float xi = ar[k] + br[k]; // binbcast.cu's op_add
                sum[col] = xi;
                tmp += xi * xi;
                xr[k] = xi;
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = a[col] + b[col]; // binbcast.cu's op_add
            sum[col] = xi;
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (n_reg > 0) {
        [[maybe_unused]] float * s_x = s_sum + 32;
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                const float y = scale * xr[k] * wr[k];
                dst[col] = y;
                if constexpr (qpn) {
                    s_x[col] = y;
                }
            }
        }
        if constexpr (qpn) {
            __syncthreads();
            qpn_source_row_smem(s_x, ncols/QK_K, row, q);
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            dst[col] = scale * sum[col] * mul[col];
        }
    }
    GGML_UNUSED(q);
}

void ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * rms_norm,
        ggml_tensor * mul_tensor) {
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    const ggml_tensor * w = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
                mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) && ggml_is_contiguous(add) && ggml_is_contiguous(mul_tensor));
    GGML_ASSERT(ggml_are_same_shape(a, add) && ggml_are_same_shape(b, add) && ggml_are_same_shape(mul_tensor, add) &&
                add->ne[2] == 1 && add->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(w) && ggml_nelements(w) == add->ne[0]);

    const int     ncols = add->ne[0];
    const int64_t sa1   = a->nb[1]/sizeof(float);
    const int64_t sb1   = b->nb[1]/sizeof(float);

    // the same block size as rms_norm_mul_f32_cuda picks, so the reduction runs in the same order
    const dim3 blocks_num(add->ne[1], 1, 1);
    const auto launch = [&](auto kernel, int block_size, size_t smem = 32 * sizeof(float), const ggml_cuda_qpn_dst & q = {}) {
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(block_size, 1, 1), smem, ctx.stream()};
        ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) a->data, (const float *) b->data, (float *) add->data, (float *) mul_tensor->data, ncols, sa1, sb1, eps,
            (const float *) w->data, q);
    };
    constexpr int n_reg = 8;
    // the normed output is the input of products on repacked weights, prepared here
    ggml_cuda_qpn_dst q;
    const size_t smem_q = (32 + (size_t) ncols) * sizeof(float);
    if (ncols % QK_K == 0 && ncols <= n_reg*1024 && smem_q <= 48*1024 &&
        ggml_cuda_qpn_source_begin(mul_tensor, ncols, (int) add->ne[1], ctx.stream(), &q)) {
        if (ncols < 1024) {
            launch(add_rms_norm_mul_f32<256, n_reg, true>, 256, smem_q, q);
        } else {
            launch(add_rms_norm_mul_f32<1024, n_reg, true>, 1024, smem_q, q);
        }
        ggml_cuda_qpn_source_end(mul_tensor, ncols, ctx.stream(), q);
        return;
    }
    if (ncols < 1024) {
        launch(add_rms_norm_mul_f32<256, n_reg>, 256);
    } else if (ncols <= n_reg*1024) {
        launch(add_rms_norm_mul_f32<1024, n_reg>, 1024);
    } else {
        launch(add_rms_norm_mul_f32<1024, 0>, 1024);
    }
}
