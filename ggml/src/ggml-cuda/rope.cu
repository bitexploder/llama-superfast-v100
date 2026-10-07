#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T *            x,
                                  T *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const bool           inplace) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_offs/2 + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T *            x,
                            T *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
template <int block_size, bool has_ff, typename D>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox) {
    ggml_cuda_pdl_lc();
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x += sample*s03 + channel*s02 + row*s01;

    const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
    mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float scale = rsqrtf(tmp/ncols + eps);

    int64_t idst = sample*s3 + channel*s2 + row*s1;
    if (set_rows_stride != 0) {
        idst = row*s1 + row_indices[channel]*set_rows_stride;
    }
    dst += idst;

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        int ix0;
        int ix1;
        if (is_neox && i0 < n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }

        const float x0 = scale * x[ix0] * mul[fastmodulo(ix0, mul_ncols_packed)];
        const float x1 = scale * x[ix1] * mul[fastmodulo(ix1, mul_ncols_packed)];

        if (i0 >= n_dims) {
            dst[ix0] = ggml_cuda_cast<D>(x0);
            dst[ix1] = ggml_cuda_cast<D>(x1);
            continue;
        }

        const float theta_base  = pos[channel]*powf(theta_scale, i0/2.0f);
        const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

        dst[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
        dst[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);

    const dim3 blocks_num(nrows, nchannels, nsamples);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        }
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    void *          dst_d           = rope->data;
    ggml_type       dst_type        = rope->type;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    const int32_t * pos = (const int32_t *) rope->src[1]->data;

    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    const int64_t s1 = rope->nb[1] / ts_dst;
    const int64_t s2 = rope->nb[2] / ts_dst;
    const int64_t s3 = rope->nb[3] / ts_dst;

    cudaStream_t stream = ctx.stream();

    if (dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda((const float *) x->data, (float *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, stream);
    } else if (dst_type == GGML_TYPE_F16) {
        rms_norm_mul_rope_cuda((const float *) x->data, (half *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, stream);
    } else {
        GGML_ABORT("fatal error");
    }
}

// RMS_NORM -> MUL by the norm weight -> ROPE multi (M-RoPE or IMRoPE, not vision), F32, in one launch
// (LLAMA_FOLD_NORM_ROPE): one block per row. rms_norm_f32<block_size, true>'s loops, reduction and store
// expression give the normed row, which stays in shared memory; then each pair is rotated with rope_multi's
// statements, reading x0 and x1 from there, and the channels outside the rotated window are copied, as
// rope_multi copies them when not in place. Every value is rounded as the separate kernels round it.
template <int block_size, bool has_ff>
static __global__ void rms_norm_mul_rope_multi_f32(
        const float * x, float * dst, const int ncols,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps, const float * mul,
        const int n_dims, const int n_offs, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors,
        const mrope_sections sections, const bool is_imrope) {
    ggml_cuda_pdl_lc();
    const int ne02    = gridDim.y;
    const int row     = blockIdx.x; // i1
    const int channel = blockIdx.y; // i2, the token
    const int sample  = blockIdx.z; // i3
    const int tid     = threadIdx.x;

    x   += sample*s03 + channel*s02 + row*s01;
    dst += sample*s3  + channel*s2  + row*s1;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums; the normed row follows the reduction's 32 slots
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    float * y = s_sum + 32;
    for (int col = tid; col < ncols; col += block_size) {
        y[col] = scale * x[col] * mul[col];
    }
    __syncthreads();

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        if (i0 < n_offs || i0 >= n_offs + n_dims) {
            dst[i0 + 0] = y[i0 + 0];
            dst[i0 + 1] = y[i0 + 1];
            continue;
        }

        const int iw = i0 - n_offs; // relative idx
        const int sector = (iw / 2) % sect_dims;

        float theta_base = 0.0;
        if (is_imrope) {
            if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
                theta_base = pos[channel + ne02 * 1] * powf(theta_scale, iw / 2.0f);
            } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
                theta_base = pos[channel + ne02 * 2] * powf(theta_scale, iw / 2.0f);
            } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
                theta_base = pos[channel] * powf(theta_scale, iw / 2.0f);
            } else {
                theta_base = pos[channel + ne02 * 3] * powf(theta_scale, iw / 2.0f);
            }
        } else {
            if (sector < sections.v[0]) {
                theta_base = pos[channel] * powf(theta_scale, iw / 2.0f);
            } else if (sector >= sections.v[0] && sector < sec_w) {
                theta_base = pos[channel + ne02 * 1] * powf(theta_scale, iw / 2.0f);
            } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
                theta_base = pos[channel + ne02 * 2] * powf(theta_scale, iw / 2.0f);
            } else if (sector >= sec_w + sections.v[2]) {
                theta_base = pos[channel + ne02 * 3] * powf(theta_scale, iw / 2.0f);
            }
        }

        const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

        float cos_theta;
        float sin_theta;

        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

        const float x0 = y[i0/2 + n_offs/2 + 0];
        const float x1 = y[i0/2 + n_offs/2 + n_dims/2];

        dst[i0/2 + n_offs/2 + 0]        = x0*cos_theta - x1*sin_theta;
        dst[i0/2 + n_offs/2 + n_dims/2] = x0*sin_theta + x1*cos_theta;
    }
}

void ggml_cuda_op_rms_norm_mul_rope_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul, ggml_tensor * rope) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    GGML_ASSERT(x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && rope->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && rope->nb[0] == sizeof(float) && ggml_is_contiguous(w) && ggml_nelements(w) == x->ne[0]);
    GGML_ASSERT(ggml_are_same_shape(x, rope) && x->ne[0] % 2 == 0);

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];
    const int n_offs     = ((const int32_t *) rope->op_params)[15];
    GGML_ASSERT((mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    mrope_sections sections;
    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (const int32_t *) rope->op_params + 11, sizeof(int)*4);
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);
    // as rope_multi_cuda computes it
    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    const int32_t * pos          = (const int32_t *) rope->src[1]->data;
    const float *   freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    const int     ncols = x->ne[0];
    const int64_t s01 = x->nb[1]/sizeof(float),    s02 = x->nb[2]/sizeof(float),    s03 = x->nb[3]/sizeof(float);
    const int64_t s1  = rope->nb[1]/sizeof(float), s2  = rope->nb[2]/sizeof(float), s3  = rope->nb[3]/sizeof(float);

    // the same block size as rms_norm_mul_f32_cuda picks, so the reduction runs in the same order
    const dim3   blocks_num(x->ne[1], x->ne[2], x->ne[3]);
    const size_t smem = (32 + ncols)*sizeof(float);
    if (ncols < 1024) {
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(256, 1, 1), smem, ctx.stream()};
        ggml_cuda_kernel_launch(freq_factors ? rms_norm_mul_rope_multi_f32<256, true> : rms_norm_mul_rope_multi_f32<256, false>,
            launch_params, (const float *) x->data, (float *) rope->data, ncols, s01, s02, s03, s1, s2, s3, eps,
            (const float *) w->data, n_dims, n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
            freq_factors, sections, is_imrope);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(1024, 1, 1), smem, ctx.stream()};
        ggml_cuda_kernel_launch(freq_factors ? rms_norm_mul_rope_multi_f32<1024, true> : rms_norm_mul_rope_multi_f32<1024, false>,
            launch_params, (const float *) x->data, (float *) rope->data, ncols, s01, s02, s03, s1, s2, s3, eps,
            (const float *) w->data, n_dims, n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
            freq_factors, sections, is_imrope);
    }
}

// LLAMA_QSA_POOL_FUSE: the QSA indexer key chain in one pass, one warp per pooled block. Every step
// repeats the arithmetic of the kernel it replaces, in the same order, so the keys are bit-identical:
//   k_get_rows_float  exact conversion to f32
//   k_bin_bcast add   ((m0 + m1) + m2) + ...
//   scale_f32         scale*x + bias
//   rms_norm_f32<256> lane l of warp w holds column 32*w + l: a warp butterfly per warp, then one over
//                     the warp sums with the absent warps as 0, then (rsqrtf(sum/ncols + eps)*x)*w
//   rope_multi        the same theta and rope_yarn, x0 and x1 read back from shared memory
template <typename src_t, int n_per_lane>
static __global__ void qsa_pool_rope_f32(
        const char * __restrict__ k, const int64_t k_nb1, const int64_t k_nb2,
        const int32_t * __restrict__ cells, const int64_t cells_s1,
        const int r, const int n_blocks, const int n_rows,
        const float scale, const float bias, const int ncols, const float eps, const float * __restrict__ w,
        float * __restrict__ dst,
        const int n_dims, const int n_offs, const int32_t * __restrict__ pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const mrope_sections sections, const bool is_imrope) {
    constexpr int n_warps = 8;
    constexpr int n_cols  = n_per_lane*WARP_SIZE;
    __shared__ float s_y[n_warps][n_cols];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = blockIdx.x*n_warps + warp;

    if (row >= n_rows) {
        return;
    }

    const int s = row / n_blocks;
    const int b = row - s*n_blocks;

    const int32_t * blk = cells + s*cells_s1 + (int64_t) b*r;

    float x[n_per_lane];
    for (int i = 0; i < r; ++i) {
        const src_t * src = (const src_t *) (k + blk[i]*k_nb1 + s*k_nb2);
#pragma unroll
        for (int m = 0; m < n_per_lane; ++m) {
            const float v = ggml_cuda_cast<float>(src[m*WARP_SIZE + lane]);
            x[m] = i == 0 ? v : x[m] + v;
        }
    }

    float warp_sum[n_per_lane];
#pragma unroll
    for (int m = 0; m < n_per_lane; ++m) {
        x[m] = scale * x[m] + bias;

        float tmp = 0.0f;
        tmp += x[m] * x[m];
        warp_sum[m] = warp_reduce_sum(tmp);
    }

    // rms_norm_f32<256>'s second stage: lane i < 8 holds warp i's sum, and warps without a column hold 0
    float tmp = 0.0f;
#pragma unroll
    for (int m = 0; m < n_per_lane; ++m) {
        if (lane == m) {
            tmp = warp_sum[m];
        }
    }
    tmp = warp_reduce_sum(tmp);

    const float mean   = tmp / ncols;
    const float rscale = rsqrtf(mean + eps);

    float * y = s_y[warp];
#pragma unroll
    for (int m = 0; m < n_per_lane; ++m) {
        y[m*WARP_SIZE + lane] = rscale * x[m] * w[m*WARP_SIZE + lane];
    }
    __syncwarp();

    constexpr bool forward = true;

    const int ne02 = n_rows;
    const int i2   = row;

    float * dst_row = dst + (int64_t) row*ncols;

#pragma unroll
    for (int m = 0; m < n_per_lane; ++m) {
        const int e = m*WARP_SIZE + lane;

        if (e < n_offs || e >= n_offs + n_dims) {
            dst_row[e] = y[e];
            continue;
        }

        const bool first = e - n_offs < n_dims/2;
        const int  iw    = 2*(first ? e - n_offs : e - n_offs - n_dims/2); // relative idx

        const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
        const int sec_w = sections.v[1] + sections.v[0];
        const int sector = (iw / 2) % sect_dims;

        float theta_base = 0.0;
        if (is_imrope) {
            if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
                theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
            } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
                theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
            } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
                theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
            } else {
                theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
            }
        } else {
            if (sector < sections.v[0]) {
                theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
            } else if (sector >= sections.v[0] && sector < sec_w) {
                theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
            } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
                theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
            } else if (sector >= sec_w + sections.v[2]) {
                theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
            }
        }

        const float freq_factor = 1.0f; // rope_multi<forward, has_ff = false>

        float cos_theta;
        float sin_theta;

        rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

        const float x0 = y[n_offs + iw/2];
        const float x1 = y[n_offs + iw/2 + n_dims/2];

        if (first) {
            dst_row[e] = x0*cos_theta - x1*sin_theta;
        } else {
            dst_row[e] = x0*sin_theta + x1*cos_theta;
        }
    }
}

template <typename src_t>
static void qsa_pool_rope_f32_cuda(const int n_per_lane, const dim3 grid, const dim3 block, cudaStream_t stream,
        const char * k, const int64_t k_nb1, const int64_t k_nb2, const int32_t * cells, const int64_t cells_s1,
        const int r, const int n_blocks, const int n_rows,
        const float scale, const float bias, const int ncols, const float eps, const float * w, float * dst,
        const int n_dims, const int n_offs, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const mrope_sections sections, const bool is_imrope) {
#define QSA_POOL_ROPE(n) \
    qsa_pool_rope_f32<src_t, n><<<grid, block, 0, stream>>>(k, k_nb1, k_nb2, cells, cells_s1, r, n_blocks, n_rows, \
        scale, bias, ncols, eps, w, dst, n_dims, n_offs, pos, freq_scale, ext_factor, attn_factor, \
        corr_dims, theta_scale, sections, is_imrope)
    switch (n_per_lane) {
        case 1: QSA_POOL_ROPE(1); break;
        case 2: QSA_POOL_ROPE(2); break;
        case 4: QSA_POOL_ROPE(4); break;
        case 8: QSA_POOL_ROPE(8); break;
        default: GGML_ABORT("qsa pool: unsupported row width");
    }
#undef QSA_POOL_ROPE
}

void ggml_cuda_op_qsa_pool_rope(ggml_backend_cuda_context & ctx, ggml_tensor * rope, const ggml_cuda_qsa_pool & p) {
    const ggml_tensor * k     = p.k;
    const ggml_tensor * cells = p.cells;
    const ggml_tensor * pos   = rope->src[1];

    const int ncols    = (int) k->ne[0];
    const int n_stream = (int) cells->ne[1];
    const int n_blocks = (int) (cells->ne[0] / p.r);
    const int n_rows   = n_blocks*n_stream;

    GGML_ASSERT(rope->type == GGML_TYPE_F32 && ggml_is_contiguous(rope) && ggml_nrows(rope) == n_rows && rope->ne[0] == ncols);
    GGML_ASSERT(cells->type == GGML_TYPE_I32 && cells->nb[0] == sizeof(int32_t) && cells->ne[0] == (int64_t) p.r*n_blocks);
    GGML_ASSERT(k->nb[0] == ggml_type_size(k->type) && k->ne[2] == n_stream);
    GGML_ASSERT(pos->type == GGML_TYPE_I32 && pos->ne[0] == 4*(int64_t) n_rows);
    GGML_ASSERT(rope->src[2] == nullptr);
    GGML_ASSERT(ncols % WARP_SIZE == 0);

    const int n_dims     = ((int32_t *) rope->op_params)[1];
    const int mode       = ((int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((int32_t *) rope->op_params)[4];
    const int n_offs     = ((int32_t *) rope->op_params)[15];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    mrope_sections sections;

    memcpy(&freq_base,   (int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) rope->op_params + 11, sizeof(int)*4);

    GGML_ASSERT((mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION);
    GGML_ASSERT(n_dims % 2 == 0 && n_offs % 2 == 0 && n_offs + n_dims <= ncols);

    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // as rope_multi_cuda computes it
    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    const dim3 block(8*WARP_SIZE, 1, 1);
    const dim3 grid((n_rows + 7) / 8, 1, 1);

    const int64_t cells_s1 = cells->nb[1] / sizeof(int32_t);

#define QSA_POOL_ROPE_ARGS(src_t) \
        ncols / WARP_SIZE, grid, block, ctx.stream(), (const char *) k->data, (int64_t) k->nb[1], (int64_t) k->nb[2], \
        (const int32_t *) cells->data, cells_s1, p.r, n_blocks, n_rows, p.scale, p.bias, ncols, p.eps, \
        (const float *) p.w->data, (float *) rope->data, n_dims, n_offs, (const int32_t *) pos->data, \
        freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, sections, is_imrope
    switch (k->type) {
        case GGML_TYPE_F32:  qsa_pool_rope_f32_cuda<float>        (QSA_POOL_ROPE_ARGS(float));         break;
        case GGML_TYPE_F16:  qsa_pool_rope_f32_cuda<half>         (QSA_POOL_ROPE_ARGS(half));          break;
        case GGML_TYPE_BF16: qsa_pool_rope_f32_cuda<nv_bfloat16>  (QSA_POOL_ROPE_ARGS(nv_bfloat16));   break;
        default: GGML_ABORT("qsa pool: unsupported key type");
    }
#undef QSA_POOL_ROPE_ARGS
    CUDA_CHECK(cudaGetLastError());
}
