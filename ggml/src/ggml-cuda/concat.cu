#include "concat.cuh"
#include "unary.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}

// One thread per (channel, sequence): write the concatenated row, then each rollback slot's tail
// from it. Everything moves as 32-bit words, as concat_non_cont<unsigned int> and the memcpys it
// replaces do, so every byte is unchanged. A thread only reads and writes its own row.
static __global__ void concat_conv_slots_f32(
        const char * __restrict__ src0, const char * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne0, const int64_t ne1,
        const size_t nb00, const size_t nb01, const size_t nb02,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2,
        const ggml_cuda_conv_slots slots) {
    const int64_t i1 = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i2 = blockIdx.y;
    if (i1 >= ne1) {
        return;
    }

    uint32_t * row = (uint32_t *) (dst + i2*nb2 + i1*nb1);
    // with a folded gather the cache row may be one this thread's slot saves overwrite below: it reads its own
    // channel's elements here, before it writes them there
    const char * s0 = slots.gather_ids != nullptr ? slots.gather_rows + (int64_t) slots.gather_ids[i2]*slots.gather_nb1
                                                  : src0 + i2*nb02;
    for (int64_t i0 = 0; i0 < ne00; ++i0) {
        row[i0] = *(const uint32_t *) (s0 + i1*nb01 + i0*nb00);
    }
    for (int64_t i0 = ne00; i0 < ne0; ++i0) {
        row[i0] = *(const uint32_t *) (src1 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
    }

    for (int k = 0; k < slots.n_slots; ++k) {
        uint32_t * s = (uint32_t *) (slots.data[k] + i2*slots.nb1[k]) + i1*slots.n_cols;
        for (int64_t c = 0; c < slots.n_cols; ++c) {
            s[c] = row[slots.off[k] + c];
        }
    }

    if (slots.conv_y != nullptr) {
        // ssm_conv_f32<true, 128, 4> for channel i1: the same products summed in the same order, then silu.
        // conv_y may be src1 itself, laid out alike: this thread has already read the elements it writes
        constexpr int d_conv = 4;
        const float * w_row = (const float *) ((const char *) slots.conv_w + i1*slots.conv_w_nb1);
        float w[d_conv];
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            w[j] = w_row[j];
        }
        char * y = (char *) slots.conv_y + i2*slots.conv_y_nb2 + i1*sizeof(float);
        for (int64_t i = 0; i < ne0 - (d_conv - 1); ++i) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sumf += __uint_as_float(row[i + j]) * w[j]; // read back as written, not through a float alias
            }
            sumf += slots.conv_b;
            *(float *) (y + i*slots.conv_y_nb1) = ggml_cuda_op_silu_single(sumf);
        }
    }
}

// concat_conv_slots_f32 for a verify or decode batch (NT tokens a sequence, 1-8; NS sequences, 1-4; slots of
// NC = 3 columns; channels a multiple of 64), one thread per channel for every sequence. All of the channel's elements of
// every sequence are loaded into registers before anything is written, so a sequence's cache row may be one that another
// sequence's slot saves overwrite: the gather folds for several sequences too (LLAMA_FOLD_CONV_GATHER). Each warp then
// stages its 32 channels' rows of a sequence in shared memory and writes the concat rows and every slot's columns as
// whole 128-byte lines (the kernel above writes each thread's words 44 or 12 bytes apart, and the stores bound it). The
// concat output is not written when the slot saves and the conv are its only readers (slots.write_dst). Every word moves
// as the same 32 bits, and the conv sums the same products in the same order, so every output is bit-identical.
template <int NT, int NS>
static __global__ void __launch_bounds__(64) concat_conv_slots_reg_f32(
        const char * __restrict__ src0, const char * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne1,
        const size_t nb00, const size_t nb01, const size_t nb02,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2,
        const ggml_cuda_conv_slots slots) {
    constexpr int NC  = 3;
    constexpr int NE0 = NC + NT;
    const int     lane = threadIdx.x % WARP_SIZE;
    const int     wid  = threadIdx.x / WARP_SIZE;
    const int64_t i1   = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;  // every thread has a channel (ne1 % 64 == 0)
    const int64_t c0   = i1 - lane;                                      // the warp's first channel

    __shared__ uint32_t stage[2][WARP_SIZE*NE0];
    uint32_t * sw = stage[wid];

    uint32_t r[NS][NE0];
#pragma unroll
    for (int s = 0; s < NS; ++s) {
        const char * s0 = slots.gather_ids != nullptr ? slots.gather_rows + (int64_t) slots.gather_ids[s]*slots.gather_nb1
                                                      : src0 + s*nb02;
#pragma unroll
        for (int i0 = 0; i0 < NC; ++i0) {
            r[s][i0] = *(const uint32_t *) (s0 + i1*nb01 + i0*nb00);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            r[s][NC + t] = *(const uint32_t *) (src1 + s*nb12 + i1*nb11 + t*nb10);
        }
    }

    // ssm_conv_f32<true, 128, 4>'s weights for this channel
    constexpr int d_conv = 4;
    float w[d_conv];
    if (slots.conv_y != nullptr) {
        const float * w_row = (const float *) ((const char *) slots.conv_w + i1*slots.conv_w_nb1);
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            w[j] = w_row[j];
        }
    }

#pragma unroll
    for (int s = 0; s < NS; ++s) {
        // the warp's rows of this sequence, channel after channel as the concat lays them out
        __syncwarp();
#pragma unroll
        for (int i0 = 0; i0 < NE0; ++i0) {
            sw[lane*NE0 + i0] = r[s][i0];
        }
        __syncwarp();

        if (slots.write_dst) {
            uint32_t * row = (uint32_t *) (dst + s*nb2 + c0*nb1);
#pragma unroll
            for (int q = 0; q < NE0; ++q) {
                row[q*WARP_SIZE + lane] = sw[q*WARP_SIZE + lane];
            }
        }

#pragma unroll
        for (int k = 0; k < GGML_CUDA_CONV_SLOTS_MAX; ++k) {
            if (k < slots.n_slots) {
                // the warp's 32 channels' columns [off, off + NC) are 32*NC consecutive words of the slot row
                uint32_t * d = (uint32_t *) (slots.data[k] + s*slots.nb1[k]) + c0*NC;
                const int off = (int) slots.off[k];
#pragma unroll
                for (int q = 0; q < NC; ++q) {
                    const int e = q*WARP_SIZE + lane;
                    d[e] = sw[(e / NC)*NE0 + off + e % NC];
                }
            }
        }

        if (slots.conv_y != nullptr) {
            char * y = (char *) slots.conv_y + s*slots.conv_y_nb2 + i1*sizeof(float);
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                float sumf = 0.0f;
#pragma unroll
                for (int j = 0; j < d_conv; ++j) {
                    sumf += __uint_as_float(r[s][i + j]) * w[j];
                }
                sumf += slots.conv_b;
                *(float *) (y + i*slots.conv_y_nb1) = ggml_cuda_op_silu_single(sumf);
            }
        }
    }
}

bool ggml_cuda_conv_slots_reg_ok(const ggml_tensor * dst, const ggml_cuda_conv_slots & slots) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_GDN_CONV_REG");
        return e == nullptr || atoi(e) != 0;
    }();
    const int64_t nt = dst->ne[0] - 3;
    return on && dst->src[0]->ne[0] == 3 && slots.n_cols == 3 && nt >= 1 && nt <= 8 && dst->ne[2] >= 1 && dst->ne[2] <= 4 &&
           dst->ne[1] % 64 == 0 && dst->nb[1] == (size_t) dst->ne[0]*sizeof(float) && (slots.conv_y == nullptr || slots.conv_w != nullptr);
}

void ggml_cuda_op_concat_conv_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_conv_slots & slots) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_get_op_params_i32(dst, 0) == 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->nb[0] == sizeof(float) && dst->ne[3] == 1 && src0->ne[3] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(slots.n_slots > 0 && slots.n_slots <= GGML_CUDA_CONV_SLOTS_MAX);

    if (ggml_cuda_conv_slots_reg_ok(dst, slots)) {
        const int block = 64;
        const dim3 grid((dst->ne[1] + block - 1) / block, 1, 1);
#define CONV_REG_LAUNCH(NT, NS) \
        concat_conv_slots_reg_f32<NT, NS><<<grid, block, 0, ctx.stream()>>>( \
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data, dst->ne[1], \
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[0], src1->nb[1], src1->nb[2], dst->nb[1], dst->nb[2], slots)
#define CONV_REG_NS(NT) \
        switch (dst->ne[2]) { \
            case 1:  CONV_REG_LAUNCH(NT, 1); break; \
            case 2:  CONV_REG_LAUNCH(NT, 2); break; \
            case 3:  CONV_REG_LAUNCH(NT, 3); break; \
            default: CONV_REG_LAUNCH(NT, 4); break; \
        }
        switch (dst->ne[0] - 3) {
            case 1:  CONV_REG_NS(1); break;
            case 2:  CONV_REG_NS(2); break;
            case 3:  CONV_REG_NS(3); break;
            case 4:  CONV_REG_NS(4); break;
            case 5:  CONV_REG_NS(5); break;
            case 6:  CONV_REG_NS(6); break;
            case 7:  CONV_REG_NS(7); break;
            default: CONV_REG_NS(8); break;
        }
#undef CONV_REG_NS
#undef CONV_REG_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // several sequences with the gather folded, or the concat output left unwritten, need the kernel above
    // (ggml_cuda_try_fuse_conv_slots checks it)
    GGML_ASSERT((slots.gather_ids == nullptr || dst->ne[2] == 1) && slots.write_dst);

    const int block = 128;
    const dim3 grid((dst->ne[1] + block - 1) / block, dst->ne[2], 1);
    concat_conv_slots_f32<<<grid, block, 0, ctx.stream()>>>(
            (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
            src0->ne[0], dst->ne[0], dst->ne[1],
            src0->nb[0], src0->nb[1], src0->nb[2],
            src1->nb[0], src1->nb[1], src1->nb[2],
            dst->nb[1], dst->nb[2], slots);
    CUDA_CHECK(cudaGetLastError());
}
