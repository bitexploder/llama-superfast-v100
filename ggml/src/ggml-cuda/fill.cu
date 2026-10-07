#include "fill.cuh"
#include "convert.cuh"

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T>
static __global__ void fill_kernel(T * dst, const int64_t k, const T value) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;

    switch (dst->type) {
        case GGML_TYPE_F32:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>((float *)dst_d, k, value);
            break;
        case GGML_TYPE_F16:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>((half *)dst_d, k, ggml_cuda_cast<half>(value));
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}

// qwen4exp's QSA attention mask in one launch (LLAMA_FOLD_QSA_MASK): FILL(kq_mask, fill) -> SET_ROWS of
// the zero-filled rows at the selected cells -> ADD kq_mask. One block per (token, stream) row: every cell gets
// fill + mask, then, after a barrier, each selected cell gets zero + mask. The values are converted as fill_kernel,
// k_set_rows and k_bin_bcast convert them (the fill value on the host, the zero with ggml_cuda_cast, the sum in
// float and cast back), so dst is bit-identical; a cell selected twice gets the same value twice, as in set_rows.
template <typename T, typename idx_t>
static __global__ void qsa_mask_fill_rows_add(const T * mask, const idx_t * idx, T * dst, const int64_t n_kv,
        const int64_t n_top_k, const int64_t sm1, const int64_t sm3, const int64_t s1, const int64_t s3,
        const int64_t s10, const int64_t s11, const int64_t s12, const int64_t idx_ne1, const int64_t idx_ne2,
        const T fill, const float zero) {
    const int64_t t = blockIdx.x;
    const int64_t s = blockIdx.y;
    const T * m = mask + t*sm1 + s*sm3;
    T *       d = dst  + t*s1  + s*s3;

    for (int64_t kv = threadIdx.x; kv < n_kv; kv += blockDim.x) {
        const float result = (float) fill + (float) m[kv];
        d[kv] = (T) result;
    }
    __syncthreads();

    const T z = ggml_cuda_cast<T>(zero);
    const idx_t * row = idx + (t % idx_ne1)*s11 + (s % idx_ne2)*s12;
    for (int64_t k = threadIdx.x; k < n_top_k; k += blockDim.x) {
        const int64_t kv = row[k*s10];
        const float result = (float) z + (float) m[kv];
        d[kv] = (T) result;
    }
}

// The same when the sum overlaps the cells: one block reads every row's cells into shared memory before it writes
template <typename T, typename idx_t>
static __global__ void qsa_mask_fill_rows_add_staged(const T * mask, const idx_t * idx, T * dst, const int64_t n_kv,
        const int64_t n_top_k, const int64_t sm1, const int64_t sm3, const int64_t s1, const int64_t s3,
        const int64_t s10, const int64_t s11, const int64_t s12, const int64_t idx_ne1, const int64_t idx_ne2,
        const T fill, const float zero, const int64_t n_t, const int64_t n_s) {
    extern __shared__ char qsa_mask_smem[];
    idx_t * cells = (idx_t *) qsa_mask_smem; // [n_top_k, idx_ne1, idx_ne2]
    for (int64_t k = threadIdx.x; k < n_top_k*idx_ne1*idx_ne2; k += blockDim.x) {
        const int64_t c = k % n_top_k, r1 = (k / n_top_k) % idx_ne1, r2 = k / (n_top_k*idx_ne1);
        cells[k] = idx[c*s10 + r1*s11 + r2*s12];
    }
    __syncthreads();

    const T z = ggml_cuda_cast<T>(zero);
    for (int64_t s = 0; s < n_s; ++s) {
        for (int64_t t = 0; t < n_t; ++t) {
            const T * m = mask + t*sm1 + s*sm3;
            T *       d = dst  + t*s1  + s*s3;
            for (int64_t kv = threadIdx.x; kv < n_kv; kv += blockDim.x) {
                const float result = (float) fill + (float) m[kv];
                d[kv] = (T) result;
            }
            __syncthreads();
            const idx_t * row = cells + ((t % idx_ne1) + (s % idx_ne2)*idx_ne1)*n_top_k;
            for (int64_t k = threadIdx.x; k < n_top_k; k += blockDim.x) {
                const int64_t kv = row[k];
                const float result = (float) z + (float) m[kv];
                d[kv] = (T) result;
            }
            __syncthreads();
        }
    }
}

void ggml_cuda_op_qsa_mask_fill_rows_add(ggml_backend_cuda_context & ctx, const ggml_tensor * fill,
        const ggml_tensor * zeros, const ggml_tensor * set_rows, ggml_tensor * add) {
    const ggml_tensor * mask = fill->src[0];
    const ggml_tensor * idx  = set_rows->src[1]; // set_rows' legacy order: rows, cells, destination
    GGML_ASSERT(mask->type == add->type && (add->type == GGML_TYPE_F16 || add->type == GGML_TYPE_F32));
    GGML_ASSERT(idx->type == GGML_TYPE_I32 || idx->type == GGML_TYPE_I64);
    GGML_ASSERT(ggml_are_same_shape(mask, add) && mask->ne[2] == 1 && mask->nb[0] == ggml_type_size(mask->type));
    GGML_ASSERT(add->nb[0] == ggml_type_size(add->type));

    float fill_value, zero_value;
    memcpy(&fill_value, fill->op_params,  sizeof(float));
    memcpy(&zero_value, zeros->op_params, sizeof(float));

    const size_t  ts   = ggml_type_size(add->type);
    const size_t  its  = ggml_type_size(idx->type);
    const int64_t n_kv = mask->ne[0], n_top_k = zeros->ne[1];
    const int64_t sm1 = mask->nb[1]/ts, sm3 = mask->nb[3]/ts, s1 = add->nb[1]/ts, s3 = add->nb[3]/ts;
    const int64_t s10 = idx->nb[0]/its, s11 = idx->nb[1]/its, s12 = idx->nb[2]/its;

    // the sum may sit on the cells' bytes: then one block reads all of them before anything is written
    const uintptr_t a0 = (uintptr_t) add->data, a1 = a0 + ggml_nbytes(add);
    const uintptr_t i0 = (uintptr_t) idx->data, i1 = i0 + ggml_nbytes(idx);
    // LLAMA_FOLD_QSA_MASK=2 takes the staged kernel always, so that the tests reach it
    static const bool force_staged = [] {
        const char * e = getenv("LLAMA_FOLD_QSA_MASK");
        return e != nullptr && atoi(e) == 2;
    }();
    const bool staged = (a0 < i1 && i0 < a1) || (force_staged && n_top_k*idx->ne[1]*idx->ne[2]*its <= 32768);
    const size_t smem = staged ? n_top_k*idx->ne[1]*idx->ne[2]*its : 0;
    GGML_ASSERT(smem <= 32768);

    const dim3 grid(mask->ne[1], mask->ne[3], 1);
    cudaStream_t stream = ctx.stream();
#define QSA_MASK_LAUNCH(T, idx_t) \
    if (staged) { \
        qsa_mask_fill_rows_add_staged<T, idx_t><<<1, 1024, smem, stream>>>((const T *) mask->data, (const idx_t *) idx->data, \
            (T *) add->data, n_kv, n_top_k, sm1, sm3, s1, s3, s10, s11, s12, idx->ne[1], idx->ne[2], ggml_cuda_cast<T>(fill_value), \
            zero_value, mask->ne[1], mask->ne[3]); \
    } else { \
        qsa_mask_fill_rows_add<T, idx_t><<<grid, CUDA_FILL_BLOCK_SIZE, 0, stream>>>((const T *) mask->data, (const idx_t *) idx->data, \
            (T *) add->data, n_kv, n_top_k, sm1, sm3, s1, s3, s10, s11, s12, idx->ne[1], idx->ne[2], ggml_cuda_cast<T>(fill_value), zero_value); \
    }
    if (add->type == GGML_TYPE_F16) {
        if (idx->type == GGML_TYPE_I32) { QSA_MASK_LAUNCH(half, int32_t); } else { QSA_MASK_LAUNCH(half, int64_t); }
    } else {
        if (idx->type == GGML_TYPE_I32) { QSA_MASK_LAUNCH(float, int32_t); } else { QSA_MASK_LAUNCH(float, int64_t); }
    }
#undef QSA_MASK_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
