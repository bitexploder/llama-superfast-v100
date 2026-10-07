#include "qsa-union.cuh"

// qwen4exp QSA: the union of the cells the queries of a stream select (see ggml_qsa_union).
//
// One thread block per stream. The cells are marked in a shared-memory bitmap, so the union
// comes out in ascending cell order, and a prefix sum over the bitmap gives each cell its
// union slot. Each query's slots then raise its mask plane at their cell's union slot with an
// atomic max, which does not depend on the order the threads run in.

#define QSA_UNION_NT 1024

// raise *a to v; *a starts at -inf and v is never -0
static __device__ __forceinline__ void qsa_union_atomic_max(float * a, const float v) {
    if (!(v > -INFINITY)) {
        return;
    }
    if (v >= 0.0f) {
        atomicMax((int *) a, __float_as_int(v));
    } else {
        atomicMin((unsigned int *) a, __float_as_uint(v));
    }
}

template <int NT>
static __global__ void qsa_union_kernel(
        const int32_t * __restrict__ top_k,
        const half    * __restrict__ mask,
        const float   * __restrict__ bias,
        float         *              dst,
        const int     width,
        const int     n_tps,
        const int     n_words,
        const int     n_max,
        const int     n_planes,
        const int     pad_to,
        const int64_t s_m1,
        const int64_t s_m3) {
    static_assert(NT % WARP_SIZE == 0 && NT/WARP_SIZE <= WARP_SIZE, "bad block size");

    extern __shared__ uint32_t s_bits[];
    __shared__ int s_base[NT];
    __shared__ int s_warp[NT/WARP_SIZE];

    const int s    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int n_e  = n_tps*width;

    const int32_t * tk  = top_k + (int64_t) s*n_e;
    const float   * bi  = bias ? bias + (int64_t) s*n_e : nullptr;
    const half    * m   = mask + s*s_m3;
    float         * out = dst + (int64_t) s*n_planes*n_max;

    for (int w = tid; w < n_words; w += NT) {
        s_bits[w] = 0;
    }
    __syncthreads();

    for (int e = tid; e < n_e; e += NT) {
        const int c = tk[e];
        atomicOr(&s_bits[c >> 5], 1u << (c & 31));
    }
    __syncthreads();

    // each thread owns a run of words; an exclusive prefix sum of their counts gives the slots
    const int per = (n_words + NT - 1)/NT;
    const int w0  = min(tid*per, n_words);
    const int w1  = min(w0 + per, n_words);

    int cnt = 0;
    for (int w = w0; w < w1; ++w) {
        cnt += __popc(s_bits[w]);
    }

    int inc = cnt;
#pragma unroll
    for (int o = 1; o < WARP_SIZE; o <<= 1) {
        const int x = __shfl_up_sync(0xFFFFFFFF, inc, o, WARP_SIZE);
        if (lane >= o) {
            inc += x;
        }
    }
    if (lane == WARP_SIZE - 1) {
        s_warp[warp] = inc;
    }
    __syncthreads();
    if (warp == 0) {
        int x = lane < NT/WARP_SIZE ? s_warp[lane] : 0;
#pragma unroll
        for (int o = 1; o < WARP_SIZE; o <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFF, x, o, WARP_SIZE);
            if (lane >= o) {
                x += y;
            }
        }
        if (lane < NT/WARP_SIZE) {
            s_warp[lane] = x;
        }
    }
    __syncthreads();

    const int base  = (warp > 0 ? s_warp[warp - 1] : 0) + inc - cnt;
    const int total = s_warp[NT/WARP_SIZE - 1];
    s_base[tid] = base;

    // plane 0: the cells in ascending order
    int r = base;
    for (int w = w0; w < w1; ++w) {
        uint32_t b = s_bits[w];
        while (b) {
            const int bit = __ffs(b) - 1;
            out[r++] = (float) (w*32 + bit);
            b &= b - 1;
        }
    }
    for (int i = tid; i < (n_planes - 1)*n_max; i += NT) {
        out[n_max + i] = -INFINITY;
    }
    __syncthreads();

    // plane 0: the first cell up to a multiple of pad_to, then rows nothing reads
    const float first = out[0];
    const int   n_p   = ((total + pad_to - 1)/pad_to)*pad_to;
    for (int u = total + tid; u < n_max; u += NT) {
        out[u] = u < n_p ? first : -1.0f;
    }

    // planes 1 + t: each slot's mask entry plus bias at its cell's union slot
    for (int e = tid; e < n_e; e += NT) {
        const int t = e/width;
        const int c = tk[e];
        const int w = c >> 5;
        const int ch = w/per;

        int u = s_base[ch];
        for (int w2 = ch*per; w2 < w; ++w2) {
            u += __popc(s_bits[w2]);
        }
        u += __popc(s_bits[w] & ((1u << (c & 31)) - 1u));

        float v = __half2float(m[t*s_m1 + c]) + (bi ? bi[e] : 0.0f);
        if (v == 0.0f) {
            v = 0.0f; // -0 counts as +0
        }
        qsa_union_atomic_max(out + (int64_t) (1 + t)*n_max + u, v);
    }
}

bool ggml_cuda_qsa_union_supported(const ggml_tensor * op) {
    const ggml_tensor * top_k = op->src[0];
    const ggml_tensor * mask  = op->src[1];
    const ggml_tensor * bias  = op->src[2];

    return ggml_is_contiguous(top_k) && ggml_is_contiguous(op) &&
           (!bias || ggml_is_contiguous(bias)) &&
           mask->nb[0] == ggml_type_size(GGML_TYPE_F16) &&
           (mask->ne[0] + 31)/32 <= QSA_UNION_MAX_WORDS &&
           top_k->ne[0]*top_k->ne[1] < INT_MAX && op->ne[0]*op->ne[1] < INT_MAX;
}

void ggml_cuda_op_qsa_union(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * top_k = dst->src[0];
    const ggml_tensor * mask  = dst->src[1];
    const ggml_tensor * bias  = dst->src[2];

    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && mask->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!bias || bias->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_cuda_qsa_union_supported(dst));

    const int64_t width    = top_k->ne[0];
    const int64_t n_tps    = top_k->ne[1];
    const int64_t n_stream = top_k->ne[3];
    const int64_t n_words  = (mask->ne[0] + 31)/32;
    const int     pad_to   = ggml_get_op_params_i32(dst, 0);

    qsa_union_kernel<QSA_UNION_NT><<<n_stream, QSA_UNION_NT, n_words*sizeof(uint32_t), ctx.stream()>>>(
        (const int32_t *) top_k->data, (const half *) mask->data, bias ? (const float *) bias->data : nullptr,
        (float *) dst->data, (int) width, (int) n_tps, (int) n_words, (int) dst->ne[0], (int) dst->ne[1], pad_to,
        mask->nb[1]/sizeof(half), mask->nb[3]/sizeof(half));
    CUDA_CHECK(cudaGetLastError());
}
