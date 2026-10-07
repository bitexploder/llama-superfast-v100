#include "qsa-select.cuh"

#include <climits>

// qwen4exp QSA block selection, one thread block per row (see ggml_qsa_select).
//
// A radix select over the 32-bit keys finds the k-th largest key; every key above it is taken,
// and among the keys equal to it the lowest block indices are taken. The histogram counts are
// integers, so their order of accumulation does not matter, and the selected blocks are written
// in ascending block order through a prefix sum. The result is therefore the same on every run
// and equal to the CPU backend's.

#define QSA_SELECT_NT        512
#define QSA_SELECT_SMEM_KEYS 11264 // keys staged in shared memory up to this many blocks (44 KiB)

// must match ggml_qsa_select_key in the CPU backend. larger float, larger key; zeros and
// subnormals all count as +0 (the device flushes subnormals); 0 means "not selectable".
static __device__ __forceinline__ uint32_t qsa_select_key(const float v) {
    if (!(v > -INFINITY)) {
        return 0;
    }
    uint32_t b = __float_as_uint(v);
    if ((b & 0x7F800000u) == 0) {
        b = 0;
    }
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// exclusive prefix sum of two counters over the thread block; *tot receives the totals
template <int NT>
static __device__ __forceinline__ int2 qsa_select_scan2(const int2 v, int2 * tot, int2 * s_warp) {
    static_assert(NT % WARP_SIZE == 0 && NT/WARP_SIZE <= WARP_SIZE, "bad block size");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    int2 inc = v;
#pragma unroll
    for (int o = 1; o < WARP_SIZE; o <<= 1) {
        const int x = __shfl_up_sync(0xFFFFFFFF, inc.x, o, WARP_SIZE);
        const int y = __shfl_up_sync(0xFFFFFFFF, inc.y, o, WARP_SIZE);
        if (lane >= o) {
            inc.x += x;
            inc.y += y;
        }
    }
    if (lane == WARP_SIZE - 1) {
        s_warp[warp] = inc;
    }
    __syncthreads();

    if (warp == 0) {
        int2 w = lane < NT/WARP_SIZE ? s_warp[lane] : make_int2(0, 0);
#pragma unroll
        for (int o = 1; o < WARP_SIZE; o <<= 1) {
            const int x = __shfl_up_sync(0xFFFFFFFF, w.x, o, WARP_SIZE);
            const int y = __shfl_up_sync(0xFFFFFFFF, w.y, o, WARP_SIZE);
            if (lane >= o) {
                w.x += x;
                w.y += y;
            }
        }
        if (lane < NT/WARP_SIZE) {
            s_warp[lane] = w;
        }
    }
    __syncthreads();

    const int2 base = warp > 0 ? s_warp[warp - 1] : make_int2(0, 0);
    *tot = s_warp[NT/WARP_SIZE - 1];
    __syncthreads(); // s_warp is reused by the next call

    return make_int2(base.x + inc.x - v.x, base.y + inc.y - v.y);
}

template <int NT>
static __global__ void qsa_select_kernel(
        const float   * __restrict__ score,
        const float   * __restrict__ bias,
        const int32_t * __restrict__ blk_cells,
        const int32_t * __restrict__ tail,
        int32_t       *              dst,
        const int     n_blocks,
        const int     r,
        const int     n_rows,
        const int     n_tail,
        const int     k,
        const int     width,
        const int64_t plane,
        const bool    keys_in_smem,
        const bool    packed) {
    static_assert(NT >= 256, "one thread per histogram bin");

    extern __shared__ uint32_t s_keys[];
    __shared__ int      s_hist[256];
    __shared__ int2     s_warp[NT/WARP_SIZE];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_rank;
    __shared__ int      s_done;

    const int64_t ir  = blockIdx.x;
    const int64_t s   = ir/n_rows;
    const int     tid = threadIdx.x;

    const float   * sc    = score + ir*n_blocks;
    const float   * bi    = bias + ir*(packed ? (n_blocks + 31)/32 : n_blocks);
    const uint32_t * bits = (const uint32_t *) bi;
    const auto bias_of = [&](const int b) -> float {
        return packed ? (((bits[b >> 5] >> (b & 31)) & 1) ? 0.0f : -INFINITY) : bi[b];
    };
    const int32_t * cells = blk_cells + s*r*n_blocks;
    const int32_t * tl    = tail + ir*n_tail;

    int32_t * out0 = dst + ir*width;
    int32_t * out1 = out0 + plane;

    int2 tot;

    // count the selectable blocks, staging their keys if they fit
    int nv = 0;
    for (int b = tid; b < n_blocks; b += NT) {
        const uint32_t key = qsa_select_key(sc[b] + bias_of(b));
        if (keys_in_smem) {
            s_keys[b] = key;
        }
        nv += key != 0;
    }
    qsa_select_scan2<NT>(make_int2(nv, 0), &tot, s_warp);
    const int n_valid = tot.x;

    const auto key_of = [&](const int b) -> uint32_t {
        return keys_in_smem ? s_keys[b] : qsa_select_key(sc[b] + bias_of(b));
    };

    // a block is taken if its masked key is above prefix, or equal to it and among the
    // first `need` such blocks by index. with pmask 0 that is every selectable block.
    uint32_t prefix = 0;
    uint32_t pmask  = 0;
    int      need   = n_valid;

    if (n_valid > k) {
        int rank = k; // rank of the k-th largest key within the keys matching prefix

        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += NT) {
                s_hist[i] = 0;
            }
            __syncthreads();

            for (int b = tid; b < n_blocks; b += NT) {
                const uint32_t key = key_of(b);
                if (key != 0 && (key & pmask) == prefix) {
                    atomicAdd(&s_hist[(key >> shift) & 0xFF], 1);
                }
            }
            __syncthreads();

            // thread t looks at bin 255 - t, so the exclusive sum counts the keys in higher bins
            const int  h  = tid < 256 ? s_hist[255 - tid] : 0;
            const int2 ex = qsa_select_scan2<NT>(make_int2(h, 0), &tot, s_warp);
            if (tid < 256 && ex.x < rank && rank <= ex.x + h) {
                s_prefix = prefix | ((uint32_t) (255 - tid) << shift);
                s_rank   = rank - ex.x;
                s_done   = h == rank - ex.x; // the whole bin is taken, so no lower bits matter
            }
            __syncthreads();

            prefix = s_prefix;
            pmask |= 0xFFu << shift;
            rank   = s_rank;

            const bool done = s_done;
            __syncthreads();

            if (done) {
                break;
            }
        }

        need = rank;
    }

    // write the taken blocks' cells in ascending block order:
    // the slot of block b is (blocks above before b) + min(blocks equal before b, need)
    int2 carry = make_int2(0, 0);
    for (int b0 = 0; b0 < n_blocks; b0 += NT) {
        const int      b   = b0 + tid;
        const uint32_t key = b < n_blocks ? key_of(b) : 0;
        const uint32_t km  = key & pmask;
        const int      gt  = key != 0 && km >  prefix;
        const int      eq  = key != 0 && km == prefix;

        const int2 ex = qsa_select_scan2<NT>(make_int2(gt, eq), &tot, s_warp);

        const int eq_before = carry.y + ex.y;
        if (gt || (eq && eq_before < need)) {
            const int pos = carry.x + ex.x + min(eq_before, need);
            for (int j = 0; j < r; ++j) {
                const int slot = pos*r + j;
                if (slot < width) {
                    out0[slot] = cells[(int64_t) b*r + j];
                }
            }
        }

        carry.x += tot.x;
        carry.y += tot.y;
    }

    const int n_sel = carry.x + min(carry.y, need);

    int n_t = 0;
    while (n_t < n_tail && tl[n_t] >= 0) {
        n_t++;
    }

    const int n = min(width, n_sel*r + n_t);

    for (int j = tid; j < n_t; j += NT) {
        const int slot = n_sel*r + j;
        if (slot < width) {
            out0[slot] = tl[j];
        }
    }
    __syncthreads();

    const int32_t first = n_sel > 0 ? out0[0] : (n_t > 0 ? tl[0] : 0);

    for (int j = tid; j < width; j += NT) {
        if (j >= n) {
            out0[j] = first;
        }
        out1[j] = j < n ? 0 : INT_MIN;
    }
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score     = dst->src[0];
    const ggml_tensor * bias      = dst->src[1];
    const ggml_tensor * blk_cells = dst->src[2];
    const ggml_tensor * tail      = dst->src[3];

    GGML_ASSERT(score->type == GGML_TYPE_F32 && (bias->type == GGML_TYPE_F32 || bias->type == GGML_TYPE_I32));
    GGML_ASSERT(blk_cells->type == GGML_TYPE_I32 && tail->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(score) && ggml_is_contiguous(bias));
    GGML_ASSERT(ggml_is_contiguous(blk_cells) && ggml_is_contiguous(tail) && ggml_is_contiguous(dst));

    const int64_t n_blocks = score->ne[0];
    const int64_t n_rows   = score->ne[1];
    const int64_t n_stream = score->ne[2];
    const int64_t r        = blk_cells->ne[0]/n_blocks;
    const int64_t n_tail   = tail->ne[0];
    const int64_t width    = dst->ne[0];
    const int     k        = ggml_get_op_params_i32(dst, 0);
    const int64_t plane    = width*n_rows*n_stream;

    GGML_ASSERT(n_blocks*r < INT_MAX && width < INT_MAX);

    const bool   keys_in_smem = n_blocks <= QSA_SELECT_SMEM_KEYS;
    const size_t smem         = keys_in_smem ? n_blocks*sizeof(uint32_t) : 0;

    qsa_select_kernel<QSA_SELECT_NT><<<n_rows*n_stream, QSA_SELECT_NT, smem, ctx.stream()>>>(
        (const float *) score->data, (const float *) bias->data,
        (const int32_t *) blk_cells->data, (const int32_t *) tail->data, (int32_t *) dst->data,
        (int) n_blocks, (int) r, (int) n_rows, (int) n_tail, k, (int) width, plane, keys_in_smem,
        bias->type == GGML_TYPE_I32);
    CUDA_CHECK(cudaGetLastError());
}
