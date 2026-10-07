#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

// LLAMA_TOPK_TILED (default OFF: measured slower than the CUB chain, +20 us a call at 248,320 logits; 1 = this kernel): k <= 32 of a long row without CUB's sort-like chain (its three
// DeviceTopKKernel passes, the last filter and, in the sampler, the gathers cost 4 launches and about 20 us a row at 248,320 logits).
// Kernel A: each block takes a contiguous slice of TOPK_TILE_IPT*TOPK_TILE_THREADS logits, k rounds of a block-wide arg-max over
// 64-bit keys (the logit's order-preserving bits, then the inverted index, so equal logits rank lowest index first) and writes its
// k best keys. Kernel B: one block merges the blocks' keys the same way and writes the k indices, best first. CUB's unsorted output
// order is decided by atomics and varies from run to run; this one is fixed (logit descending, ties by index), the set is the same.
#define TOPK_TILE_THREADS 256
#define TOPK_TILE_IPT     8
#define TOPK_MERGE_IPT    16
#define TOPK_TILE_MAXK    32

static __device__ __forceinline__ unsigned long long topk_tile_key(const float v, const uint32_t idx) {
    const uint32_t bits = __float_as_uint(v);
    const uint32_t ord  = bits ^ ((uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U);
    return ((unsigned long long) ord << 32) | (uint32_t) (0xFFFFFFFFu - idx);
}

static __device__ __forceinline__ unsigned long long topk_tile_shfl_xor_max(unsigned long long m, const int off) {
    const uint32_t lo = __shfl_xor_sync(0xFFFFFFFF, (uint32_t) m, off);
    const uint32_t hi = __shfl_xor_sync(0xFFFFFFFF, (uint32_t) (m >> 32), off);
    const unsigned long long o = ((unsigned long long) hi << 32) | lo;
    return o > m ? o : m;
}

// k rounds of the block-wide maximum over IPT keys a thread (0 = no key); the winner of round r goes to out(r, key)
template <int IPT, typename Out>
static __device__ __forceinline__ void topk_tile_select(unsigned long long (&key)[IPT], const int k, Out out) {
    __shared__ unsigned long long wmax[2][TOPK_TILE_THREADS/32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int r = 0; r < k; ++r) {
        unsigned long long m = 0;
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            m = key[i] > m ? key[i] : m;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = topk_tile_shfl_xor_max(m, off);
        }
        if (lane == 0) {
            wmax[r & 1][warp] = m;
        }
        __syncthreads();
        unsigned long long w = wmax[r & 1][0];
#pragma unroll
        for (int i = 1; i < TOPK_TILE_THREADS/32; ++i) {
            w = wmax[r & 1][i] > w ? wmax[r & 1][i] : w;
        }
        if (threadIdx.x == 0) {
            out(r, w);
        }
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            key[i] = key[i] == w ? 0 : key[i];
        }
    }
}

static __global__ void __launch_bounds__(TOPK_TILE_THREADS) topk_tile_a(const float * __restrict__ src, unsigned long long * __restrict__ cand,
        const int ncols, const int k, const size_t row_stride) {
    const float * row = src + blockIdx.y*row_stride;
    const int base = blockIdx.x*(TOPK_TILE_THREADS*TOPK_TILE_IPT);
    unsigned long long key[TOPK_TILE_IPT];
#pragma unroll
    for (int i = 0; i < TOPK_TILE_IPT; ++i) {
        const int idx = base + i*TOPK_TILE_THREADS + threadIdx.x;
        key[i] = idx < ncols ? topk_tile_key(row[idx], idx) : 0;
    }
    unsigned long long * out = cand + ((size_t) blockIdx.y*gridDim.x + blockIdx.x)*k;
    topk_tile_select<TOPK_TILE_IPT>(key, k, [&](const int r, const unsigned long long w) { out[r] = w; });
}

static __global__ void __launch_bounds__(TOPK_TILE_THREADS) topk_tile_b(const unsigned long long * __restrict__ cand, int * __restrict__ dst,
        const int ncand, const int k) {
    const unsigned long long * in = cand + (size_t) blockIdx.x*ncand;
    unsigned long long key[TOPK_MERGE_IPT];
#pragma unroll
    for (int i = 0; i < TOPK_MERGE_IPT; ++i) {
        const int j = i*TOPK_TILE_THREADS + threadIdx.x;
        key[i] = j < ncand ? in[j] : 0;
    }
    int * out = dst + (size_t) blockIdx.x*k;
    topk_tile_select<TOPK_MERGE_IPT>(key, k, [&](const int r, const unsigned long long w) { out[r] = (int) (0xFFFFFFFFu - (uint32_t) w); });
}

static bool top_k_tiled(ggml_cuda_pool & pool, const float * src, int * dst, const int64_t ncols, const int64_t nrows, const int k,
                        const size_t row_stride, cudaStream_t stream) {
    static const bool on = [] { const char * e = getenv("LLAMA_TOPK_TILED"); return e != nullptr && atoi(e) != 0; }();
    const int64_t per_block = TOPK_TILE_THREADS*TOPK_TILE_IPT;
    const int64_t nblk      = (ncols + per_block - 1)/per_block;
    if (!on || k < 1 || k > TOPK_TILE_MAXK || nblk < 2 || nblk*k > TOPK_TILE_THREADS*TOPK_MERGE_IPT || ncols >= (1ll << 31) || nrows > 65535) {
        return false;
    }
    ggml_cuda_pool_alloc<unsigned long long> cand(pool, (size_t) nrows*nblk*k);
    topk_tile_a<<<dim3((unsigned) nblk, (unsigned) nrows), TOPK_TILE_THREADS, 0, stream>>>(src, cand.get(), (int) ncols, k, row_stride);
    CUDA_CHECK(cudaGetLastError());
    topk_tile_b<<<(unsigned) nrows, TOPK_TILE_THREADS, 0, stream>>>(cand.get(), dst, (int) (nblk*k), k);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    if (src0->nb[1] % sizeof(float) == 0 && top_k_tiled(pool, src0_d, dst_d, ncols, nrows, (int) k, src0->nb[1]/sizeof(float), stream)) {
        return;
    }
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

// GGML_OP_TOP_K_SPLIT: the k highest of this device's slice by ggml_cuda_op_top_k, written into this device's
// slot of dst (values, then their indices in the whole row as floats) with 0 in every other slot (ggml.h)
static __global__ void top_k_split_write(const float * src, const int * ids, float * dst, const int k, const int n_dev,
        const int rank, const int offset) {
    const int n = 2*k*n_dev;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int half = i / (k*n_dev);
        const int slot = (i % (k*n_dev)) / k;
        const int e    = i % k;
        float v = 0.0f;
        if (slot == rank) {
            const int id = ids[e];
            v = half == 0 ? src[id] : (float) (id + offset);
        }
        dst[i] = v;
    }
}

// the split's top k of one row in two kernels, for the MTP draft step's head slice (49,152 logits, k 26), in place of
// the one-card top-k (cub's DeviceTopK, four kernels) and top_k_split_write. Each block holds its keys in registers (ordered keys;
// -0.0 as +0.0, as the float compare the pick ranks by), finds the bin of its k-th highest key 11 bits at a time (a shared
// histogram and a suffix scan), keeps every key above it and the bin's keys once the bin is small, and ranks those candidates
// by (key descending, index ascending), a total order. First 48 blocks of 1,024 keys each keep their k highest, then one block
// takes the k highest of those and writes them, in that order, into this device's slot of dst. The k highest are the one-card
// op's set whenever the k-th and (k+1)-th values differ. LLAMA_TOPK_SPLIT_FAST=0: the one-card op, as before
#define TOPK_SPLIT_THREADS 1024
#define TOPK_SPLIT_BINS    2048   // 11 bits a level
#define TOPK_SPLIT_CAP     1024   // candidates kept at most

static __device__ __forceinline__ unsigned int topk_split_key(const float v) {
    unsigned int u = __float_as_uint(v == 0.0f ? 0.0f : v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block: the k highest of its n keys (key, index) by (key descending, index ascending), into out_v/out_i (FINAL: dst's slot)
template <int VPT, bool FINAL>
static __global__ void __launch_bounds__(TOPK_SPLIT_THREADS) topk_split_select(const float * __restrict__ src, const float * __restrict__ in_v,
        const int * __restrict__ in_i, const int n_all, const int k, float * __restrict__ out_v, int * __restrict__ out_i,
        const int n_dev, const int rank, const int offset) {
    __shared__ int       hist[TOPK_SPLIT_BINS];
    __shared__ int       warp_sum[32];
    __shared__ unsigned long long cand[TOPK_SPLIT_CAP];
    __shared__ int       order[64];
    __shared__ int       n_cand;
    __shared__ int       s_bin, s_gt, s_in;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int base = blockIdx.x*TOPK_SPLIT_THREADS*VPT;
    const int n    = min(n_all - base, TOPK_SPLIT_THREADS*VPT);

    unsigned int key[VPT];
    int          idx[VPT];
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        const int p = tid + j*TOPK_SPLIT_THREADS;
        key[j] = p < n ? topk_split_key(in_v[base + p]) : 0u;
        idx[j] = p < n ? (in_i ? in_i[base + p] : base + p) : -1;
    }
    if (tid == 0) {
        n_cand = 0;
    }

    unsigned int prefix = 0, pmask = 0;
    int need = min(k, n);
    int shift = 21;
    for (int level = 0; level < 3; ++level) {
        const int nbits = level < 2 ? 11 : 10;
        const unsigned int bmask = (1u << nbits) - 1;
        for (int b = tid; b < TOPK_SPLIT_BINS; b += TOPK_SPLIT_THREADS) {
            hist[b] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            if (idx[j] >= 0 && (key[j] & pmask) == prefix) {
                atomicAdd(&hist[(key[j] >> shift) & bmask], 1);
            }
        }
        __syncthreads();
        // suffix sums from the top bin down: thread t takes bins 2047 - 2t and 2046 - 2t
        const int h0 = hist[TOPK_SPLIT_BINS - 1 - 2*tid], h1 = hist[TOPK_SPLIT_BINS - 2 - 2*tid];
        int v = h0 + h1;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xFFFFFFFF, v, o);
            v += lane >= o ? u : 0;
        }
        if (lane == 31) {
            warp_sum[wid] = v;
        }
        __syncthreads();
        if (wid == 0) {
            int w = warp_sum[lane];
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int u = __shfl_up_sync(0xFFFFFFFF, w, o);
                w += lane >= o ? u : 0;
            }
            warp_sum[lane] = w;
        }
        __syncthreads();
        const int incl = v + (wid > 0 ? warp_sum[wid - 1] : 0);
        const int excl = incl - h0 - h1;
        if (excl < need && incl >= need) {
            const bool first = excl + h0 >= need;
            s_bin = first ? TOPK_SPLIT_BINS - 1 - 2*tid : TOPK_SPLIT_BINS - 2 - 2*tid;
            s_gt  = first ? excl : excl + h0;
            s_in  = first ? h0 : h1;
        }
        __syncthreads();
        const unsigned int bin = (unsigned int) s_bin;
        const int gt = s_gt, in = s_in;
        const bool last = gt + in <= TOPK_SPLIT_CAP - n_cand || level == 2;
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            if (idx[j] >= 0 && (key[j] & pmask) == prefix) {
                const unsigned int bj = (key[j] >> shift) & bmask;
                if (bj > bin || (last && bj == bin)) {
                    const int c = atomicAdd(&n_cand, 1);
                    if (c < TOPK_SPLIT_CAP) {
                        cand[c] = ((unsigned long long) key[j] << 32) | (unsigned long long) (0xFFFFFFFFu - (unsigned int) idx[j]);
                    }
                }
            }
        }
        __syncthreads();
        if (last) {
            break;
        }
        need  -= gt;
        prefix |= bin << shift;
        pmask  |= bmask << shift;
        shift  -= level == 0 ? 11 : 10;
    }
    const int nc = min(n_cand, TOPK_SPLIT_CAP);
    const int kk = min(k, n);
    for (int c = tid; c < nc; c += TOPK_SPLIT_THREADS) {
        const unsigned long long kc = cand[c];
        int r = 0;
        for (int d = 0; d < nc; ++d) {
            r += cand[d] > kc;
        }
        if (r < kk) {
            order[r] = c;
        }
    }
    __syncthreads();
    if constexpr (FINAL) {
        // as top_k_split_write: values, then indices in the whole row as floats, in this device's slot; 0 in the others
        const int nw = 2*k*n_dev;
        for (int i = tid; i < nw; i += TOPK_SPLIT_THREADS) {
            const int half = i / (k*n_dev);
            const int slot = (i % (k*n_dev)) / k;
            const int e    = i % k;
            float val = 0.0f;
            if (slot == rank) {
                const int id = (int) (0xFFFFFFFFu - (unsigned int) (cand[order[e]] & 0xFFFFFFFFull));
                val = half == 0 ? src[id] : (float) (id + offset);
            }
            out_v[i] = val;
        }
    } else {
        for (int e = tid; e < k; e += TOPK_SPLIT_THREADS) {
            const int o = blockIdx.x*k + e;
            if (e < kk) {
                const int id = (int) (0xFFFFFFFFu - (unsigned int) (cand[order[e]] & 0xFFFFFFFFull));
                out_v[o] = src[id];
                out_i[o] = id;
            } else {
                out_v[o] = -INFINITY; // a block of fewer than k keys: never chosen over a real one
                out_i[o] = 0x7FFFFFFF;
            }
        }
    }
}

void ggml_cuda_op_top_k_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_nrows(src0) == 1);

    const int k      = ggml_get_op_params_i32(dst, 0);
    const int n_dev  = ggml_get_op_params_i32(dst, 1);
    const int rank   = ggml_get_op_params_i32(dst, 2);
    const int offset = ggml_get_op_params_i32(dst, 3);
    GGML_ASSERT(src0->ne[0] >= k && rank < n_dev && ggml_nelements(dst) == 2*(int64_t) k*n_dev);

    static const bool fast = [] { const char * e = getenv("LLAMA_TOPK_SPLIT_FAST"); return e == nullptr || atoi(e) != 0; }();
    const int64_t ncols   = src0->ne[0];
    const int     n_tiles = (int) ((ncols + TOPK_SPLIT_THREADS - 1)/TOPK_SPLIT_THREADS);
    // (rows of at most 65,536: the draft's head slices; the target's full-vocabulary rows keep the one-card op, whatever their k)
    if (fast && k <= 64 && ncols <= 65536 && (int64_t) n_tiles*k <= 2*TOPK_SPLIT_THREADS) {
        ggml_cuda_pool_alloc<float> cv(ctx.pool(), (size_t) n_tiles*k);
        ggml_cuda_pool_alloc<int>   ci(ctx.pool(), (size_t) n_tiles*k);
        const float * x = (const float *) src0->data;
        topk_split_select<1, false><<<n_tiles, TOPK_SPLIT_THREADS, 0, ctx.stream()>>>(x, x, nullptr, (int) ncols, k, cv.get(), ci.get(), n_dev, rank, offset);
        CUDA_CHECK(cudaGetLastError());
        topk_split_select<2, true><<<1, TOPK_SPLIT_THREADS, 0, ctx.stream()>>>(x, cv.get(), ci.get(), n_tiles*k, k, (float *) dst->data, nullptr, n_dev, rank, offset);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    ggml_cuda_pool_alloc<int> ids(ctx.pool(), k);

    // the plain top-k of the local slice, through the one-card op
    ggml_tensor tk = {};
    tk.type   = GGML_TYPE_I32;
    tk.op     = GGML_OP_TOP_K;
    tk.ne[0]  = k; tk.ne[1] = 1; tk.ne[2] = 1; tk.ne[3] = 1;
    tk.nb[0]  = sizeof(int32_t); tk.nb[1] = k*sizeof(int32_t); tk.nb[2] = tk.nb[1]; tk.nb[3] = tk.nb[1];
    tk.src[0] = const_cast<ggml_tensor *>(src0);
    tk.data   = ids.get();
    ggml_cuda_op_top_k(ctx, &tk);

    top_k_split_write<<<1, 256, 0, ctx.stream()>>>((const float *) src0->data, ids.get(), (float *) dst->data, k, n_dev, rank, offset);
    CUDA_CHECK(cudaGetLastError());
}
