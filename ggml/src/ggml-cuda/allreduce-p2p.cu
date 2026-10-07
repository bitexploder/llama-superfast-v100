#include "allreduce-p2p.cuh"
#include "qpn-source.cuh"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

// The plain kernel's grid is sized for the copy: one 16-byte vector per thread up to P2P_AR_MAX_BLOCKS
// blocks (at most one block per SM of a V100, so every block of a call is resident and a spinning block never keeps a
// pushing one out), and the threads loop beyond that. The fused kernel runs one block per row.
#define P2P_AR_THREADS    256
#define P2P_AR_MAX_BLOCKS 72
#define P2P_AR_MAX_RANKS 8
// reductions larger than 64 rows of a 5120-wide trunk (prefill) go through the copy engine: on this
// board's root complex it moves 40 MB both ways at once in 6.5 ms against the 16-byte kernel push's 7.8 ms
// Two ranks only; decided on the size alone, so both ranks take the same path.
#define P2P_AR_CE_MIN_BYTES ((int64_t) 64 * 5120 * 4)
// the copy engine's landing area holds two slots on every rank, used by call parity, so one call's copy can still be in
// flight (its sum deferred, the meta backend's two-batch overlap) while the next call's copy lands
#define P2P_AR_CE_SLOTS 2

// Layout on every rank:
//   stage: [2 halves][n ranks][cap floats]  slot s of a half is written by rank s
// The flag-free protocol of FlashInfer's TP2 push kernel (ipc_tp2_remote_push_kernel) and TensorRT-LLM's
// Lamport one-shot: the payload is its own ready signal. Every slot starts filled with a sentinel bit pattern that no
// pushed value carries (an input holding it is pushed as another NaN). A call pushes the local partial into slot `rank`
// of every peer's stage, then polls its own stage until the peers' values have replaced the sentinel, sums all slots in
// rank order (so every rank gets bit-identical results, the same sums as before), and writes the sentinel back. No
// fence, no flag, no second crossing of the bus; each value is summed as soon as it lands.
// Double buffering by call parity is enough: a peer writes a half in call k+2 only after its call k+1 has read this
// rank's call-k+1 values, which this rank pushes only after its call k (which read and reset that half) has ended.
// The call counter is one per rank: every block reads it at its start and the last block to finish advances it.
#define P2P_AR_SENTINEL 0xffffffffu
#define P2P_AR_NAN      0x7fffffffu

// op params of an ALLREDUCE node; cap is included so a regrown staging area changes the node and the
// captured CUDA graph is not reused with stale pointers
struct p2p_ar_op_params {
    ggml_cuda_p2p_ar * ar;
    int64_t            cap;
    int32_t            rank;
    int32_t            zero; // the local partial was not computed (zero-sized slice): add 0 instead
};
static_assert(sizeof(p2p_ar_op_params) <= GGML_MAX_OP_PARAMS, "p2p_ar_op_params too large");

// GGML_CUDA_P2P_AR_STAMPS_CTL=<path> with it: the stamps start off and are written only while <path> exists (the
// dumper checks every 20 ms and flips a device-memory switch the kernels read), so one launch can measure with and without them.
// GGML_CUDA_P2P_AR_STAMPS=<file>: every call writes %globaltimer stamps per block (start, pushed, fenced,
// ready, end) into a host-mapped ring per rank, and a host thread appends each finished record to <file> (a record is
// finished once the next call's record exists: calls on one rank run in stream order). Off: no stamp is written.
#define P2P_AR_STAMP_RING   (1 << 15)
#define P2P_AR_STAMP_BLOCKS 16
struct p2p_ar_stamp {
    unsigned long long t[5][P2P_AR_STAMP_BLOCKS];
    long long          n_el;
    int                id;
    int                kind; // 0 plain, 1 with the add + rms_norm + mul epilogue, 2 the same with the qpn source, 3 copy engine;
                             // + 4 when the producing product pushed the partial
    unsigned           nb;   // blocks in the call
    unsigned           seq;  // the call's epoch, written last by block 0
};

static __device__ __forceinline__ unsigned long long p2p_ar_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

struct p2p_ar_args {
    float    * stage[P2P_AR_MAX_RANKS]; // stage of each rank (peer memory except [rank])
    // unused: without it nvcc 12.8 copies this struct to local memory for the dynamic stage[r] index
    // (128 to 184 bytes of stack per kernel); with it the index reads the parameter bank directly (no stack)
    void     * pad_[P2P_AR_MAX_RANKS];
    unsigned * epoch;                   // local call counter [0] and the blocks finished in this call [1]
    float    * ce_stage;                // the copy engine's landing slot on this rank; the call's slot of two
    unsigned short * ce_pack;          // local packed source for this call's landing slot
    unsigned * ce;                      // [0] copy-engine calls this rank has read, [1] blocks finished, [2] the peer's [0]
    unsigned * ce_peer;                 // the peer's ce array (peer memory)
    int64_t    cap;                     // floats per slot
    int        rank;
    int        n;
    int        zero;
    int64_t    max_spins; // GGML_CUDA_P2P_AR_DEBUG: report a stuck call and return instead of trapping
    int        id;        // GGML_CUDA_P2P_AR_DEBUG: index of the node in its graph
    p2p_ar_stamp * stamps; // this rank's stamp ring, nullptr when off
    int        kind;
    int        pushed; // the producing product pushed this rank's partial already: only poll and sum
    int        wire16; // kernel path uses bf16; independent of the copy-engine setting
    const unsigned * stamp_on; // with stamps: nonzero while they are written (GGML_CUDA_P2P_AR_STAMPS_CTL)
    unsigned   ce_call;            // the copy-engine call's index on this rank (calls 0, 1, 2, ... in order on both ranks)
};

// the call writes its stamps: the ring exists and the switch is on
static __device__ __forceinline__ bool p2p_ar_stamping(const p2p_ar_args & p) {
    return p.stamps != nullptr && *(volatile const unsigned *) p.stamp_on != 0;
}

template <bool qpn>
static __global__ void __launch_bounds__(1024) k_allreduce_p2p_add_rms_norm_mul(
        float * data, const int64_t n_el, const p2p_ar_args p,
        const float * b, const int64_t sb1, float * sum, float * dst, const int ncols, const float eps, const float * mul,
        const ggml_cuda_qpn_dst q, const int n_ar, const ggml_cuda_p2p_ar_pf pf);

struct ggml_cuda_p2p_ar {
    int        n = 0;
    int        devices[P2P_AR_MAX_RANKS];
    float    * stage[P2P_AR_MAX_RANKS] = {};
    unsigned * epoch[P2P_AR_MAX_RANKS] = {};
    float    * ce_stage[P2P_AR_MAX_RANKS] = {}; // [cap] per rank, written by the peer's copy engine; [2][cap]
    unsigned short * ce_pack[P2P_AR_MAX_RANKS][P2P_AR_CE_SLOTS] = {};
    bool ce_wire16[P2P_AR_MAX_RANKS][P2P_AR_CE_SLOTS] = {};
    unsigned * ce[P2P_AR_MAX_RANKS] = {};
    // the asynchronous copy-engine call: the copy runs on a side stream of its own, the sum is enqueued later on the
    // compute stream (ggml_cuda_p2p_ar_ce_finish), after the compute that does not need the sum
    cudaStream_t ce_stream[P2P_AR_MAX_RANKS] = {};
    cudaEvent_t  ce_ev_pre[P2P_AR_MAX_RANKS][P2P_AR_CE_SLOTS] = {};  // the pre kernel is done: the copy may read the partial
    cudaEvent_t  ce_ev_copy[P2P_AR_MAX_RANKS][P2P_AR_CE_SLOTS] = {}; // the copy is done: the sum may write the partial in place
    uint64_t     ce_calls[P2P_AR_MAX_RANKS] = {};                    // copy-engine calls started on each rank (host side)
    int64_t    cap = 0;
    p2p_ar_stamp * stamps[P2P_AR_MAX_RANKS] = {}; // host-mapped, one ring per rank
    unsigned     * stamp_on[P2P_AR_MAX_RANKS] = {}; // device memory, one switch per rank
    std::string    stamp_ctl;                       // GGML_CUDA_P2P_AR_STAMPS_CTL: stamps on while this path exists
    cudaStream_t   stamp_stream[P2P_AR_MAX_RANKS] = {}; // the switch's writes
    std::thread       stamp_thread;
    std::atomic<bool> stamp_stop{false};
    // GGML_CUDA_P2P_AR_PF_CTL: the prefetch's switch per rank (device memory, nonzero: skip) and its poller
    unsigned        * pf_off[P2P_AR_MAX_RANKS] = {};
    cudaStream_t      pf_stream[P2P_AR_MAX_RANKS] = {};
    std::string       pf_ctl;
    std::thread       pf_thread;
    std::atomic<bool> pf_stop{false};
};

// GGML_CUDA_P2P_AR_PF_CTL: skip the prefetch while the control path exists
static void p2p_ar_pf_poll(ggml_cuda_p2p_ar * ar) {
    bool off = false;
    while (!ar->pf_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        FILE * c = fopen(ar->pf_ctl.c_str(), "r");
        const bool want = c != nullptr;
        if (c != nullptr) {
            fclose(c);
        }
        if (want != off) {
            const unsigned v = want ? 1u : 0u;
            for (int r = 0; r < ar->n; ++r) {
                ggml_cuda_set_device(ar->devices[r]);
                CUDA_CHECK(cudaMemcpyAsync(ar->pf_off[r], &v, sizeof(v), cudaMemcpyHostToDevice, ar->pf_stream[r]));
                CUDA_CHECK(cudaStreamSynchronize(ar->pf_stream[r]));
            }
            off = want;
            GGML_LOG_INFO("%s: prefetch under the AllReduce %s\n", __func__, off ? "off" : "on");
        }
    }
}

int64_t ggml_cuda_p2p_ar_pf_budget() {
    static const int64_t b = [] {
        const char * e = getenv("GGML_CUDA_P2P_AR_PF_MB");
        return (int64_t) (e != nullptr ? atof(e) : 6.0) * 1048576;
    }();
    return b;
}

// the prefetch blocks' work: thread tid of nthr walks the 32-byte lines of every tile's first pre_bytes
static __device__ void p2p_ar_prefetch(const ggml_cuda_p2p_ar_pf & pf, const int64_t tid, const int64_t nthr) {
    if (pf.off != nullptr && *(volatile const unsigned *) pf.off != 0) {
        return;
    }
    // unrolled, so the struct is read from the parameter bank (a dynamic index would copy it to the stack)
#pragma unroll
    for (int w = 0; w < GGML_CUDA_P2P_AR_PF_MAX; ++w) {
        if (w >= pf.n) {
            break;
        }
        const int64_t lpt     = pf.pre_bytes[w] / 32;
        const int64_t n_lines = lpt * pf.ntiles[w];
        for (int64_t i = tid; i < n_lines; i += nthr) {
            const int64_t t = i / lpt;
            const char * a = pf.W[w] + t * pf.tile_bytes[w] + (i - t * lpt) * 32;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
        }
    }
}

// append each finished stamp record to the file, as (int32 rank, p2p_ar_stamp); a record of call k is
// finished once call k+1's record carries its seq, so the last call of a run is not written
static void p2p_ar_stamp_dump(ggml_cuda_p2p_ar * ar, std::string path) {
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        GGML_LOG_ERROR("%s: cannot open %s\n", __func__, path.c_str());
        return;
    }
    unsigned next[P2P_AR_MAX_RANKS];
    for (int r = 0; r < ar->n; ++r) {
        next[r] = 1;
    }
    bool on = ar->stamp_ctl.empty();
    for (bool stop = false; !stop; ) {
        stop = ar->stamp_stop.load();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // GGML_CUDA_P2P_AR_STAMPS_CTL: switch the stamps on while the control path exists, off otherwise
        if (!ar->stamp_ctl.empty()) {
            FILE * c = fopen(ar->stamp_ctl.c_str(), "r");
            const bool want = c != nullptr;
            if (c != nullptr) {
                fclose(c);
            }
            if (want != on) {
                const unsigned v = want ? 1u : 0u;
                for (int r = 0; r < ar->n; ++r) {
                    // on a stream of its own that does not wait for the graphs (the legacy stream would)
                    ggml_cuda_set_device(ar->devices[r]);
                    CUDA_CHECK(cudaMemcpyAsync(ar->stamp_on[r], &v, sizeof(v), cudaMemcpyHostToDevice, ar->stamp_stream[r]));
                    CUDA_CHECK(cudaStreamSynchronize(ar->stamp_stream[r]));
                }
                on = want;
                GGML_LOG_INFO("%s: stamps %s\n", __func__, on ? "on" : "off");
            }
        }
        for (int r = 0; r < ar->n; ++r) {
            for (;;) {
                const volatile p2p_ar_stamp * a = ar->stamps[r] + (next[r]     & (P2P_AR_STAMP_RING - 1));
                const volatile p2p_ar_stamp * b = ar->stamps[r] + ((next[r]+1) & (P2P_AR_STAMP_RING - 1));
                if (a->seq != next[r] || b->seq != next[r] + 1) {
                    // a ring overrun would leave the dumper behind for good: jump to the newest record
                    if (a->seq > next[r] && a->seq != 0) {
                        GGML_LOG_WARN("%s: rank %d skipped from call %u to %u\n", __func__, r, next[r], (unsigned) a->seq);
                        next[r] = a->seq;
                        continue;
                    }
                    break;
                }
                p2p_ar_stamp copy;
                memcpy(&copy, (const void *) a, sizeof(copy));
                const int32_t rank = r;
                fwrite(&rank, sizeof(rank), 1, f);
                fwrite(&copy, sizeof(copy), 1, f);
                next[r]++;
            }
        }
        fflush(f);
    }
    fclose(f);
}

#define P2P_AR_STAMP(k, blk) \
    if (st != nullptr && (blk) < P2P_AR_STAMP_BLOCKS) { st->t[k][blk] = p2p_ar_now(); }
#define P2P_AR_STAMP_CLOSE(blk) \
    if (st != nullptr && (blk) == 0 && threadIdx.x == 0) { \
        st->n_el = n_el; st->id = p.id; st->kind = p.kind; st->nb = gridDim.x; \
        __threadfence_system(); \
        st->seq = ep; \
    }

// a value as it goes on the wire: never the sentinel
static __device__ __forceinline__ float p2p_ar_wire(const float v) {
    static_assert(P2P_AR_SENTINEL == 0xffffffffu && P2P_AR_NAN == 0x7fffffffu, "ggml_cuda_p2p_wire's patterns");
    return ggml_cuda_p2p_wire(v);
}

// the poll's load: a volatile asm, so the compiler can not hoist it out of the loop (CUDA 12.8's __ldcv(float4 *) is a
// plain asm and was hoisted, which spun forever)
static __device__ __forceinline__ float p2p_ar_poll(const float * src) {
    float v;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(v) : "l"(src) : "memory");
    return v;
}
static __device__ __forceinline__ float4 p2p_ar_poll(const float4 * src) {
    float4 v;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(src) : "memory");
    return v;
}

// wait until a peer's value replaced the sentinel at src, take it and put the sentinel back. A peer that never arrives
// traps (after 2^28 polls, or 2^24 with GGML_CUDA_P2P_AR_DEBUG) instead of hanging; no call or printf here, which would
// put the callers' register arrays on the stack
static __device__ __forceinline__ float p2p_ar_take(float * src, const int64_t max_spins) {
    float v;
    int64_t spins = 0;
    while (__float_as_uint(v = p2p_ar_poll(src)) == P2P_AR_SENTINEL) {
        if (++spins > max_spins) {
            __trap();
        }
    }
    *src = __uint_as_float(P2P_AR_SENTINEL);
    return v;
}

// the call's epoch, read by every block at its start
static __device__ __forceinline__ unsigned p2p_ar_begin(const unsigned * epoch) {
    return epoch[0] + 1;
}

// the last block of the call to finish advances the epoch for the next call on this stream
// (n_blocks, the call's blocks that take part, the fused kernel's row blocks; its prefetch blocks do not)
static __device__ __forceinline__ void p2p_ar_end(unsigned * epoch, const unsigned ep, const unsigned n_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = atomicAdd(&epoch[1], 1u);
        if (done == n_blocks - 1) {
            epoch[1] = 0;
            epoch[0] = ep;
        }
    }
}

// 16-byte vectors on the wire (FlashInfer's st.volatile.global.v4 push, the survey's vector-store point)
template <typename T> struct p2p_ar_vec;
template <> struct p2p_ar_vec<float> {
    static __device__ __forceinline__ float wire(const float v) { return p2p_ar_wire(v); }
    static __device__ __forceinline__ bool  missing(const float v) { return __float_as_uint(v) == P2P_AR_SENTINEL; }
    static __device__ __forceinline__ float sentinel() { return __uint_as_float(P2P_AR_SENTINEL); }
    static __device__ __forceinline__ float add(const float a, const float b) { return a + b; }
};
template <> struct p2p_ar_vec<float4> {
    static __device__ __forceinline__ float4 wire(const float4 v) {
        return make_float4(p2p_ar_wire(v.x), p2p_ar_wire(v.y), p2p_ar_wire(v.z), p2p_ar_wire(v.w));
    }
    static __device__ __forceinline__ bool missing(const float4 v) {
        // each 4-byte lane is checked: a vector that lands in pieces is still taken whole
        return __float_as_uint(v.x) == P2P_AR_SENTINEL || __float_as_uint(v.y) == P2P_AR_SENTINEL ||
               __float_as_uint(v.z) == P2P_AR_SENTINEL || __float_as_uint(v.w) == P2P_AR_SENTINEL;
    }
    static __device__ __forceinline__ float4 sentinel() {
        const float s = __uint_as_float(P2P_AR_SENTINEL);
        return make_float4(s, s, s, s);
    }
    static __device__ __forceinline__ float4 add(const float4 a, const float4 b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
};

template <typename T>
static __device__ __forceinline__ T p2p_ar_take_v(T * src, const int64_t max_spins) {
    T v;
    int64_t spins = 0;
    while (p2p_ar_vec<T>::missing(v = p2p_ar_poll(src))) {
        if (++spins > max_spins) {
            __trap();
        }
    }
    *src = p2p_ar_vec<T>::sentinel();
    return v;
}

// sum element j (a vector) of all ranks in rank order into data
template <typename T>
static __device__ __forceinline__ void p2p_ar_finish(T * data, T * mine, const int64_t cap_v, const int64_t j,
        const int n, const int rank, const int zero, const int64_t max_spins) {
    T acc;
    for (int s = 0; s < n; ++s) {
        T v;
        if (s != rank) {
            v = p2p_ar_take_v(mine + s * cap_v + j, max_spins);
        } else if (zero) {
            v = T{};
        } else {
            v = p2p_ar_vec<T>::wire(data[j]);
        }
        acc = s == 0 ? v : p2p_ar_vec<T>::add(acc, v);
    }
    data[j] = acc;
}

// T is float4 when the tensor, its address and the slot size allow it, else float. Each thread pushes, then sums, the
// same elements, so the in-place output never overtakes a push; with several vectors per thread (prefill) element j is
// summed after element j+stride is pushed, so the sums run behind the copy instead of after it.
template <typename T>
static __global__ void k_allreduce_p2p(T * data, const int64_t n_v, const int64_t n_el, const p2p_ar_args p) {
    const int b = blockIdx.x;
    const unsigned long long t_start = p.stamps != nullptr ? p2p_ar_now() : 0;

    __shared__ unsigned ep;
    if (threadIdx.x == 0) {
        ep = p2p_ar_begin(p.epoch);
    }
    __syncthreads();
    p2p_ar_stamp * st = p2p_ar_stamping(p) ? p.stamps + (ep & (P2P_AR_STAMP_RING - 1)) : nullptr;
    if (threadIdx.x == 0) {
        if (st != nullptr && b < P2P_AR_STAMP_BLOCKS) { st->t[0][b] = t_start; }
    }

    constexpr int64_t V = sizeof(T) / sizeof(float);
    const int64_t cap_v   = p.cap / V;
    const int64_t half    = (int64_t) (ep & 1) * p.n * cap_v;
    const int64_t my_slot = half + p.rank * cap_v;
    T * mine = (T *) p.stage[p.rank] + half;

    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    int64_t prev = -1;
    for (int64_t j = (int64_t) b * blockDim.x + threadIdx.x; j < n_v; j += stride) {
        if (!p.pushed) {
            const T v = p.zero ? T{} : p2p_ar_vec<T>::wire(data[j]);
            for (int r = 0; r < p.n; ++r) {
                if (r != p.rank) {
                    ((T *) p.stage[r])[my_slot + j] = v;
                }
            }
        }
        if (prev >= 0) {
            p2p_ar_finish(data, mine, cap_v, prev, p.n, p.rank, p.zero, p.max_spins);
        }
        prev = j;
    }
    if (st != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            P2P_AR_STAMP(1, b);
            P2P_AR_STAMP(2, b);
        }
    }
    if (prev >= 0) {
        p2p_ar_finish(data, mine, cap_v, prev, p.n, p.rank, p.zero, p.max_spins);
    }
    if (st != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            P2P_AR_STAMP(3, b);
            P2P_AR_STAMP(4, b);
        }
        P2P_AR_STAMP_CLOSE(b);
    }
    p2p_ar_end(p.epoch, ep, gridDim.x);
}

// The kernel wire uses the first half of each fp32 slot's bytes. Both formats reset to 0xff,
// so calls of different formats can alternate without changing the double-buffer protocol.
static bool p2p_ar_wire16_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_P2P_WIRE");
        const bool v = e == nullptr || atoi(e) != 32;
        if (e != nullptr && atoi(e) != 32 && atoi(e) != 16) {
            GGML_LOG_WARN("%s: GGML_CUDA_P2P_WIRE=%s is not 32 or 16; using 16\n", __func__, e);
        }
        return v;
    }();
    return on;
}

static bool p2p_ar_wire16_for(const ggml_tensor * partial, const int64_t n_el) {
    return p2p_ar_wire16_on() && n_el % 8 == 0 && (partial == nullptr || partial->op != GGML_OP_TOP_K_SPLIT);
}
#define P2P_AR_SENTINEL16 0xffffu

static __device__ __forceinline__ unsigned short p2p_ar_poll16(const unsigned short * src) {
    unsigned short v;
    asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(v) : "l"(src) : "memory");
    return v;
}
static __device__ __forceinline__ unsigned short p2p_ar_take16(unsigned short * src, const int64_t max_spins) {
    unsigned short v;
    int64_t spins = 0;
    while ((v = p2p_ar_poll16(src)) == P2P_AR_SENTINEL16) {
        if (++spins > max_spins) {
            __trap();
        }
    }
    *src = P2P_AR_SENTINEL16;
    return v;
}
static __device__ __forceinline__ uint4 p2p_ar_poll16(const uint4 * src) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src) : "memory");
    return v;
}
static __device__ __forceinline__ bool p2p_ar_missing16(const unsigned w) {
    return (w & 0xffffu) == P2P_AR_SENTINEL16 || (w >> 16) == P2P_AR_SENTINEL16;
}
static __device__ __forceinline__ bool p2p_ar_missing16(const uint4 v) {
    // each 16-bit lane is checked: a vector that lands in pieces is still taken whole
    return p2p_ar_missing16(v.x) || p2p_ar_missing16(v.y) || p2p_ar_missing16(v.z) || p2p_ar_missing16(v.w);
}
static __device__ __forceinline__ uint4 p2p_ar_take16(uint4 * src, const int64_t max_spins) {
    uint4 v;
    int64_t spins = 0;
    while (p2p_ar_missing16(v = p2p_ar_poll16(src))) {
        if (++spins > max_spins) {
            __trap();
        }
    }
    *src = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
    return v;
}

// 8 floats from data[8j..8j+7] (16-byte loads when data is 16-byte aligned) as 8 bf16 on the wire
static __device__ __forceinline__ uint4 p2p_ar_load_wire16(const float * data, const int64_t j, const bool aligned) {
    float f[8];
    if (aligned) {
        const float4 a = ((const float4 *) data)[2*j];
        const float4 b = ((const float4 *) data)[2*j + 1];
        f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            f[k] = data[8*j + k];
        }
    }
    return make_uint4(ggml_cuda_p2p_wire16x2(f[0], f[1]), ggml_cuda_p2p_wire16x2(f[2], f[3]),
                      ggml_cuda_p2p_wire16x2(f[4], f[5]), ggml_cuda_p2p_wire16x2(f[6], f[7]));
}

// acc[0..7] (+)= the 8 bf16 of w, widened to fp32
static __device__ __forceinline__ void p2p_ar_acc16(float * acc, const uint4 w, const bool first) {
    const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float lo = __uint_as_float(u[k] << 16);
        const float hi = __uint_as_float(u[k] & 0xffff0000u);
        acc[2*k]     = first ? lo : acc[2*k]     + lo;
        acc[2*k + 1] = first ? hi : acc[2*k + 1] + hi;
    }
}

static __device__ __forceinline__ void p2p_ar_store8(float * data, const int64_t j, const float * acc, const bool aligned) {
    if (aligned) {
        ((float4 *) data)[2*j]     = make_float4(acc[0], acc[1], acc[2], acc[3]);
        ((float4 *) data)[2*j + 1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            data[8*j + k] = acc[k];
        }
    }
}

// every rank sums the same rounded operands in rank order.
static __device__ __forceinline__ void p2p_ar_finish16(float * data, uint4 * mine, const int64_t cap_v, const int64_t j,
        const int n, const int rank, const uint4 own, const int64_t max_spins, const bool aligned) {
    float acc[8];
    for (int s = 0; s < n; ++s) {
        const uint4 v = s != rank ? p2p_ar_take16(mine + s * cap_v + j, max_spins) : own;
        p2p_ar_acc16(acc, v, s == 0);
    }
    p2p_ar_store8(data, j, acc, aligned);
}

static __global__ void k_allreduce_p2p16(float * data, const int64_t n_v, const int64_t n_el, const p2p_ar_args p, const bool aligned) {
    const int b = blockIdx.x;
    const unsigned long long t_start = p.stamps != nullptr ? p2p_ar_now() : 0;
    __shared__ unsigned ep;
    if (threadIdx.x == 0) {
        ep = p2p_ar_begin(p.epoch);
    }
    __syncthreads();
    p2p_ar_stamp * st = p2p_ar_stamping(p) ? p.stamps + (ep & (P2P_AR_STAMP_RING - 1)) : nullptr;
    if (threadIdx.x == 0) {
        if (st != nullptr && b < P2P_AR_STAMP_BLOCKS) { st->t[0][b] = t_start; }
    }
    const int64_t cap_v   = p.cap / 4;
    const int64_t half    = (int64_t) (ep & 1) * p.n * cap_v;
    const int64_t my_slot = half + p.rank * cap_v;
    uint4 * mine = (uint4 *) p.stage[p.rank] + half;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    int64_t prev = -1;
    uint4 prev_own = {};
    for (int64_t j = (int64_t) b * blockDim.x + threadIdx.x; j < n_v; j += stride) {
        const uint4 own = p.zero ? uint4{} : p2p_ar_load_wire16(data, j, aligned);
        if (!p.pushed) {
            for (int r = 0; r < p.n; ++r) {
                if (r != p.rank) {
                    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"((uint4 *) p.stage[r] + my_slot + j),
                        "r"(own.x), "r"(own.y), "r"(own.z), "r"(own.w) : "memory");
                }
            }
        }
        if (prev >= 0) {
            p2p_ar_finish16(data, mine, cap_v, prev, p.n, p.rank, prev_own, p.max_spins, aligned);
        }
        prev = j;
        prev_own = own;
    }
    if (st != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            P2P_AR_STAMP(1, b);
            P2P_AR_STAMP(2, b);
        }
    }
    if (prev >= 0) {
        p2p_ar_finish16(data, mine, cap_v, prev, p.n, p.rank, prev_own, p.max_spins, aligned);
    }
    if (st != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            P2P_AR_STAMP(3, b);
            P2P_AR_STAMP(4, b);
        }
        P2P_AR_STAMP_CLOSE(b);
    }
    p2p_ar_end(p.epoch, ep, gridDim.x);
}

// the copy-engine path, two ranks. Per call on each rank, in stream order:
//   k_allreduce_p2p_ce_pre: wait until the peer has read this rank's copy-engine call of two calls ago out of its landing slot
//     (two slots by call parity; the peer's [0] counter, the calls it has summed, which it writes into this rank's
//     [2], must reach call - 1), and make the partial safe for the wire (the sentinel rewritten, or zeros for a partial that was
//     not computed);
//   a copy-engine copy of the partial into the peer's landing slot (a memcpy node in the captured graph);
//   k_allreduce_p2p_ce_sum: poll the own landing slot until the peer's values have replaced the sentinel, sum in rank
//     order, put the sentinel back, and tell the peer (its [2]) that the slot is free again.
// The sum's rank order and the values are the kernel path's, so the outputs are bitwise the same.
template <typename T>
static __global__ void k_allreduce_p2p_ce_pre(T * data, const int64_t n_v, const p2p_ar_args p) {
    if (threadIdx.x == 0 && blockIdx.x < P2P_AR_STAMP_BLOCKS && p2p_ar_stamping(p)) {
        p.stamps[p2p_ar_begin(p.epoch) & (P2P_AR_STAMP_RING - 1)].t[0][blockIdx.x] = p2p_ar_now();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && p.ce_call >= P2P_AR_CE_SLOTS) {
        int64_t spins = 0;
        while (*(volatile unsigned *) &p.ce[2] < p.ce_call - (P2P_AR_CE_SLOTS - 1)) {
            if (++spins > p.max_spins) {
                __trap();
            }
        }
    }
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; j < n_v; j += stride) {
        if (p.zero) {
            data[j] = T{};
        } else {
            const T v = data[j];
            if (p2p_ar_vec<T>::missing(v)) {
                data[j] = p2p_ar_vec<T>::wire(v);
            }
        }
    }
}

template <typename T>
static __global__ void k_allreduce_p2p_ce_sum(T * data, const int64_t n_v, const int64_t n_el, const p2p_ar_args p) {
    const int b = blockIdx.x;
    __shared__ unsigned ep;
    if (threadIdx.x == 0) {
        ep = p2p_ar_begin(p.epoch);
    }
    __syncthreads();
    p2p_ar_stamp * st = p2p_ar_stamping(p) ? p.stamps + (ep & (P2P_AR_STAMP_RING - 1)) : nullptr;
    if (st != nullptr && threadIdx.x == 0) {
        P2P_AR_STAMP(1, b);
        P2P_AR_STAMP(2, b);
    }
    T * slot = (T *) p.ce_stage;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t j = (int64_t) b * blockDim.x + threadIdx.x; j < n_v; j += stride) {
        const T peer = p2p_ar_take_v(slot + j, p.max_spins);
        const T mine = data[j]; // already made safe for the wire by the pre kernel, as the kernel path's sum sees it
        data[j] = p.rank == 0 ? p2p_ar_vec<T>::add(mine, peer) : p2p_ar_vec<T>::add(peer, mine);
    }
    if (st != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            P2P_AR_STAMP(3, b);
            P2P_AR_STAMP(4, b);
        }
        P2P_AR_STAMP_CLOSE(b);
    }
    // the slot's resets must be in memory before the peer may copy into it again
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = atomicAdd(&p.ce[1], 1u);
        if (done == gridDim.x - 1) {
            p.ce[1] = 0;
            const unsigned k = p.ce[0] + 1;
            p.ce[0] = k;
            __threadfence_system();
            *(volatile unsigned *) &p.ce_peer[2] = k;
        }
    }
    p2p_ar_end(p.epoch, ep, gridDim.x);
}

// the copy-engine path at the 16-bit wire: the pre kernel packs the partial to bf16 into this rank's ce_pack (the
// partial itself is not touched), the copy engine moves the packed half-size partial into the peer's landing slot, and the sum
// kernel adds the peer's bf16 and this rank's own packed values in rank order
static __global__ void k_allreduce_p2p_ce_pack16(const float * data, const int64_t n_v, const p2p_ar_args p, const bool aligned) {
    if (threadIdx.x == 0 && blockIdx.x < P2P_AR_STAMP_BLOCKS && p2p_ar_stamping(p)) {
        p.stamps[p2p_ar_begin(p.epoch) & (P2P_AR_STAMP_RING - 1)].t[0][blockIdx.x] = p2p_ar_now();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && p.ce_call >= P2P_AR_CE_SLOTS) {
        int64_t spins = 0;
        while (*(volatile unsigned *) &p.ce[2] < p.ce_call - (P2P_AR_CE_SLOTS - 1)) {
            if (++spins > p.max_spins) {
                __trap();
            }
        }
    }
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; j < n_v; j += stride) {
        ((uint4 *) p.ce_pack)[j] = p.zero ? uint4{} : p2p_ar_load_wire16(data, j, aligned);
    }
}

static __global__ void k_allreduce_p2p_ce_sum16(float * data, const int64_t n_v, const int64_t n_el, const p2p_ar_args p, const bool aligned) {
    const int b = blockIdx.x;
    __shared__ unsigned ep;
    if (threadIdx.x == 0) {
        ep = p2p_ar_begin(p.epoch);
    }
    __syncthreads();
    p2p_ar_stamp * st = p2p_ar_stamping(p) ? p.stamps + (ep & (P2P_AR_STAMP_RING - 1)) : nullptr;
    if (st != nullptr && threadIdx.x == 0) {
        P2P_AR_STAMP(1, b);
        P2P_AR_STAMP(2, b);
    }
    uint4 * slot = (uint4 *) p.ce_stage;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t j = (int64_t) b * blockDim.x + threadIdx.x; j < n_v; j += stride) {
        const uint4 peer = p2p_ar_take16(slot + j, p.max_spins);
        const uint4 mine = ((const uint4 *) p.ce_pack)[j];
        float acc[8];
        p2p_ar_acc16(acc, p.rank == 0 ? mine : peer, true);
        p2p_ar_acc16(acc, p.rank == 0 ? peer : mine, false);
        p2p_ar_store8(data, j, acc, aligned);
    }
    if (st != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            P2P_AR_STAMP(3, b);
            P2P_AR_STAMP(4, b);
        }
        P2P_AR_STAMP_CLOSE(b);
    }
    // the slot's resets must be in memory before the peer may copy into it again
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = atomicAdd(&p.ce[1], 1u);
        if (done == gridDim.x - 1) {
            p.ce[1] = 0;
            const unsigned k = p.ce[0] + 1;
            p.ce[0] = k;
            __threadfence_system();
            *(volatile unsigned *) &p.ce_peer[2] = k;
        }
    }
    p2p_ar_end(p.epoch, ep, gridDim.x);
}
static p2p_ar_args p2p_ar_make_args(const ggml_cuda_p2p_ar * ar, int rank, float * data, int64_t n_el, int zero, int id, int pushed) {
    GGML_ASSERT(n_el <= ar->cap);

    p2p_ar_args p = {};
    for (int r = 0; r < ar->n; ++r) {
        p.stage[r] = ar->stage[r];
    }
    p.epoch = ar->epoch[rank];
    p.cap   = ar->cap;
    p.rank  = rank;
    p.n     = ar->n;
    p.zero  = zero;
    static const bool debug = getenv("GGML_CUDA_P2P_AR_DEBUG") != nullptr;
    p.max_spins = debug ? int64_t(1) << 24 : int64_t(1) << 28;
    p.id        = id;
    p.stamps    = ar->stamps[rank];
    p.stamp_on  = ar->stamp_on[rank];
    p.kind      = pushed ? 4 : 0;
    p.pushed    = pushed;
    if (debug) {
        GGML_LOG_WARN("p2p allreduce launch: rank %d id %d n=%lld zero=%d pushed=%d\n", rank, id, (long long) n_el, zero, pushed);
    }
    return p;
}

// whether a call of n_el floats takes the copy-engine path (decided on the size alone, so both ranks agree)
static bool p2p_ar_is_ce(const ggml_cuda_p2p_ar * ar, const int64_t n_el) {
    return ar->n == 2 && n_el * (int64_t) sizeof(float) > P2P_AR_CE_MIN_BYTES;
}

static bool p2p_ar_vec_ok(const ggml_cuda_p2p_ar * ar, const float * data, const int64_t n_el) {
    return n_el % 4 == 0 && ar->cap % 4 == 0 && (uintptr_t) data % 16 == 0;
}

static int p2p_ar_blocks(const int64_t n_v) {
    return (int) std::max<int64_t>(1, std::min<int64_t>(P2P_AR_MAX_BLOCKS, (n_v + P2P_AR_THREADS - 1) / P2P_AR_THREADS));
}

// Serving-width KL passed: default to 16-bit CE wire; set 32 for fp32. Independent of the kernel wire.
static bool p2p_ar_ce_wire16_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_P2P_AR_CE_WIRE");
        return e == nullptr || atoi(e) == 16;
    }();
    return on;
}

static bool p2p_ar_ce_wire16_for(const ggml_tensor * t) {
    return p2p_ar_ce_wire16_on() && ggml_nelements(t) % 8 == 0 &&
        t->op != GGML_OP_TOP_K_SPLIT && strstr(t->name, "mtp_") == nullptr;
}

// the copy-engine call in two halves, so the meta backend can run independent compute between them:
//   start:  the pre kernel on the compute stream, then on the rank's copy stream the copy of the partial into the peer's landing
//           slot of this call's parity (the copy engine runs it while the compute stream goes on);
//   finish: the compute stream waits for this rank's own copy (the sum writes the partial in place), then polls its landing slot
//           until the peer's values have replaced the sentinel, sums in rank order, puts the sentinel back and frees the slot.
// At the fp32 wire the values and rank order are unchanged; bf16 rounds each operand before the same sum.
// Deferred and immediate finishes agree at either wire. Calls start and finish in the same order on both ranks;
// the host's call index is exact because these calls are always eager (the meta backend never merges a CE graph
// into a captured graph).
static uint64_t p2p_ar_ce_start(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, float * data, int64_t n_el, int zero, int id,
        bool wire16) {
    GGML_ASSERT(ar->n == 2);
    const int      peer = 1 - rank;
    const uint64_t call = ar->ce_calls[rank]++;
    const int      slot = (int) (call % P2P_AR_CE_SLOTS);
    ar->ce_wire16[rank][slot] = wire16;
    p2p_ar_args p = p2p_ar_make_args(ar, rank, data, n_el, zero, id, 0);
    p.ce       = ar->ce[rank];
    p.ce_peer  = ar->ce[peer];
    p.kind     = 3;
    p.ce_call  = (unsigned) call;
    p.ce_pack  = ar->ce_pack[rank][slot];
    const bool    vec    = p2p_ar_vec_ok(ar, data, n_el);
    const int64_t n_v    = wire16 ? n_el / 8 : vec ? n_el / 4 : n_el;
    const int     blocks = p2p_ar_blocks(n_v);
    if (wire16) {
        k_allreduce_p2p_ce_pack16<<<blocks, P2P_AR_THREADS, 0, stream>>>(data, n_v, p, vec);
    } else if (vec) {
        k_allreduce_p2p_ce_pre<float4><<<blocks, P2P_AR_THREADS, 0, stream>>>((float4 *) data, n_v, p);
    } else {
        k_allreduce_p2p_ce_pre<float><<<blocks, P2P_AR_THREADS, 0, stream>>>(data, n_v, p);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ar->ce_ev_pre[rank][slot], stream));
    CUDA_CHECK(cudaStreamWaitEvent(ar->ce_stream[rank], ar->ce_ev_pre[rank][slot], 0));
    // a plain async copy between the two devices' pointers (UVA, peer access on)
    const void * source = wire16 ? (const void *) p.ce_pack : (const void *) data;
    CUDA_CHECK(cudaMemcpyAsync(ar->ce_stage[peer] + (int64_t) slot * ar->cap, source,
        n_el * (wire16 ? sizeof(unsigned short) : sizeof(float)), cudaMemcpyDeviceToDevice, ar->ce_stream[rank]));
    CUDA_CHECK(cudaEventRecord(ar->ce_ev_copy[rank][slot], ar->ce_stream[rank]));
    return call;
}

static void p2p_ar_ce_finish(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, float * data, int64_t n_el, int zero, int id, uint64_t call) {
    const int peer = 1 - rank;
    const int slot = (int) (call % P2P_AR_CE_SLOTS);
    const bool wire16 = ar->ce_wire16[rank][slot];
    p2p_ar_args p = p2p_ar_make_args(ar, rank, data, n_el, zero, id, 0);
    p.ce_stage = ar->ce_stage[rank] + (int64_t) slot * ar->cap;
    p.ce       = ar->ce[rank];
    p.ce_peer  = ar->ce[peer];
    p.kind     = 3;
    p.ce_call  = (unsigned) call;
    p.ce_pack  = ar->ce_pack[rank][slot];
    const bool    vec    = p2p_ar_vec_ok(ar, data, n_el);
    const int64_t n_v    = wire16 ? n_el / 8 : vec ? n_el / 4 : n_el;
    const int     blocks = p2p_ar_blocks(n_v);
    CUDA_CHECK(cudaStreamWaitEvent(stream, ar->ce_ev_copy[rank][slot], 0));
    if (wire16) {
        k_allreduce_p2p_ce_sum16<<<blocks, P2P_AR_THREADS, 0, stream>>>(data, n_v, n_el, p, vec);
    } else if (vec) {
        k_allreduce_p2p_ce_sum<float4><<<blocks, P2P_AR_THREADS, 0, stream>>>((float4 *) data, n_v, n_el, p);
    } else {
        k_allreduce_p2p_ce_sum<float><<<blocks, P2P_AR_THREADS, 0, stream>>>(data, n_v, n_el, p);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_p2p_ar_launch_impl(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, float * data, int64_t n_el, int zero, int id,
        int pushed = 0, bool wire16 = false) {
    if (p2p_ar_is_ce(ar, n_el)) {
        // one rank's start and finish back to back: the meta backend's eager reductions start both ranks before
        // either finishes (ggml_cuda_p2p_ar_ce_start / _finish); a graph node never takes this path (copy-engine graphs are not
        // merged, so the host's call count stays exact)
        GGML_ASSERT(!pushed);
        const uint64_t call = p2p_ar_ce_start(ar, rank, stream, data, n_el, zero, id, wire16);
        p2p_ar_ce_finish(ar, rank, stream, data, n_el, zero, id, call);
        return;
    }
    p2p_ar_args p = p2p_ar_make_args(ar, rank, data, n_el, zero, id, pushed);
    if (wire16) {
        GGML_ASSERT(n_el % 8 == 0 && ar->cap % 4 == 0);
        const int64_t n_v8 = n_el / 8;
        k_allreduce_p2p16<<<p2p_ar_blocks(n_v8), P2P_AR_THREADS, 0, stream>>>(data, n_v8, n_el, p, (uintptr_t) data % 16 == 0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool vec = p2p_ar_vec_ok(ar, data, n_el);
    const int64_t n_v = vec ? n_el / 4 : n_el;
    const int blocks = p2p_ar_blocks(n_v);
    if (vec) {
        k_allreduce_p2p<float4><<<blocks, P2P_AR_THREADS, 0, stream>>>((float4 *) data, n_v, n_el, p);
    } else {
        k_allreduce_p2p<float><<<blocks, P2P_AR_THREADS, 0, stream>>>(data, n_v, n_el, p);
    }
    CUDA_CHECK(cudaGetLastError());
}

// whether the product that computed this node's partial pushed it (it is always the node just before: the meta
// backend puts the ALLREDUCE right after its partial, and the graph loop asks only there)
static int p2p_ar_take_pushed(ggml_backend_cuda_context & ctx, const ggml_tensor * ar_node) {
    GGML_ASSERT(ctx.p2p_pushed == nullptr || ctx.p2p_pushed == ar_node);
    const int pushed = ctx.p2p_pushed == ar_node;
    ctx.p2p_pushed = nullptr;
    return pushed;
}

bool ggml_cuda_p2p_ar_push_target(const ggml_tensor * ar_node, ggml_cuda_p2p_push * push) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_P2P_PUSH"); return e == nullptr || atoi(e) != 0; }();
    if (!on || ar_node == nullptr || ar_node->op != GGML_OP_ALLREDUCE) {
        return false;
    }
    p2p_ar_op_params op;
    memcpy(&op, ar_node->op_params, sizeof(op));
    const ggml_cuda_p2p_ar * ar = op.ar;
    const int64_t n_el = ggml_nelements(ar_node);
    // two ranks, a computed partial, the kernel path's sizes (the copy engine's pre kernel rewrites the partial in place)
    if (ar == nullptr || ar->n != 2 || op.zero || op.cap != ar->cap || n_el > ar->cap || n_el % 4 != 0 ||
        n_el * (int64_t) sizeof(float) > P2P_AR_CE_MIN_BYTES || ar_node->type != GGML_TYPE_F32 || !ggml_is_contiguous(ar_node)) {
        return false;
    }
    push->wire16 = p2p_ar_wire16_for(ar_node->src[0], n_el);
    push->peer   = ar->stage[1 - op.rank] + (int64_t) op.rank * ar->cap;
    push->half   = (int64_t) ar->n * ar->cap * (push->wire16 ? 2 : 1);
    push->epoch  = ar->epoch[op.rank];
    return true;
}

void ggml_cuda_op_allreduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->data == dst->src[0]->data);

    p2p_ar_op_params op;
    memcpy(&op, dst->op_params, sizeof(op));
    ggml_cuda_p2p_ar_launch_impl(op.ar, op.rank, ctx.stream(), (float *) dst->data, ggml_nelements(dst), op.zero,
        atoi(dst->name + strlen("allreduce_")), p2p_ar_take_pushed(ctx, dst), p2p_ar_wire16_for(dst->src[0], ggml_nelements(dst)));
}

void ggml_cuda_p2p_ar_launch(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    ggml_cuda_p2p_ar_launch_impl(ar, rank, stream, (float *) t->data, ggml_nelements(t),
        (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0, -1, 0,
        p2p_ar_is_ce(ar, ggml_nelements(t)) ? p2p_ar_ce_wire16_for(t) : p2p_ar_wire16_for(t, ggml_nelements(t)));
}

size_t ggml_cuda_p2p_ar_ce_min_bytes(const ggml_cuda_p2p_ar * ar) {
    return ar->n == 2 ? (size_t) P2P_AR_CE_MIN_BYTES : SIZE_MAX;
}

bool ggml_cuda_p2p_ar_ce_eligible(const ggml_cuda_p2p_ar * ar, const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->data != nullptr && ggml_nelements(t) <= ar->cap &&
        p2p_ar_is_ce(ar, ggml_nelements(t));
}

uint64_t ggml_cuda_p2p_ar_ce_start(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, ggml_tensor * t) {
    GGML_ASSERT(ggml_cuda_p2p_ar_ce_eligible(ar, t));
    return p2p_ar_ce_start(ar, rank, stream, (float *) t->data, ggml_nelements(t),
        (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0, -1, p2p_ar_ce_wire16_for(t));
}

void ggml_cuda_p2p_ar_ce_finish(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, ggml_tensor * t, uint64_t call) {
    GGML_ASSERT(ggml_cuda_p2p_ar_ce_eligible(ar, t));
    p2p_ar_ce_finish(ar, rank, stream, (float *) t->data, ggml_nelements(t), (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0, -1, call);
}

// set once an in-graph AllReduce exists: the process runs under --split-mode tensor
static std::atomic<bool> p2p_ar_active{false};

bool ggml_cuda_p2p_ar_active() {
    return p2p_ar_active.load(std::memory_order_relaxed);
}

ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices) {
    if (n_devices < 2 || n_devices > P2P_AR_MAX_RANKS || getenv("GGML_CUDA_P2P_AR_DISABLE") != nullptr) {
        return nullptr;
    }
    // every pair needs peer access: each rank writes into every peer
    for (size_t i = 0; i < n_devices; ++i) {
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) {
                continue;
            }
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, devices[i], devices[j]));
            if (!can) {
                GGML_LOG_INFO("%s: no peer access from device %d to %d, in-graph AllReduce off\n", __func__, devices[i], devices[j]);
                return nullptr;
            }
        }
    }
    const int n = (int) n_devices;
    ggml_cuda_p2p_ar * ar = new ggml_cuda_p2p_ar;
    ar->n = n;
    for (int i = 0; i < n; ++i) {
        ar->devices[i] = devices[i];
        ggml_cuda_set_device(devices[i]);
        for (int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(err);
            }
        }
        (void) cudaGetLastError();
        // load the kernel now: a lazy load while a peer spins in the AllReduce can deadlock
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p<float>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p<float4>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p_add_rms_norm_mul<false>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p_add_rms_norm_mul<true>));
        CUDA_CHECK(cudaMalloc(&ar->epoch[i], 2 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ar->epoch[i], 0, 2 * sizeof(unsigned)));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p_ce_pre<float>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p_ce_pre<float4>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p_ce_sum<float>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p_ce_sum<float4>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p16));
        CUDA_CHECK(cudaMalloc(&ar->ce[i], 4 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ar->ce[i], 0, 4 * sizeof(unsigned)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ar->ce_stream[i], cudaStreamNonBlocking));
        for (int s = 0; s < P2P_AR_CE_SLOTS; ++s) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ar->ce_ev_pre[i][s],  cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ar->ce_ev_copy[i][s], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (const char * path = getenv("GGML_CUDA_P2P_AR_STAMPS")) {
        const char * ctl = getenv("GGML_CUDA_P2P_AR_STAMPS_CTL");
        ar->stamp_ctl = ctl != nullptr ? ctl : "";
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(devices[i]);
            CUDA_CHECK(cudaHostAlloc((void **) &ar->stamps[i], P2P_AR_STAMP_RING * sizeof(p2p_ar_stamp), cudaHostAllocMapped | cudaHostAllocPortable));
            memset(ar->stamps[i], 0, P2P_AR_STAMP_RING * sizeof(p2p_ar_stamp));
            const unsigned v = ctl != nullptr ? 0u : 1u;
            CUDA_CHECK(cudaMalloc(&ar->stamp_on[i], sizeof(unsigned)));
            CUDA_CHECK(cudaMemcpy(ar->stamp_on[i], &v, sizeof(v), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaStreamCreateWithFlags(&ar->stamp_stream[i], cudaStreamNonBlocking));
        }
        ar->stamp_thread = std::thread(p2p_ar_stamp_dump, ar, std::string(path));
        GGML_LOG_INFO("%s: GGML_CUDA_P2P_AR_STAMPS: per-call stamps to %s (%zu-byte records)\n", __func__, path, sizeof(p2p_ar_stamp) + 4);
    }
    if (const char * ctl = getenv("GGML_CUDA_P2P_AR_PF_CTL")) {
        ar->pf_ctl = ctl;
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(devices[i]);
            CUDA_CHECK(cudaMalloc(&ar->pf_off[i], sizeof(unsigned)));
            CUDA_CHECK(cudaMemset(ar->pf_off[i], 0, sizeof(unsigned)));
            CUDA_CHECK(cudaStreamCreateWithFlags(&ar->pf_stream[i], cudaStreamNonBlocking));
        }
        ar->pf_thread = std::thread(p2p_ar_pf_poll, ar);
        GGML_LOG_INFO("%s: GGML_CUDA_P2P_AR_PF_CTL: the prefetch under the AllReduce is off while %s exists\n", __func__, ctl);
    }
    GGML_LOG_INFO("%s: in-graph P2P AllReduce on %d devices, %s wire (GGML_CUDA_P2P_WIRE)\n", __func__, n,
        p2p_ar_wire16_on() ? "bf16 (16-bit)" : "fp32 (32-bit)");
    p2p_ar_active = true;
    return ar;
}

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar) {
    if (ar == nullptr) {
        return;
    }
    if (ar->stamp_thread.joinable()) {
        ar->stamp_stop = true;
        ar->stamp_thread.join();
    }
    if (ar->pf_thread.joinable()) {
        ar->pf_stop = true;
        ar->pf_thread.join();
    }
    for (int i = 0; i < ar->n; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (ar->stamps[i] != nullptr) {
            CUDA_CHECK(cudaFreeHost(ar->stamps[i]));
            CUDA_CHECK(cudaFree(ar->stamp_on[i]));
            CUDA_CHECK(cudaStreamDestroy(ar->stamp_stream[i]));
        }
        if (ar->pf_off[i] != nullptr) {
            CUDA_CHECK(cudaFree(ar->pf_off[i]));
            CUDA_CHECK(cudaStreamDestroy(ar->pf_stream[i]));
        }
        CUDA_CHECK(cudaFree(ar->stage[i]));
        CUDA_CHECK(cudaFree(ar->ce_stage[i]));
        for (int s = 0; s < P2P_AR_CE_SLOTS; ++s) {
            CUDA_CHECK(cudaFree(ar->ce_pack[i][s]));
        }
        CUDA_CHECK(cudaFree(ar->ce[i]));
        if (ar->ce_stream[i] != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(ar->ce_stream[i]));
            for (int s = 0; s < P2P_AR_CE_SLOTS; ++s) {
                CUDA_CHECK(cudaEventDestroy(ar->ce_ev_pre[i][s]));
                CUDA_CHECK(cudaEventDestroy(ar->ce_ev_copy[i][s]));
            }
        }
        CUDA_CHECK(cudaFree(ar->epoch[i]));
    }
    delete ar;
}

bool ggml_cuda_p2p_ar_reserve(ggml_cuda_p2p_ar * ar, size_t nbytes) {
    const int64_t cap = ((int64_t) ((nbytes + sizeof(float) - 1) / sizeof(float)) + 3) / 4 * 4; // 16-byte slots
    if (cap <= ar->cap) {
        return true;
    }
    // growing moves the staging buffers: no call may be in flight
    for (int i = 0; i < ar->n; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int i = 0; i < ar->n; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        CUDA_CHECK(cudaFree(ar->stage[i]));
        CUDA_CHECK(cudaMalloc(&ar->stage[i], 2 * (size_t) ar->n * cap * sizeof(float)));
        static_assert(P2P_AR_SENTINEL == 0xffffffffu, "the stage is filled bytewise");
        CUDA_CHECK(cudaMemset(ar->stage[i], 0xff, 2 * (size_t) ar->n * cap * sizeof(float)));
        CUDA_CHECK(cudaFree(ar->ce_stage[i]));
        ar->ce_stage[i] = nullptr;
        for (int s = 0; s < P2P_AR_CE_SLOTS; ++s) {
            CUDA_CHECK(cudaFree(ar->ce_pack[i][s]));
            ar->ce_pack[i][s] = nullptr;
        }
        if (ar->n == 2 && cap * (int64_t) sizeof(float) > P2P_AR_CE_MIN_BYTES) {
            CUDA_CHECK(cudaMalloc(&ar->ce_stage[i], (size_t) P2P_AR_CE_SLOTS * cap * sizeof(float)));
            CUDA_CHECK(cudaMemset(ar->ce_stage[i], 0xff, (size_t) P2P_AR_CE_SLOTS * cap * sizeof(float)));
            if (p2p_ar_ce_wire16_on()) {
                for (int s = 0; s < P2P_AR_CE_SLOTS; ++s) {
                    CUDA_CHECK(cudaMalloc(&ar->ce_pack[i][s], (size_t) cap * sizeof(unsigned short)));
                }
            }
        }
    }
    for (int i = 0; i < ar->n; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ar->cap = cap;
    return true;
}

void ggml_cuda_p2p_ar_set_params(ggml_cuda_p2p_ar * ar, size_t rank, ggml_tensor * node) {
    GGML_ASSERT(rank < (size_t) ar->n);
    p2p_ar_op_params op = {};
    op.ar   = ar;
    op.cap  = ar->cap;
    op.rank = (int32_t) rank;
    op.zero = (node->src[0]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
    memset(node->op_params, 0, sizeof(node->op_params));
    memcpy(node->op_params, &op, sizeof(op));
}

// the AllReduce with the trunk's residual ADD -> RMS_NORM -> MUL (norm.cu's add_rms_norm_mul_f32,
// LLAMA_FOLD_ADD_NORM) as its epilogue, one kernel per reduction. Block b owns the flat range [b*per, (b+1)*per), which
// is whole rows (one row per block); it pushes that range and then forms its rows. Under the
// sentinel protocol a peer may split the same tensor differently (the plain kernel), since every value is its own
// signal. Each row is formed exactly as add_rms_norm_mul_f32<1024, 8> forms it (the same columns per thread, the same
// block reduction), so the outputs are bitwise those of the AllReduce followed by the fold.
template <bool qpn>
static __global__ void __launch_bounds__(1024) k_allreduce_p2p_add_rms_norm_mul(
        float * data, const int64_t n_el, const p2p_ar_args p,
        const float * b, const int64_t sb1, float * sum, float * dst, const int ncols, const float eps, const float * mul,
        const ggml_cuda_qpn_dst q, const int n_ar, const ggml_cuda_p2p_ar_pf pf) {
    constexpr int block_size = 1024;
    constexpr int n_reg      = 8;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    if (blk >= n_ar) {
        // a prefetch block: the next products' first weight bytes into L2 while the row blocks wait
        p2p_ar_prefetch(pf, (int64_t) (blk - n_ar)*block_size + tid, (int64_t) (gridDim.x - n_ar)*block_size);
        return;
    }
    const unsigned long long t_start = p.stamps != nullptr ? p2p_ar_now() : 0;

    __shared__ unsigned ep;
    if (tid == 0) {
        ep = p2p_ar_begin(p.epoch);
    }
    __syncthreads();
    p2p_ar_stamp * st = p2p_ar_stamping(p) ? p.stamps + (ep & (P2P_AR_STAMP_RING - 1)) : nullptr;
    if (tid == 0) {
        if (st != nullptr && blk < P2P_AR_STAMP_BLOCKS) { st->t[0][blk] = t_start; }
    }

    const int64_t per  = (n_el + n_ar - 1) / n_ar;
    const int64_t i0   = blk * per;
    const int64_t i1   = min(n_el, i0 + per);
    const int64_t half = (int64_t) (ep & 1) * p.n * p.cap;

    const int64_t my_slot = half + p.rank * p.cap;
    const int64_t half16    = 2*half;
    const int64_t my_slot16 = 2*my_slot;
    if (p.pushed) {
        // the producing product pushed this rank's partial already
    } else if (p.wire16) {
        if ((i0 % 8) == 0 && (i1 - i0) % 8 == 0) {
            const bool aligned = (uintptr_t) data % 16 == 0;
            for (int64_t c = tid; c < (i1 - i0) / 8; c += block_size) {
                const uint4 v = p.zero ? uint4{} : p2p_ar_load_wire16(data, i0 / 8 + c, aligned);
                for (int r = 0; r < p.n; ++r) {
                    if (r != p.rank) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"((unsigned short *) p.stage[r] + my_slot16 + i0 + 8*c),
                            "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
                    }
                }
            }
        } else {
            for (int64_t i = i0 + tid; i < i1; i += block_size) {
                const unsigned short v = p.zero ? 0 : ggml_cuda_p2p_wire16(data[i]);
                for (int r = 0; r < p.n; ++r) {
                    if (r != p.rank) {
                        ((unsigned short *) p.stage[r])[my_slot16 + i] = v;
                    }
                }
            }
        }
    } else if ((i0 % 4) == 0 && (i1 - i0) % 4 == 0 && (uintptr_t) data % 16 == 0 && p.cap % 4 == 0) {
        // 16-byte vectors on the wire
        const float4 * src4 = (const float4 *) (data + i0);
        for (int64_t c = tid; c < (i1 - i0) / 4; c += block_size) {
            const float4 v = p.zero ? float4{} : p2p_ar_vec<float4>::wire(src4[c]);
            for (int r = 0; r < p.n; ++r) {
                if (r != p.rank) {
                    ((float4 *) (p.stage[r] + my_slot + i0))[c] = v;
                }
            }
        }
    } else {
        for (int64_t i = i0 + tid; i < i1; i += block_size) {
            const float v = p.zero ? 0.0f : p2p_ar_wire(data[i]);
            for (int r = 0; r < p.n; ++r) {
                if (r != p.rank) {
                    p.stage[r][my_slot + i] = v;
                }
            }
        }
    }
    __syncthreads(); // the rows below are written in place by other threads than the ones that pushed them
    if (tid == 0) {
        P2P_AR_STAMP(1, blk);
        P2P_AR_STAMP(2, blk);
    }

    float * stage_mine = p.stage[p.rank] + half;
    unsigned short * stage16_mine = (unsigned short *) p.stage[p.rank] + half16;
    extern __shared__ float s_sum[];
    for (int64_t row = i0 / ncols; row < i1 / ncols; ++row) {
        const int64_t o = row * ncols;
        float tmp = 0.0f;
        float xr[n_reg];
        float wr[n_reg];
        float ar[n_reg], br[n_reg];
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                float acc = 0.0f;
                if (p.wire16) {
                    for (int s = 0; s < p.n; ++s) {
                        const unsigned short w = s != p.rank ? p2p_ar_take16(stage16_mine + s * 2*p.cap + o + col, p.max_spins) :
                            p.zero ? (unsigned short) 0 : ggml_cuda_p2p_wire16(data[o + col]);
                        const float v = ggml_cuda_p2p_unwire16(w);
                        acc = s == 0 ? v : acc + v;
                    }
                } else {
                    for (int s = 0; s < p.n; ++s) {
                        const float v = s != p.rank ? p2p_ar_take(stage_mine + s * p.cap + o + col, p.max_spins) : p.zero ? 0.0f : p2p_ar_wire(data[o + col]);
                        acc = s == 0 ? v : acc + v;
                    }
                }
                ar[k] = acc;
                br[k] = b[row*sb1 + col];
                wr[k] = mul[col];
            }
        }
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                data[o + col] = ar[k]; // the AllReduce's own output, as the plain kernel leaves it
                const float xi = ar[k] + br[k]; // binbcast.cu's op_add (commutative: either operand order)
                sum[o + col] = xi;
                tmp += xi * xi;
                xr[k] = xi;
            }
        }

        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);

        float * s_x = s_sum + 32;
#pragma unroll
        for (int k = 0; k < n_reg; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                const float y = scale * xr[k] * wr[k];
                dst[o + col] = y;
                if constexpr (qpn) {
                    s_x[col] = y;
                }
            }
        }
        if constexpr (qpn) {
            __syncthreads();
            qpn_source_row_smem(s_x, ncols/QK_K, (int) row, q);
        }
        __syncthreads(); // s_sum and s_x are reused by the next row
    }
    if (st != nullptr && tid == 0) {
        P2P_AR_STAMP(3, blk);
        P2P_AR_STAMP(4, blk);
    }
    if (st != nullptr && blk == 0 && tid == 0) {
        st->n_el = n_el; st->id = p.id; st->kind = p.kind; st->nb = n_ar;
        __threadfence_system();
        st->seq = ep;
    }
    p2p_ar_end(p.epoch, ep, n_ar);
    GGML_UNUSED(q);
}

bool ggml_cuda_op_allreduce_add_rms_norm_mul_supported(const ggml_tensor * ar, const ggml_tensor * add) {
    const int64_t ncols  = ar->ne[0];
    const int64_t n_rows = ggml_nrows(ar);
    return ar->type == GGML_TYPE_F32 && ggml_is_contiguous(ar) && ncols >= 1024 && ncols <= 8*1024 &&
        n_rows % 8 == 0 && n_rows <= 64 && ggml_are_same_shape(ar, add); // the same set as the 8-block kernel's
}

void ggml_cuda_op_allreduce_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * ar_node, const ggml_tensor * ar_in,
        const ggml_tensor * add, const ggml_tensor * rms_norm, ggml_tensor * mul_tensor, const ggml_cuda_p2p_ar_pf * pf_in) {
    GGML_ASSERT(ar_node->data == ar_node->src[0]->data);
    // the ADD may read a reshape of the AllReduce (the same contiguous floats)
    GGML_ASSERT(ar_in == ar_node || (ar_in->data == ar_node->data && ggml_is_contiguous(ar_in) && ggml_is_contiguous(ar_node) &&
                                     ggml_nelements(ar_in) == ggml_nelements(ar_node) && ar_in->ne[0] == ar_node->ne[0]));
    GGML_ASSERT(ggml_cuda_op_allreduce_add_rms_norm_mul_supported(ar_in, add));
    GGML_ASSERT(add->src[0] == ar_in || add->src[1] == ar_in);
    const ggml_tensor * b = add->src[0] == ar_in ? add->src[1] : add->src[0];
    const ggml_tensor * w = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];

    p2p_ar_op_params op;
    memcpy(&op, ar_node->op_params, sizeof(op));
    const ggml_cuda_p2p_ar * ar = op.ar;
    const int64_t n_el = ggml_nelements(ar_node);
    GGML_ASSERT(n_el <= ar->cap);

    p2p_ar_args p = {};
    for (int r = 0; r < ar->n; ++r) {
        p.stage[r] = ar->stage[r];
    }
    p.epoch = ar->epoch[op.rank];
    p.cap   = ar->cap;
    p.rank  = op.rank;
    p.n     = ar->n;
    p.zero  = op.zero;
    static const bool debug = getenv("GGML_CUDA_P2P_AR_DEBUG") != nullptr;
    p.max_spins = debug ? int64_t(1) << 24 : int64_t(1) << 28;
    p.id        = atoi(ar_node->name + strlen("allreduce_"));
    p.stamps    = ar->stamps[op.rank];
    p.stamp_on  = ar->stamp_on[op.rank];
    p.pushed    = p2p_ar_take_pushed(ctx, ar_node);
    p.wire16    = p2p_ar_wire16_for(ar_node->src[0], n_el);

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int     ncols = (int) add->ne[0];
    const int64_t sb1   = b->nb[1]/sizeof(float);
    const int     n_rows = (int) ggml_nrows(add);

    // the prefetch blocks after the n_rows row blocks: one per SM the row blocks leave (72 on this board)
    ggml_cuda_p2p_ar_pf pf;
    int n_pf = 0;
    if (pf_in != nullptr && pf_in->n > 0) {
        pf     = *pf_in;
        pf.off = ar->pf_off[op.rank];
        n_pf   = std::max(8, ggml_cuda_info().devices[ggml_cuda_get_device()].nsm - n_rows);
    }

    ggml_cuda_qpn_dst q;
    const size_t smem_q = (32 + (size_t) ncols) * sizeof(float);
    if (ncols % QK_K == 0 && smem_q <= 48*1024 && ggml_cuda_qpn_source_begin(mul_tensor, ncols, n_rows, ctx.stream(), &q)) {
        p.kind = p.pushed ? 6 : 2;
        k_allreduce_p2p_add_rms_norm_mul<true><<<n_rows + n_pf, 1024, smem_q, ctx.stream()>>>((float *) ar_node->data, n_el, p,
            (const float *) b->data, sb1, (float *) add->data, (float *) mul_tensor->data, ncols, eps, (const float *) w->data, q,
            n_rows, pf);
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_qpn_source_end(mul_tensor, ncols, ctx.stream(), q);
        return;
    }
    p.kind = p.pushed ? 5 : 1;
    k_allreduce_p2p_add_rms_norm_mul<false><<<n_rows + n_pf, 1024, 32*sizeof(float), ctx.stream()>>>((float *) ar_node->data, n_el, p,
        (const float *) b->data, sb1, (float *) add->data, (float *) mul_tensor->data, ncols, eps, (const float *) w->data, q,
        n_rows, pf);
    CUDA_CHECK(cudaGetLastError());
}
