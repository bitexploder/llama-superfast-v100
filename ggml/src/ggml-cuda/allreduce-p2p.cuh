#pragma once

#include "common.cuh"

// In-graph AllReduce for 2-8 GPUs with peer access between every pair (NVLink or PCIe). The meta backend puts one
// GGML_OP_ALLREDUCE node per partial tensor into each GPU's graph, so a whole pass runs as one CUDA graph per GPU.
// Each call pushes the local partial into every peer's staging buffer and polls its own buffer until the peers' values
// have replaced a sentinel (no flags). Call counters live in device memory, so replays of a captured graph
// stay in step on all GPUs.

struct ggml_cuda_p2p_ar;

// nullptr if the devices can not use peer access
ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices);
void               ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar);

// true once an in-graph AllReduce was created: the process runs under --split-mode tensor (false on one card)
bool ggml_cuda_p2p_ar_active();

// make the staging buffers hold nbytes per call; synchronizes the devices if they must grow
bool ggml_cuda_p2p_ar_reserve(ggml_cuda_p2p_ar * ar, size_t nbytes);

// fill the op params of an ALLREDUCE node that runs on device index rank
void ggml_cuda_p2p_ar_set_params(ggml_cuda_p2p_ar * ar, size_t rank, ggml_tensor * node);

void ggml_cuda_op_allreduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the same kernel launched eagerly on stream for tensor t on device index rank, outside any graph: the meta
// backend's unmerged first run of a graph shape uses it, with the same wire rounding and fp32 sum as graph replays
void ggml_cuda_p2p_ar_launch(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, ggml_tensor * t);

// the copy-engine reduction (two ranks, a partial over ggml_cuda_p2p_ar_ce_min_bytes) in two calls, so other compute
// can run between them while the copy engine moves the partial: start enqueues the pre kernel on stream and the copy on the rank's
// own copy stream, and returns the call's index; finish enqueues on stream the wait for that copy and the sum in place. Start the
// calls of every rank before finishing any, and finish them in the order they started; outputs are bitwise the eager call's.
size_t   ggml_cuda_p2p_ar_ce_min_bytes(const ggml_cuda_p2p_ar * ar);
bool     ggml_cuda_p2p_ar_ce_eligible(const ggml_cuda_p2p_ar * ar, const ggml_tensor * t);
uint64_t ggml_cuda_p2p_ar_ce_start(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, ggml_tensor * t);
void     ggml_cuda_p2p_ar_ce_finish(ggml_cuda_p2p_ar * ar, int rank, cudaStream_t stream, ggml_tensor * t, uint64_t call);

// the AllReduce node ar followed by the trunk fold's ADD (add, reading ar) -> RMS_NORM -> MUL in one kernel,
// bitwise the AllReduce followed by norm.cu's add_rms_norm_mul fold; for few rows only (a multiple of 8, at most 64)
bool ggml_cuda_op_allreduce_add_rms_norm_mul_supported(const ggml_tensor * ar, const ggml_tensor * add);
// While its row blocks wait for the peer, the fused kernel's extra blocks pull the first bytes of every row tile of
// the next products (QPN-repacked weights, tile-major: tile t is nsb tile-superblocks at t*tile_bytes) into L2 with
// prefetch.global.L2, pre_bytes from each tile's start, which are the first superblocks each of the product's split-K warps
// reads. Values are untouched. GGML_CUDA_P2P_AR_PF_MB (default 6, 0 off) caps the bytes; GGML_CUDA_P2P_AR_PF_CTL=<path>: the
// prefetch is skipped while <path> exists (a device-memory switch, so one run can time both arms in its captured graphs).
#define GGML_CUDA_P2P_AR_PF_MAX 3
struct ggml_cuda_p2p_ar_pf {
    const char * W[GGML_CUDA_P2P_AR_PF_MAX]          = {};
    int64_t      tile_bytes[GGML_CUDA_P2P_AR_PF_MAX] = {};
    int64_t      pre_bytes[GGML_CUDA_P2P_AR_PF_MAX]  = {};
    int          ntiles[GGML_CUDA_P2P_AR_PF_MAX]     = {};
    int          n = 0;
    const unsigned * off = nullptr; // GGML_CUDA_P2P_AR_PF_CTL's switch (nonzero: skip), or nullptr
};
// the byte budget (GGML_CUDA_P2P_AR_PF_MB), 0 when off
int64_t ggml_cuda_p2p_ar_pf_budget();

// ar_in is the tensor the ADD reads: ar itself, or a reshape of it (the same contiguous floats); pf (nullable) the
// next products' weights to prefetch
void ggml_cuda_op_allreduce_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * ar, const ggml_tensor * ar_in,
        const ggml_tensor * add, const ggml_tensor * rms_norm, ggml_tensor * mul, const ggml_cuda_p2p_ar_pf * pf = nullptr);

// The producer's push (the overlap design; Flux's per-tile epilogue push without its flags). A QPN product whose
// output is the partial of the next ALLREDUCE node writes each finished output tile, made safe for the wire, straight into this
// rank's landing slot on the peer, as soon as the tile's split-K sum is done, so the transfer of the first tiles runs under the
// product's remaining weight stream. The ALLREDUCE node then only polls, sums in rank order and runs its epilogue: the sums are
// identical to the plain reduction at the selected wire precision (bitwise the same at fp32).
// Two ranks, the kernel path's sizes only. GGML_CUDA_P2P_PUSH=0 turns it off.
struct ggml_cuda_p2p_push {
    float          * peer  = nullptr; // this rank's slot on the peer, half 0, at the partial's first element
    int64_t          half  = 0;       // wire elements from half 0 to half 1 of the peer's stage
    const unsigned * epoch = nullptr; // this rank's call counter: the call about to run is epoch[0] + 1, its half that parity
    int              wire16 = 0;      // bf16 wire: offsets count 16-bit elements
};

static __host__ __device__ __forceinline__ void ggml_cuda_p2p_push_advance(ggml_cuda_p2p_push & push, const int64_t n) {
    push.peer = push.wire16 ? (float *) ((unsigned short *) push.peer + n) : push.peer + n;
}

// the push target of ALLREDUCE node ar, whose partial is n_el floats; false if the call can not take a pushed partial
bool ggml_cuda_p2p_ar_push_target(const ggml_tensor * ar, ggml_cuda_p2p_push * push);

// a value as it goes on the wire: never the sentinel bit pattern 0xffffffff (an input holding it goes as another NaN)
static __device__ __forceinline__ float ggml_cuda_p2p_wire(const float v) {
    return __float_as_uint(v) == 0xffffffffu ? __uint_as_float(0x7fffffffu) : v;
}

// bf16 wire arithmetic, shared by the kernel wire and the copy-engine wire.
// RNE; quiet NaNs, reserving 0xffff for the sentinel.
static __host__ __device__ __forceinline__ unsigned short ggml_cuda_p2p_wire16(const unsigned u) {
    unsigned short r;
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        r = (unsigned short) ((u >> 16) | 0x0040u);
    } else {
        r = (unsigned short) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return r == 0xffffu ? (unsigned short) 0x7fffu : r;
}
static __device__ __forceinline__ unsigned short ggml_cuda_p2p_wire16(const float v) {
    return ggml_cuda_p2p_wire16(__float_as_uint(v));
}
static __device__ __forceinline__ unsigned ggml_cuda_p2p_wire16x2(const float a, const float b) {
    return (unsigned) ggml_cuda_p2p_wire16(a) | ((unsigned) ggml_cuda_p2p_wire16(b) << 16);
}
static __device__ __forceinline__ float ggml_cuda_p2p_unwire16(const unsigned short w) {
    return __uint_as_float((unsigned) w << 16);
}

// one warp pushes the n_cols columns of rows [row0, row0 + 32) of a finished output (column c at src + c*stride, the same
// offsets on the peer) as 16-byte stores of whole 128-byte rows (64 bytes at bf16); own writes precede reads
static __device__ __forceinline__ void ggml_cuda_p2p_push_tile(const ggml_cuda_p2p_push & push, const float * src, const int64_t stride,
        const int64_t row0, const int n_cols) {
    __syncwarp();
    unsigned ep;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(ep) : "l"(push.epoch) : "memory");
    src += row0;
    if (push.wire16) {
        unsigned short * dst = (unsigned short *) push.peer + (int64_t) ((ep + 1) & 1)*push.half + row0;
        for (int i = threadIdx.x % WARP_SIZE; i < 4*n_cols; i += WARP_SIZE) {
            const int64_t o = (i >> 2)*stride + 8*(i & 3);
            const float4 a = __ldcg((const float4 *) (src + o));
            const float4 b = __ldcg((const float4 *) (src + o + 4));
            const unsigned w0 = ggml_cuda_p2p_wire16x2(a.x, a.y), w1 = ggml_cuda_p2p_wire16x2(a.z, a.w);
            const unsigned w2 = ggml_cuda_p2p_wire16x2(b.x, b.y), w3 = ggml_cuda_p2p_wire16x2(b.z, b.w);
            asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + o), "r"(w0), "r"(w1), "r"(w2), "r"(w3) : "memory");
        }
        return;
    }
    float * dst = push.peer + (int64_t) ((ep + 1) & 1)*push.half + row0;
    for (int i = threadIdx.x % WARP_SIZE; i < 8*n_cols; i += WARP_SIZE) {
        const int64_t o = (i >> 3)*stride + 4*(i & 7);
        float4 v = __ldcg((const float4 *) (src + o));
        v.x = ggml_cuda_p2p_wire(v.x); v.y = ggml_cuda_p2p_wire(v.y); v.z = ggml_cuda_p2p_wire(v.z); v.w = ggml_cuda_p2p_wire(v.w);
        // one 16-byte store: nvcc 12.8 split a plain float4 store here into four 4-byte stores, which cross PCIe as partial writes
        // at about a tenth of the link's rate (290 us per 160 KB instead of 31)
        asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(dst + o), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
}
