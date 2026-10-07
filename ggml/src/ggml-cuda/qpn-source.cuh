#pragma once

// fp16 at the source (LLAMA_QPN_PREP_AT_SOURCE, default on). A kernel that produces the input of products
// on repacked weights (mmvq-qpn.cu) writes that input's prepared form (qpn-prep.cuh: xh, xsc and, when a consumer
// reads them, the per-32 sums xs) as a second output, next to its fp32 output, which stays: the residual, the dp4a
// products and QPN's non-finite fallback read it. The share planner (ggml_cuda_plan_q8_share) gives the prepared input
// its slot from the producer's node on, and the consumers then find it ready and launch no prep. The producers:
//   add_rms_norm_mul_f32 (the trunk's ADD + RMS_NORM + MUL fold), rms_norm_mul_qpn_f32 (a lone RMS_NORM + MUL, layer 0),
//   rms_norm_mul_gate_qpn_f32 (the GDN gated norm, silu form), unary_gated_qpn_kernel (the FFN SwiGLU),
//   cont_sigmoid_mul_qpn_f32 (the attention output gate), k_get_rows_qpn (the output rows, before the head).
// Each forms its fp32 values exactly as its ordinary kernel does, then hands one warp one 256-column slice of one token
// (8 consecutive columns per lane) to qpn_prep_slice_warp, the one definition of the prepared input.
//
// LLAMA_QPN_PREP_CHECK (a bitmask, default 0), the byte-identity instrument: 1 = after each producer that wrote a
// prepared input, run qpn_prep_kernel on its fp32 output and compare every byte (xh, xsc, and xs where written) on the
// device; 2 = also make one value per launch non-finite (NaN, +inf, -inf in turn) in the prepared copy only, and in the
// reference's copy of the fp32 output, so the non-finite path is compared too (the fp32 output is untouched; QPN's
// fallback recomputes that token's column from it); 4 = every eligible producer output writes a prepared input into a
// scratch buffer even with no QPN consumer (test-backend-ops). Totals are logged when the backend context is freed.

#include "qpn-prep.cuh"

struct ggml_cuda_qpn_dst {
    half  * xh     = nullptr; // nullptr: nothing to write
    half  * xs     = nullptr; // nullptr: no consumer reads the per-32 sums
    float * xsc    = nullptr;
    int     T      = 0;       // tokens: the producer's rows
    int     inj_sb = -1;      // LLAMA_QPN_PREP_CHECK & 2: the value (slice, token, lane, element) made non-finite
    int     inj_t  = -1;
    int     inj_g  = -1;
    int     inj_i  = -1;
    float   inj_v  = 0.0f;
};

// the most tokens a producer writes the prepared input for (one pass of the tensor-core kernels; 8 with LLAMA_QPN_W16=0)
#define GGML_CUDA_QPN_SOURCE_MAX_TOKENS 16

// defined in ggml-cuda.cu. Called by a producer's launcher once it knows it can write the prepared input, before its
// launch: true (and d filled, the input marked prepared on this stream) if out is the input of products on repacked
// weights, of K = out->ne[0]*... columns (a multiple of 256) and T <= 8 tokens (16), that the share plan prepares at source
bool ggml_cuda_qpn_source_begin(const ggml_tensor * out, int64_t K, int T, cudaStream_t stream, ggml_cuda_qpn_dst * d);
// after the launch: the LLAMA_QPN_PREP_CHECK comparison (nothing when it is off)
void ggml_cuda_qpn_source_end(const ggml_tensor * out, int64_t K, cudaStream_t stream, const ggml_cuda_qpn_dst & d);

// the slice (sb, t) by one warp, lane g holding columns sb*256 + 8*g .. + 7 in v
static __device__ __forceinline__ void qpn_source_slice(float * v, const int sb, const int t, const ggml_cuda_qpn_dst & d) {
    if (d.inj_sb == sb && d.inj_t == t && d.inj_g == (int) (threadIdx.x % WARP_SIZE)) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = i == d.inj_i ? d.inj_v : v[i];
        }
    }
    qpn_prep_slice_warp(v, sb, t, d.T, d.xh, d.xs, d.xsc);
}

// the slice (sb, t) by the calling warp from its 256 fp32 values in shared memory
static __device__ __forceinline__ void qpn_source_slice_smem(const float * s_x, const int sb, const int t, const ggml_cuda_qpn_dst & d) {
    const int lane = threadIdx.x % WARP_SIZE;
    const float4 a = *(const float4 *) (s_x + lane*8);
    const float4 b = *(const float4 *) (s_x + lane*8 + 4);
    float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    qpn_source_slice(v, sb, t, d);
}

// the same from a row of K fp32 values in shared memory, one warp per slice, the block's warps in turn
static __device__ __forceinline__ void qpn_source_row_smem(const float * s_x, const int nsb, const int t, const ggml_cuda_qpn_dst & d) {
    const int lane = threadIdx.x % WARP_SIZE;
    for (int sb = threadIdx.x / WARP_SIZE; sb < nsb; sb += blockDim.x / WARP_SIZE) {
        const float4 a = *(const float4 *) (s_x + sb*QK_K + lane*8);
        const float4 b = *(const float4 *) (s_x + sb*QK_K + lane*8 + 4);
        float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        qpn_source_slice(v, sb, t, d);
    }
}
