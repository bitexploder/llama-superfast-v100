#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// LLAMA_FOLD_NORM_ROPE: RMS_NORM -> MUL by the weight -> ROPE multi (M-RoPE, IMRoPE), F32, one launch
void ggml_cuda_op_rms_norm_mul_rope_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul, ggml_tensor * rope);

// qwen4exp's QSA indexer keys (build_qsa_top_k) in one pass: GET_ROWS of each block's r member rows,
// the mean over them (r - 1 ADDs of CONT slices, then SCALE), RMS_NORM, MUL by the norm weight and
// ROPE (multi, no freq factors), written to the ROPE node's output. The ROPE node carries the rope
// parameters and positions; the rest comes from the other nodes of the chain.
struct ggml_cuda_qsa_pool {
    const ggml_tensor * k     = nullptr;   // GET_ROWS src0: [ncols, n_kv, n_stream]
    const ggml_tensor * cells = nullptr;   // GET_ROWS src1: I32 [r*n_blocks, n_stream]
    const ggml_tensor * w     = nullptr;   // MUL weight: F32 [ncols]
    int   r     = 0;
    float scale = 0.0f;                    // SCALE: scale*x + bias
    float bias  = 0.0f;
    float eps   = 0.0f;                    // RMS_NORM
};

void ggml_cuda_op_qsa_pool_rope(ggml_backend_cuda_context & ctx, ggml_tensor * rope, const ggml_cuda_qsa_pool & p);
