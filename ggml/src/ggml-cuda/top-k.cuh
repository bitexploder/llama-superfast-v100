#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_TOP_K_SPLIT (ggml.h)
void ggml_cuda_op_top_k_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
