#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// LLAMA_FOLD_MOE_TAIL: add = the weighted sum of the experts + shexp * sigmoid(gate), in one launch
void ggml_cuda_op_moe_weighted_reduction_shexp_tail(ggml_backend_cuda_context & ctx,
                                                    const ggml_tensor *         experts,
                                                    const ggml_tensor *         expert_scale,
                                                    const ggml_tensor *         weights,
                                                    const ggml_tensor *         sigmoid,
                                                    const ggml_tensor *         mul,
                                                    ggml_tensor *               add);
