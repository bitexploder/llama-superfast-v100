#include "common.cuh"

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// LLAMA_FOLD_QSA_MASK: add = SET_ROWS(FILL(kq_mask), zeros, idx) + kq_mask in one launch
void ggml_cuda_op_qsa_mask_fill_rows_add(ggml_backend_cuda_context & ctx, const ggml_tensor * fill,
        const ggml_tensor * zeros, const ggml_tensor * set_rows, ggml_tensor * add);
