#include "common.cuh"

// bitmap words in shared memory, one bit per cell: the op covers n_kv <= 32*QSA_UNION_MAX_WORDS
#define QSA_UNION_MAX_WORDS 8192

void ggml_cuda_op_qsa_union(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_qsa_union_supported(const ggml_tensor * op);
