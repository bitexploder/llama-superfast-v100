#include "common.cuh"

// GGML_OP_DRAFT_PICK, the MTP draft's pick of its next token on the device (ggml-draft-pick.h)
void ggml_cuda_op_draft_pick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
