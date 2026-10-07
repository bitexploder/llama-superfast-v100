#pragma once

#include "common.cuh"

// block-sparse QSA prompt attention: a FLASH_ATTN_EXT node carrying selected cell ids in src[5] (see qsa-attn.cu)
bool ggml_cuda_qsa_attn_is_sparse(const ggml_tensor * dst);
bool ggml_cuda_qsa_attn_supported(const ggml_tensor * dst);
void ggml_cuda_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
