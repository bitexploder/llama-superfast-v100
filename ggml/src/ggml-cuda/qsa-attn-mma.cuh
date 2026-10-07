#pragma once

#include "common.cuh"

// the QSA prompt attention (a sparse FLASH_ATTN_EXT node, see qsa-attn.cu) on Volta's mma.m8n8k4 tensor
// cores, a port of Strata-V100's prompt_attn_v70_kernel (MIT, see qsa-attn-mma.cu). One block per (token, KV head)
// walks the token's n_sel selected cells in chunks of 32 with an online softmax, no split. Needs head dim 256,
// 12 query heads per KV head and an sm_70 device; qsa-attn.cu routes prompt chunks (n_split == 1) here unless
// LLAMA_QSA_ATTN_MMA=0.
void ggml_cuda_qsa_attn_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
