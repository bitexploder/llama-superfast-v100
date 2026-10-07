#pragma once

#include "common.cuh"

// Volta prompt attention that packs all 6 query heads of each KV head into the rows of two CUTLASS GEMMs
// (a port of 1Cat-vLLM's sm70_v37 operator, fattn-gqa6.cu). Taken only for D 256, F16 K/V, 6 query heads per KV head,
// a mask and at least LLAMA_FA_GQA6_MIN_ROWS rows and LLAMA_FA_GQA6_MIN_KV keys; LLAMA_FA_GQA6=0 keeps today's kernels.
bool ggml_cuda_fattn_gqa6_applies(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_gqa6(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
