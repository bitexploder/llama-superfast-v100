#pragma once

#include "common.cuh"

// MUL_MAT_ID's routed expert products on Volta's FP16 tensor cores (a port of Strata-V100's
// expert_kernel_wmma, MIT, see mmid-fp16tc.cu), for the seven expert formats of Flash-Next: IQ2_XXS, IQ2_XS, IQ2_S,
// IQ3_XXS, IQ3_S, IQ4_NL and Q2_0. It reads MMQ's compact, expert-sorted q8_1 activations and writes FP32 rows where
// MMQ writes them (through ids_dst). Taken on Volta for those types with ne00 % 64 == 0, ne01 % 128 == 0 and at least
// LLAMA_MMID_FP16TC_MIN_TOKENS tokens (default MMID_FP16TC_MIN_TOKENS); LLAMA_MMID_FP16TC=0 keeps MMQ.
// Returns false (nothing launched) when the route does not apply.
bool ggml_cuda_mmid_fp16tc(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, int64_t n_tokens,
        const void * src1_q8_1, int64_t act_rows, const int32_t * ids_dst, const int32_t * expert_bounds,
        float * dst, int64_t stride_dst, cudaStream_t stream);
