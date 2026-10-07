#pragma once

#include "common.cuh"

// Volta tensor-core path for the verify's multi-token dense K-quant GEMVs.
// LLAMA_MMVQ_TC: atoi bitmask, default all on, 0 restores the dp4a path.
// One bit per type; only Q5_K has a kernel so far (the output head is the one product it wins on).
#define GGML_CUDA_MMVQ_TC_Q5_K 1
#define GGML_CUDA_MMVQ_TC_ALL  GGML_CUDA_MMVQ_TC_Q5_K

// true if this dense mul_mat runs on the tensor-core path; such a product never quantizes src1 to q8_1,
// so the q8_1 sharing plan must neither lead nor join a group with it
bool ggml_cuda_mmvq_tc_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int device);

void ggml_cuda_mul_mat_vec_q_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
