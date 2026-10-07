#include "common.cuh"

// Elementwise tail that a gated-residual product may carry out of its own output store, in place of
// the separate SCALE and UNARY nodes that would otherwise follow it (LLAMA_HC_EPILOGUE).
enum ggml_cuda_hc_epilogue_kind {
    GGML_CUDA_HC_EPI_NONE                = 0,
    GGML_CUDA_HC_EPI_SCALE_SILU          = 1, // silu(s0*x + b0): the low-rank down projection
    GGML_CUDA_HC_EPI_SCALE_SIGMOID_SCALE = 2, // s1*sigmoid(s0*x + b0) + b1: the injection projection
};

struct ggml_cuda_hc_epilogue {
    ggml_cuda_hc_epilogue_kind kind = GGML_CUDA_HC_EPI_NONE;
    float s0 = 1.0f, b0 = 0.0f, s1 = 1.0f, b1 = 0.0f;
};

// Matrix-vector products for the BF16 gated-residual (hyper-connection) projections, used only for
// mul_mats tagged GGML_HINT_HC_PROJ. Returns false, leaving dst untouched, for shapes it does not handle.
// With an epilogue, the result goes to out (the last node of the fused chain, same shape as dst)
// instead of dst; only the long-row (hc_dim) kernel takes one.
bool ggml_cuda_mul_mat_vec_hc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                              const ggml_cuda_hc_epilogue * epi = nullptr, ggml_tensor * out = nullptr);

// The same long-row kernel for untagged n_embd-wide (2560) BF16 x F32 products at one to eight columns,
// where mul_mat_vec_f would run; bit-identical to it. Returns false for other shapes (LLAMA_BF16_GEMV).
bool ggml_cuda_mul_mat_vec_bf16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Whether ggml_cuda_mul_mat_vec_bf16 takes this product on this device.
bool ggml_cuda_mul_mat_vec_bf16_ok(int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// n such products of one shared src1 in one launch, each bit-identical to its own ggml_cuda_mul_mat_vec_bf16
// launch. The caller has checked ggml_cuda_mul_mat_vec_bf16_ok for every one.
#define GGML_CUDA_BF16_GEMV_GROUP_MAX 4
void ggml_cuda_mul_mat_vec_bf16_group(ggml_backend_cuda_context & ctx, int n, const ggml_tensor * const * src0,
                                      const ggml_tensor * src1, ggml_tensor * const * dst);
