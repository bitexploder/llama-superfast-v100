#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

// The MoE weighted reduction and the shared expert's gated tail in one launch (LLAMA_FOLD_MOE_TAIL):
// dst = moe_weighted_reduction_f32's sum + shexp * sigmoid(gate), where the separate kernels store the sum and
// shexp_gate_tail_f32 adds to it. The sum is formed with moe_weighted_reduction_f32's statements, and the tail
// rounds each operation on its own as shexp_gate_tail_f32 does, so dst is bit-identical.
static __global__ void moe_weighted_reduction_shexp_tail_f32(const float * __restrict__ experts,
                                                             const float * __restrict__ expert_scale,
                                                             const float * __restrict__ weights,
                                                             const float *              shexp,
                                                             const float * __restrict__ gate,
                                                             float *                    dst,
                                                             const int64_t n_embd,
                                                             const int     n_expert_used,
                                                             const int64_t s_shexp1,
                                                             const int64_t s_gate1,
                                                             const int64_t s_dst1) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }

    const float sig   = 1.0f / (1.0f + expf(-gate[token*s_gate1])); // unary.cu's op_sigmoid
    const float gated = __fmul_rn(shexp[col + token*s_shexp1], sig);
    dst[col + token*s_dst1] = __fadd_rn(sum, gated);
}

void ggml_cuda_op_moe_weighted_reduction_shexp_tail(ggml_backend_cuda_context & ctx,
                                                    const ggml_tensor *         experts,
                                                    const ggml_tensor *         expert_scale,
                                                    const ggml_tensor *         weights,
                                                    const ggml_tensor *         sigmoid,
                                                    const ggml_tensor *         mul,
                                                    ggml_tensor *               add) {
    const ggml_tensor * gate  = sigmoid->src[0];
    const ggml_tensor * shexp = mul->src[0] == sigmoid ? mul->src[1] : mul->src[0];
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32);
    GGML_ASSERT(shexp->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || (expert_scale->type == GGML_TYPE_F32 && ggml_is_contiguous(expert_scale)));
    GGML_ASSERT(ggml_is_contiguous(experts) && ggml_is_contiguous(weights) && ggml_is_contiguous(add));
    GGML_ASSERT(shexp->nb[0] == sizeof(float) && gate->ne[0] == 1);

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    GGML_ASSERT(add->ne[0] == n_embd && add->ne[1] == n_tokens);

    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_shexp_tail_f32<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *) experts->data, expert_scale ? (const float *) expert_scale->data : nullptr,
        (const float *) weights->data, (const float *) shexp->data, (const float *) gate->data, (float *) add->data,
        n_embd, (int) n_expert_used, (int64_t) (shexp->nb[1]/sizeof(float)), (int64_t) (gate->nb[1]/sizeof(float)),
        (int64_t) (add->nb[1]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}
