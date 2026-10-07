// launcher and route of the streaming verify kernel (fattn-verify-stream.cuh)
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-verify-stream.cuh"

// The verify and draft-step call (1 to 8 tokens, a GQA group of 6, D = 256) over an F16 cache of at least LLAMA_FATTN_VERIFY_STREAM keys
// (default 4096; -1 or 0: off, the mma routes are then used). And 9 to 16 tokens (two slots'
// verify), as pairs of 8-token blocks over the same K/V stream (LLAMA_FATTN_VERIFY_STREAM_W16=0: 1 to 8 only, as before). And over
// several KV streams (the non-unified cache, one per sequence: ne[3] of Q, K, V and the mask), each block on its own stream
// (LLAMA_FATTN_VERIFY_STREAM_NS=0: one stream only, as before).
static int fvs_min_keys() {
    static const int min_keys = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM"); return e ? atoi(e) : 4096; }();
    return min_keys;
}
static int fvs_max_tok() {
    static const int max_tok = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_W16"); return e == nullptr || atoi(e) != 0 ? 16 : 8; }();
    return max_tok;
}

// the kernel's limits, for the KV cache's whole-cache decode graphs: the least key count it takes and the most tokens
// (min_keys = 0: the kernel is off, or a CUDA device is not Volta)
void ggml_cuda_fattn_verify_stream_limits(int * min_keys, int * max_tok) {
    int mk = fvs_min_keys();
    for (int id = 0; id < ggml_cuda_info().device_count; ++id) {
        if (ggml_cuda_info().devices[id].cc != GGML_CUDA_CC_VOLTA) {
            mk = 0;
        }
    }
    *min_keys = std::max(mk, 0);
    *max_tok  = fvs_max_tok();
}

bool ggml_cuda_fattn_verify_stream_applies(const ggml_tensor * dst, int device) {
    const int min_keys = fvs_min_keys();
    if (min_keys <= 0 || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || sinks != nullptr || ggml_get_op_params_i32(dst, 4) != 0) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || !mask || mask->type != GGML_TYPE_F16) {
        return false;
    }
    const int max_tok = fvs_max_tok();
    static const bool multi_stream = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_NS"); return e == nullptr || atoi(e) != 0; }();
    const int64_t ns = Q->ne[3];
    if (Q->ne[0] != fvs::D || K->ne[0] != fvs::D || V->ne[0] != fvs::D || Q->ne[1] < 1 || Q->ne[1] > max_tok ||
            ns < 1 || (ns > 1 && !multi_stream) || K->ne[3] != ns || V->ne[3] != ns ||
            K->ne[2] == 0 || Q->ne[2] != 6*K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) {
        return false;
    }
    if (K->ne[1] < min_keys || K->ne[1] % FATTN_KQ_STRIDE != 0 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 ||
            mask->ne[3] != ns) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if ((uintptr_t) t->data % 16 != 0 || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0) {
            return false;
        }
    }
    return Q->nb[0] == 4 && K->nb[0] == 2 && V->nb[0] == 2;
}

void ggml_cuda_flash_attn_ext_verify_stream(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // by default the PV warps load and store the last 8 of the 16 K rows per lane (KP = 8; the same mma sequence and
    // reductions, so bitwise); LLAMA_FATTN_VERIFY_STREAM_KP=0 restores the earlier kernel (all of K in the QK warps)
    static const bool kp8 = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_KP"); return !e || atoi(e) != 0; }();
    constexpr size_t nbytes_shared = sizeof(fvs::smem_t);
    // the shared-memory limit is a per-device attribute (the macro keeps one flag per device), so it is raised on every
    // device that launches the kernel, not once inside the static below (the second card of --split-mode tensor failed the launch)
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false>),    nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 8>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, true>),     nbytes_shared);
    static int occ = [] {
        int o = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, kp8 ? fvs::verify_stream<6, false, 8> : fvs::verify_stream<6, false>, fvs::NTH, nbytes_shared));
        return std::max(o, 1);
    }();

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_tok  = Q->ne[1];
    const int n_head = Q->ne[2];
    const int nkvh   = K->ne[2];
    const int n_kv   = K->ne[1];
    const int n_strm = Q->ne[3]; // KV streams, one grid row each
    const int ntiles = n_kv / fvs::TK;
    const int nsplit = std::max(1, std::min(occ*nsm / nkvh, ntiles)); // one wave (of 8-token blocks)
    // src[6] is the used key count of a decode graph over the whole cache (read by the kernel). The split above is computed
    // from the whole cache's tiles; under this assert it equals the one computed from any used count of at least min_keys keys
    const int32_t * kv_count = dst->src[6] ? (const int32_t *) dst->src[6]->data : nullptr;
    if (kv_count) {
        GGML_ASSERT(occ*nsm / nkvh <= fvs_min_keys() / fvs::TK);
    }
    // 9 to 16 tokens, two 8-token blocks per (KV head, split), side by side; the split is the 8-token call's, so each token's
    // result is bit-identical to it
    const int nhalf  = (n_tok + 7)/8;

    ggml_cuda_pool_alloc<float>  parts(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) n_strm*n_tok*n_head*nsplit*fvs::D);
        meta.alloc((size_t) n_strm*n_tok*n_head*nsplit);
    }

    static const bool stub = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_STUB"); return e && atoi(e) != 0; }(); // ceiling probe, wrong output
    (stub ? fvs::verify_stream<6, true> : kp8 ? fvs::verify_stream<6, false, 8> : fvs::verify_stream<6, false>)<<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, nbytes_shared, stream>>>(
        (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
        (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
        Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
        Q->nb[3], K->nb[3], V->nb[3], mask->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    if (nsplit > 1) {
        const dim3 blocks_num_combine(n_tok, n_head, n_strm);
        flash_attn_combine_results<fvs::D><<<blocks_num_combine, fvs::D, nsplit*sizeof(float2), stream>>>(parts.ptr, meta.ptr, (float *) dst->data, nsplit);
        CUDA_CHECK(cudaGetLastError());
    }
}
