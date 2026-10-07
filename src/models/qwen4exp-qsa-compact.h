#pragma once

// Compact selected-KV attention for qwen4exp QSA decode and small verification batches.
//
// build_attn_qsa's original path attends over the whole cache view with a mask that is
// -inf off the top_k cells and the ordinary KQ mask on them. That is the same softmax as
// attending over only the top_k cells with their ordinary mask entries, so this gathers,
// for every query, its selected K/V rows and mask entries and runs the existing flash
// attention on that compact set, one query per ne[3] slot. The selection itself (top_k,
// ties, tail, causal mask) is not touched, and the cache keeps its full history.
//
// The gathers write f16 directly (ggml_get_rows_as) when dev, the device holding the cache,
// supports that node; otherwise, including dev == nullptr, they gather f32 and cast.
//
// The compact length is padded up to pad_to (the CUDA GQA-shared tile path needs a
// multiple of 256) with repeats of already selected cells whose mask entries are -inf.
//
// top_k may repeat a cell if top_k_bias gives each repeat -inf: the bias is added to the
// gathered mask entry of every slot, as ggml_qsa_select's plane 1 is meant to be.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

// the gathered length: the selection width padded up to pad_to. the compact path applies only to a view wider than this
static inline int64_t llama_qsa_compact_width(int64_t width, int64_t pad_to) {
    return GGML_PAD(width, pad_to);
}

// k/v: cache views [D, n_head_kv, n_kv, n_stream] with contiguous rows (v not transposed)
// q: [D, n_head, n_tokens]; kq_mask: F16 [n_kv, >= n_tps, 1, n_stream]
// top_k: I32 [width, n_tps, 1, n_stream], contiguous, distinct cells per query unless a
//        top_k_bias of -inf drops the repeats
static inline bool llama_qsa_compact_applies(
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * v,
        const ggml_tensor * kq_mask,
        const ggml_tensor * top_k,
        int64_t             max_tokens,
        int64_t             pad_to) {
    if (!q || !k || !v || !kq_mask || !top_k || pad_to <= 0) {
        return false;
    }

    const int64_t n_kv     = k->ne[2];
    const int64_t n_stream = k->ne[3];
    const int64_t width    = top_k->ne[0];
    const int64_t n_tps    = top_k->ne[1];
    const int64_t wp       = llama_qsa_compact_width(width, pad_to);

    auto rows_contiguous = [](const ggml_tensor * t) {
        return !ggml_is_quantized(t->type) && t->nb[0] == ggml_type_size(t->type) &&
               t->nb[1] == t->ne[0]*t->nb[0] && t->nb[2] == t->ne[1]*t->nb[1];
    };

    return n_tps*n_stream <= max_tokens &&        // decode and small verification only; prefill keeps the original path
           wp < n_kv &&                            // otherwise there is nothing to compact
           wp - width <= width &&                  // padding repeats selected cells
           top_k->type == GGML_TYPE_I32 && ggml_is_contiguous(top_k) &&
           top_k->ne[2] == 1 && top_k->ne[3] == n_stream &&
           kq_mask->type == GGML_TYPE_F16 && kq_mask->ne[0] == n_kv &&
           kq_mask->ne[1] >= n_tps && kq_mask->ne[3] == n_stream &&
           q->ne[2] == n_tps*n_stream && q->ne[0] == k->ne[0] &&
           v->ne[1] == k->ne[1] && v->ne[2] == n_kv && v->ne[3] == n_stream &&
           rows_contiguous(k) && rows_contiguous(v);
}

// returns F32 [D_v*n_head, n_tokens] like build_attn_mha; *fa receives the attention node
// top_k_bias: F32 [width, n_tps, 1, n_stream] added to the gathered mask entries, or nullptr
static inline ggml_tensor * llama_qsa_compact_attn(
        ggml_context * ctx,
        ggml_tensor *  q,
        ggml_tensor *  k,
        ggml_tensor *  v,
        ggml_tensor *  kq_mask,
        ggml_tensor *  top_k,
        float          kq_scale,
        float          max_bias,
        float          softcap,
        int64_t            pad_to,
        ggml_backend_dev_t dev,
        ggml_tensor **     fa,
        ggml_tensor *      top_k_bias = nullptr) {
    const int64_t n_hkv    = k->ne[1];
    const int64_t n_kv     = k->ne[2];
    const int64_t n_stream = k->ne[3];
    const int64_t width    = top_k->ne[0];
    const int64_t n_tps    = top_k->ne[1];
    const int64_t n_tokens = n_tps*n_stream;
    const int64_t wp       = GGML_PAD(width, pad_to);
    const int64_t n_pad    = wp - width;

    // row indices per stream: query t of stream s owns rows [t*wp, (t+1)*wp)
    ggml_tensor * idx = top_k;
    if (n_pad > 0) {
        ggml_tensor * rep = ggml_view_4d(ctx, top_k, n_pad, n_tps, 1, n_stream, top_k->nb[1], top_k->nb[2], top_k->nb[3], 0);
        idx = ggml_concat(ctx, top_k, rep, 0);
    }
    idx = ggml_reshape_2d(ctx, idx, wp*n_tps, n_stream);

    // [d, n_hkv, n_kv, n_stream] -> gathered f16 [d, wp, n_hkv, n_tokens]
    auto gather = [&](ggml_tensor * c) {
        const int64_t d = c->ne[0];
        ggml_tensor * rows = ggml_view_3d(ctx, c, d*n_hkv, n_kv, n_stream, c->nb[2], c->nb[3], 0);
        ggml_tensor * g = dev ? ggml_get_rows_as(ctx, rows, idx, GGML_TYPE_F16) : nullptr;
        if (g == nullptr || !ggml_backend_dev_supports_op(dev, g)) {
            g = ggml_cast(ctx, ggml_get_rows(ctx, rows, idx), GGML_TYPE_F16);
        }
        g = ggml_reshape_4d(ctx, g, d, n_hkv, wp, n_tokens);
        return ggml_permute(ctx, g, 0, 2, 1, 3);
    };
    ggml_tensor * kc = gather(k);
    ggml_tensor * vc = gather(v);

    // mask entries of the selected cells, then -inf for the padding
    ggml_tensor * m = ggml_view_4d(ctx, kq_mask, 1, n_kv, n_tps, n_stream,
            kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[3], 0);
    m = ggml_get_rows(ctx, m, ggml_reshape_3d(ctx, top_k, width, n_tps, n_stream));
    m = ggml_reshape_4d(ctx, m, width, n_tps, 1, n_stream);
    if (top_k_bias) {
        m = ggml_add(ctx, m, top_k_bias);
    }
    if (n_pad > 0) {
        ggml_tensor * ninf = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_pad, n_tps, 1, n_stream);
        ninf = ggml_fill(ctx, ninf, -INFINITY);
        m = ggml_concat(ctx, m, ninf, 0);
    }
    m = ggml_reshape_4d(ctx, m, wp, 1, 1, n_tokens);
    m = ggml_cast(ctx, m, GGML_TYPE_F16);

    // one query per ne[3] slot: [D, n_head, n_tokens] -> [D, 1, n_head, n_tokens]
    ggml_tensor * qc = ggml_view_4d(ctx, q, q->ne[0], q->ne[1], 1, n_tokens, q->nb[1], q->nb[2], q->nb[2], 0);
    qc = ggml_permute(ctx, qc, 0, 2, 1, 3);

    ggml_tensor * cur = ggml_flash_attn_ext(ctx, qc, kc, vc, m, kq_scale, max_bias, softcap);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    if (fa) {
        *fa = cur;
    }

    return ggml_reshape_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
}

// The union form, for verification batches: the queries of a stream share one gather of the
// distinct cells any of them selects (ggml_qsa_union), and each keeps its own mask entries,
// -inf on the cells only the others select. Adjacent queries select mostly the same blocks, so
// this gathers far fewer rows than one set per query. The gathered length is static (the
// worst case, n_tps*width) but only the union, padded to pad_to, is written: the CUDA gather
// skips the rest and flash attention skips the all -inf tiles past it (mask scan). It needs a
// device that gathers straight to f16, so dev == nullptr or a CPU cache keeps the form above.
//
// The softmax is the same set of cells with the same mask entries, so the result equals the
// per-query form up to the order flash attention accumulates in.

// the union bitmap lives in shared memory (QSA_UNION_MAX_WORDS in ggml-cuda/qsa-union.cuh)
#define LLAMA_QSA_UNION_MAX_KV (32*8192)

static inline bool llama_qsa_union_applies(
        const ggml_tensor * k,
        const ggml_tensor * top_k,
        ggml_backend_dev_t  dev) {
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }
    return top_k->ne[1] > 1 && k->ne[2] <= LLAMA_QSA_UNION_MAX_KV;
}

// the gathered length: the union can hold neither more than every slot nor more than the view
static inline int64_t llama_qsa_union_rows(const ggml_tensor * k, const ggml_tensor * top_k, int64_t pad_to) {
    return GGML_PAD(std::min(top_k->ne[0]*top_k->ne[1], k->ne[2]), pad_to);
}

// same contract as llama_qsa_compact_attn, for a case llama_qsa_compact_applies and
// llama_qsa_union_applies accept
static inline ggml_tensor * llama_qsa_union_attn(
        ggml_context * ctx,
        ggml_tensor *  q,
        ggml_tensor *  k,
        ggml_tensor *  v,
        ggml_tensor *  kq_mask,
        ggml_tensor *  top_k,
        float          kq_scale,
        float          max_bias,
        float          softcap,
        int64_t            pad_to,
        ggml_backend_dev_t dev,
        ggml_tensor **     fa,
        ggml_tensor *      top_k_bias = nullptr) {
    const int64_t n_hkv    = k->ne[1];
    const int64_t n_kv     = k->ne[2];
    const int64_t n_stream = k->ne[3];
    const int64_t n_tps    = top_k->ne[1];
    const int64_t n_max    = llama_qsa_union_rows(k, top_k, pad_to);
    // flash attention's mask scan reads whole column tiles of up to 8 queries
    const int64_t n_rows   = GGML_PAD(n_tps, 8);

    // [n_max, 1 + n_rows, n_stream]: plane 0 the cells (-1: not gathered), then the masks
    ggml_tensor * un = ggml_qsa_union(ctx, top_k, kq_mask, top_k_bias, (int) n_max, (int) n_rows, (int) pad_to);

    ggml_tensor * idx = ggml_view_2d(ctx, un, n_max, n_stream, un->nb[2], 0);
    idx = ggml_cast(ctx, idx, GGML_TYPE_I32);

    // [d, n_hkv, n_kv, n_stream] -> gathered f16 [d, n_max, n_hkv, n_stream]
    auto gather = [&](ggml_tensor * c) {
        const int64_t d = c->ne[0];
        ggml_tensor * rows = ggml_view_3d(ctx, c, d*n_hkv, n_kv, n_stream, c->nb[2], c->nb[3], 0);
        ggml_tensor * g = ggml_get_rows_as(ctx, rows, idx, GGML_TYPE_F16);
        GGML_ASSERT(ggml_backend_dev_supports_op(dev, g));
        g = ggml_reshape_4d(ctx, g, d, n_hkv, n_max, n_stream);
        return ggml_permute(ctx, g, 0, 2, 1, 3);
    };
    ggml_tensor * kc = gather(k);
    ggml_tensor * vc = gather(v);

    ggml_tensor * m = ggml_view_4d(ctx, un, n_max, n_rows, 1, n_stream, un->nb[1], un->nb[2], un->nb[2], un->nb[1]);
    m = ggml_cast(ctx, m, GGML_TYPE_F16);

    // [D, n_head, n_tokens] -> [D, n_tps, n_head, n_stream]
    ggml_tensor * qc = ggml_view_4d(ctx, q, q->ne[0], q->ne[1], n_tps, n_stream, q->nb[1], q->nb[2], q->nb[2]*n_tps, 0);
    qc = ggml_permute(ctx, qc, 0, 2, 1, 3);

    ggml_tensor * cur = ggml_flash_attn_ext(ctx, qc, kc, vc, m, kq_scale, max_bias, softcap);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    ggml_flash_attn_ext_set_mask_scan(cur, 1);
    if (fa) {
        *fa = cur;
    }

    // [D_v, n_head, n_tps, n_stream] -> [D_v*n_head, n_tokens]
    return ggml_reshape_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
}
