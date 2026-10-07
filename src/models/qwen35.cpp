#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache.h"

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // LLAMA_GDN_AB_MERGE (0 turns it off): a GDN layer's a and b projections read the same input, so
    // the graph runs them as one product (build_layer_attn_linear). Their two [n_embd, n_v_heads] weights are
    // loaded into the halves of one [n_embd, 2*n_v_heads] tensor, alpha's rows first, and ssm_alpha and ssm_beta
    // become views of it: nothing is copied and no memory is added. The merged tensor is created just before the
    // two, in the context the layer's weights go to, so the allocator places it before its views. Only a device
    // context is merged, where every tensor is allocated here and the loader fills each view by its file name.
    static const bool ab_merge = [] {
        const char * e = getenv("LLAMA_GDN_AB_MERGE");
        return e == nullptr || atoi(e) != 0;
    }();
    auto ctx_of = [&](const ggml_tensor * t, const llama_model_loader::ctx_key ** key) -> ggml_context * {
        for (auto & [k, c] : ml.ctx_map) {
            if (ggml_get_tensor(c.get(), ggml_get_name(t)) == t) {
                if (key != nullptr) {
                    *key = &k;
                }
                return c.get();
            }
        }
        return nullptr;
    };
    bool ab_split_logged = false;
    auto ab_create = [&](int il, const ggml_tensor * anchor, int64_t n_rows) -> ggml_tensor * {
        const ggml_tensor * ma = ml.get_tensor_meta(tn(LLM_TENSOR_SSM_ALPHA, "weight", il).str().c_str());
        const ggml_tensor * mb = ml.get_tensor_meta(tn(LLM_TENSOR_SSM_BETA,  "weight", il).str().c_str());
        // under --split-mode tensor the meta device splits tensors by name and has no rule for the merged weight, so
        // alpha and beta stay upstream's two tensors (ssm_alpha, ssm_beta), each split by its own rule
        if (ab_merge && split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            if (!ab_split_logged) {
                LLAMA_LOG_WARN("%s: --split-mode tensor: the GDN a/b projections are not merged\n", __func__);
                ab_split_logged = true;
            }
            return nullptr;
        }
        if (!ab_merge || anchor == nullptr || ma == nullptr || mb == nullptr || ma->type != mb->type ||
            ma->ne[0] != n_embd || mb->ne[0] != n_embd || ma->ne[1] != n_rows || mb->ne[1] != n_rows ||
            ma->ne[2] != 1 || mb->ne[2] != 1 || ma->ne[3] != 1 || mb->ne[3] != 1) {
            return nullptr;
        }
        const llama_model_loader::ctx_key * key = nullptr;
        ggml_context * ctx = ctx_of(anchor, &key);
        if (ctx == nullptr || key->lazy || ggml_backend_buft_is_host(key->buft) ||
            ggml_get_mem_size(ctx) - ggml_used_mem(ctx) < 8*ggml_tensor_overhead()) {
            return nullptr;
        }
        ggml_tensor * ab = ggml_new_tensor_2d(ctx, ma->type, n_embd, 2*n_rows);
        ggml_format_name(ab, "blk.%d.ssm_ab.weight", il);
        return ab;
    };
    auto ab_bind = [&](ggml_tensor * ab, ggml_tensor * alpha, ggml_tensor * beta) {
        if (ab == nullptr) {
            return;
        }
        ggml_context * ctx = ctx_of(ab, nullptr);
        if (alpha == nullptr || beta == nullptr || ctx_of(alpha, nullptr) != ctx || ctx_of(beta, nullptr) != ctx ||
            alpha->type != ab->type || beta->type != ab->type || alpha->data != nullptr || beta->data != nullptr ||
            alpha->view_src != nullptr || beta->view_src != nullptr) {
            // the merged tensor stays unused (allocated, never read)
            LLAMA_LOG_WARN("%s: %s not merged: alpha and beta went elsewhere\n", __func__, ggml_get_name(ab));
            return;
        }
        alpha->view_src  = ab;
        alpha->view_offs = 0;
        beta->view_src   = ab;
        beta->view_offs  = ggml_nbytes(alpha);
    };

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            ggml_tensor * ab     = flags == 0 ? ab_create(il, layer.ssm_a, n_v_heads) : nullptr;
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            ab_bind(ab, layer.ssm_alpha, layer.ssm_beta);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        GGML_ASSERT(params.tu_mode == LLM_GRAPH_TU_NONE);
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// a model with its trunk (not an MTP-only draft file)
bool llama_model_qwen35::supports_layer_groups() const {
    return hparams.n_layer() > 0 && layers[0].attn_norm != nullptr && layer_group_period() > 0;
}

// the shortest period of layer types that repeats over the trunk and holds both types, so every group has a GDN layer and an
// attention layer (and so builds every graph input it creates); 0 if there is none
int32_t llama_model_qwen35::layer_group_period() const {
    const int32_t n = (int32_t) hparams.n_layer();
    for (int32_t p = 2; p <= n; ++p) {
        if (n % p != 0) {
            continue;
        }
        bool ok = true;
        for (int32_t il = 0; il < n && ok; ++il) {
            ok = hparams.is_recr(il) == hparams.is_recr(il % p);
        }
        bool has_recr = false;
        bool has_attn = false;
        for (int32_t il = 0; il < p; ++il) {
            (hparams.is_recr(il) ? has_recr : has_attn) = true;
        }
        if (ok && has_recr && has_attn) {
            return p;
        }
    }
    return 0;
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // a prompt job's group runs layers [il0, il1): the first starts from the token embeddings, a later one from the saved
    // residual; a group before the last ends by writing the residual there, the last builds the outputs. No input is created that
    // the group's layers do not read (every group holds both layer types, llama_model_qwen35::layer_group_period)
    const bool tu       = tu_mode != LLM_GRAPH_TU_NONE;
    const int  il0      = tu ? tu_il_begin : 0;
    const int  il1      = tu ? tu_il_end   : (int) n_layer;
    const bool from_saved = tu_mode == LLM_GRAPH_TU_MID   || tu_mode == LLM_GRAPH_TU_LAST;
    const bool to_saved   = tu_mode == LLM_GRAPH_TU_FIRST || tu_mode == LLM_GRAPH_TU_MID;
    if (tu) {
        GGML_ASSERT(tu_resid != nullptr && tu_resid->ne[0] == n_embd && tu_resid->ne[1] >= n_tokens);
        GGML_ASSERT(0 <= il0 && il0 < il1 && il1 <= n_layer);
        GGML_ASSERT((il0 == 0) == !from_saved && (il1 == n_layer) == !to_saved);
    }

    if (from_saved) {
        inpL = ggml_view_2d(ctx0, tu_resid, n_embd, n_tokens, tu_resid->nb[1], 0);
        cb(inpL, "tu_resid_in", il0);
    } else {
        inpL = build_inp_embd(model.tok_embd);
        cb(inpL, "model.input_embed", -1);
    }

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = to_saved ? nullptr : build_inp_out_ids();

    // under --split-mode tensor a prefill ubatch runs as two halves interleaved layer by layer (build_tbo)
    if (tbo_applies()) {
        build_tbo(inpL, inp, inp_pos, inp_out_ids, sections, il0, il1, to_saved);
        return;
    }

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = il0; il < il1; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    // the residual after layer il1 - 1, for the next group
    if (to_saved) {
        ggml_tensor * dst = ggml_view_2d(ctx0, tu_resid, n_embd, n_tokens, tu_resid->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, inpL, dst));
        return;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Two-batch overlap under --split-mode tensor (Domino, ISO, SGLang's two-batch overlap; the meta backend side is in
// ggml-backend-meta.cpp's unmerged loop). Each prefill-size reduction of the trunk is a 2-card exchange of n_embd x n_tokens floats
// on the copy engine, and the next node needs its sum, so on one stream the copy never overlaps a kernel. Here a ubatch of one
// sequence is built as two halves, rows [0, n_a) and [n_a, n_tokens), n_a a multiple of 64, interleaved layer by layer with the
// first half one phase ahead:
//   layer L, attention part: A, then B (B's GDN starts from A's state after layer L, B's attention reads A's K/V of layer L)
//   layer L, FFN part:       A, then B
// so independent compute can run while the copy engine moves a half's reduction. Products run at half width;
// the engaged synthetic harness is not bitwise against the whole ubatch, so numerical validation is required.
// The GDN restarts at A's fp32 state on a chunk boundary (64 | n_a); the conv reads A's last columns;
// attention runs once over all rows (flash attention's stream-k splits a tile's keys by the tile count, so a half would sum
// in another order), between the halves' projections and their output products. LLAMA_TP_TBO=0 turns it off;
// LLAMA_TP_TBO_MIN (default 256) is the smallest ubatch split.
bool llama_model_qwen35::graph::tbo_applies() const {
    static const bool on = [] {
        const char * e = getenv("LLAMA_TP_TBO");
        return e == nullptr || atoi(e) != 0;
    }();
    static const int64_t n_min = [] {
        const char * e = getenv("LLAMA_TP_TBO_MIN");
        return e ? (int64_t) atoll(e) : (int64_t) 256;
    }();
    if (!on || model.split_mode() != LLAMA_SPLIT_MODE_TENSOR || n_tokens < std::max<int64_t>(n_min, 128)) {
        return false;
    }
    if (ubatch.n_seqs != 1 || (int64_t) ubatch.n_seq_tokens != n_tokens || !ubatch.equal_seqs()) {
        return false;
    }
    if (cparams.embeddings_nextn_masked || !cparams.flash_attn || hparams.f_clamp_kqv > 0.0f || hparams.n_pos_per_embd() == 0) {
        return false;
    }
    // a GDN half runs the fused chunked op, as the whole ubatch does with the recurrent snapshots on or the fused op chosen
    if (cparams.n_rs_seq == 0 && !cparams.fused_gdn_ch) {
        return false;
    }
    for (bool b : cparams.embeddings_layer_inp) {
        if (b) {
            return false;
        }
    }
    const int64_t n_a = (n_tokens / 2) / 64 * 64;
    if (n_a < 64 || n_tokens - n_a < std::max<int64_t>(64, (int64_t) cparams.n_rs_seq + 1)) {
        return false;
    }
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        if (hparams.is_recr(il)) {
            // the a/b weights are not merged under the split
            if (layer.ssm_alpha == nullptr || layer.ssm_alpha->view_src != nullptr || layer.wqkv == nullptr || layer.wqkv_gate == nullptr) {
                return false;
            }
        } else if (layer.wq_b || layer.wk_b || layer.wv_b || layer.wqkv_b || layer.wo_b) {
            return false;
        }
    }
    return true;
}

// the fp16 expansions build_tbo made for the layer it is building: the halves' builders read each weight through tbo_w,
// which returns its expansion if it has one (the builders are build_tbo's own, so the table is set only while it runs)
namespace {
struct tbo_w16_table {
    const ggml_tensor * w[8];
    ggml_tensor *       w16[8];
    int                 n = 0;
};
thread_local const tbo_w16_table * tbo_w16_cur = nullptr;
struct tbo_w16_scope {
    explicit tbo_w16_scope(const tbo_w16_table * t) { tbo_w16_cur = t; }
    ~tbo_w16_scope() { tbo_w16_cur = nullptr; }
};
}

// LLAMA_TP_W16=0 leaves out the fp16 copies build_tbo shares between the two halves of a prompt ubatch
// The outputs are the same either way. Off, the compute buffer no longer holds a layer's gate, up and down
// copies together (588 MiB a card less on the 27B), and each half's product expands its weight itself, as before,
// so prompt reading is slower. It is for more context in the same memory, for example two users. Default on.
static bool tbo_w16_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_TP_W16");
        const bool v = e == nullptr || atoi(e) != 0;
        LLAMA_LOG_WARN("%s: shared fp16 weight copies for both prompt halves: %s (LLAMA_TP_W16=0 turns them off and frees their buffer)\n",
            __func__, v ? "on" : "off");
        return v;
    }();
    return on;
}

static ggml_tensor * tbo_w(ggml_tensor * w) {
    if (tbo_w16_cur != nullptr) {
        for (int i = 0; i < tbo_w16_cur->n; ++i) {
            if (tbo_w16_cur->w[i] == w) {
                return tbo_w16_cur->w16[i];
            }
        }
    }
    return w;
}

void llama_model_qwen35::graph::build_tbo(ggml_tensor * inpL, llm_graph_input_mem_hybrid * inp, ggml_tensor * inp_pos,
        ggml_tensor * inp_out_ids, int * sections, int il0, int il1, bool to_saved) {
    llm_graph_input_attn_kv * inp_attn = inp->get_attn();
    GGML_ASSERT(inp_attn->self_k_rot == nullptr && inp_attn->self_v_rot == nullptr);
    GGML_ASSERT(inp_attn->self_k_idxs->ne[0] == n_tokens && inp_attn->self_v_idxs->ne[0] == n_tokens); // V not transposed

    const int64_t n_a     = (n_tokens / 2) / 64 * 64;
    const int64_t r0[2]   = { 0, n_a };
    const int64_t nr[2]   = { n_a, n_tokens - n_a };
    const int64_t n_pe    = inp_pos->ne[0] / n_tokens;
    const size_t  es_pos  = ggml_element_size(inp_pos);

    ggml_tensor * x[2];
    ggml_tensor * pos[2];
    for (int h = 0; h < 2; ++h) {
        x[h] = ggml_view_2d(ctx0, inpL, n_embd, nr[h], inpL->nb[1], r0[h]*inpL->nb[1]);
        if (n_pe == 1) {
            pos[h] = ggml_view_1d(ctx0, inp_pos, nr[h], r0[h]*es_pos);
        } else {
            // the positions are [n_pe][n_tokens]: this half's rows of each, contiguous
            ggml_tensor * p = ggml_view_2d(ctx0, inp_pos, nr[h], n_pe, n_tokens*es_pos, r0[h]*es_pos);
            pos[h] = ggml_reshape_1d(ctx0, ggml_cont(ctx0, p), nr[h]*n_pe);
        }
    }

    // a weight the CUDA backend repacked (GGML_TENSOR_FLAG_BACKEND_LAYOUT) is expanded to fp16 for cuBLAS above the
    // tensor-core kernel's widths (at most GGML_CUDA_QPN_MAX_TOKENS = 64 rows, LLAMA_QPN_MAX_TOKENS lowers it), and each half used
    // to expand it again inside its product. When both halves are wider than 64 rows, each such weight of the layer is expanded
    // once by a graph node (a CPY to F16, which the CUDA backend runs as that expansion) and both halves' products read it; the
    // allocator owns its memory, from the first half's product to the second's. Not under a LoRA (it looks a weight up by its
    // tensor). The attention layers' fused wqkv stays on the products' own expansion: the tensor split cuts it in two segments,
    // which a copy's split state does not keep
    tbo_w16_table w16;
    const bool w16_on = tbo_w16_enabled() && nr[0] > 64 && nr[1] > 64 && loras->empty();
    int n_w16 = 0;
    auto expand_once = [&](ggml_tensor * w, int il) {
        if (!w16_on || w == nullptr || w->view_src != nullptr || !(w->flags & GGML_TENSOR_FLAG_BACKEND_LAYOUT)) {
            return;
        }
        GGML_ASSERT(w16.n < (int) (sizeof(w16.w)/sizeof(w16.w[0])));
        ggml_tensor * t = ggml_cast(ctx0, w, GGML_TYPE_F16);
        cb(t, "w16", il);
        w16.w[w16.n]   = w;
        w16.w16[w16.n] = t;
        w16.n++;
        n_w16++;
    };
    tbo_w16_scope w16_scope(&w16);

    for (int il = il0; il < il1; ++il) {
        const auto & layer = model.layers[il];
        w16.n = 0;
        if (hparams.is_recr(il)) {
            expand_once(layer.wqkv, il);
            expand_once(layer.wqkv_gate, il);
            expand_once(layer.ssm_beta, il);
            expand_once(layer.ssm_alpha, il);
            expand_once(layer.ssm_out, il);
        } else {
            expand_once(layer.wq, il);
            expand_once(layer.wk, il);
            expand_once(layer.wv, il);
            expand_once(layer.wo, il);
        }
        expand_once(layer.ffn_gate, il);
        expand_once(layer.ffn_up, il);
        expand_once(layer.ffn_down, il);

        ggml_tensor * out[2];
        if (hparams.is_recr(il)) {
            ggml_tensor * conv_tail = nullptr;
            ggml_tensor * state     = nullptr;
            for (int h = 0; h < 2; ++h) {
                ggml_tensor * cur = build_norm(x[h], model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
                ggml_build_forward_expand(gf, cur);
                out[h] = build_layer_attn_linear_half(inp->get_recr(), cur, nr[h], h == 0, conv_tail, state, il);
                ggml_build_forward_expand(gf, out[h]);
            }
        } else {
            build_layer_attn_halves(inp_attn, x, pos, r0, nr, sections, il, out);
        }
        ggml_tensor * ffn_residual[2];
        ggml_tensor * ffn_out[2];
        for (int h = 0; h < 2; ++h) {
            ggml_tensor * cur = ggml_add(ctx0, out[h], x[h]);
            cb(cur, "attn_residual", il);

            ffn_residual[h] = cur;

            ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(attn_post_norm, "attn_post_norm", il);

            // build_layer_ffn's product, its weights read through the table
            GGML_ASSERT(layer.ffn_gate_inp == nullptr);
            cur = build_ffn(attn_post_norm,
                tbo_w(layer.ffn_up),   NULL, layer.ffn_up_s,
                tbo_w(layer.ffn_gate), NULL, layer.ffn_gate_s,
                tbo_w(layer.ffn_down), NULL, layer.ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
            ffn_out[h] = cur;
            ggml_build_forward_expand(gf, cur);
        }
        // Both products precede either reader: A's reduction can overlap B's FFN.
        for (int h = 0; h < 2; ++h) {
            ggml_tensor * cur = ggml_add(ctx0, ffn_out[h], ffn_residual[h]);
            cb(cur, "post_ffn", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            x[h] = cur;
            // B's is left to its next reader, B's attention norm in the next layer, which comes after that layer's A half:
            // the meta backend finishes B's FFN reduction just before it, so the transfer runs under A's half. The last layer's
            // concat reads both. A group's last layer completes both halves: the saved residual is written after it
            if (h == 0 || il == il1 - 1) {
                ggml_build_forward_expand(gf, cur);
            }
        }
    }
    w16.n = 0;

    // both halves' residual, each to its rows of the saved residual
    if (to_saved) {
        for (int h = 0; h < 2; ++h) {
            ggml_tensor * dst = ggml_view_2d(ctx0, tu_resid, n_embd, nr[h], tu_resid->nb[1], r0[h]*tu_resid->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, x[h], dst));
        }
        return;
    }

    {
        static int n_logged = 0;
        if (n_logged < 4) {
            n_logged++;
            LLAMA_LOG_INFO("%s: two-batch overlap: a ubatch of %lld rows as %lld + %lld (LLAMA_TP_TBO=0 turns it off); %d weights expanded once for both halves\n",
                __func__, (long long) n_tokens, (long long) n_a, (long long) (n_tokens - n_a), n_w16);
        }
    }

    // the rest as the whole ubatch's graph builds it
    ggml_tensor * cur = ggml_concat(ctx0, x[0], x[1], 1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

void llama_model_qwen35::graph::build_layer_attn_halves(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             x[2],
        ggml_tensor *             pos[2],
        const int64_t             r0[2],
        const int64_t             nr[2],
        int *                     sections,
        int                       il,
        ggml_tensor *             out[2]) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    const auto & layer = model.layers[il];
    const auto * mctx_cur = inp->mctx;

    const int64_t n_embd_q = n_embd_head * 2 * n_head; // query + gate
    const int64_t n_embd_k = n_embd_head * n_head_kv;
    const int64_t n_embd_v = n_embd_head * n_head_kv;

    ggml_tensor * Q[2];
    ggml_tensor * gate[2];
    for (int h = 0; h < 2; ++h) {
        const int64_t n = nr[h];

        ggml_tensor * cur = build_norm(x[h], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        // the projections as build_qkv(..., reshape = false) makes them, for this half's rows
        ggml_tensor * Qcur_full;
        ggml_tensor * Kcur;
        ggml_tensor * Vcur;
        if (layer.wqkv) {
            ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
            cb(qkv, "wqkv", il);
            Qcur_full = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_q, n, qkv->nb[1], 0));
            Kcur      = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_k, n, qkv->nb[1], ggml_row_size(qkv->type, n_embd_q)));
            Vcur      = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_v, n, qkv->nb[1], ggml_row_size(qkv->type, n_embd_q + n_embd_k)));
        } else {
            Qcur_full = build_lora_mm(tbo_w(layer.wq), cur, layer.wq_s);
            Kcur      = build_lora_mm(tbo_w(layer.wk), cur, layer.wk_s);
            Vcur      = build_lora_mm(tbo_w(layer.wv), cur, layer.wv_s);
        }
        cb(Qcur_full, "Qcur_full", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
        cb(Qcur, "Qcur_reshaped", il);

        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        ggml_tensor * g = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
        gate[h] = ggml_cont_2d(ctx0, g, n_embd_head * n_head, n);
        cb(gate[h], "gate_reshaped", il);

        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n);

        Qcur = ggml_rope_multi(
                ctx0, Qcur, pos[h], nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        Kcur = ggml_rope_multi(
                ctx0, Kcur, pos[h], nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // as build_attn: q, v and k together, then this half's rows into the KV cache
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);
        {
            ggml_tensor * k_idxs = inp->get_k_idxs();
            ggml_tensor * v_idxs = inp->get_v_idxs();
            k_idxs = ggml_view_1d(ctx0, k_idxs, n, r0[h]*ggml_element_size(k_idxs));
            v_idxs = ggml_view_1d(ctx0, v_idxs, n, r0[h]*ggml_element_size(v_idxs));
            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, v_idxs, il));
        }
        Q[h] = Qcur;
    }

    // one attention over all rows, exactly the whole ubatch's call
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    ggml_tensor * q = ggml_concat(ctx0, Q[0], Q[1], 2);
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    GGML_ASSERT(!mctx_cur->is_whole()); // prompt batches are never over the whole cache
    ggml_tensor * attn = build_attn_mha(q, k, v, nullptr, inp->get_kq_mask(), nullptr, nullptr, 0, kq_scale, il);
    cb(attn, "kqv_out", il);
    cb(attn, "attn_pregate", il);
    GGML_ASSERT(ggml_is_contiguous(attn) && attn->ne[1] == n_tokens);

    for (int h = 0; h < 2; ++h) {
        ggml_tensor * cur = ggml_view_2d(ctx0, attn, attn->ne[0], nr[h], attn->nb[1], r0[h]*attn->nb[1]);

        ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate[h]);
        cb(gate_sigmoid, "gate_sigmoid", il);

        cur = ggml_mul(ctx0, cur, gate_sigmoid);
        cb(cur, "attn_gated", il);

        cur = build_lora_mm(tbo_w(layer.wo), cur, layer.wo_s);
        cb(cur, "attn_output", il);

        out[h] = cur;
        ggml_build_forward_expand(gf, cur);
    }
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear_half(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int64_t              n,
        bool                 first,
        ggml_tensor *&       conv_tail,
        ggml_tensor *&       state,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    const auto & layer_w  = model.layers[il];

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = 1;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = n;

    // input projections, as build_qkvz and build_layer_attn_linear order them (unmerged a/b, as under the split)
    ggml_tensor * qkv_mixed = build_lora_mm(tbo_w(layer_w.wqkv), cur, layer_w.wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(tbo_w(layer_w.wqkv_gate), cur, layer_w.wqkv_gate_s);
    cb(z, "z", il);

    static const bool gdn_adj = [] {
        const char * e = getenv("LLAMA_QPN_GDN_GROUP");
        return e == nullptr || atoi(e) != 0;
    }();
    if (gdn_adj) {
        ggml_build_forward_expand(gf, qkv_mixed);
        ggml_build_forward_expand(gf, z);
    }

    ggml_tensor * beta  = build_lora_mm(tbo_w(layer_w.ssm_beta), cur, layer_w.ssm_beta_s);
    ggml_tensor * alpha = build_lora_mm(tbo_w(layer_w.ssm_alpha), cur, layer_w.ssm_alpha_s);
    ggml_build_forward_expand(gf, beta);
    ggml_build_forward_expand(gf, alpha);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, layer_w.ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, layer_w.ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = layer_w.ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    // the conv input: the first half from the cache's conv states, the second from the first half's last columns
    ggml_tensor * conv_states;
    if (first) {
        conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        cb(conv_states, "conv_states", il);
        conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    } else {
        conv_states = conv_tail;
    }
    cb(conv_states, "conv_states_reshaped", il);

    ggml_tensor * qkv_t = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_t, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_t, 0);
    cb(conv_input, "conv_input", il);

    if (first) {
        conv_tail = ggml_cont(ctx0, ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs,
            conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, conv_input->ne[0] - (conv_kernel_size - 1))));
        cb(conv_tail, "conv_tail", il);
    } else {
        // the cache's conv states after the ubatch, as build_conv_state writes them
        const auto    kv_head   = mctx_cur->get_head();
        const auto    mem_size  = mctx_cur->get_size();
        const int64_t row_count = (conv_kernel_size - 1) * conv_channels;
        const size_t  row_size  = ggml_row_size(conv_states_all->type, row_count);
        const int64_t K         = (int64_t) cparams.n_rs_seq + 1;
        for (int64_t t = 1; t <= K; ++t) {
            const int64_t s_idx  = cparams.n_rs_seq == 0 ? conv_input->ne[0] - conv_states->ne[0]
                                                        : std::max<int64_t>(0, conv_input->ne[0] - conv_states->ne[0] - K + t);
            const int64_t s_slot = cparams.n_rs_seq == 0 ? 0 : K - t;

            ggml_tensor * conv_state_last = ggml_view_3d(ctx0, conv_input,
                conv_kernel_size - 1, conv_channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

            ggml_tensor * conv_state_update = ggml_view_2d(ctx0, conv_states_all,
                row_count, n_seqs, conv_states_all->nb[1],
                (s_slot * mem_size + kv_head) * row_size);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
        }
    }

    ggml_tensor * s;
    if (first) {
        s = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        s = ggml_reshape_4d(ctx0, s, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    } else {
        s = state;
    }
    cb(s, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_qkv_mix = ggml_silu(ctx0, conv_output_proper);
    cb(conv_qkv_mix, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));
    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output;
    if (first) {
        // the fused chunked op over this half, its final fp32 state kept for the second half (nothing to the cache)
        auto attn_out = build_delta_net_fused(q_conv, k_conv, v_conv, gate, beta, s, il);
        output = attn_out.first;
        state  = attn_out.second;
        cb(output, "attn_output", il);
        cb(state, "new_state", il);
    } else {
        output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, s, il);
    }

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * attn_out_norm = build_norm_gated(output, layer_w.ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(tbo_w(layer_w.ssm_out), final_output, layer_w.ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    // LLAMA_QPN_GDN_GROUP (0 turns it off): z is read only by the gated norm after the recurrence, so the graph's depth-first
    // order put its product far from qkv's, and the CUDA sibling planner could not run the two as one launch (an early z would write
    // buffers the nodes between them still read). Expanding qkv, z and (below) ab here puts the three products of one input side by side.
    static const bool gdn_adj = [] {
        const char * e = getenv("LLAMA_QPN_GDN_GROUP");
        return e == nullptr || atoi(e) != 0;
    }();
    if (gdn_adj) {
        ggml_build_forward_expand(gf, qkv_mixed);
        ggml_build_forward_expand(gf, z);
    }

    // With the a and b weights merged at load (see load_arch_tensors), up to 8 tokens run one product and
    // take alpha and beta as views of it. There the product is mul_mat_vec_q, where each row reduces alone whatever
    // the row count, so both halves are bit-identical to the two products; a larger batch keeps the two (MMQ's
    // stream-k splits K by the tile count). The gated_delta_net launch folds both views' gates: the
    // CONT keeps the unfolded sigmoid on a contiguous input, and the fold reads through it.
    const auto & layer_w = model.layers[il];
    ggml_tensor * ab_w = layer_w.ssm_alpha->view_src;
    const bool ab_merged = ab_w != nullptr && ab_w == layer_w.ssm_beta->view_src && layer_w.ssm_alpha->view_offs == 0 &&
                           layer_w.ssm_beta->view_offs == ggml_nbytes(layer_w.ssm_alpha) && ab_w->ne[1] == 2*num_v_heads &&
                           layer_w.ssm_alpha_s == nullptr && layer_w.ssm_beta_s == nullptr && loras->empty() &&
                           (ubatch.n_tokens <= 8 || (ab_w->flags & GGML_TENSOR_FLAG_BACKEND_LAYOUT) != 0);

    ggml_tensor * beta;
    ggml_tensor * alpha;
    if (ab_merged) {
        ggml_tensor * ab = ggml_mul_mat(ctx0, ab_w, cur);
        cb(ab, "ab", il);
        if (gdn_adj) {
            ggml_build_forward_expand(gf, ab);
        }
        const size_t es = ggml_element_size(ab);
        beta = ggml_view_4d(ctx0, ab, 1, num_v_heads, n_seq_tokens, n_seqs, es, ab->nb[1], ab->nb[1]*n_seq_tokens, num_v_heads*es);
        beta = ggml_cont(ctx0, beta);
        alpha = ggml_view_3d(ctx0, ab, num_v_heads, n_seq_tokens, n_seqs, ab->nb[1], ab->nb[1]*n_seq_tokens, 0);
    } else {
        // a merged a/b weight the backend repacked (GGML_TENSOR_FLAG_BACKEND_LAYOUT) is read only through ab
        GGML_ASSERT(ab_w == nullptr || !(ab_w->flags & GGML_TENSOR_FLAG_BACKEND_LAYOUT));
        beta = build_lora_mm(layer_w.ssm_beta, cur, layer_w.ssm_beta_s);
        // the alpha product right after beta's, so the CUDA backend runs the two in one launch (ggml_cuda_mul_mat_vec_q_group:
        // under --split-mode tensor each is a 24-row dp4a product); built after the sigmoid, alpha's output could take beta's freed memory
        alpha = build_lora_mm(layer_w.ssm_alpha, cur, layer_w.ssm_alpha_s);
        ggml_build_forward_expand(gf, beta);
        ggml_build_forward_expand(gf, alpha);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    if (!ab_merged) {
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");

    // one draft step: the token embedding and the hidden row in, the step's h_nextn out (the hidden state after the head's
    // norm, which the next step takes as its hidden row). shared by the one-step graph and the chained steps
    auto step = [&](ggml_tensor * tok_embd, ggml_tensor * h_embd, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn) -> ggml_tensor * {
        const int64_t n_tok = h_embd->ne[1];

        ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
        cb(h_norm, "mtp_hnorm", il);

        ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
        cb(e_norm, "mtp_enorm", il);

        ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
        cb(concat, "mtp_concat", il);

        ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
        cb(cur, "mtp_eh_proj", il);

        ggml_tensor * inpSA = cur;

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_attn_norm", il);

        auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
                n_embd_head * 2, n_head,
                n_embd_head,     n_head_kv,
                n_embd_head,     n_head_kv,
                il, false);
        cb(Qcur_full, "mtp_Qcur_full", il);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
                n_embd_head, n_head, n_tok,
                ggml_element_size(Qcur_full) * n_embd_head * 2,
                ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
                0);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);

        ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
                n_embd_head, n_head, n_tok,
                ggml_element_size(Qcur_full) * n_embd_head * 2,
                ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
                ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tok);
        cb(gate, "mtp_gate", il);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tok);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "mtp_Kcur_normed", il);

        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tok);
        cb(Vcur, "mtp_Vcur", il);

        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        const float kq_scale = hparams.f_attention_scale == 0.0f
                ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "mtp_attn_pregate", il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        cb(cur, "mtp_attn_out", il);
        if (cur->op == GGML_OP_MUL && cur->src[0]->op == GGML_OP_MUL_MAT) {
            ggml_set_name(cur->src[0], "mtp_attn_out_partial");
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "mtp_attn_residual", il);

        ggml_tensor * ffn_residual = cur;
        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_attn_post_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "mtp_ffn_out", il);
        if (cur->op == GGML_OP_MUL && cur->src[0]->op == GGML_OP_MUL_MAT) {
            ggml_set_name(cur->src[0], "mtp_ffn_out_partial");
        }

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "mtp_post_ffn", il);

        cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

        cb(cur, "h_nextn", -1);
        return cur;
    };

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");

    if (mtp_chain > 0) {
        build_chain(model, step);
        return;
    }

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * cur = step(tok_embd, h_embd, inp_pos, inp_attn);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    if (model.head_subset_w && head_w == model.output && head_s == nullptr) {
        // [TAG_DRAFT_VOCAB] the draft's logits over the subset rows only; build_sampling maps rows back to token ids
        cur = ggml_mul_mat(ctx0, model.head_subset_w, cur);
        res->t_logits_ids = model.head_subset_ids;
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

// the MTP draft's steps chained in one graph (llama_mtp_chain_set): step 0 takes the ubatch's token and hidden
// row, each later step the token the previous one picked on the device (GGML_OP_DRAFT_PICK, its embedding from the
// subset's rows) and that step's h_nextn. Each step's logits over the draft vocabulary subset take the device top-k the
// draft's backend sampler takes (the split top-k under --split-mode tensor), and its picks are the graph's outputs.
// Nothing crosses to the host between the steps
// LLAMA_MTP_CHAIN_PACK (default 1): the chain's inputs and its picks packed (llm_graph_input_mtp_chain::pos_all)
static bool mtp_chain_pack() {
    static const bool v = [] { const char * e = getenv("LLAMA_MTP_CHAIN_PACK"); return e == nullptr || atoi(e) != 0; }();
    return v;
}

void llama_model_qwen35::graph_mtp::build_chain(const llama_model & model,
        const std::function<ggml_tensor * (ggml_tensor *, ggml_tensor *, ggml_tensor *, llm_graph_input_attn_kv *)> & step) {
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(model.head_subset_w && model.head_subset_ids && model.head_subset_embd && "a chained MTP draft needs the draft vocabulary subset");
    GGML_ASSERT(!layer.nextn.shared_head_head && model.output_s == nullptr);

    auto * inp = build_inp_mtp_chain();

    const uint32_t n_steps = inp->n_steps;
    const uint32_t n_seq   = inp->n_seq;
    const int32_t  k       = (int32_t) mtp_chain_k;
    const int32_t  n_split = (int32_t) cparams.n_tensor_split;
    const int64_t  n_sub   = model.head_subset_w->ne[1];

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tok0);
    cb(tok_embd, "mtp_tok_embd", il);
    ggml_tensor * h_embd = inp->h0;

    for (uint32_t i = 0; i < n_steps; ++i) {
        ggml_tensor * h = step(tok_embd, h_embd, inp->pos[i], inp->attn[i].get());

        ggml_tensor * logits = ggml_mul_mat(ctx0, model.head_subset_w, h); // [n_sub, n_seq]
        cb(logits, "result_output", -1);

        // each sequence's candidates: under the split each device's top k of its slice, gathered by the AllReduce
        // (ggml_top_k_split, as the draft's backend sampler takes them), on one card the row's top k; the pick
        // takes the k highest of the gathered ones itself, and maps the subset rows to token ids (fewer kernels a step)
        ggml_tensor * c_a[3];
        ggml_tensor * c_b[3];
        ggml_tensor * c_c[3];
        for (uint32_t j = 0; j < n_seq; ++j) {
            ggml_tensor * row = ggml_view_1d(ctx0, logits, n_sub, j*logits->nb[1]);
            ggml_format_name(row, "mtp_chain_logits_%u_%u", i, j);
            if (n_split > 1) {
                c_a[j] = ggml_top_k_split(ctx0, row, k, n_split);
                ggml_set_name(c_a[j], "top_k_split");
                c_b[j] = nullptr;
            } else {
                c_a[j] = row;
                c_b[j] = ggml_top_k(ctx0, row, k);
                ggml_set_name(c_b[j], "top_k");
            }
            c_c[j] = model.head_subset_ids;
        }

        ggml_tensor * pick = ggml_draft_pick(ctx0, (int) n_seq, k, n_split > 1 ? 1 : 2, c_a, c_b, c_c, inp->prms[i]);
        ggml_format_name(pick, "mtp_chain_pick_%u", i);
        ggml_set_output(pick); // read back after the whole chain: its memory is not reused by a later step
        ggml_build_forward_expand(gf, pick);
        res->t_mtp_chain.push_back(pick);

        // (LLAMA_MTP_CHAIN_PACK) the steps' picks joined as they come, so they cross to the host in one copy and not
        // one a step (each a get and its wait: about 10 us apart under the split)
        if (mtp_chain_pack()) {
            ggml_tensor * p2 = ggml_reshape_2d(ctx0, pick, ggml_nelements(pick), 1);
            res->t_mtp_chain_all = i == 0 ? p2 : ggml_concat(ctx0, res->t_mtp_chain_all, p2, 1);
            if (i + 1 == n_steps) {
                ggml_set_name(res->t_mtp_chain_all, "mtp_chain_picks");
                ggml_set_output(res->t_mtp_chain_all);
                ggml_build_forward_expand(gf, res->t_mtp_chain_all);
            }
        }

        if (i + 1 < n_steps) {
            // the next step's token: its embedding row of the subset, and its hidden row: this step's h_nextn
            ggml_tensor * picked = ggml_view_1d(ctx0, pick, n_seq, 0);
            tok_embd = ggml_get_rows(ctx0, model.head_subset_embd, picked);
            cb(tok_embd, "mtp_tok_embd", il);
            h_embd = h;
        }
    }
}
