#include "models.h"
#include "qwen4exp-qsa-compact.h"

#include <cstdlib>
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>
#include <climits>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // must precede the per-layer arrays: n_layer() == n_layer_all - n_layer_nextn.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // the converter writes 0 (dense) for the MTP block, but the checkpoint's MTP layer is a
    // full_attention layer with its own indexer, and the reference runs every full-attention layer,
    // the MTP module's included, as QSA. give a block that ships indexer weights the trunk's ratio.
    // opt-in with LLAMA_MTP_QSA=1: it is faster from about 20K up (-3.4% per verify step at 64K) but
    // still 0.7-1.2% slower below 8K, the headline range. unset or 0 keeps the file's 0: a plain cache and dense attention.
    static const bool mtp_qsa = [] {
        const char * e = getenv("LLAMA_MTP_QSA");
        return e != nullptr && atoi(e) != 0;
    }();
    if (mtp_qsa && hparams.n_layer_nextn > 0) {
        uint32_t r_trunk = 0;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            r_trunk = std::max(r_trunk, hparams.dsv4_compress_ratios[il]);
        }
        for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
            const std::string k_proj = format("blk.%u.indexer.k_proj.weight", il);
            if (r_trunk > 0 && hparams.dsv4_compress_ratios[il] == 0 && ml.get_weight(k_proj.c_str()) != nullptr) {
                hparams.dsv4_compress_ratios[il] = r_trunk;
                LLAMA_LOG_INFO("%s: MTP block %u attends with QSA, compress ratio %u (LLAMA_MTP_QSA=1)\n",
                        __func__, il, r_trunk);
            }
        }
    }

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // a draft-only export declares the full block count but ships the MTP block alone.
    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    // the MTP head has its own in nextn.hc_head_*
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        // absent when mtp_use_dedicated_embeddings=false (qwen4exp); the head falls back to the trunk's.
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    // the trunk and output head (not the MTP block, il == n_layer) tag their projections for the
    // backend's gated-residual matrix-vector kernel. LLAMA_HC_GEMV=0 keeps the generic kernel.
    static const bool hc_gemv = [] {
        const char * e = getenv("LLAMA_HC_GEMV");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool hc_hint = hc_gemv && il < (int) hparams.n_layer();
    const auto tag = [hc_hint](ggml_tensor * t) {
        if (hc_hint && t->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_hint(t, GGML_HINT_HC_PROJ);
        }
        return t;
    };

    ggml_tensor * lo = tag(build_lora_mm(w_down, xn));
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = tag(build_lora_mm(w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = tag(build_lora_mm(w_inject, xn));
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs every token's row, so it defers the gather until after t_h_nextn.
        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // export res_hc itself, never a reshape view: a pure view gets no backend assignment to read back.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// [TAG_QSA_SEL_KEEP] the host side of the MTP draft's kept selection (llama_qsa_sel_keep), one per graph_mtp:
//   REUSE  a draft step: its tokens attend with their kept rows plus the cells added since (inputs below)
//   STORE  any other decode whose block selection fits: its rows replace the kept ones
//   FORGET the rest (a prompt ubatch, several streams): nothing is kept past it
// set_input records what the graph does, so the host side always describes the tensor
class llama_model_qwen4exp::llm_graph_input_qsa_keep : public llm_graph_input_i {
public:
    enum mode_t { FORGET, STORE, REUSE };

    llm_graph_input_qsa_keep(const llama_memory_hybrid_idx_context * mctx, mode_t mode, bool blk_sel, int64_t n_tokens, int64_t w) :
        mctx(mctx), mode(mode), blk_sel(blk_sel), n_tokens(n_tokens), w(w) {}
    virtual ~llm_graph_input_qsa_keep() = default;

    // blk_sel: the decode's own selection would be block selection, which is the only kind kept
    static mode_t mode_of(const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & ub, bool blk_sel) {
        const llama_qsa_sel_keep * keep = mctx->get_sel_keep();

        if (mctx->get_n_stream() != 1 || mctx->get_idx() == nullptr) {
            return FORGET;
        }
        if (keep->match(ub, nullptr, nullptr, nullptr)) {
            return REUSE;
        }
        if (blk_sel && keep->sel != nullptr && (int64_t) ub.n_tokens <= llama_qsa_sel_keep::n_rows) {
            return STORE;
        }

        return FORGET;
    }

    void set_input(const llama_ubatch * ubatch) override {
        llama_qsa_sel_keep * keep = mctx->get_sel_keep();

        if (mode != REUSE) {
            keep->forget();

            if (mode == STORE) {
                keep->w = w;

                for (uint32_t t = 0; t < ubatch->n_tokens; ++t) {
                    if (ubatch->n_seq_id[t] == 1) {
                        keep->rows.push_back({ ubatch->seq_id[t][0], ubatch->pos[t], (int32_t) t });
                    }
                }
            }

            return;
        }

        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);

        GGML_ASSERT(ggml_backend_buffer_is_host(rows->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(app->buffer));

        // the whole-view attention path reads no bias, so app_mask may not be in the graph
        const bool has_mask = app_mask->buffer != nullptr;
        GGML_ASSERT(!has_mask || ggml_backend_buffer_is_host(app_mask->buffer));

        const int64_t n_app = llama_qsa_sel_keep::n_app;

        int32_t * dst_rows = (int32_t *) rows->data;
        int32_t * dst_app  = (int32_t *) app->data;
        int32_t * dst_mask = has_mask ? (int32_t *) app_mask->data : nullptr;

        std::vector<int32_t> n_add(ubatch->n_tokens);

        // mode_of chose REUSE for this ubatch, and nothing changed the kept rows since
        GGML_ASSERT(keep->match(*ubatch, dst_rows, dst_app, n_add.data()));

        std::vector<int32_t> own(ubatch->n_tokens);

        for (uint32_t t = 0; t < ubatch->n_tokens; ++t) {
            own[t] = (int32_t) mctx->get_attn()->get_cell(0, t);

            // the added cells, the token's own, then its own again under INT_MIN as padding
            for (int64_t j = n_add[t]; j < n_app; ++j) {
                dst_app[t*n_app + j] = own[t];
            }
            for (int64_t j = 0; dst_mask && j < n_app; ++j) {
                dst_mask[t*n_app + j] = j <= n_add[t] ? 0 : INT_MIN;
            }
        }

        for (uint32_t t = 0; t < ubatch->n_tokens; ++t) {
            keep->adds.push_back({ ubatch->seq_id[t][0], ubatch->pos[t], own[t] });
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        if ((int64_t) params.ubatch.n_tokens != n_tokens || mode_of(mctx, params.ubatch, blk_sel) != mode) {
            return false;
        }

        return mode != REUSE || mctx->get_sel_keep()->w == w;
    }

    // REUSE only
    ggml_tensor * k_idxs   = nullptr;   // I64 [n_tokens], indexer cache rows of the step's keys
    ggml_tensor * rows     = nullptr;   // I32 [n_tokens], kept row of each token
    ggml_tensor * app      = nullptr;   // I32 [n_app, n_tokens], cells added after it, the token's own, padding
    ggml_tensor * app_mask = nullptr;   // I32 [n_app, n_tokens], 0 for those cells, INT_MIN for the padding

    const llama_memory_hybrid_idx_context * mctx;

    const mode_t  mode;
    const bool    blk_sel;
    const int64_t n_tokens;
    const int64_t w;          // width of the rows this graph writes (STORE) or reads (REUSE)
};

// LLM_GRAPH_TYPE_DECODER_MTP draft head for qwen4exp. The MTP block is a QSA layer with its own
// indexer, as in the reference: it selects over its own indexer cache with the trunk's machinery.
// Only with LLAMA_MTP_QSA=1; by default its ratio stays at the file's 0 and it attends densely over a plain cache.
// With LLAMA_MTP_QSA_REUSE (default 1) the draft steps reuse the selection of the last verified row, as the
// reference reuses its top-k across draft steps (see llm_graph_input_qsa_keep).
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // a nonzero ratio means create_memory gave this context the hybrid-idx wrapper (LLAMA_MTP_QSA)
    const bool mtp_qsa = hparams.dsv4_compress_ratios[il] > 0;

    llm_graph_input_attn_kv * inp_attn = nullptr;
    const llama_memory_hybrid_idx_context * mctx_hyb = nullptr;

    if (mtp_qsa) {
        auto * inp_mem = build_inp_mem_hybrid();

        // nothing consumes the recurrent half, so s_copy never enters the graph while set_input
        // would still read its buffer (as glm5next's MTP graph does)
        ggml_build_forward_expand(gf, inp_mem->get_recr()->s_copy);

        mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_mem->mctx);
        GGML_ASSERT(mctx_hyb->get_idx() != nullptr &&
                mctx_hyb->get_idx()->get_n_kv() == inp_mem->mctx->get_attn()->get_n_kv() &&
                "the MTP indexer cache must track its attention cache cell for cell");

        inp_attn = inp_mem->get_attn();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // per stream, not pooled: pooling before the projection discards the hyper-connection residual.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    ggml_tensor * top_k_bias = nullptr;
    ggml_tensor * top_k      = nullptr;

    // [TAG_QSA_SEL_KEEP] LLAMA_MTP_QSA_REUSE (default 1): a draft step attends with the selection the catch-up
    // computed for the last verified row, plus the cells written since, and runs no indexer query. every other
    // decode selects as before and keeps its selection. 0 selects in every decode.
    static const bool mtp_qsa_reuse = [] {
        const char * e = getenv("LLAMA_MTP_QSA_REUSE");
        return e == nullptr || atoi(e) != 0;
    }();

    llama_qsa_sel_keep * keep = mtp_qsa && mtp_qsa_reuse ? mctx_hyb->get_sel_keep() : nullptr;

    using keep_input = llm_graph_input_qsa_keep;

    if (keep && keep_input::mode_of(mctx_hyb, ubatch, false) == keep_input::REUSE) {
        const int64_t w_max = keep->w_max;
        const int64_t w     = keep->w;
        const int64_t n_app = llama_qsa_sel_keep::n_app;

        auto inp = std::make_unique<keep_input>(mctx_hyb, keep_input::REUSE, false, n_tokens, w);

        inp->k_idxs   = mctx_hyb->get_idx()->build_input_k_idxs(ctx0, ubatch);
        inp->rows     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        inp->app      = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_app, n_tokens);
        inp->app_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_app, n_tokens);

        ggml_set_input(inp->rows);
        ggml_set_input(inp->app);
        ggml_set_input(inp->app_mask);

        // the step's indexer key still goes into the cache, so every cell holds its token's key in both caches
        ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
        k_raw = ggml_reshape_3d(ctx0, k_raw, hparams.indexer_head_size, 1, n_tokens);
        ggml_build_forward_expand(gf, mctx_hyb->get_idx()->cpy_k(ctx0, k_raw, inp->k_idxs, il));

        // each token's kept row, both planes, then the added cells after it: [w + n_app, n_tokens]
        ggml_tensor * kept  = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, keep->sel, 2*w_max, llama_qsa_sel_keep::n_rows), inp->rows);
        ggml_tensor * cells = ggml_view_2d(ctx0, kept, w, n_tokens, kept->nb[1], 0);
        ggml_tensor * mask  = ggml_view_2d(ctx0, kept, w, n_tokens, kept->nb[1], w_max*ggml_element_size(kept));

        top_k = ggml_concat(ctx0, cells, inp->app, 0);
        top_k = ggml_reshape_4d(ctx0, top_k, w + n_app, n_tokens, 1, 1);
        cb(top_k, "indexer_top_k_kept", il);

        top_k_bias = ggml_cast(ctx0, ggml_concat(ctx0, mask, inp->app_mask, 0), GGML_TYPE_F32);
        top_k_bias = ggml_reshape_4d(ctx0, top_k_bias, w + n_app, n_tokens, 1, 1);
        cb(top_k_bias, "indexer_top_k_bias_kept", il);

        res->add_input(std::move(inp));
    } else if (mtp_qsa) {
        // the indexer reads the same block input as q/k/v, as in build_layer_attn
        top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp_attn->get_kq_mask(), sections, &top_k_bias, il);

        if (keep) {
            // block selection's result is [width, n_tokens, n_stream, 2]; its bias is the second plane
            const bool blk_sel = top_k_bias != nullptr && top_k->view_src != nullptr;

            if (blk_sel && keep->sel == nullptr) {
                const ggml_tensor * k_idx = mctx_hyb->get_idx()->get_k(ctx0, il);
                const ggml_tensor * k_src = k_idx->view_src ? k_idx->view_src : k_idx;

                const int64_t r = hparams.dsv4_compress_ratios[il];

                if (k_src->buffer == nullptr || !keep->alloc(ggml_backend_buffer_get_type(k_src->buffer), hparams.indexer_top_k + r - 1)) {
                    LLAMA_LOG_WARN("%s: no kept-selection buffer, the MTP draft steps select anew\n", __func__);
                }
            }

            const auto mode = keep_input::mode_of(mctx_hyb, ubatch, blk_sel);
            const int64_t w = top_k->ne[0];

            if (mode == keep_input::STORE) {
                GGML_ASSERT(w <= keep->w_max);

                ggml_tensor * sel = top_k->view_src;
                GGML_ASSERT(sel->type == GGML_TYPE_I32 && sel->ne[0] == w && sel->ne[1] == n_tokens && sel->ne[2] == 1 && sel->ne[3] == 2);

                // rows 0..n_tokens-1 of the kept tensor, cells then mask
                ggml_tensor * dst = ggml_view_4d(ctx0, keep->sel, w, n_tokens, 1, 2,
                        keep->sel->nb[2], keep->sel->nb[2]*n_tokens, keep->sel->nb[1], 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, sel, dst));
            }

            res->add_input(std::make_unique<keep_input>(mctx_hyb, mode, blk_sel, n_tokens, w));
        }
    }

    // [TAG_MTP_CATCHUP_NOREAD] a decode with no outputs (the catch-up after every verify, and the draft's prompt
    // ubatches) leaves nothing behind but its cache writes: no logits or h_nextn row is extracted, and every row
    // after the attention is dropped by inp_out_ids. so build only K and V and their writes (and, with QSA, the
    // indexer above, unchanged), and skip Q, the gate, the attention and wo. LLAMA_MTP_CATCHUP_NOREAD=0 builds them.
    static const bool mtp_catchup_noread = [] {
        const char * e = getenv("LLAMA_MTP_CATCHUP_NOREAD");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool noread = mtp_catchup_noread && n_outputs == 0;

    ggml_tensor * attn_in = cur;

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur = nullptr;
    ggml_tensor * gate = nullptr;

    if (!noread) {
        ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur_full, "mtp_Qcur_full", il);

        Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);

        gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
        cb(gate, "mtp_gate", il);
    }

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    if (Qcur) {
        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "mtp_Qcur", il);
    }
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Kcur, "mtp_Kcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (noread) {
        // the selection stays in the graph even where nothing reads it (a prompt ubatch), as with the attention
        if (top_k) {
            ggml_build_forward_expand(gf, top_k);
        }

        // the store half of build_attn / build_attn_qsa, in the same order
        if (inp_attn->self_k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
        }
        if (inp_attn->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
        }

        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);

        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));

        // the attention output would be n_embd wide like its input; inp_out_ids keeps 0 rows of either
        cur = attn_in;
    } else {
        if (top_k) {
            cur = build_attn_qsa(inp_attn, Qcur, Kcur, Vcur, top_k, top_k_bias, kq_scale, il);
        } else {
            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }
        cb(cur, "mtp_attn_pregate", il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        cb(cur, "mtp_attn_gated", il);

        cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        cb(cur, "mtp_attn_out", il);
    }

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // the next draft step re-enters here, so export the wide stream before it is collapsed.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // no res->t_embd: it is n_embd wide, but the context sizes that buffer by n_embd_out.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN4EXP MTP: missing LM head (nextn.shared_head_head or model.output)");

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

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
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

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// LLAMA_QSA_PACK=0 keeps the F16 KQ mask of the sparse prompt path and the F32 block selection bias. on by
// default, both go to one bit per entry: the block bias as I32 words (ggml_qsa_select) and the mask as the visibility
// of llm_graph_input_qsa_vis (ggml_qsa_vis_rows), so no prompt graph reads the cache-sized F16 mask
static bool llama_qsa_pack() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_QSA_PACK");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// the KQ mask of the sparse prompt path as one bit per cell, I32 [(n_kv + 31)/32, n_tokens] (one stream)
class llm_graph_input_qsa_vis : public llm_graph_input_i {
public:
    llm_graph_input_qsa_vis(const llama_kv_cache_context * mctx, bool causal_attn, int64_t n_kv) :
        mctx(mctx), causal_attn(causal_attn), n_kv(n_kv) {}
    virtual ~llm_graph_input_qsa_vis() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_kq_vis(vis, ubatch, causal_attn);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();

        bool res = true;

        res &= (int64_t) mctx->get_n_kv() == n_kv;
        res &= vis->ne[1] == params.ubatch.n_tokens;
        res &= params.cparams.causal_attn == causal_attn;

        return res;
    }

    ggml_tensor * vis = nullptr;

    const llama_kv_cache_context * mctx;
    const bool    causal_attn;
    const int64_t n_kv;
};

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias, int64_t n_kv) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias), n_kv(n_kv) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, tail, ubatch, ratio, blk_bias);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= this->n_kv       == n_kv;
        res &= blk_cells->ne[1] == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0]      == (bias->type == GGML_TYPE_I32 ? (n_blocks + 31)/32 : (blk_bias ? n_blocks : n_kv));
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;
        res &= tail == nullptr  || tail->ne[1] == params.ubatch.n_tokens/n_stream;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream], cell ranking only
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream], or packed (block selection,
                                         // LLAMA_QSA_PACK) I32 [(n_blocks + 31)/32, n_tokens/n_stream, n_stream]
    ggml_tensor * tail      = nullptr;   // I32 [max(ratio - 1, 1), n_tokens/n_stream, n_stream], block selection only

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    const int64_t n_kv;
};

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        ggml_tensor **                          top_k_bias,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // with the per-block bias, rank whole blocks as the reference does: indexer_top_k/r blocks plus
    // the incomplete tail, in ascending position, with nothing picked by a tie-break the hardware
    // chooses (ggml_qsa_select). LLAMA_QSA_BLOCK_TOPK=0 ranks every cell by its block's score instead.
    static const bool qsa_block_topk = [] {
        const char * e = getenv("LLAMA_QSA_BLOCK_TOPK");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool blk_sel = qsa_block_topk && blk_bias;

    *top_k_bias = nullptr;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, n_kv);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        qsa->bias      = blk_sel && llama_qsa_pack() ?
            ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, (n_blocks + 31)/32, n_tps, n_stream) :
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);

        if (blk_sel) {
            qsa->tail = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, std::max<int64_t>(r - 1, 1), n_tps, n_stream);
            ggml_set_input(qsa->tail);
        } else {
            qsa->cell_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
            ggml_set_input(qsa->cell_blk);
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                        members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, "indexer_k_pooled", il);

    // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks*n_stream, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
    pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // the score of query rows t0 .. t0+nt, summed over the heads: [n_blocks, nt, n_stream]. the whole chunk takes q as
    // today; a tile (one stream) reads its rows of q, which are contiguous
    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    const auto score_rows = [&](int64_t t0, int64_t nt) {
        ggml_tensor * q_rows = nt == n_tps ?
            ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream) :
            ggml_view_3d(ctx0, q, idx_dim, n_idx_h*nt, 1, q->nb[1], q->nb[1]*n_idx_h*nt, t0*q->nb[2]);

        ggml_tensor * s = ggml_mul_mat(ctx0, pooled, q_rows);
        s = ggml_reshape_4d(ctx0, s, n_blocks, n_idx_h, nt, n_stream);
        s = ggml_relu(ctx0, s);

        // the heads sit side by side on ne[1] and there are only a few of them. keep this chain node for node:
        // LLAMA_FOLD_IDX_SUM matches it in each tile
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, s, n_blocks, nt, n_stream,
                    s->nb[2], s->nb[3], h*s->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }

        cb(summed, "indexer_score", il);
        return summed;
    };

    // block selection scores the chunk in tiles of LLAMA_QSA_IDX_TILE queries (512; 0 scores it whole), so the
    // F32 score [n_blocks, n_idx_h, n_tps] and its head sum are a tile's size, not the chunk's. each row's selection
    // reads only its own score, bias and tail rows, so the result is the same; the graph orders each tile's chain
    // whole before the next, and the allocator reuses the slabs. decode, verify and draft chunks never tile
    static const int64_t qsa_idx_tile = [] {
        const char * e = getenv("LLAMA_QSA_IDX_TILE");
        return e == nullptr ? (int64_t) 512 : (int64_t) std::max(atoll(e), 0LL);
    }();
    const bool idx_tile = blk_sel && n_stream == 1 && qsa_idx_tile > 0 && n_tps > qsa_idx_tile;

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);
    // [TAG_SPEC_PIPE_LOOSE] the unclamped width, for llama_pipe_rows_independent's view forecast
    res->qsa.sel = std::max<int64_t>(res->qsa.sel, (int64_t) hparams.indexer_top_k + r - 1);

    if (blk_sel) {
        // the bias leaves each query the blocks it sees whole, below its tail; the tail cells come
        // from set_input. unused slots repeat a selected cell, and plane 1 of the result masks them.
        ggml_tensor * sel = nullptr;
        if (!idx_tile) {
            sel = ggml_qsa_select(ctx0, score_rows(0, n_tps), inp->bias, inp->blk_cells, inp->tail,
                    (int) (hparams.indexer_top_k/r), (int) width);
        } else {
            // not in the memory-fit probe (no_alloc), whose logger drops it: the real load prints it
            static bool logged = false;
            if (!logged && !hparams.no_alloc) {
                logged = true;
                LLAMA_LOG_WARN("QSA indexer tiles (LLAMA_QSA_IDX_TILE): %lld queries, n_tps %lld in %lld tiles\n",
                        (long long) qsa_idx_tile, (long long) n_tps, (long long) ((n_tps + qsa_idx_tile - 1)/qsa_idx_tile));
            }

            // the rows t0 .. t0+nt of the bias and the tail, in their own type and width
            for (int64_t t0 = 0; t0 < n_tps; t0 += qsa_idx_tile) {
                const int64_t nt = std::min(qsa_idx_tile, n_tps - t0);

                ggml_tensor * s      = score_rows(t0, nt);
                ggml_tensor * bias_t = ggml_view_3d(ctx0, inp->bias, inp->bias->ne[0], nt, 1,
                        inp->bias->nb[1], inp->bias->nb[1]*nt, t0*inp->bias->nb[1]);
                ggml_tensor * tail_t = ggml_view_3d(ctx0, inp->tail, inp->tail->ne[0], nt, 1,
                        inp->tail->nb[1], inp->tail->nb[1]*nt, t0*inp->tail->nb[1]);

                ggml_tensor * sel_t = ggml_qsa_select(ctx0, s, bias_t, inp->blk_cells, tail_t,
                        (int) (hparams.indexer_top_k/r), (int) width);
                sel = sel ? ggml_concat(ctx0, sel, sel_t, 1) : sel_t;
            }
        }
        cb(sel, "indexer_select", il);

        // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
        ggml_tensor * top_k = ggml_view_4d(ctx0, sel, width, n_tps, 1, n_stream, sel->nb[1], sel->nb[2], sel->nb[2], 0);
        cb(top_k, "indexer_top_k", il);

        *top_k_bias = ggml_cast(ctx0,
                ggml_view_4d(ctx0, sel, width, n_tps, 1, n_stream, sel->nb[1], sel->nb[2], sel->nb[2], sel->nb[3]),
                GGML_TYPE_F32);
        cb(*top_k_bias, "indexer_top_k_bias", il);

        return top_k;
    }

    ggml_tensor * score = score_rows(0, n_tps);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, inp->bias);
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        ggml_tensor *             top_k_bias,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // decode and small verification batches attend over the gathered top_k rows instead of the
    // whole cache view (see qwen4exp-qsa-compact.h). prefill, selections covering the whole view
    // and anything the compact form cannot express keep the original path below.
    // LLAMA_QSA_COMPACT=0 forces the original path.
    static const bool qsa_compact = [] {
        const char * e = getenv("LLAMA_QSA_COMPACT");
        return e == nullptr || atoi(e) != 0;
    }();
    // LLAMA_QSA_UNION=0 gathers every query's selection separately in verification batches
    // instead of the union once (see qwen4exp-qsa-compact.h).
    static const bool qsa_union = [] {
        const char * e = getenv("LLAMA_QSA_UNION");
        return e == nullptr || atoi(e) != 0;
    }();
    // [TAG_SPEC_PIPE_LOOSE] the record llama_pipe_rows_independent reads: the union is the one place a
    // verify row reads the other tokens of its batch
    // a selection clamped to a narrow view widens with it up to sel (build_qsa_top_k); an unclamped one stays
    if (top_k) {
        auto & rec = res->qsa;
        const int64_t wp = llama_qsa_compact_width(top_k->ne[0] < k->ne[2] ? top_k->ne[0] : std::max(top_k->ne[0], rec.sel), 256);
        rec.n_kv      = rec.has ? std::max(rec.n_kv, k->ne[2]) : k->ne[2];
        rec.wp        = rec.has ? std::min(rec.wp, wp) : wp;
        rec.union_env = rec.union_env || (qsa_compact && cparams.flash_attn && qsa_union);
        rec.has       = true;
    }
    if (qsa_compact && cparams.flash_attn &&
            llama_qsa_compact_applies(q_cur, k, v, kq_mask, top_k, /*max_tokens*/ 8, /*pad_to*/ 256)) {
        // the cache's device decides whether the gathers may write f16 directly
        const ggml_tensor *         k_src = k->view_src ? k->view_src : k;
        ggml_backend_dev_t          k_dev = k_src->buffer ?
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(k_src->buffer)) : nullptr;

        // the union only for single-sequence batches: after two concurrent requests shared one union,
        // every later request decoded garbage; the per-query form is correct there
        const bool use_union = qsa_union && ubatch.n_seqs_unq == 1 && llama_qsa_union_applies(k, top_k, k_dev);
        res->qsa.union_used = res->qsa.union_used || use_union;
        {
            ggml_tensor * fa  = nullptr;
            ggml_tensor * cur = (use_union ? llama_qsa_union_attn : llama_qsa_compact_attn)(ctx0, q_cur, k, v, kq_mask, top_k, kq_scale,
                    hparams.f_max_alibi_bias, hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f,
                    /*pad_to*/ 256, k_dev, &fa, top_k_bias);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, fa, il});
            cb(cur, "kqv_out", il);

            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }

            return cur;
        }
    }

    // prompt batches (PentaCoxian's block-sparse prompt attention): one FLASH_ATTN_EXT node
    // carrying each query's selected cells in src[5] (ggml-cuda/qsa-attn.cu) reads them straight from
    // the cache, instead of the cache-sized mask and dense attention over every cell below. Same set
    // of cells with the same mask entries (repeats dropped by top_k_bias), so the softmax is the same.
    // LLAMA_QSA_SPARSE_PREFILL=0 keeps the dense form.
    static const bool qsa_sparse_prefill = [] {
        const char * e = getenv("LLAMA_QSA_SPARSE_PREFILL");
        return e == nullptr || atoi(e) != 0;
    }();
    if (qsa_sparse_prefill && cparams.flash_attn && n_tokens > 8 && top_k_bias != nullptr &&
            hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap) {
        const int64_t n_kv   = k->ne[2];
        const int64_t width  = top_k->ne[0];
        const int64_t n_tps  = top_k->ne[1];
        const int64_t n_hkv  = k->ne[1];
        const int64_t gqa    = n_hkv > 0 && q_cur->ne[1] % n_hkv == 0 ? q_cur->ne[1]/n_hkv : 0;
        const bool sparse_ok =
            width < n_kv && k->ne[3] == 1 && v->ne[3] == 1 && top_k->ne[3] == 1 && n_tps == n_tokens &&
            top_k->type == GGML_TYPE_I32 && ggml_is_contiguous(top_k) &&
            top_k_bias->type == GGML_TYPE_F32 && ggml_is_contiguous(top_k_bias) && top_k_bias->ne[0] == width &&
            kq_mask->type == GGML_TYPE_F16 && kq_mask->ne[0] == n_kv && kq_mask->ne[1] >= n_tps &&
            q_cur->type == GGML_TYPE_F32 && q_cur->ne[0] == 256 && q_cur->ne[2] == n_tokens &&
            k->type == v->type && (k->type == GGML_TYPE_BF16 || k->type == GGML_TYPE_F16) &&
            k->ne[0] == 256 && v->ne[0] == 256 && v->ne[1] == n_hkv && v->ne[2] == n_kv &&
            k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) && v->nb[1] <= v->nb[2] &&
            (gqa == 12 || gqa == 8 || gqa == 16 || gqa == 2 || gqa == 1);
        if (sparse_ok) {
            ggml_tensor * q = ggml_is_contiguous(q_cur) ? q_cur : ggml_cont(ctx0, q_cur);
            q = ggml_reshape_3d(ctx0, q, q->ne[0], q->ne[1], n_tokens);

            // mask entries of the selected cells, as llama_qsa_compact_attn gathers them
            // one view per graph (qsa_mask_rows): a view per layer is a separate 128 MiB split input per card at 131K
            // when packed, qsa_mask_rows holds the graph's one visibility input instead, and the entries come from its bits
            ggml_tensor *& m_rows = qsa_mask_rows[kq_mask];
            ggml_tensor * m = nullptr;
            if (llama_qsa_pack() && !hparams.use_alibi && (int64_t) mctx_cur->get_n_kv() == n_kv) {
                if (m_rows == nullptr) {
                    auto inp_vis = std::make_unique<llm_graph_input_qsa_vis>(mctx_cur, cparams.causal_attn, n_kv);
                    inp_vis->vis = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, (n_kv + 31)/32, n_tps);
                    ggml_set_input(inp_vis->vis);
                    ggml_set_name(inp_vis->vis, "qsa_vis");
                    m_rows = inp_vis->vis;
                    res->add_input(std::move(inp_vis));

                    // the engagement line, once: the visibility and the packed block bias of the first such graph
                    // not in the memory-fit probe (no_alloc), whose logger drops it: the real load prints it
                    static bool logged = false;
                    if (!logged && !hparams.no_alloc) {
                        logged = true;
                        const ggml_tensor * b = nullptr;
                        for (const auto & it : qsa_inps) {
                            if (it.second->bias->type == GGML_TYPE_I32) {
                                b = it.second->bias;
                                break;
                            }
                        }
                        LLAMA_LOG_WARN("QSA packed visibility (LLAMA_QSA_PACK): on, mask visibility I32 [%lld, %lld], block bias %s [%lld, %lld, %lld]\n",
                                (long long) m_rows->ne[0], (long long) m_rows->ne[1], b ? "I32" : "none",
                                (long long) (b ? b->ne[0] : 0), (long long) (b ? b->ne[1] : 0), (long long) (b ? b->ne[2] : 0));
                    }
                }
                GGML_ASSERT(m_rows->type == GGML_TYPE_I32);
                m = ggml_qsa_vis_rows(ctx0, m_rows, ggml_reshape_2d(ctx0, top_k, width, n_tps),
                        ggml_reshape_2d(ctx0, top_k_bias, width, n_tps));
            } else {
                if (m_rows == nullptr) {
                    m_rows = ggml_view_4d(ctx0, kq_mask, 1, n_kv, n_tps, 1, kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[3], 0);
                }
                GGML_ASSERT(m_rows->type == kq_mask->type);
                m = ggml_get_rows(ctx0, m_rows, ggml_reshape_3d(ctx0, top_k, width, n_tps, 1));
                m = ggml_reshape_2d(ctx0, m, width, n_tps);
                m = ggml_add(ctx0, m, ggml_reshape_2d(ctx0, top_k_bias, width, n_tps));
            }
            cb(m, "qsa_sparse_mask", il);

            ggml_tensor * ids = ggml_reshape_2d(ctx0, top_k, width, n_tps);
            ggml_tensor * k3  = ggml_view_3d(ctx0, k, k->ne[0], k->ne[1], k->ne[2], k->nb[1], k->nb[2], 0);
            ggml_tensor * v3  = ggml_view_3d(ctx0, v, v->ne[0], v->ne[1], v->ne[2], v->nb[1], v->nb[2], 0);

            ggml_tensor * out = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 256, q->ne[1], n_tokens);
            // FLASH_ATTN_EXT params: scale, max_bias, softcap, prec; [6] the group size the kernel expects
            const float   fparams[3] = { kq_scale, 0.0f, 0.0f };
            const int32_t prec       = GGML_PREC_F32;
            const int32_t group      = 4;
            memcpy((char *) out->op_params + 0*sizeof(int32_t), fparams, sizeof(fparams));
            memcpy((char *) out->op_params + 3*sizeof(int32_t), &prec, sizeof(prec));
            memcpy((char *) out->op_params + 6*sizeof(int32_t), &group, sizeof(group));
            out->op     = GGML_OP_FLASH_ATTN_EXT;
            out->src[0] = q;
            out->src[1] = k3;
            out->src[2] = v3;
            out->src[3] = m;
            out->src[5] = ids;
            cb(out, "qsa_sparse_attn", il);

            ggml_tensor * cur = ggml_reshape_2d(ctx0, out, 256*q->ne[1], n_tokens);
            cb(cur, "kqv_out", il);

            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }

            return cur;
        }
    }

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;

    // TODO: enable sparse attention when we are ready
    // ref: https://github.com/ggml-org/llama.cpp/pull/27970
    //ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k_bias = nullptr;
    ggml_tensor * top_k      = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, &top_k_bias, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

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

    // Apply IMRoPE
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

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, top_k_bias, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
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

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

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

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    // the table stays host side and is read by 16 gathers per token, no two on the same page.
    // queued here they are in flight before the graph runs
    pmodel.prefetch_rows(pmodel.per_layer_tok_embd, idx.data(), idx.size());

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
