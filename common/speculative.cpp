#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "mtp-carry.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>

#include "ggml-rtimer.h"

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    // (optional) the prompt about to be prefilled for seq_id ends at position n_end, and the server may take a
    // checkpoint at each position in marks; see common_speculative_prefill_plan
    virtual void prefill_plan(llama_seq_id /*seq_id*/, llama_pos /*n_end*/, const std::vector<llama_pos> & /*marks*/) {}

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}
    // (optional) a lineage boundary for seq_id: drop the state carried for it
    virtual void reset_state(llama_seq_id /*seq_id*/) {}
    // (optional) whether the state carried for seq_id lets it continue at position pos_next
    virtual bool state_valid(llama_seq_id /*seq_id*/, llama_pos /*pos_next*/) const { return true; }

    // [TAG_SPEC_PIPELINE] (optional) pipelined drafting, see common_speculative_pipe_* in speculative.h
    virtual bool pipe_enable() { return false; }
    virtual const std::vector<float> * pipe_probs(llama_seq_id /*seq_id*/) const { return nullptr; }
    virtual bool pipe_process(llama_seq_id /*seq_id*/, const llama_token * /*toks*/, int32_t /*n*/, llama_pos /*pos0*/, llama_token /*extra*/) { return false; }
    virtual bool pipe_chain(llama_seq_id /*seq_id*/, const llama_token * /*known*/, int32_t /*n_known*/, llama_pos /*pos*/, int32_t /*n_new*/, llama_tokens & /*out*/, common_sampler * /*coupled*/,
            std::mt19937 * /*q_rng*/, std::vector<common_rejection_q> * /*q*/) { return false; }
    virtual bool pipe_rebase(llama_seq_id /*seq_id*/) { return false; }
};

// [TAG_SPEC_REJECTION] the draft token sampled from the draft's distribution q (its candidates cur_p through the
// copy of the target's sampler, LLAMA_SPEC_REJECTION_TEMP), with a draw from the target's sampler, and accepted
// into the copy; q is appended to dp.q. where the copy cannot follow the target, the forced token or the argmax,
// with an empty row. p: the draft's probability of the token (q(x); 1 for a forced token). cur_p->data[0] is the
// argmax. [TAG_SPEC_REJECTION_PIPE] with gen, the draw comes from gen instead. shared by MTP and
// DFlash2's sampled draft
static llama_token common_speculative_rejection_pick(common_sampler * copy, common_sampler * smpl_tgt, std::mt19937 * gen,
        std::vector<common_rejection_q> & q, const llama_token_data_array * cur_p, float & p) {
    GGML_RT_SCOPE("spec.rejection_pick");
    common_rejection_q row;
    llama_token forced = LLAMA_TOKEN_NULL;
    llama_token id = common_sampler_rejection_draft(copy, smpl_tgt, cur_p, common_speculative_rejection_temp(), row, &forced, gen);
    if (id == LLAMA_TOKEN_NULL) {
        id = forced != LLAMA_TOKEN_NULL ? forced : cur_p->data[0].id;
        p  = forced != LLAMA_TOKEN_NULL ? 1.0f : cur_p->data[0].p;
    } else {
        p = 0.0f;
        for (const auto & e : row) {
            if (e.id == id) {
                p = e.p;
                break;
            }
        }
    }
    q.push_back(std::move(row));
    common_sampler_accept_draft(copy, id); // the argmax fallback need not fit a triggered grammar
    return id;
}

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// [TAG_DRAFT_VOCAB] (MTP, also DFlash2) LLAMA_SPEC_DRAFT_VOCAB=<K>: the draft computes its logits over K tokens only: the vocabulary's real
// special tokens (control and user-defined, e.g. <|im_end|>, <think>, <tool_call>; never the unused padding ids), then
// the first distinct ids of the ranking in LLAMA_SPEC_DRAFT_VOCAB_FILE (whitespace-separated token ids, most frequent
// first). There is no built-in size, but K unset takes 98304, -1 takes every id of the file, 0 turns the subset off,
// and a file with fewer ids than K gives what it has. K is at least the draft's candidate count top_k, and is rounded
// up to a multiple of 4 with the next ranked ids (down, if the file runs out), because the 1-column head product takes
// 4 rows per block only then (mmvq.cu, LLAMA_MMVQ_DENSE1) and otherwise falls back to a slower kernel. A draft token
// changes only where the full head's argmax lies outside the subset; the target verifies every draft, so the output
// does not depend on it where the verify is batch-independent
static bool common_speculative_draft_vocab_ids(const llama_model * model, const int32_t top_k, std::vector<llama_token> & ids_out, std::string & what, const bool arch_default = false) {
    const char * ek = getenv("LLAMA_SPEC_DRAFT_VOCAB");
    const char * ef = getenv("LLAMA_SPEC_DRAFT_VOCAB_FILE");
    // K unset takes 98304 (the deployed launch's value; -1 still takes every id of the file). The file path stays
    // a per-launch choice (a model file): with neither variable set the subset stays off
    int32_t k = 98304;
    // with arch_default (the MTP draft; not DFlash2) and no LLAMA_SPEC_DRAFT_VOCAB_FILE, the ranking
    // file shipped in models/ for the draft's architecture: a qwen4exp draft (Flash-Next) takes its file with K 65536,
    // a qwen35 draft (the 27B) its file with K 98304. Any other architecture runs with no subset
    // and no warning unless LLAMA_SPEC_DRAFT_VOCAB was set
    std::string ef_default;
#ifdef LLAMA_SPEC_DRAFT_VOCAB_DIR
    if (arch_default && ef == nullptr) {
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (strcmp(arch, "qwen4exp") == 0) {
            k = 65536;
            ef_default = std::string(LLAMA_SPEC_DRAFT_VOCAB_DIR) + "/draft-vocab-qwen3.8-flash-next.txt";
        } else if (strcmp(arch, "qwen35") == 0) {
            ef_default = std::string(LLAMA_SPEC_DRAFT_VOCAB_DIR) + "/draft-vocab-qwen3.8-27b.txt";
        }
        ef = ef_default.empty() ? nullptr : ef_default.c_str();
    }
#endif
    if (ek != nullptr && *ek != '\0') {
        char * end = nullptr;
        errno = 0;
        const long v = strtol(ek, &end, 10);
        if (*end != '\0' || errno != 0 || v < -1 || v > INT32_MAX) {
            SPC_WRN("LLAMA_SPEC_DRAFT_VOCAB=%s is not an integer of at least -1; draft vocabulary subset off\n", ek);
            return false;
        }
        k = (int32_t) v;
    }
    if (k == 0 || (ek == nullptr && ef == nullptr)) {
        return false;
    }
    FILE * f = ef ? fopen(ef, "r") : nullptr;
    if (f == nullptr) {
        SPC_WRN("LLAMA_SPEC_DRAFT_VOCAB=%d needs a readable LLAMA_SPEC_DRAFT_VOCAB_FILE (%s); draft vocabulary subset off\n", k, ef ? ef : "unset");
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // the ranking, checked whole: every word must be a token id, and each id counts once
    std::vector<llama_token> ranked;
    std::vector<uint8_t> in_file(n_vocab, 0);
    size_t n_word = 0, n_bad = 0, n_range = 0, n_dup = 0;
    char word[64];
    while (fscanf(f, "%63s", word) == 1) {
        ++n_word;
        char * end = nullptr;
        errno = 0;
        const long v = strtol(word, &end, 10);
        if (end == word || *end != '\0' || errno != 0) {
            if (n_bad++ == 0) {
                SPC_WRN("%s: word %zu, '%s', is not a token id; skipped\n", ef, n_word, word);
            }
        } else if (v < 0 || v >= n_vocab) {
            if (n_range++ == 0) {
                SPC_WRN("%s: word %zu, %ld, is outside the vocabulary [0, %d); skipped\n", ef, n_word, v, n_vocab);
            }
        } else if (in_file[v]) {
            if (n_dup++ == 0) {
                SPC_WRN("%s: word %zu, %ld, repeats an earlier id; its first place counts\n", ef, n_word, v);
            }
        } else {
            in_file[v] = 1;
            ranked.push_back((llama_token) v);
        }
    }
    fclose(f);
    if (n_bad + n_range + n_dup > 0) {
        SPC_WRN("%s: %zu words, %zu not token ids, %zu out of range, %zu repeated; %zu distinct ids used\n", ef, n_word, n_bad, n_range, n_dup, ranked.size());
    }

    // the real specials first, so no K drops them
    std::vector<llama_token> ids;
    std::vector<uint8_t> seen(n_vocab, 0);
    for (llama_token t = 0; t < n_vocab; ++t) {
        if (llama_vocab_get_attr(vocab, t) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
            seen[t] = 1;
            ids.push_back(t);
        }
    }
    const size_t n_forced = ids.size();

    if (k > 0 && k < top_k) {
        SPC_WRN("LLAMA_SPEC_DRAFT_VOCAB=%d is below the draft's %d candidates per row; using %d\n", k, top_k, top_k);
        k = top_k;
    }
    const size_t want = k < 0 ? SIZE_MAX : (size_t) k;
    for (const llama_token t : ranked) {
        if (ids.size() >= want && ids.size() % 4 == 0) {
            break;
        }
        if (!seen[t]) {
            seen[t] = 1;
            ids.push_back(t);
        }
    }
    if (k > 0 && ids.size() < want) {
        SPC_WRN("LLAMA_SPEC_DRAFT_VOCAB=%d: %s gives only %zu tokens with the %zu specials; using them all\n", k, ef, ids.size(), n_forced);
    }
    if (ids.size() % 4 != 0 && ids.size() >= n_forced + 4) {
        ids.resize(ids.size() / 4 * 4); // the file ran out: drop the lowest-ranked ids, never a special
    }
    if ((int32_t) ids.size() < top_k || ids.size() % 4 != 0) {
        SPC_WRN("%s: %zu tokens is fewer than the draft's %d candidates per row or not a multiple of 4; draft vocabulary subset off\n", ef, ids.size(), top_k);
        return false;
    }

    const int32_t n = (int32_t) ids.size();
    // ascending ids keep the subset rows in vocabulary order, so the device top-k meets ties in the same order
    std::sort(ids.begin(), ids.end());
    what = string_format("(LLAMA_SPEC_DRAFT_VOCAB=%d): %d of %d tokens, %zu of them specials, from %s", k, n, n_vocab, n_forced, ef);
    ids_out = std::move(ids);
    return true;
}


// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // the prefill plan per seq (prefill_plan): the prompt's end and the checkpoint marks, and the draft's window: a
    // query at position p sees keys at positions > p - n_swa, so a draft at pos0 >= n_end needs the rows from n_end - (n_swa - 1)
    std::vector<llama_pos>              plan_end;
    std::vector<std::vector<llama_pos>> plan_marks;
    int32_t                             n_swa_dft = 0;
    std::vector<int64_t>                plan_rows_seen, plan_rows_injected; // per seq, since the plan was set (logged at begin)

    // the target keeps a verify's layer inputs on the device and the injection copies them there
    bool feat_dev = false;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        // [TAG_DRAFT_VOCAB] LLAMA_DFLASH2_VOCAB (default 1; 0 = the full head): DFlash2's LM head over the draft
        // vocabulary of LLAMA_SPEC_DRAFT_VOCAB and LLAMA_SPEC_DRAFT_VOCAB_FILE (MTP's), its rows taken from output.weight of the GGUF
        // in LLAMA_DFLASH2_HEAD_FILE (the MTP draft's own Q3_K head) when set, otherwise from the target's head it borrows. The
        // selector's top-k then runs over the subset and maps its candidates back to token ids (dflash.cpp); a drafted token
        // changes only where the full head's candidates lie outside the subset, and the target verifies every draft
        if (is_dflash2) {
            const char * ev = getenv("LLAMA_DFLASH2_VOCAB");
            std::vector<llama_token> ids;
            std::string what;
            if ((ev == nullptr || atoi(ev) != 0) && common_speculative_draft_vocab_ids(model_tgt, selector_top_k, ids, what)) {
                const char * hf = getenv("LLAMA_DFLASH2_HEAD_FILE");
                hf = hf && *hf ? hf : nullptr;
                if (llama_model_set_head_subset_dflash(model_dft, model_tgt, hf, ids.data(), (int32_t) ids.size())) {
                    LOG_INF("%s: DFlash2 head over the draft vocabulary %s, rows from %s\n", __func__, what.c_str(), hf ? hf : "the target's head");
                } else {
                    SPC_WRN("DFlash2 head subset %s could not be made (head file %s); full head\n", what.c_str(), hf ? hf : "unset");
                }
            }
        }

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_ubatch(ctx_dft), n_embd_enc, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // [TAG_DFLASH2_FEAT_DEV] LLAMA_DFLASH2_FEAT_DEV (default 1): a verify-sized target decode (one ubatch of at most
        // 64 rows) keeps its layer inputs on the device, and the injection copies them device to device on the GPU's streams, instead
        // of device -> host -> device with a wait; larger decodes (prefill) still go through the host. The device path takes the
        // layers in ascending order, as the host path's slots are, so it needs ascending target_layer_ids
        {
            const char * e = getenv("LLAMA_DFLASH2_FEAT_DEV");
            bool ascending = true;
            for (uint32_t k = 1; k < target_layer_ids_n; ++k) {
                ascending = ascending && target_layer_ids[k] > target_layer_ids[k - 1];
            }
            feat_dev = is_dflash2 && ascending && (e == nullptr || atoi(e) != 0) && n_seq == 1;
            if (feat_dev) {
                llama_set_embeddings_layer_inp_dev(ctx_tgt, 64);
            }
        }

        // the prefill plan, LLAMA_DFLASH2_INJECT_WINDOW (default 1): only for a draft whose every layer is a sliding window
        {
            const char * e = getenv("LLAMA_DFLASH2_INJECT_WINDOW");
            n_swa_dft = (e == nullptr || atoi(e) != 0) && llama_model_all_swa(model_dft) ? llama_model_n_swa(model_dft) : 0;
            plan_end.assign(n_seq, -1);
            plan_marks.assign(n_seq, {});
            plan_rows_seen.assign(n_seq, 0);
            plan_rows_injected.assign(n_seq, 0);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // the prefill is done: every later row is injected
        if (plan_end[seq_id] >= 0) {
            LOG_INF("%s: seq %d: the prefill injected %" PRId64 " of %" PRId64 " rows (prompt end %d, %zu checkpoint marks, window %d)\n", __func__,
                    (int) seq_id, plan_rows_injected[seq_id], plan_rows_seen[seq_id], (int) plan_end[seq_id], plan_marks[seq_id].size(), n_swa_dft - 1);
        }
        plan_end[seq_id] = -1;
        plan_marks[seq_id].clear();

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    void prefill_plan(llama_seq_id seq_id, llama_pos n_end, const std::vector<llama_pos> & marks) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || n_swa_dft <= 0) {
            return;
        }
        plan_end[seq_id]   = n_end;
        plan_marks[seq_id] = marks;
        plan_rows_seen[seq_id]     = 0;
        plan_rows_injected[seq_id] = 0;
    }

    // whether the row at position p is injected: with a prefill plan, only rows a later draft's window reaches, i.e.
    // within n_swa - 1 before the prompt's end or before a checkpoint mark (a restore there resumes with the rows before it). Without
    // a plan (decode, or no sliding window), every row
    bool inject_row(llama_seq_id seq_id, llama_pos p) const {
        const llama_pos end = plan_end[seq_id];
        if (end < 0) {
            return true;
        }
        const llama_pos w = n_swa_dft - 1;
        if (p >= end - w) {
            return true;
        }
        for (const llama_pos m : plan_marks[seq_id]) {
            if (p < m && p >= m - w) {
                return true;
            }
        }
        return false;
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // embeddings are injected too, except the pinned ones skipped below.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        // [TAG_DFLASH2_FEAT_DEV] the target held this decode's layer inputs on the device (a verify): the one sequence's
        // rows, all of them, go to the injection device to device. Otherwise they are read back from the host copies
        const int32_t n_dev = feat_dev ? llama_get_embeddings_layer_inp_dev_rows(ctx_tgt) : 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // an M-RoPE image pins all its rows to one position, so a windowed draft
            // cache cannot free cells for it - skip it, the draft can jump over the gap
            const bool pos_pinned = batch_in.pos[i_batch_beg[seq_id]] == batch_in.pos[i_batch_end[seq_id]];
            if (has_embeddings && n_rows > 1 && pos_pinned) {
                continue;
            }

            // [TAG_DFLASH2_FEAT_DEV] the target held this decode's features on the device only (no host copy): the injection takes
            // all its rows from there. It does so only for one sequence's small batch (feat_dev needs n_seq == 1)
            const bool dev = n_dev > 0;
            GGML_ASSERT(!dev || (n_dev == n_tokens && n_rows == n_tokens && n_rows <= n_ubatch));

            // the rows a later draft can see (inject_row), in batch order; with the device path every row (a small
            // prefill batch ends at a checkpoint mark or at the prompt's end, so its rows are within the window anyway)
            std::vector<int32_t> rows;
            rows.reserve(n_rows);
            for (int32_t r = i_batch_beg[seq_id]; r <= i_batch_end[seq_id]; ++r) {
                if (dev || inject_row(seq_id, batch_in.pos[r])) {
                    rows.push_back(r);
                }
            }
            if (plan_end[seq_id] >= 0) {
                plan_rows_seen[seq_id]     += n_rows;
                plan_rows_injected[seq_id] += (int64_t) rows.size();
            }
            if (rows.empty()) {
                continue;
            }
            const int32_t n_keep = (int32_t) rows.size();

            for (int32_t offset = 0; offset < n_keep; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_keep - offset);

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                batch_inject.n_tokens = n_chunk;
                if (!dev) {
                    for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                        const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                        if (!layer) {
                            GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                        }
                        for (int32_t i = 0; i < n_chunk; ++i) {
                            float       * dst = batch_inject.embd + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                            const float * src = layer + (size_t) rows[offset + i] * n_embd_tgt;
                            std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                        }
                    }
                }

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.pos[rows[offset + i]];
                    batch_inject.pos[i] = p;
                    if (is_mrope) {
                        batch_inject.pos[1 * n_chunk + i] = p;
                        batch_inject.pos[2 * n_chunk + i] = p;
                        batch_inject.pos[3 * n_chunk + i] = 0;
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                if (dev) {
                    llama_set_inject_dev(ctx_dft, true);
                }
                const int32_t rc = llama_decode(ctx_dft, batch_inject);
                if (dev) {
                    llama_set_inject_dev(ctx_dft, false);
                }
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.pos0;

            // [TAG_SPEC_ADAPT_WIDTH] a narrowed round drafts a block of that width, as a launch at that n-max does
            const int32_t n_draft = dp.n_cap > 0 ? std::min(params.n_max, dp.n_cap) : params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            // [TAG_SPEC_REJECTION] with LLAMA_SPEC_REJECTION and a target sampler that can take the
            // rejection step, each position's candidates (distinct ids: ggml_top_k), with their selector scores given
            // the predecessor as logits, go through common_speculative_rejection_pick as MTP's do: q is the candidates
            // through a copy of the target's sampler chain (LLAMA_SPEC_REJECTION_TEMP), the draw is the target's (or
            // q_rng), the copy accepts each drafted token, and a forced token or a row the copy cannot follow gets the
            // forced token or the argmax with an empty row. the drawn candidate is the next position's predecessor; a
            // forced token outside the candidates ends the draft. p_min truncates on the largest probability of the
            // token's row (1 for a forced token, the selector's softmax at the argmax for an argmax fallback).
            // otherwise (the toggle off, T=0, a chain that does not end in dist) the argmax loop below, unchanged
            // the sampled draft is the default under LLAMA_SPEC_REJECTION; LLAMA_DFLASH2_GREEDY=1 keeps the argmax walk
            static const bool greedy = [] { const char * e = getenv("LLAMA_DFLASH2_GREEDY"); return e != nullptr && atoi(e) != 0; }();
            if (is_dflash2 && !greedy && dp.q && dp.smpl_tgt && common_speculative_rejection() && common_sampler_rejection_ok(dp.smpl_tgt)) {
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                dp.q->clear();

                common_sampler_ptr copy(common_sampler_clone(dp.smpl_tgt));

                std::vector<llama_token_data> cand(selector_top_k);
                int32_t predecessor = 0;
                // [TAG_SPEC_ADAPT_WIDTH] the server's cap on this round's draft length, by depth and acceptance
                const int32_t n_take = dp.n_cap > 0 ? std::min(dp.n_cap, n_block_tokens - 1) : n_block_tokens - 1;
                for (int32_t i = 1; i < n_block_tokens && (int32_t) result.size() < n_take; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    // the argmax first, as the fallback reads it; p the selector's softmax
                    const float s_max = *std::max_element(scores, scores + selector_top_k);
                    float sum = 0.0f;
                    for (int32_t k = 0; k < selector_top_k; ++k) {
                        cand[k] = { (llama_token) row[k], scores[k], std::exp(scores[k] - s_max) };
                        sum += cand[k].p;
                    }
                    for (auto & c : cand) {
                        c.p /= sum;
                    }
                    std::stable_sort(cand.begin(), cand.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
                    const llama_token_data_array cur_p = { cand.data(), cand.size(), -1, true };

                    float p_max = 0.0f;
                    const llama_token id = common_speculative_rejection_pick(copy.get(), dp.smpl_tgt, dp.q_rng, *dp.q, &cur_p, p_max);
                    for (const auto & e : dp.q->back()) {
                        p_max = std::max(p_max, e.p);
                    }
                    if (p_max < params.p_min) {
                        dp.q->pop_back();
                        break;
                    }
                    result.push_back(id);

                    predecessor = -1;
                    for (int32_t k = 0; k < selector_top_k; ++k) {
                        if ((llama_token) row[k] == id) {
                            predecessor = k;
                            break;
                        }
                    }
                    if (predecessor < 0) {
                        break;
                    }
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    dp.q->clear();
                }
                continue;
            }

            if (is_dflash2) {
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                int32_t predecessor = 0;
                const int32_t n_take = dp.n_cap > 0 ? std::min(dp.n_cap, n_block_tokens - 1) : n_block_tokens - 1;
                for (int32_t i = 1; i < n_block_tokens && (int32_t) result.size() < n_take; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + selector_top_k));
                    if (params.p_min > 0.0f) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                        }
                        if (1.0f / sum < params.p_min) {
                            break;
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // the carry: the target row the first input of the next batch pairs with, with its position and lineage,
    // and the verification window of the latest target batch (row 0: the round's first token, row N: the Nth draft token).
    // See common/mtp-carry.h
    common_mtp_carry carry;

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // [TAG_SPEC_PIPELINE] the draft's hidden row that pairs with the next token to feed, per seq: set by draft() for
    // its last drafted token, by pipe_process() for its extra token, and by pipe_chain() for its last drafted token
    bool                            pipe_on = false;
    std::vector<std::vector<float>> pipe_h;
    std::vector<std::vector<float>> pipe_p; // the draft's probability of each token drafted since the last draft() start

    // [TAG_SPEC_COUPLED] per seq, the copy of the target's sampler that picks the draft tokens of the current draft()
    std::vector<common_sampler_ptr> coupled;
    // [TAG_SPEC_REJECTION] per seq, whether the current draft() samples its tokens for the rejection step (the copy
    // is then coupled[seq_id], and the draws come from the target's sampler)
    std::vector<bool> rejection;
    // [TAG_SPEC_REJECTION_ADAPT] --spec-draft-n-max, the draft length of a request that does not take
    // the rejection step, and per seq the product of q over the current draft()'s drafted tokens
    int32_t             n_max_base = 0;
    std::vector<double> rs_prod;

    // [TAG_DRAFT_VOCAB] the draft's LM head subset over common_speculative_draft_vocab_ids (shared with DFlash2)
    static void draft_vocab_init(const llama_model * model, const int32_t top_k) {
        std::vector<llama_token> ids;
        std::string what;
        if (!common_speculative_draft_vocab_ids(model, top_k, ids, what, /*arch_default*/ true)) {
            return;
        }
        if (!llama_model_set_head_subset(model, ids.data(), (int32_t) ids.size())) {
            SPC_WRN("%s: this draft's head does not allow a subset; draft vocabulary subset off\n", what.c_str());
            return;
        }
        LOG_INF("%s: draft vocabulary subset %s\n", __func__, what.c_str());
    }

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        // [TAG_SPEC_COUPLED] the candidates per row: 10, or with coupled drafting as many as the target's device
        // top-k returns (params.top_k), so the target's chain applied to them gives what it gives on the full row
        const int32_t top_k = std::max(10, this->params.top_k);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = top_k;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }
        coupled.resize(n_seq);
        rejection.assign(n_seq, false);

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // [TAG_DRAFT_VOCAB] the head subset exists only for the backend samplers, so every sequence needs one
        if (std::all_of(backend_chains.begin(), backend_chains.end(), [](llama_sampler * c) { return c != nullptr; })) {
            draft_vocab_init(llama_get_model(ctx_dft), top_k);
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        LOG_INF("%s: the draft's steps chained on the device (LLAMA_MTP_DRAFT_CHAIN): %s\n", __func__,
                !chain_on() ? "off" : llama_mtp_chain_supported(ctx_dft) ? "on" : "off, this draft cannot chain them (needs the qwen35 MTP head, its vocabulary subset and flash attention)");

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        // [TAG_SPEC_REJECTION_ADAPT] drafts up to LLAMA_SPEC_REJECTION_NMAX under rejection sampling
        n_max_base = this->params.n_max;
        rs_prod.assign(n_seq, 1.0);
        if (common_speculative_rejection_nmax() > this->params.n_max) {
            this->params.n_max = common_speculative_rejection_nmax();
        }

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        // the zero rows of a new context: the fresh boundary of every sequence
        carry.init(n_seq, n_embd);

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // the target's rows of this batch, one per token, in batch order (the context puts the unmasked nextn
        // rows in batch order, whatever order its ubatches held them in): row k is batch token k's. One synchronization
        const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // each token paired with its sequence's previous target row: the previous batch row of the same
            // sequence, or for its first row the carry (position 0: the fresh boundary). Sequences need not be contiguous
            carry.gather(batch_in, h_tgt, batch.embd);

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        // each sequence's rows of this batch become its verification window, and its last row the carry
        carry.set_windows(batch_in, h_tgt);

        return true;
    }

    // LLAMA_MTP_DRAFT_CHAIN (default 1; 0: the host loop): the draft's steps chained on the device
    static bool chain_on() {
        static const bool v = [] { const char * e = getenv("LLAMA_MTP_DRAFT_CHAIN"); return e == nullptr || atoi(e) != 0; }();
        return v;
    }

    // draft() with the steps chained in one graph (llama_mtp_chain_set): each step's token is picked on the
    // device from the merged candidates (GGML_OP_DRAFT_PICK) and fed with its h_nextn to the next step there; the picks
    // come back once, after the last step. The host then walks the steps as the loop below does (p_min, the rows the
    // rejection step can take, the q rows, the adaptive length, n_max) and keeps the drafted tokens up to where the loop
    // would have stopped, which is exact: a draft may end anywhere. The draws are the loop's: each uniform comes from the
    // target's dist generator in the same order (taken from a copy, then the generator advanced by the ones used), so a
    // seed gives what it gave. false (nothing drafted, nothing changed) when a sequence needs the loop: a sampler the
    // pick does not do, coupled drafting, the pipeline, chained heads or a shared cache, or a rejection-mode
    // sequence whose first row the rejection step cannot take (a grammar applies, or the reasoning budget forces a
    // token): the loop proposes the forced token or the argmax there with an empty row, which the verify takes by
    // sample-and-match, and the chain has no such row to give; the loop then drafts
    bool draft_chain(common_speculative_draft_params_vec & dparams) {
        auto * ctx_dft = params.ctx_dft;

        if (!chain_on() || chain_heads || is_mem_shared || pipe_on || !llama_mtp_chain_supported(ctx_dft) || params.n_max <= 0) {
            return false;
        }
        struct count_guard {
            bool chained = false;
            ~count_guard() {
                if (chained) {
                    GGML_RT_COUNT("spec.chain_rounds", 1);
                } else {
                    GGML_RT_COUNT("spec.chain_fallback", 1);
                }
            }
        } count_g;
        for (llama_sampler * c : backend_chains) {
            if (c == nullptr) {
                return false;
            }
        }

        const int32_t n_steps = params.n_max;
        const int32_t top_k   = std::max(10, params.top_k);

        std::vector<llama_seq_id> seqs;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (dparams[seq_id].drafting) {
                seqs.push_back(seq_id);
            }
        }
        if (seqs.empty() || seqs.size() > 3 || (int32_t) (seqs.size()*n_steps) > (int32_t) llama_n_batch(ctx_dft)) {
            return false;
        }

        // every sequence's mode: the rejection step's pick (the target's chain on the device) or the argmax
        std::vector<bool> rej(n_seq, false);
        std::vector<std::vector<float>> prm(n_seq);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_dft));
        for (llama_seq_id seq_id : seqs) {
            auto & dp = dparams[seq_id];
            const bool r = common_speculative_rejection() && dp.q && dp.smpl_tgt && common_sampler_rejection_ok(dp.smpl_tgt);
            if (!r && common_speculative_coupled()) {
                return false;
            }
            // the first row, read before anything is decoded or changed: where it cannot take the rejection
            // step (every tool call runs under a grammar) the round is the loop's, which drafts it as with the chain off
            if (r && !common_sampler_rejection_row_ready(dp.smpl_tgt)) {
                return false;
            }
            float desc[32] = {};
            if (r && !common_sampler_device_chain(dp.smpl_tgt, vocab, common_speculative_rejection_temp(), desc)) {
                return false;
            }
            rej[seq_id] = r;
            prm[seq_id].assign((size_t) n_steps*32, 0.0f);
            for (int32_t i = 0; i < n_steps; ++i) {
                std::copy(desc, desc + 32, prm[seq_id].begin() + (size_t) i*32);
            }
        }

        // from here on the sequences draft through the chain
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        common_batch_clear(batch);
        for (llama_seq_id seq_id : seqs) {
            auto & dp = dparams[seq_id];

            common_sampler_reset(smpls[seq_id].get());
            rejection[seq_id] = rej[seq_id];
            if (dp.q) {
                dp.q->clear();
            }
            rs_prod[seq_id] = 1.0;
            coupled[seq_id].reset(rej[seq_id] ? common_sampler_clone(dp.smpl_tgt) : nullptr);

            // the draws, from a copy of the target's generator (the clone carries its state)
            if (rej[seq_id]) {
                common_sampler_ptr gen { common_sampler_clone(dp.smpl_tgt) };
                for (int32_t i = 0; i < n_steps; ++i) {
                    double u = 0.0;
                    GGML_ASSERT(common_sampler_draw_uniform(gen.get(), &u));
                    // the double as two floats (ggml-draft-pick.h)
                    const float hi = (float) u;
                    prm[seq_id][(size_t) i*32 + 6]  = hi;
                    prm[seq_id][(size_t) i*32 + 29] = (float) (u - (double) hi);
                }
            }

            common_mtp_carry::use_t use;
            const float * h_row = carry.row_for(seq_id, dp.pos0, use);
            for (int32_t i = 0; i < n_steps; ++i) {
                common_batch_add(batch, dp.id_last, dp.pos0 + i, { seq_id }, false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
            }
            llama_mtp_chain_set_params(ctx_dft, seq_id, prm[seq_id].data(), (int32_t) prm[seq_id].size());
        }

        llama_mtp_chain_set(ctx_dft, n_steps, top_k);
        const int ret = llama_decode(ctx_dft, batch);
        common_batch_clear(batch);
        if (ret != 0) {
            SPC_ERR("chained draft llama_decode returned %d; drafting with the host loop\n", ret);
            return false;
        }

        GGML_RT_SCOPE("spec.chain_walk");
        for (llama_seq_id seq_id : seqs) {
            auto & dp     = dparams.at(seq_id);
            auto & result = *dp.result;
            auto * smpl   = smpls[seq_id].get();

            int32_t n_draws = 0;
            for (int32_t i = 0; i < n_steps; ++i) {
                int32_t w = 0;
                const int32_t * o = llama_mtp_chain_get(ctx_dft, seq_id, i, &w);
                GGML_ASSERT(o != nullptr && w >= 8);
                const int32_t k = (w - 8)/2;

                float p0;
                std::memcpy(&p0, &o[3], sizeof(float));
                // only collect very high-confidence draft tokens
                if (p0 < params.p_min) {
                    break;
                }

                const llama_token id = o[0];
                float p_id = p0;
                if (rejection[seq_id]) {
                    // the copy's row, read after the tokens drafted so far: a row that became one the rejection step cannot
                    // take mid-draft (a lazy grammar's trigger fired on a drafted token, or the reasoning budget now forces
                    // one) ends the draft here, which is valid (a draft may end anywhere). the chain's picks past it were
                    // drawn on the device with no grammar, and the loop's draws do not advance on such a row, so they
                    // are not followed. the same end where the chain left no candidate (o[4] == 0): the loop goes on there
                    // with the argmax and an empty row, but its next row would then use the draw this step's pick took
                    // (reported, not changed; the target's chain leaves no candidate only for a filter that empties it)
                    if (!common_sampler_rejection_row_ready(coupled[seq_id].get()) || o[4] <= 0) {
                        break;
                    }
                    std::memcpy(&p_id, &o[2], sizeof(float));
                    common_rejection_q row;
                    row.reserve(o[4]);
                    for (int32_t t = 0; t < o[4]; ++t) {
                        float pq;
                        std::memcpy(&pq, &o[8 + k + t], sizeof(float));
                        row.push_back({ o[8 + t], 0.0f, pq });
                    }
                    dp.q->push_back(std::move(row));
                    n_draws++;
                    common_sampler_accept_draft(coupled[seq_id].get(), id);
                }

                common_sampler_accept(smpl, id, true);
                result.push_back(id);

                // [TAG_SPEC_REJECTION_ADAPT], as the loop
                const bool adapt = rejection[seq_id] && common_speculative_rejection_nmax() > 0;
                if (adapt) {
                    rs_prod[seq_id] *= p_id;
                }
                if ((adapt ? params.n_max : std::min(params.n_max, n_max_base)) <= (int) result.size() ||
                        (adapt && rs_prod[seq_id] < common_speculative_rejection_cutoff())) {
                    break;
                }
            }

            // the target's generator advanced past the draws the loop would have taken
            for (int32_t i = 0; i < n_draws; ++i) {
                double u = 0.0;
                common_sampler_draw_uniform(dp.smpl_tgt, &u);
            }
        }

        for (llama_seq_id seq_id : seqs) {
            auto & dp = dparams[seq_id];
            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
                if (dp.q) {
                    dp.q->clear();
                }
            }
        }

        count_g.chained = true;
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        // a draft starts from the carry, which holds pos0 - 1 after every prompt and round; a sequence without
        // it drafts nothing this round (the round decodes its sampled token alone)
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (dp.drafting && !is_mem_shared && !carry.valid_for(seq_id, dp.pos0)) {
                LOG_WRN("%s: seq %d: no target row for pos %d (carry pos %d), no draft\n", __func__, (int) seq_id, (int) dp.pos0 - 1,
                        (int) carry.pos[seq_id]);
                GGML_RT_COUNT("mtp.draft_no_carry", 1);
                dp.drafting = false;
            }
        }

        if (draft_chain(dparams)) {
            return;
        }

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());
            if (pipe_on) {
                pipe_p[seq_id].clear();
            }
            // [TAG_SPEC_REJECTION] rejection sampling takes precedence over coupled drafting
            rejection[seq_id] = common_speculative_rejection() && dp.q && dp.smpl_tgt && common_sampler_rejection_ok(dp.smpl_tgt);
            if (dp.q) {
                dp.q->clear();
            }
            rs_prod[seq_id] = 1.0;
            coupled[seq_id].reset((rejection[seq_id] || common_speculative_coupled()) && dp.smpl_tgt && common_sampler_coupled_ok(dp.smpl_tgt) ?
                    common_sampler_clone(dp.smpl_tgt) : nullptr);

            common_mtp_carry::use_t use;
            const float * h_row = carry.row_for(seq_id, dp.pos0, use);
            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(h_row, h_row + n_embd);
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < pos0
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                // add drafted token for each sequence: the argmax, or [TAG_SPEC_COUPLED] the pick through the copy of
                // the target's sampler
                float p_id = cur_p->data[0].p;
                const llama_token id = rejection[seq_id] ? common_speculative_rejection_pick(coupled[seq_id].get(), dparams[seq_id].smpl_tgt, dparams[seq_id].q_rng, *dparams[seq_id].q, cur_p, p_id) :
                    coupled[seq_id] ? coupled_pick(coupled[seq_id].get(), cur_p, p_id) : cur_p->data[0].id;

                if (coupled[seq_id] && coupled_debug()) {
                    LOG_INF("coupled draft seq %d pos %d argmax %d %.4f pick %d %.4f n %zu\n", (int) seq_id, (int) (dparams[seq_id].pos0 + i + 1),
                            cur_p->data[0].id, cur_p->data[0].p, id, p_id, cur_p->size);
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (pipe_on) {
                    std::memcpy(pipe_h[seq_id].data(), h_row, row_bytes);
                    pipe_p[seq_id].push_back(p_id);
                }

                // [TAG_SPEC_REJECTION_ADAPT] a request that takes the rejection step drafts up to
                // LLAMA_SPEC_REJECTION_NMAX, and drafts the next token only while the product of q over the tokens
                // drafted so far stays at least the cutoff; others keep --spec-draft-n-max
                const bool adapt = rejection[seq_id] && common_speculative_rejection_nmax() > 0;
                if (adapt) {
                    rs_prod[seq_id] *= p_id;
                }
                if ((adapt ? params.n_max : std::min(params.n_max, n_max_base)) <= (int) result.size() ||
                        (adapt && rs_prod[seq_id] < common_speculative_rejection_cutoff())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.pos0 + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.pos0, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
                if (dp.q) {
                    dp.q->clear();
                }
            }
        }
    }

    // [TAG_SPEC_COUPLED] the draft token picked from the draft's candidates cur_p (sorted) through the copy of the
    // target's sampler, accepted into it; the draft's argmax when the copy cannot follow the target. p: the draft's
    // probability of the picked token (1 for a token the reasoning budget forces, 0 for one outside the candidates)
    static bool coupled_debug() {
        static const bool v = [] { const char * e = getenv("LLAMA_SPEC_COUPLED_DEBUG"); return e != nullptr && atoi(e) != 0; }();
        return v;
    }

    // near-tie guard (LLAMA_SPEC_COUPLED_TIE, default 0.5, 0: off): a pick other than the argmax whose draft
    // probability is at least that fraction of the argmax's proposes the argmax. near ties are where the draft's and
    // the target's orders differ most, and the per-position log showed those swaps losing
    static float coupled_tie() {
        static const float v = [] { const char * e = getenv("LLAMA_SPEC_COUPLED_TIE"); return e ? (float) atof(e) : 0.5f; }();
        return v;
    }

    static llama_token coupled_pick(common_sampler * coupled, const llama_token_data_array * cur_p, float & p) {
        bool forced = false;
        llama_token id = common_sampler_coupled_pick(coupled, cur_p, &forced);
        if (id == LLAMA_TOKEN_NULL) {
            id = cur_p->data[0].id;
        }
        p = forced ? 1.0f : 0.0f;
        for (size_t k = 0; !forced && k < cur_p->size; ++k) {
            if (cur_p->data[k].id == id) {
                p = cur_p->data[k].p;
                break;
            }
        }
        if (!forced && id != cur_p->data[0].id && coupled_tie() > 0.0f && p >= coupled_tie()*cur_p->data[0].p) {
            id = cur_p->data[0].id;
            p  = cur_p->data[0].p;
        }
        common_sampler_accept_draft(coupled, id); // the argmax fallback need not fit a triggered grammar
        return id;
    }

    const std::vector<float> * pipe_probs(llama_seq_id seq_id) const override {
        return pipe_on && seq_id >= 0 && seq_id < (llama_seq_id) n_seq ? &pipe_p[seq_id] : nullptr;
    }

    bool pipe_enable() override {
        if (chain_heads || is_mem_shared) {
            return false;
        }
        pipe_on = true;
        pipe_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        pipe_p.assign(n_seq, {});
        return true;
    }

    // the catch-up of process() for one sequence's verified chunk (the target's current outputs hold its n rows),
    // plus an optional extra token after it, paired with the chunk's last target row; the draft's hidden row for
    // the extra token is kept in pipe_h. the draft cache is cleared from pos0 on first
    bool pipe_process(llama_seq_id seq_id, const llama_token * toks, int32_t n, llama_pos pos0, llama_token extra) override {
        if (!pipe_on || n <= 0 || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t  row_bytes = (size_t) n_embd * sizeof(float);
        const bool    has_extra = extra != LLAMA_TOKEN_NULL;
        const int32_t n_all     = n + (has_extra ? 1 : 0);

        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pos0, -1);

        common_batch_clear(batch);
        for (int32_t k = 0; k < n_all; ++k) {
            common_batch_add(batch, k < n ? toks[k] : extra, pos0 + k, { seq_id }, has_extra && k == n_all - 1);
        }

        // the chunk's rows (one sequence, in batch order); its first token pairs with the carry
        const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
        common_mtp_carry::use_t use;
        std::memcpy(batch.embd, carry.row_for(seq_id, pos0, use), row_bytes);
        std::memcpy(batch.embd + (size_t) n_embd, h_tgt, row_bytes * (n_all - 1));

        const int32_t rc = llama_decode(ctx_dft, batch);
        if (rc != 0) {
            SPC_ERR("pipelined catch-up llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", (int) rc, (int) pos0);
            return false;
        }

        carry.set_window(seq_id, h_tgt, pos0, n, common_mtp_carry::SRC_PIPE);

        if (has_extra) {
            std::memcpy(pipe_h[seq_id].data(), llama_get_embeddings_nextn_ith(ctx_dft, n_all - 1), row_bytes);
        }

        return true;
    }

    // [TAG_SPEC_REJECTION_PIPE] the next pipe_chain's first known token pairs with the target's row the
    // catch-up left for it (the carry), as the serial loop's draft() pairs its first token
    bool pipe_rebase(llama_seq_id seq_id) override {
        if (!pipe_on || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        pipe_h[seq_id].assign(carry.h[seq_id].begin(), carry.h[seq_id].end());
        return true;
    }

    // continue the draft chain from pipe_h: feed the known tokens (known[0] at pos pairs with pipe_h; the draft's
    // predictions before the last known token are not used), then draft up to n_new tokens into out
    bool pipe_chain(llama_seq_id seq_id, const llama_token * known, int32_t n_known, llama_pos pos, int32_t n_new, llama_tokens & out, common_sampler * coupled,
            std::mt19937 * q_rng, std::vector<common_rejection_q> * q) override {
        if (!pipe_on || n_known <= 0 || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }

        auto * ctx_dft = this->params.ctx_dft;
        auto * smpl    = smpls[seq_id].get();

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        std::vector<float> & h = pipe_h[seq_id];

        llama_token tok     = known[0];
        llama_pos   p       = pos;
        int32_t     i_known = 1;
        int32_t     n_drft  = 0;

        while (n_drft < n_new) {
            common_batch_clear(batch);
            common_batch_add(batch, tok, p, { seq_id }, true);
            std::memcpy(batch.embd, h.data(), row_bytes);

            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("pipelined draft llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", (int) rc, (int) p);
                return false;
            }

            if (i_known < n_known) {
                std::memcpy(h.data(), llama_get_embeddings_nextn_ith(ctx_dft, 0), row_bytes);
                tok = known[i_known++];
                p++;
                continue;
            }

            common_sampler_sample(smpl, ctx_dft, 0, true);
            std::memcpy(h.data(), llama_get_embeddings_nextn_ith(ctx_dft, 0), row_bytes);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);
            if (cur_p->data[0].p < params.p_min) {
                break;
            }
            float p_id = cur_p->data[0].p;
            // [TAG_SPEC_REJECTION_PIPE] sampled from q with the draft's own stream
            const llama_token id = coupled && q && q_rng ? common_speculative_rejection_pick(coupled, nullptr, q_rng, *q, cur_p, p_id) :
                coupled ? coupled_pick(coupled, cur_p, p_id) : cur_p->data[0].id;
            if (coupled && coupled_debug()) {
                LOG_INF("coupled chain pos %d argmax %d %.4f pick %d %.4f n %zu\n", (int) p + 1, cur_p->data[0].id, cur_p->data[0].p, id, p_id, cur_p->size);
            }
            common_sampler_accept(smpl, id, true);

            out.push_back(id);
            pipe_p[seq_id].push_back(p_id);
            n_drft++;

            tok = id;
            p++;
        }

        return true;
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        carry.accept(seq_id, n_accepted);
    }

    // the boundary state: the carry (position and row). The pipeline's chain row and the verification window
    // are derived within a round and never saved; a restore or reset drops them
    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        return carry.get_state(seq_id, data);
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        carry.set_state(seq_id, data);
        pipe_reset(seq_id);
    }

    void reset_state(llama_seq_id seq_id) override {
        carry.reset(seq_id);
        pipe_reset(seq_id);
    }

    bool state_valid(llama_seq_id seq_id, llama_pos pos_next) const override {
        // the shared-memory drafter (gemma4) reads the carry only at draft(), after process() has set it from the batch
        return is_mem_shared || carry.valid_for(seq_id, pos_next);
    }

    void pipe_reset(llama_seq_id seq_id) {
        if (pipe_on && seq_id >= 0 && seq_id < (llama_seq_id) n_seq) {
            std::fill(pipe_h[seq_id].begin(), pipe_h[seq_id].end(), 0.0f);
            pipe_p[seq_id].clear();
        }
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                // [TAG_SPEC_REJECTION_ADAPT] the adaptive draft's depth sizes the verify's outputs
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                n_max = std::max(n_max, common_speculative_rejection_nmax());
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        // default to global devices value
        if (!params_spec.devices.empty()) {
            result.devices           = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        const size_t n_devs = std::count_if(params_spec.devices.begin(), params_spec.devices.end(),
                [](ggml_backend_dev_t d) { return d != nullptr; });
        if (n_devs == 1) {
            result.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }

        if (params_spec.n_ubatch > 0) {
            result.n_ubatch = std::min(params.n_ubatch, params_spec.n_ubatch);
            LOG_WRN("draft ubatch %d (target %d) (--spec-draft-ubatch)\n", result.n_ubatch, params.n_ubatch);
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        mparams.model_shared = model_tgt;

        llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

void common_speculative_prefill_plan(common_speculative * spec, llama_seq_id seq_id, llama_pos n_end, const std::vector<llama_pos> & marks) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->prefill_plan(seq_id, n_end, marks);
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    GGML_RT_SCOPE("spec.process");
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    GGML_RT_SCOPE("spec.draft");
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                    // [TAG_SPEC_REJECTION] each q row depends only on the tokens before it, so the kept rows stay valid
                    if (dp.q && (int) dp.q->size() > dp.n_max) {
                        dp.q->resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    GGML_RT_SCOPE("spec.accept");
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

bool common_speculative_rejection() {
    // default on for Flash-Next and the 27B; LLAMA_SPEC_REJECTION=0 turns it off
    static const bool v = [] { const char * e = getenv("LLAMA_SPEC_REJECTION"); return e == nullptr || atoi(e) != 0; }();
    return v;
}

float common_speculative_rejection_temp() {
    static const float v = [] { const char * e = getenv("LLAMA_SPEC_REJECTION_TEMP"); return e ? (float) atof(e) : 0.0f; }();
    return v;
}

int32_t common_speculative_rejection_nmax() {
    static const int32_t v = [] {
        const char * e = getenv("LLAMA_SPEC_REJECTION_ADAPT");
        if (!common_speculative_rejection() || e == nullptr || atoi(e) == 0) {
            return 0;
        }
        // a 5-token verify (4 drafts) emits garbage at 16K, on main too: at most 3 drafts
        const char * en = getenv("LLAMA_SPEC_REJECTION_NMAX");
        return std::clamp(en ? atoi(en) : 3, 1, 3);
    }();
    return v;
}

float common_speculative_rejection_cutoff() {
    static const float v = [] { const char * e = getenv("LLAMA_SPEC_REJECTION_CUTOFF"); return e ? (float) atof(e) : 0.5f; }();
    return v;
}

bool common_speculative_coupled() {
    static const bool v = [] { const char * e = getenv("LLAMA_SPEC_COUPLED"); return e != nullptr && atoi(e) != 0; }();
    return v;
}

bool common_speculative_pipe_enable(common_speculative * spec) {
    if (spec == nullptr || spec->impls.size() != 1) {
        return false;
    }
    return spec->impls[0]->pipe_enable();
}

const std::vector<float> * common_speculative_pipe_probs(const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || spec->impls.size() != 1) {
        return nullptr;
    }
    return spec->impls[0]->pipe_probs(seq_id);
}

bool common_speculative_pipe_process(common_speculative * spec, llama_seq_id seq_id, const llama_token * toks, int32_t n, llama_pos pos0, llama_token extra) {
    if (spec == nullptr || spec->impls.size() != 1) {
        return false;
    }
    return spec->impls[0]->pipe_process(seq_id, toks, n, pos0, extra);
}

bool common_speculative_pipe_rebase(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || spec->impls.size() != 1) {
        return false;
    }
    return spec->impls[0]->pipe_rebase(seq_id);
}

bool common_speculative_pipe_chain(common_speculative * spec, llama_seq_id seq_id, const llama_token * known, int32_t n_known, llama_pos pos, int32_t n_new, llama_tokens & out,
        common_sampler * coupled, std::mt19937 * q_rng, std::vector<common_rejection_q> * q) {
    if (spec == nullptr || spec->impls.size() != 1) {
        return false;
    }
    auto & impl = spec->impls[0];
    common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
    const size_t n0 = out.size();
    const bool ok = impl->pipe_chain(seq_id, known, n_known, pos, n_new, out, coupled, q_rng, q);
    impl->n_gen_tokens += out.size() - n0;
    return ok;
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_reset_state(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->reset_state(seq_id);
    }
}

bool common_speculative_state_valid(const common_speculative * spec, llama_seq_id seq_id, llama_pos pos_next) {
    if (spec == nullptr) {
        return true;
    }

    for (const auto & impl : spec->impls) {
        if (!impl->state_valid(seq_id, pos_next)) {
            return false;
        }
    }

    return true;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
