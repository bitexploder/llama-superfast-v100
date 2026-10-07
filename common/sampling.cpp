#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "../src/llama-ext.h" // [TAG_SPEC_REJECTION] llama_sampler_chain_draw_uniform
#include "reasoning-budget.h"

#include "ggml.h"
#include "ggml-rtimer.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // [TAG_BACKEND_TOPK] number of top candidates the context returns for this sampler's rows (0: none), and the
    // tokens the chain's logit bias sets to -inf (sorted)
    int32_t backend_topk = 0;

    std::vector<llama_token> topk_excl = {};

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    // [TAG_BACKEND_TOPK] the rows carry the backend's top n = backend_topk candidates, and the chain's result depends
    // only on its k = params.top_k highest logits in order, after a logit bias that can only set tokens to -inf
    // (common_sampler_topk_exact); n >= k + 1 + the number of those tokens. if the k + 1 highest candidates that are
    // not set to -inf are finite and strictly decreasing, the k highest of the whole biased vocabulary are the same
    // tokens in the same order (every token outside the candidates is at most the lowest candidate, and at least
    // k + 1 unbiased candidates are above that), so the chain applied to the candidates gives exactly what it gives
    // on the full vocabulary: load them and return true. otherwise (a tie, or anything unexpected) return false and
    // the caller loads the full logits.
    bool set_logits_topk(struct llama_context * ctx, int idx) {
        const float *       lg  = llama_get_sampled_logits_ith     (ctx, idx);
        const llama_token * ids = llama_get_sampled_candidates_ith (ctx, idx);

        const int32_t n = (int32_t) llama_get_sampled_logits_count_ith(ctx, idx);
        const int32_t k = params.top_k;

        if (lg == nullptr || ids == nullptr || llama_get_sampled_probs_ith(ctx, idx) != nullptr ||
                n != backend_topk || (int32_t) llama_get_sampled_candidates_count_ith(ctx, idx) != n ||
                k <= 0 || k + 1 + (int32_t) topk_excl.size() > n) {
            return false;
        }

        topk_tmp.clear();
        for (int32_t i = 0; i < n; ++i) {
            if (!std::isfinite(lg[i])) {
                return false;
            }
            if (!std::binary_search(topk_excl.begin(), topk_excl.end(), ids[i])) {
                topk_tmp.push_back(lg[i]);
            }
        }
        if ((int32_t) topk_tmp.size() < k + 1) {
            return false;
        }
        std::partial_sort(topk_tmp.begin(), topk_tmp.begin() + k + 1, topk_tmp.end(), std::greater<float>());
        for (int32_t i = 0; i < k; ++i) {
            if (!(topk_tmp[i] > topk_tmp[i + 1])) {
                return false;
            }
        }

        cur.resize(n);
        for (int32_t i = 0; i < n; ++i) {
            cur[i] = llama_token_data{ids[i], lg[i], 0.0f};
        }

        cur_p = { cur.data(), cur.size(), -1, false };

        return true;
    }

    std::vector<float> topk_tmp = {};

    // [TAG_BACKEND_TOPK] common_reasoning_budget_apply changes the row only in REASONING_BUDGET_FORCING; in every
    // other state it returns before touching cur_p, so the chain sees the candidates exactly as without it. the
    // state is read per row, after the tokens accepted before this row
    bool rbudget_topk_ok() const {
        return rbudget == nullptr || common_reasoning_budget_get_state(rbudget) != REASONING_BUDGET_FORCING;
    }

    // [TAG_BACKEND_TOPK] a lazy grammar's apply returns before touching cur_p until its trigger fires (the trigger is
    // found when a token is accepted); once triggered it masks candidates, so its rows read the full logits. the state
    // is read per row, after the tokens accepted before this row
    bool grmr_topk_ok() const {
        return grmr == nullptr || llama_sampler_grammar_awaiting_trigger(grmr);
    }

    void set_logits_full(const float * logits, int n_vocab) {
        GGML_ASSERT(logits != nullptr);
        cur.resize(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    // grammar_masks: false when this row's draw does not apply the grammar (common_sampler_sample's first
    // draw, whose single-token check follows it), so a triggered lazy grammar does not need the full logits there
    void set_logits(struct llama_context * ctx, int idx, bool grammar_masks = true) {
        if (backend_topk != 0) {
            // [TAG_BACKEND_TOPK] a reasoning budget that is forcing its end sequence sets every token but one to -inf,
            // and the forced token need not be a candidate: those rows read the full logits (rbudget_topk_ok), as do
            // the rows after a lazy grammar has triggered (grmr_topk_ok) when the draw applies the grammar
            n_topk_rows++;
            if (backend_topk > 0 && backend_topk_check() != 2 && rbudget_topk_ok() && (grmr_topk_ok() || !grammar_masks) &&
                    set_logits_topk(ctx, idx)) {
                return;
            }
            n_topk_full++;
            set_logits_full(llama_get_logits_full_ith(ctx, idx), llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx))));
            return;
        }

        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    // [TAG_BACKEND_TOPK] rows sampled with backend candidates, and how many of them needed the full logits
    uint64_t n_topk_rows = 0;
    uint64_t n_topk_full = 0;

    // LLAMA_BACKEND_TOPK_CHECK: 1 also samples every candidate row from the full logits with a clone of the chain and
    // compares the two, 2 always uses the full logits (the fallback path)
    static int backend_topk_check() {
        static const int v = [] {
            const char * e = getenv("LLAMA_BACKEND_TOPK_CHECK");
            return e ? atoi(e) : 0;
        }();
        return v;
    }

    // LLAMA_TOPK_FIRST_DRAW (default 1): under a triggered lazy grammar, common_sampler_sample's first draw
    // (which does not apply the grammar) takes the backend candidates; the full logits are read only if its token fails
    // the grammar and the row is resampled. 0 reads them for the first draw too, as before
    static bool topk_first_draw() {
        static const bool v = [] {
            const char * e = getenv("LLAMA_TOPK_FIRST_DRAW");
            return e == nullptr || atoi(e) != 0;
        }();
        return v;
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        GGML_RT_SCOPE("grmr.accept"); // the lazy grammar's trigger scan runs here while it awaits its trigger
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    GGML_RT_SCOPE("cs.clone");
    auto * result = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
    };

    result->backend_topk = gsmpl->backend_topk;
    result->topk_excl    = gsmpl->topk_excl;

    return result;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    GGML_RT_SCOPE("cs.copy");
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->t_total_us = src->t_total_us;
    dst->backend_topk = src->backend_topk;
    dst->topk_excl    = src->topk_excl;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

// [TAG_BACKEND_TOPK]
static bool common_backend_topk_enabled() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_BACKEND_TOPK");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

static bool common_backend_topk_rbudget() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_BACKEND_TOPK_RBUDGET");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

static bool common_backend_topk_grammar() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_BACKEND_TOPK_GRAMMAR");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

// the tokens the chain's logit bias changes (the request's biases and the vocabulary's suppressed tokens, as
// common_sampler_init merges them), or false if a bias is not -inf
static bool common_sampler_logit_bias_excl(const struct common_sampler * gsmpl, const llama_vocab * vocab, std::vector<llama_token> & excl) {
    excl.clear();
    for (const auto & lb : gsmpl->params.logit_bias) {
        if (!(lb.bias == -INFINITY)) {
            return false;
        }
        excl.push_back(lb.token);
    }
    int32_t n_suppress = 0;
    const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
    excl.insert(excl.end(), suppress, suppress + n_suppress);
    std::sort(excl.begin(), excl.end());
    excl.erase(std::unique(excl.begin(), excl.end()), excl.end());
    return true;
}

int32_t common_sampler_topk_exact(const struct common_sampler * gsmpl, const struct llama_vocab * vocab, int32_t * n_cand,
        std::string * reason) {
    *n_cand = 0;

    // [TAG_BACKEND_TOPK] why the chain is refused, for the server's one-time log line
    const auto refuse = [&](std::string why) {
        if (reason) {
            *reason = std::move(why);
        }
        return 0;
    };
    if (reason) {
        reason->clear();
    }

    // a reasoning budget is eligible: it changes a row only while forcing, and those rows read the full logits
    // (common_sampler::rbudget_topk_ok). LLAMA_BACKEND_TOPK_RBUDGET=0 refuses it as before. so is a lazy grammar that
    // has not triggered: it changes no row until it triggers, and the rows after that read the full logits
    // (common_sampler::grmr_topk_ok). LLAMA_BACKEND_TOPK_GRAMMAR=0 refuses it as before. any other grammar refuses
    if (!gsmpl) {
        return refuse("no sampler");
    }
    if (!common_backend_topk_enabled()) {
        return refuse("LLAMA_BACKEND_TOPK=0");
    }
    if (gsmpl->grmr) {
        if (!gsmpl->params.grammar_lazy) {
            return refuse("a grammar");
        }
        if (!common_backend_topk_grammar()) {
            return refuse("a lazy grammar (the chat format's tool calls) with LLAMA_BACKEND_TOPK_GRAMMAR=0");
        }
        if (!llama_sampler_grammar_awaiting_trigger(gsmpl->grmr)) {
            return refuse("a lazy grammar (the chat format's tool calls) that is not awaiting its trigger");
        }
    }
    if (gsmpl->rbudget && !common_backend_topk_rbudget()) {
        return refuse("a reasoning budget with LLAMA_BACKEND_TOPK_RBUDGET=0");
    }

    const int32_t k = gsmpl->params.top_k;
    if (k <= 0 || k > 255) {
        return refuse("top_k = " + std::to_string(k) + " outside 1..255");
    }

    std::vector<llama_token> excl;
    const bool bias_ok = common_sampler_logit_bias_excl(gsmpl, vocab, excl);

    // every sampler before the first top-k must be an empty one ("?name"), or a logit bias that only sets tokens to
    // -inf, and that top-k is params.top_k
    const int n = llama_sampler_chain_n(gsmpl->chain);
    for (int i = 0; i < n; ++i) {
        const char * name = llama_sampler_name(llama_sampler_chain_get(gsmpl->chain, i));
        if (name[0] == '?') {
            continue;
        }
        if (name[0] == '+' || name[0] == '-') {
            name++;
        }
        if (strcmp(name, "logit-bias") == 0 && bias_ok && excl.size() <= 256) {
            continue;
        }
        if (strcmp(name, "top-k") != 0) {
            return refuse(std::string("sampler '") + name + "' before top-k");
        }
        *n_cand = k + 1 + (int32_t) excl.size();
        return k;
    }

    return refuse("no top-k in the chain");
}

struct llama_sampler * common_sampler_backend_topk_init(int32_t n) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(n));
    return chain;
}

int32_t common_sampler_backend_topk_n(const struct llama_vocab * vocab, const struct common_params_sampling & params) {
    if (!common_backend_topk_enabled() || params.top_k <= 0 || params.top_k > 255) {
        return 0;
    }

    std::vector<llama_token> excl;
    for (llama_token t = 0; t < llama_vocab_n_tokens(vocab); t++) {
        if (llama_vocab_is_eog(vocab, t)) {
            excl.push_back(t);
        }
    }
    int32_t n_suppress = 0;
    const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
    excl.insert(excl.end(), suppress, suppress + n_suppress);
    std::sort(excl.begin(), excl.end());
    excl.erase(std::unique(excl.begin(), excl.end()), excl.end());

    return params.top_k + 1 + (int32_t) excl.size();
}

void common_sampler_set_backend_topk(struct common_sampler * gsmpl, const struct llama_vocab * vocab, int32_t n) {
    if (!gsmpl) {
        return;
    }
    gsmpl->backend_topk = n;
    if (n > 0 && !common_sampler_logit_bias_excl(gsmpl, vocab, gsmpl->topk_excl)) {
        gsmpl->backend_topk = -1; // not reached: common_sampler_topk_exact checked the biases
    }
}

void common_sampler_backend_topk_stats(const struct common_sampler * gsmpl, uint64_t * n_rows, uint64_t * n_full) {
    *n_rows = gsmpl ? gsmpl->n_topk_rows : 0;
    *n_full = gsmpl ? gsmpl->n_topk_full : 0;
}

static void common_sampler_backend_topk_compare(common_sampler * gsmpl, llama_sampler * chain_ref, llama_context * ctx, int idx) {
    static uint64_t n_checked = 0;
    static uint64_t n_bad     = 0;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_full_ith(ctx, idx);
    GGML_ASSERT(logits != nullptr);

    std::vector<llama_token_data> ref(n_vocab);
    for (llama_token t = 0; t < n_vocab; t++) {
        ref[t] = llama_token_data{t, logits[t], 0.0f};
    }
    llama_token_data_array ref_p = { ref.data(), ref.size(), -1, false };
    llama_sampler_apply(chain_ref, &ref_p);

    const auto & cur_p = gsmpl->cur_p;

    bool same = ref_p.size == cur_p.size && ref_p.selected == cur_p.selected && ref_p.sorted == cur_p.sorted;
    for (size_t i = 0; same && i < ref_p.size; ++i) {
        same = ref_p.data[i].id == cur_p.data[i].id &&
               memcmp(&ref_p.data[i].logit, &cur_p.data[i].logit, sizeof(float)) == 0 &&
               memcmp(&ref_p.data[i].p,     &cur_p.data[i].p,     sizeof(float)) == 0;
    }

    n_checked++;
    if (!same) {
        n_bad++;
        LOG_WRN("%s: MISMATCH at row %d: candidates give %d (of %zu), full logits give %d (of %zu)\n", __func__, idx,
                cur_p.selected >= 0 ? cur_p.data[cur_p.selected].id : -1, cur_p.size,
                ref_p.selected >= 0 ? ref_p.data[ref_p.selected].id : -1, ref_p.size);
    }
    if (!same || n_checked % 1000 == 0) {
        LOG_INF("%s: checked %llu rows, %llu mismatches, this sampler %llu candidate rows, %llu full\n", __func__,
                (unsigned long long) n_checked, (unsigned long long) n_bad,
                (unsigned long long) gsmpl->n_topk_rows, (unsigned long long) gsmpl->n_topk_full);
    }
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    GGML_RT_SCOPE("cs.sample");
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // the first draw applies the grammar only with grammar_first; otherwise the backend candidates give the
    // chain's exact result on this row whatever the grammar's state (the resample below reads the full logits)
    gsmpl->set_logits(ctx, idx, !common_sampler::topk_first_draw() || (grammar_first && grammar_should_apply(gsmpl)));

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // [TAG_BACKEND_TOPK] check mode: the same chain state applied to the full logits must give the same result
    llama_sampler * chain_ref = nullptr;
    if (gsmpl->backend_topk > 0 && common_sampler::backend_topk_check() == 1 && cur_p.size < (size_t) llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)))) {
        chain_ref = llama_sampler_clone(chain);
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (chain_ref) {
        common_sampler_backend_topk_compare(gsmpl, chain_ref, ctx, idx);
        llama_sampler_free(chain_ref);
    }

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_RT_SCOPE("cs.accept_n");
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // upstream #29638: an accepted end-of-generation token ends the round, so no draft token after it is accepted
    // and no bonus row is sampled; it becomes the round's last token, as if the target had sampled it there
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id || llama_vocab_is_eog(vocab, id)) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

// [TAG_SPEC_COUPLED]
bool common_sampler_coupled_ok(const struct common_sampler * gsmpl) {
    if (!gsmpl || gsmpl->params.mirostat != 0 || gsmpl->params.backend_sampling) {
        return false;
    }
    const int n = llama_sampler_chain_n(gsmpl->chain);
    if (n <= 0) {
        return false;
    }
    const char * last = llama_sampler_name(llama_sampler_chain_get(gsmpl->chain, n - 1));
    if (last[0] == '+' || last[0] == '-') {
        last++;
    }
    if (strcmp(last, "dist") != 0) {
        return false;
    }
    for (const auto t : gsmpl->params.samplers) {
        if (t == COMMON_SAMPLER_TYPE_ADAPTIVE_P || (t == COMMON_SAMPLER_TYPE_XTC && gsmpl->params.xtc_probability > 0.0f)) {
            return false;
        }
    }
    return true;
}

void common_sampler_coupled_skip(struct common_sampler * gsmpl, llama_token tok) {
    llama_token_data       one   = { tok, 0.0f, 0.0f };
    llama_token_data_array one_p = { &one, 1, -1, false };
    llama_sampler_apply(gsmpl->chain, &one_p);
}

llama_token common_sampler_coupled_pick(struct common_sampler * gsmpl, const llama_token_data_array * cand, bool * forced) {
    *forced = false;

    const llama_token tok_forced = common_reasoning_budget_forced_token(gsmpl->rbudget);
    if (tok_forced != LLAMA_TOKEN_NULL) {
        common_sampler_coupled_skip(gsmpl, tok_forced);
        *forced = true;
        return tok_forced;
    }

    if (cand == nullptr || cand->size == 0 || (grammar_should_apply(gsmpl) && !llama_sampler_grammar_awaiting_trigger(gsmpl->grmr))) {
        common_sampler_coupled_skip(gsmpl, cand && cand->size > 0 ? cand->data[0].id : 0);
        return LLAMA_TOKEN_NULL;
    }

    auto & cur   = gsmpl->cur;
    auto & cur_p = gsmpl->cur_p;

    cur.resize(cand->size);
    for (size_t i = 0; i < cand->size; ++i) {
        cur[i] = llama_token_data{ cand->data[i].id, cand->data[i].logit, 0.0f };
    }
    cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler_apply(gsmpl->chain, &cur_p);

    if (cur_p.size == 0 || cur_p.selected < 0 || (size_t) cur_p.selected >= cur_p.size) {
        return LLAMA_TOKEN_NULL; // not reached with dist, which draws and selects whenever a candidate is left
    }

    return cur_p.data[cur_p.selected].id;
}

// [TAG_SPEC_REJECTION]
bool common_sampler_rejection_ok(const struct common_sampler * gsmpl) {
    return common_sampler_coupled_ok(gsmpl);
}

void common_sampler_accept_draft(struct common_sampler * gsmpl, llama_token token) {
    if (gsmpl && gsmpl->grmr && grammar_should_apply(gsmpl)) {
        llama_token_data       one   = { token, 1.0f, 0.0f };
        llama_token_data_array one_p = { &one, 1, -1, false };
        llama_sampler_apply(gsmpl->grmr, &one_p);
        if (one.logit == -INFINITY) {
            // everything common_sampler_accept does but the grammar (whose end-sequence replay needs a grammar that
            // does not apply, so it is not reached here)
            if (gsmpl->rbudget) {
                llama_sampler_accept(gsmpl->rbudget, token);
            }
            llama_sampler_accept(gsmpl->chain, token);
            gsmpl->prev.push_back(token);
            return;
        }
    }
    common_sampler_accept(gsmpl, token, true);
}

// the row can take the rejection step: no reasoning budget forcing its end sequence, and no grammar that applies
// (a lazy grammar awaiting its trigger changes no row). the state is read per row, after the tokens accepted before it
static bool common_sampler_rejection_row_ok(const struct common_sampler * gsmpl) {
    return gsmpl->rbudget_topk_ok() && !(grammar_should_apply(const_cast<common_sampler *>(gsmpl)) && !llama_sampler_grammar_awaiting_trigger(gsmpl->grmr));
}

// cur_p through every sampler of the chain but its final dist, then (temp > 0, and the chain's own temperature
// positive) the logits scaled by the chain's temperature / temp, then the softmax dist would take: p = exp(l - max) /
// sum, 0 for a -inf logit. false when no candidate is left
static bool common_sampler_rejection_probs(struct common_sampler * gsmpl, llama_token_data_array * cur_p, float temp) {
    const int n = llama_sampler_chain_n(gsmpl->chain);
    for (int i = 0; i + 1 < n; ++i) {
        llama_sampler_apply(llama_sampler_chain_get(gsmpl->chain, i), cur_p);
    }
    if (cur_p->size == 0) {
        return false;
    }
    if (temp > 0.0f && gsmpl->params.temp > 0.0f && temp != gsmpl->params.temp) {
        const float s = gsmpl->params.temp / temp;
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (std::isfinite(cur_p->data[i].logit)) {
                cur_p->data[i].logit *= s;
            }
        }
    }
    float max_l = -INFINITY;
    for (size_t i = 0; i < cur_p->size; ++i) {
        max_l = std::max(max_l, cur_p->data[i].logit);
    }
    if (!std::isfinite(max_l)) {
        return false;
    }
    double sum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float p = std::isfinite(cur_p->data[i].logit) ? expf(cur_p->data[i].logit - max_l) : 0.0f;
        cur_p->data[i].p = p;
        sum += p;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p = (float) (cur_p->data[i].p / sum);
    }
    return true;
}

// the index of the entry the inverse CDF of weights w (in array order) gives for a uniform u: the first entry with
// w > 0 whose running sum reaches u * the total; -1 when nothing has weight
template <typename W>
static int common_sampler_rejection_icdf(size_t n, const W & w, double u) {
    double sum = 0.0;
    int last = -1;
    for (size_t i = 0; i < n; ++i) {
        if (w(i) > 0.0) {
            sum += w(i);
            last = (int) i;
        }
    }
    if (last < 0) {
        return -1;
    }
    const double tgt = u * sum;
    double run = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (w(i) > 0.0) {
            run += w(i);
            if (run >= tgt) {
                return (int) i;
            }
        }
    }
    return last;
}

llama_token common_sampler_rejection_draft(struct common_sampler * copy, struct common_sampler * rng,
        const llama_token_data_array * cand, float temp, common_rejection_q & q, llama_token * forced, std::mt19937 * gen) {
    q.clear();
    *forced = common_reasoning_budget_forced_token(copy->rbudget);
    if (*forced != LLAMA_TOKEN_NULL || cand == nullptr || cand->size == 0 || !common_sampler_rejection_row_ok(copy)) {
        return LLAMA_TOKEN_NULL;
    }

    auto & cur   = copy->cur;
    auto & cur_p = copy->cur_p;

    cur.resize(cand->size);
    for (size_t i = 0; i < cand->size; ++i) {
        cur[i] = llama_token_data{ cand->data[i].id, cand->data[i].logit, 0.0f };
    }
    cur_p = { cur.data(), cur.size(), -1, false };

    double u = 0.0;
    if (!common_sampler_rejection_probs(copy, &cur_p, temp)) {
        return LLAMA_TOKEN_NULL;
    }
    if (gen != nullptr) {
        // [TAG_SPEC_REJECTION_PIPE] the draft's own stream, the same kind of draw as the chain's dist
        std::uniform_real_distribution<double> dist(0.0f, 1.0f);
        u = dist(*gen);
    } else if (rng == nullptr || !llama_sampler_chain_draw_uniform(rng->chain, &u)) {
        return LLAMA_TOKEN_NULL;
    }

    const int sel = common_sampler_rejection_icdf(cur_p.size, [&](size_t i) { return (double) cur_p.data[i].p; }, u);
    if (sel < 0) {
        return LLAMA_TOKEN_NULL;
    }
    cur_p.selected = sel;

    for (size_t i = 0; i < cur_p.size; ++i) {
        if (cur_p.data[i].p > 0.0f) {
            q.push_back(cur_p.data[i]);
        }
    }

    return cur_p.data[sel].id;
}

bool common_sampler_device_chain(const struct common_sampler * gsmpl, const struct llama_vocab * vocab, float temp, float * prm) {
    if (!common_sampler_coupled_ok(gsmpl)) {
        return false;
    }
    const auto & p = gsmpl->params;
    for (int i = 0; i < 32; ++i) {
        prm[i] = 0.0f;
    }
    prm[0] = 1.0f;
    prm[1] = (float) p.top_k;
    prm[2] = p.top_p;
    prm[3] = p.min_p;
    prm[4] = p.temp;
    prm[5] = (float) p.min_keep;
    prm[28] = temp;

    // the logit bias of common_sampler_init: the request's, then the vocabulary's suppressed tokens
    std::vector<llama_logit_bias> bias = p.logit_bias;
    int32_t n_suppress = 0;
    const llama_token * suppress = vocab ? llama_vocab_get_suppress_tokens(vocab, &n_suppress) : nullptr;
    for (int32_t i = 0; i < n_suppress; ++i) {
        bias.push_back({ suppress[i], -INFINITY });
    }
    if (bias.size() > 8) {
        return false;
    }
    prm[7] = (float) bias.size();
    for (size_t b = 0; b < bias.size(); ++b) {
        prm[8 + 2*b]     = (float) bias[b].token;
        prm[8 + 2*b + 1] = bias[b].bias;
    }

    int n_order = 0;
    for (const auto t : p.samplers) {
        switch (t) {
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (!(p.penalty_last_n == 0 || (p.penalty_repeat == 1.0f && p.penalty_freq == 0.0f && p.penalty_present == 0.0f))) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (!(p.dry_multiplier == 0.0f || p.dry_base < 1.0f || p.dry_penalty_last_n == 0)) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (p.top_n_sigma > 0.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                if (p.typ_p < 1.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                // common_sampler_coupled_ok: no XTC with a probability
                break;
            case COMMON_SAMPLER_TYPE_TOP_K:
            case COMMON_SAMPLER_TYPE_TOP_P:
            case COMMON_SAMPLER_TYPE_MIN_P:
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                if (n_order >= 4 || (t == COMMON_SAMPLER_TYPE_TEMPERATURE && p.dynatemp_range > 0.0f)) {
                    return false;
                }
                prm[24 + n_order++] = t == COMMON_SAMPLER_TYPE_TOP_K ? 1.0f : t == COMMON_SAMPLER_TYPE_TOP_P ? 2.0f :
                                      t == COMMON_SAMPLER_TYPE_MIN_P ? 3.0f : 4.0f;
                break;
            default:
                return false;
        }
    }
    return true;
}

bool common_sampler_draw_uniform(struct common_sampler * gsmpl, double * u) {
    return gsmpl != nullptr && llama_sampler_chain_draw_uniform(gsmpl->chain, u);
}

bool common_sampler_rejection_row_ready(struct common_sampler * gsmpl) {
    return common_reasoning_budget_forced_token(gsmpl->rbudget) == LLAMA_TOKEN_NULL && common_sampler_rejection_row_ok(gsmpl);
}

static float common_rejection_q_of(const common_rejection_q & q, llama_token id) {
    for (const auto & e : q) {
        if (e.id == id) {
            return e.p;
        }
    }
    return 0.0f;
}

llama_token common_sampler_rejection_step(const llama_token_data * p, size_t n_p, const common_rejection_q & q, llama_token x,
        const std::function<double()> & draw, bool * accepted) {
    float px = 0.0f;
    for (size_t i = 0; i < n_p; ++i) {
        if (p[i].id == x) {
            px = p[i].p;
            break;
        }
    }
    const float qx = common_rejection_q_of(q, x);

    // accept with probability min(1, px/qx): u < px/qx, written without the division
    const double u = draw();
    if (u * (double) qx < (double) px) {
        *accepted = true;
        return x;
    }
    *accepted = false;

    // the residual max(0, p - q); p itself if rounding left it no mass
    const double u2 = draw();
    int sel = common_sampler_rejection_icdf(n_p, [&](size_t i) { return std::max(0.0, (double) p[i].p - (double) common_rejection_q_of(q, p[i].id)); }, u2);
    if (sel < 0) {
        sel = common_sampler_rejection_icdf(n_p, [&](size_t i) { return (double) p[i].p; }, u2);
    }
    GGML_ASSERT(sel >= 0 && "rejection step: the target's distribution has no mass");
    return p[sel].id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<common_rejection_q> & q, int * n_step, int * n_accepted) {
    GGML_RT_SCOPE("cs.accept_n_rejection");
    // [TAG_SPEC_REJECTION_PIPE] idxs.size() == draft.size(): no bonus row
    GGML_ASSERT((idxs.size() == draft.size() + 1 || idxs.size() == draft.size()) && "idxs.size() must be draft.size() + 1 or draft.size()");
    GGML_ASSERT(q.size() == draft.size() && "one q row per drafted token");

    const auto draw = [&]() {
        double u = 0.0;
        const bool ok = llama_sampler_chain_draw_uniform(gsmpl->chain, &u);
        GGML_ASSERT(ok && "the rejection step needs a chain that ends in dist");
        return u;
    };

    if (n_step) {
        *n_step = 0;
    }

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        llama_token id = LLAMA_TOKEN_NULL;
        bool accepted = false;

        // [TAG_SPEC_REJECTION_PIPE] a row whose q lacks its drafted token (not sampled from q) takes sample-and-match
        if (common_sampler_rejection_row_ok(gsmpl) && common_rejection_q_of(q[i], draft[i]) > 0.0f) {
            llama_synchronize(ctx);

            const auto tm = gsmpl->tm();

            auto & cur_p = gsmpl->cur_p;

            gsmpl->set_logits(ctx, idxs[i]);
            llama_sampler_apply(gsmpl->rbudget, &cur_p);

            if (common_sampler_rejection_probs(gsmpl, &cur_p, 0.0f)) {
                id = common_sampler_rejection_step(cur_p.data, cur_p.size, q[i], draft[i], draw, &accepted);
                for (size_t k = 0; k < cur_p.size; ++k) {
                    if (cur_p.data[k].id == id) {
                        cur_p.selected = k;
                        break;
                    }
                }
                if (n_step) {
                    (*n_step)++;
                }
            }
        }

        if (id == LLAMA_TOKEN_NULL) {
            // today's sample-and-match at this row
            id = common_sampler_sample(gsmpl, ctx, idxs[i]);
            accepted = id == draft[i];
        }

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // upstream #29638: an accepted end-of-generation token ends the round (see common_sampler_sample_and_accept_n)
        if (!accepted || llama_vocab_is_eog(vocab, id)) {
            break;
        }
    }

    if (n_accepted) {
        *n_accepted = (int) i;
    }

    if (i == draft.size() && idxs.size() > draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i]);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

// [TAG_SPEC_BLOCK] block verification: Sun et al., "Block Verification Accelerates Speculative Decoding"
// (arXiv 2403.10444), Algorithm 2, with the paper's p_i as w. Every uniform comes from draw: gamma of them first (eta_1 ..
// eta_gamma), then one for the residual when tau < gamma. In the paper's terms, row k is M_b(. | c, X^k) for p and
// M_s(. | c, X^k) for q, and draft[k] is X_{k+1}.
int common_spec_block_verify(const std::vector<const llama_token_data *> & p, const std::vector<size_t> & n_p,
        const std::vector<common_rejection_q> & q, const llama_tokens & draft, const std::function<double()> & draw, llama_token * y) {
    const int gamma = (int) draft.size();
    GGML_ASSERT(gamma > 0 && (int) p.size() >= gamma && (int) n_p.size() >= gamma && (int) q.size() >= gamma);

    const auto p_of = [&](int k, llama_token id) {
        for (size_t j = 0; j < n_p[k]; ++j) {
            if (p[k][j].id == id) {
                return (double) p[k][j].p;
            }
        }
        return 0.0;
    };
    // the residual's weight of row k's entry j at w: max(w * p - q, 0); only the target's entries can be positive
    const auto res_w = [&](int k, double w, size_t j) {
        return std::max(0.0, w * (double) p[k][j].p - (double) common_rejection_q_of(q[k], p[k][j].id));
    };

    std::vector<double> eta(gamma);
    for (int i = 0; i < gamma; ++i) {
        eta[i] = draw();
    }

    int    tau   = 0;
    double w     = 1.0; // p_0
    double w_tau = 1.0;
    for (int i = 1; i <= gamma; ++i) {
        // p_i = min(p_{i-1} * M_b(X_i) / M_s(X_i), 1), both at row i - 1
        const double qx = common_rejection_q_of(q[i - 1], draft[i - 1]);
        w = qx > 0.0 ? std::min(w * p_of(i - 1, draft[i - 1]) / qx, 1.0) : 0.0;
        // h_gamma = p_gamma; h_i = S / (S + 1 - p_i) with S = sum_x max(p_i * M_b(x) - M_s(x), 0) at row i
        double h = w;
        if (i < gamma) {
            double s = 0.0;
            for (size_t j = 0; j < n_p[i]; ++j) {
                s += res_w(i, w, j);
            }
            const double den = s + 1.0 - w;
            h = den > 0.0 ? s / den : 1.0;
        }
        // tau is the last i that passes, not the first that fails: the loop does not stop. strict, as the token step's
        // test is: a uniform of exactly 0 must not keep a draft whose h is 0
        if (eta[i - 1] < h) {
            tau   = i;
            w_tau = w;
        }
    }

    *y = LLAMA_TOKEN_NULL;
    if (tau < gamma) {
        const double u = draw();
        int sel = common_sampler_rejection_icdf(n_p[tau], [&](size_t j) { return res_w(tau, w_tau, j); }, u);
        if (sel < 0) {
            sel = common_sampler_rejection_icdf(n_p[tau], [&](size_t j) { return (double) p[tau][j].p; }, u);
        }
        GGML_ASSERT(sel >= 0 && "block verification: the target's distribution has no mass");
        *y = p[tau][sel].id;
    }
    return tau;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<common_rejection_q> & q, int * n_step,
        int * n_accepted, bool * took) {
    GGML_RT_SCOPE("cs.accept_n_block");
    *took = false;
    const size_t gamma = draft.size();
    if (gamma == 0 || idxs.size() != gamma + 1 || q.size() != gamma) {
        return common_sampler_sample_and_accept_n_rejection(gsmpl, ctx, idxs, draft, q, n_step, n_accepted);
    }

    llama_synchronize(ctx);

    // the target's rows 0 .. gamma - 1, each read as the token step reads it on reaching that row: through a copy of
    // gsmpl that has accepted the drafts before it. gsmpl itself is untouched until the decision, so a row that cannot
    // take the step hands the whole round to the token step unchanged
    std::vector<std::vector<llama_token_data>> rows(gamma);
    {
        common_sampler_ptr copy(common_sampler_clone(gsmpl));
        for (size_t k = 0; k < gamma; ++k) {
            if (!common_sampler_rejection_row_ok(copy.get()) || common_rejection_q_of(q[k], draft[k]) <= 0.0f) {
                return common_sampler_sample_and_accept_n_rejection(gsmpl, ctx, idxs, draft, q, n_step, n_accepted);
            }
            auto & cur_p = copy->cur_p;
            copy->set_logits(ctx, idxs[k]);
            llama_sampler_apply(copy->rbudget, &cur_p);
            if (!common_sampler_rejection_probs(copy.get(), &cur_p, 0.0f)) {
                return common_sampler_sample_and_accept_n_rejection(gsmpl, ctx, idxs, draft, q, n_step, n_accepted);
            }
            rows[k].assign(cur_p.data, cur_p.data + cur_p.size);
            common_sampler_accept(copy.get(), draft[k], true);
        }
        // the copy read every row: count them on the sampler, as the token step's reads are counted
        gsmpl->n_topk_rows += copy->n_topk_rows;
        gsmpl->n_topk_full += copy->n_topk_full;
    }

    const auto draw = [&]() {
        double u = 0.0;
        const bool ok = llama_sampler_chain_draw_uniform(gsmpl->chain, &u);
        GGML_ASSERT(ok && "block verification needs a chain that ends in dist");
        return u;
    };

    std::vector<const llama_token_data *> p(gamma);
    std::vector<size_t> n_p(gamma);
    for (size_t k = 0; k < gamma; ++k) {
        p[k]   = rows[k].data();
        n_p[k] = rows[k].size();
    }
    llama_token y = LLAMA_TOKEN_NULL;
    const int tau = common_spec_block_verify(p, n_p, q, draft, draw, &y);
    *took = true;

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::vector<llama_token> result;
    result.reserve(tau + 1);
    for (int k = 0; k < tau; ++k) {
        common_sampler_accept(gsmpl, draft[k], true);
        result.push_back(draft[k]);
        // upstream #29638: a kept end-of-generation token ends the round, as in the token step
        if (llama_vocab_is_eog(vocab, draft[k])) {
            if (n_step) {
                *n_step = (int) result.size();
            }
            if (n_accepted) {
                *n_accepted = k;
            }
            return result;
        }
    }
    if (tau == (int) gamma) {
        y = common_sampler_sample(gsmpl, ctx, idxs[gamma]); // the bonus row, as the token step samples it
    }
    common_sampler_accept(gsmpl, y, true);
    result.push_back(y);

    if (n_step) {
        *n_step = (int) std::min(result.size(), gamma);
    }
    if (n_accepted) {
        *n_accepted = tau;
    }
    return result;
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
