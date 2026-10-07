#pragma once

#include "llama.h"

#include "common.h"

#include <functional>
#include <random>
#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// [TAG_BACKEND_TOPK] exact backend top-k (toggle LLAMA_BACKEND_TOPK, default on)
//
// returns k when the sampler's result depends only on the k highest logits, in order: no grammar other than a lazy one
// that has not triggered (toggle LLAMA_BACKEND_TOPK_GRAMMAR, default on), a reasoning budget only with
// LLAMA_BACKEND_TOPK_RBUDGET on (the default), and every sampler before the chain's first top-k is empty or a logit
// bias that only sets tokens to -inf. rows where the grammar has triggered or the budget is forcing read the full logits.
// *n_cand is then the number of backend candidates that makes it exact: k + 1 + the number of those tokens.
// returns 0 otherwise, or with the toggle off. reason (optional) names why it returns 0, and is empty otherwise.
int32_t common_sampler_topk_exact(const struct common_sampler * gsmpl, const struct llama_vocab * vocab, int32_t * n_cand,
        std::string * reason = nullptr);

// a backend sampler chain (for llama_set_sampler) holding only top-k(n)
struct llama_sampler * common_sampler_backend_topk_init(int32_t n);

// the number of candidates the context's backend top-k returns: enough for the default sampling (top_k of params)
// with every end-of-generation token biased to -inf (ignore_eos) and the vocabulary's suppressed tokens, or 0 when
// the toggle is off or top_k is not in [1, 255]
int32_t common_sampler_backend_topk_n(const struct llama_vocab * vocab, const struct common_params_sampling & params);

// the context returns the top n candidates for this sampler's rows: with n >= *n_cand of common_sampler_topk_exact,
// sample from them when their k + 1 highest unbiased logits are strictly ordered, and from the full logits
// otherwise. n < 0: the rows carry candidates but this sampler always reads the full logits. 0: no candidates.
void common_sampler_set_backend_topk(struct common_sampler * gsmpl, const struct llama_vocab * vocab, int32_t n);

// rows sampled with backend top-k candidates, and how many of those fell back to the full logits
void common_sampler_backend_topk_stats(const struct common_sampler * gsmpl, uint64_t * n_rows, uint64_t * n_full);

// [TAG_SPEC_COUPLED] coupled drafting: a copy of the target's sampler (common_sampler_clone) picks a
// draft token from the draft's candidates with the random draw the target's own sample will take at that row.
// true when every sampled row takes exactly one draw that can be followed without the row's logits: the chain ends
// in dist, with no other random sampler (mirostat, adaptive-p, an active xtc) and no backend sampling
bool common_sampler_coupled_ok(const struct common_sampler * gsmpl);
// take the draw of a row the target samples without its logits (an accepted draft token's row, tok): the chain
// applied to one candidate draws exactly once, as it does on the row itself. the caller then accepts the token
void common_sampler_coupled_skip(struct common_sampler * gsmpl, llama_token tok);
// the token the sampler picks at its next row from the draft's candidates, taking that row's draw; the caller then
// accepts the token. LLAMA_TOKEN_NULL (the draw still taken) when the pick cannot follow the target: a grammar that
// applies (anything but a lazy grammar awaiting its trigger), or no candidate left. a reasoning budget that is
// forcing returns its forced token, with *forced set
llama_token common_sampler_coupled_pick(struct common_sampler * gsmpl, const llama_token_data_array * cand, bool * forced);

// [TAG_SPEC_REJECTION] rejection sampling for the MTP draft, LLAMA_SPEC_REJECTION, default on (0: off). the
// draft samples each token x from q, its candidates through a copy of the target's sampler, and the verify accepts x
// with probability min(1, p(x)/q(x)), p being the target's distribution after its chain; on a rejection it samples
// the normalized residual max(0, p - q) and stops, and after every draft is accepted it samples the bonus token from
// p. the emitted tokens then follow p exactly. every uniform comes from the target sampler's dist generator, so a
// seed still fixes the output, but a different output than the sample-and-match verify gives

// a q row: the draft's distribution at one drafted position (id, p), p > 0 and summing to 1. empty when the
// position's token was not sampled from a distribution (forced by the reasoning budget, a grammar that applies, or
// no candidate left): the verify then takes today's sample-and-match path for the whole round
typedef std::vector<llama_token_data> common_rejection_q;

// true when the sampler's rows can take the rejection step: common_sampler_coupled_ok (the chain ends in dist, no
// other random sampler, no backend sampling)
bool common_sampler_rejection_ok(const struct common_sampler * gsmpl);

// [TAG_SPEC_REJECTION] accept a drafted token into a copy of the target's sampler, as
// common_sampler_accept(gsmpl, token, true) does, except that its grammar takes the token only when the token fits it.
// where the copy cannot follow the target (a triggered grammar constrains the row) the draft proposes its own argmax,
// which need not fit, and the grammar would throw on it (a tool call's <|im_end|> followed by the draft's
// <|im_start|>). the target never accepts such a token, since it samples those rows through the grammar, so what the
// copy holds after it is never used
void common_sampler_accept_draft(struct common_sampler * gsmpl, llama_token token);

// the draft's pick at its next row: the candidates cand through copy (a copy of the target's sampler, advanced to
// this row), every sampler but the final dist, with the logits then scaled by copy's temperature / temp when temp > 0
// and both are positive; q gets that distribution, and the token is drawn from it with one uniform from rng's dist
// generator (the target's own sampler). LLAMA_TOKEN_NULL with q empty and no draw when the copy cannot follow the
// target at this row: a forcing reasoning budget (*forced gets its token), a grammar that applies, or no candidate
// left. the caller then accepts the token into copy
// [TAG_SPEC_REJECTION_PIPE] with gen, the uniform comes from gen instead of rng's dist generator
llama_token common_sampler_rejection_draft(struct common_sampler * copy, struct common_sampler * rng,
        const llama_token_data_array * cand, float temp, common_rejection_q & q, llama_token * forced, std::mt19937 * gen = nullptr);

// the accept and residual step at one position: p the target's distribution (ids and probabilities, summing to 1,
// zero entries allowed), q the draft's row, x the drafted token. returns x with probability min(1, p(x)/q(x)),
// taking one uniform from draw; otherwise a token from the normalized residual max(0, p - q), taking a second one
// (from p itself if the residual has no mass, which only rounding can cause). *accepted tells which
llama_token common_sampler_rejection_step(const llama_token_data * p, size_t n_p, const common_rejection_q & q, llama_token x,
        const std::function<double()> & draw, bool * accepted);

// the verify with the rejection step: like common_sampler_sample_and_accept_n, the emitted tokens accepted into gsmpl
// one by one. requires q.size() == draft.size(), every row non-empty and holding its drafted token. a row where the
// step cannot follow the target's sample (a grammar that applies, a forcing reasoning budget, the chain leaving no
// candidate) takes today's sample-and-match at that row. *n_step: rows that took the rejection step.
// [TAG_SPEC_REJECTION_PIPE] idxs.size() == draft.size(): no bonus row, the round ends after the last
// draft (the pipeline's guess of the bonus token is that draft); a row whose q lacks its drafted token takes
// sample-and-match. *n_accepted: the drafted tokens accepted
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<common_rejection_q> & q, int * n_step = nullptr,
        int * n_accepted = nullptr);

// [TAG_SPEC_BLOCK] block verification (Sun et al., "Block Verification Accelerates Speculative Decoding",
// arXiv 2403.10444, Algorithm 2): the drafted block is judged as a whole, so a draft the token step would reject can be
// kept when the drafts before it carry enough of the target's mass, and the emitted tokens still follow the target's
// distribution exactly. p[k] (n_p[k] entries) is the target's distribution at row k (after the first k drafts) and q[k]
// the draft's row the k-th draft was drawn from, k = 0 .. gamma - 1, gamma = draft.size(). Takes gamma uniforms from
// draw, then one more when tau < gamma. Returns tau, the drafts kept; when tau < gamma, *y gets the token drawn from the
// block residual max(0, p_tau * p[tau] - q[tau]) (p[tau] itself if that has no mass, which only rounding can cause);
// when tau == gamma, *y is LLAMA_TOKEN_NULL and the caller samples the bonus row
int common_spec_block_verify(const std::vector<const llama_token_data *> & p, const std::vector<size_t> & n_p,
        const std::vector<common_rejection_q> & q, const llama_tokens & draft, const std::function<double()> & draw, llama_token * y);

// LLAMA_SPEC_BLOCK_VERIFY is read by the server (server-context.cpp): unset, on for the 27B's architecture only; set,
// on when nonzero for any target. Off, the verify takes common_sampler_sample_and_accept_n_rejection, bit for bit as before

// the verify with block verification, for a round with a bonus row (idxs.size() == draft.size() + 1): the target's rows
// 0 .. gamma - 1 read through a copy of gsmpl that accepts the drafts in turn, then common_spec_block_verify, then the kept
// drafts and the last token accepted into gsmpl. The whole round takes the token step instead, exactly as
// common_sampler_sample_and_accept_n_rejection, when there is no bonus row or any row cannot take the rejection step
// (*took false). *n_step: the rows the step covered; *n_accepted: the drafts kept (a kept end-of-generation draft ends
// the round there, as in the token step)
std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<common_rejection_q> & q, int * n_step,
        int * n_accepted, bool * took);

// the MTP draft's steps chained on the device (llama_mtp_chain_set): the target's chain but its dist as the
// device pick's parameters (prm[0, 32), ggml-draft-pick.h, mode 1; the draw prm[6] left 0) when the pick does every one of
// its samplers that is not neutral (logit bias of at most 8 tokens, top-k, top-p, min-p, temperature without a dynamic
// range); false otherwise. temp: LLAMA_SPEC_REJECTION_TEMP
bool common_sampler_device_chain(const struct common_sampler * gsmpl, const struct llama_vocab * vocab, float temp, float * prm);

// one uniform from the sampler's dist generator, as common_sampler_rejection_draft takes it
bool common_sampler_draw_uniform(struct common_sampler * gsmpl, double * u);

// whether the sampler's next row takes common_sampler_rejection_draft's pick: no token forced by the
// reasoning budget, and no grammar that applies
bool common_sampler_rejection_row_ready(struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
