#pragma once

#include "llama.h"
#include "common.h"

#include <random>

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // [TAG_SPEC_COUPLED] the target's sampler, at the row that samples the first draft token (LLAMA_SPEC_COUPLED:
    // the MTP draft picks each token through a copy of it)
    struct common_sampler * smpl_tgt = nullptr;

    // [TAG_SPEC_REJECTION] with LLAMA_SPEC_REJECTION, the MTP draft samples each token from its distribution q through
    // a copy of smpl_tgt, drawing from smpl_tgt's own generator, and puts q here, one row per drafted token (an empty
    // row where it could not sample: see common_rejection_q). nullptr: the draft picks as without the toggle
    std::vector<std::vector<llama_token_data>> * q = nullptr;

    // [TAG_SPEC_REJECTION_PIPE] with q: the draft's uniforms come from this generator instead of smpl_tgt's
    std::mt19937 * q_rng = nullptr;

    // [TAG_SPEC_ADAPT_WIDTH] with the DFlash draft: keep only the first n_cap drafted tokens (0: no cap)
    int32_t n_cap = 0;
};

// [TAG_SPEC_COUPLED] coupled drafting, LLAMA_SPEC_COUPLED=1, default off: the MTP draft picks each
// token from its candidates through a copy of the target's sampler, with the draw the target's sample of that row
// will take, instead of its argmax
bool common_speculative_coupled();

// [TAG_SPEC_REJECTION] rejection sampling, LLAMA_SPEC_REJECTION, default on (0: off): the MTP draft samples its
// tokens from its own distribution and the verify keeps the target's output distribution with the rejection step
// (common_sampler_sample_and_accept_n_rejection). LLAMA_SPEC_REJECTION_TEMP: the draft's temperature (default: the
// target's). changes seeded outputs, not their distribution
bool  common_speculative_rejection();
float common_speculative_rejection_temp();

// [TAG_SPEC_REJECTION_ADAPT] the adaptive draft length under rejection sampling, LLAMA_SPEC_REJECTION_ADAPT=1
// with LLAMA_SPEC_REJECTION=1, default off: a request that takes the rejection step drafts up to
// LLAMA_SPEC_REJECTION_NMAX tokens (default 3, clamped to 1..3, and at least --spec-draft-n-max), and drafts token
// k+1 only while the product of the draft's q over its k drafted tokens is at least LLAMA_SPEC_REJECTION_CUTOFF
// (default 0.5 = c / T * E[A]: one more drafted and verified token costs c = 5.0 ms against a round of T = 25 ms
// yielding E[A] = 2.56 tokens). The decision reads only the tokens already drafted, never the
// next one, so the output distribution is kept. the returned depth is 0 when off; the target's n_rs_seq covers it
int32_t common_speculative_rejection_nmax();
float   common_speculative_rejection_cutoff();

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// optionally call before prefilling a prompt: it ends at position n_end (exclusive), and marks are the positions at
// which a checkpoint may be taken (a checkpoint holds the state before that position). A draft whose attention reaches back a
// fixed window (DFlash2's sliding window) then injects only the prompt positions a later draft can see: those within the window
// before n_end or before any mark. The plan holds until common_speculative_begin
void common_speculative_prefill_plan(common_speculative * spec, llama_seq_id seq_id, llama_pos n_end, const std::vector<llama_pos> & marks);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// [TAG_SPEC_PIPELINE] pipelined drafting, for a single MTP implementation without chained heads
// enable it; false when the implementation does not support it
bool common_speculative_pipe_enable(common_speculative * spec);
// the draft's probability of each token drafted since the last common_speculative_draft, chains included
const std::vector<float> * common_speculative_pipe_probs(const common_speculative * spec, llama_seq_id seq_id);
// the catch-up of one verified chunk of n tokens at pos0 (the target's current outputs hold its rows), plus an extra
// token after it (LLAMA_TOKEN_NULL for none) to chain from; clears the draft cache from pos0 first
bool common_speculative_pipe_process(common_speculative * spec, llama_seq_id seq_id, const llama_token * toks, int32_t n, llama_pos pos0, llama_token extra);
// [TAG_SPEC_REJECTION_PIPE] after a catch-up: the next chain's first known token pairs with the target's
// row for the last accepted token, as the serial loop's draft pairs its first token
bool common_speculative_pipe_rebase(common_speculative * spec, llama_seq_id seq_id);
// continue the chain: feed known tokens from pos (the first pairs with the kept hidden row), then draft n_new tokens.
// [TAG_SPEC_COUPLED] with coupled (a copy of the target's sampler at the first new token's row), each new token is
// picked from the draft's candidates through it (common_sampler_coupled_pick, the draft's argmax where that returns
// no token) and accepted into it; the draft's probability of the picked token goes to common_speculative_pipe_probs
// [TAG_SPEC_REJECTION_PIPE] with q and q_rng as well, each new token is instead sampled from the draft's
// distribution through coupled (common_sampler_rejection_draft) with a draw from q_rng, its row appended to q (an
// empty row where it could not sample), and the draft's probability of it goes to common_speculative_pipe_probs
bool common_speculative_pipe_chain(common_speculative * spec, llama_seq_id seq_id, const llama_token * known, int32_t n_known, llama_pos pos, int32_t n_new, llama_tokens & out,
        struct common_sampler * coupled = nullptr, std::mt19937 * q_rng = nullptr, std::vector<std::vector<llama_token_data>> * q = nullptr);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
// MTP's state is its carried target row with its position (common/mtp-carry.h). get_state gives nothing for a
// sequence without a valid carry; set_state with an empty, malformed or foreign record resets the sequence's state instead
// of leaving the previous one in place
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// a lineage boundary for seq_id: what the sequence holds was cleared or replaced (a cleared slot, a restored
// slot file, a shift), so the state carried for it is dropped; its next input at position 0 takes the fresh boundary
void common_speculative_reset_state(common_speculative * spec, llama_seq_id seq_id);

// whether the state carried for seq_id lets it continue at position pos_next (its next input): true when no
// implementation carries state across batches, at position 0, or when the carried row is for pos_next - 1
bool common_speculative_state_valid(const common_speculative * spec, llama_seq_id seq_id, llama_pos pos_next);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
