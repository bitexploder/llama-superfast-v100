#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// [TAG_DFLASH2_FEAT_DEV] a target keeps the enabled layer inputs of a decode that is one ubatch of at most n_rows_max
// tokens on the device instead of copying them to the host (0 = never); llama_get_embeddings_layer_inp_dev_rows gives the last
// decode's row count held so (0: they went to the host, read them with llama_get_embeddings_layer_inp)
LLAMA_API void    llama_set_embeddings_layer_inp_dev     (struct llama_context * ctx, int32_t n_rows_max);
LLAMA_API int32_t llama_get_embeddings_layer_inp_dev_rows(struct llama_context * ctx);
// ... and a DFlash2 draft's next embd decodes take their features from its ctx_other's device-held layer inputs (the batch's embd
// rows are then not read) while this is set
LLAMA_API void    llama_set_inject_dev(struct llama_context * ctx, bool value);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// every layer of the model attends through a sliding window (llama_model_n_swa)
LLAMA_API bool llama_model_all_swa(const struct llama_model * model);

// [TAG_DRAFT_VOCAB] give a draft model a row subset of its LM head: the rows ids[0..n) of output, copied to the head's
// own device, so its graphs compute logits over those tokens only (the draft's backend samplers map rows back to ids).
// Call before the draft context's first decode. Returns false, and changes nothing, if the head does not allow it.
LLAMA_API bool llama_model_set_head_subset(const struct llama_model * model, const llama_token * ids, int32_t n);

// [TAG_DRAFT_VOCAB] the same for a DFlash2 draft, which borrows the target's head: the rows ids[0..n) of head_model's
// output, or, when head_file is set, of the output.weight tensor in that GGUF file (e.g. the MTP draft's Q3_K head), on the draft's
// device and repacked by its routes. The draft's graph then computes its logits over those tokens only and maps its candidates back
// to token ids. Call before the draft context's first decode. Returns false, and changes nothing, if it cannot.
LLAMA_API bool llama_model_set_head_subset_dflash(const struct llama_model * model, const struct llama_model * head_model,
        const char * head_file, const llama_token * ids, int32_t n);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

//
// [TAG_SPEC_PIPELINE] pipelined speculative decoding
//

// the output slot (0 or 1) the next decode writes, and the output getters read. a decode into one slot leaves the
// other slot's outputs (logits, backend sampling results, nextn rows) intact, so a batch can be read while the next
// one is in flight. reading a slot that is older than the latest decode waits only for that slot's decode
LLAMA_API void llama_pipe_select(struct llama_context * ctx, int32_t slot);

// reserve both output slots now (at load), for the most outputs a decode can have, so the first
// pipelined decodes do not pin and clear the n_batch-row buffers; the live slot is unchanged
LLAMA_API void llama_pipe_reserve(struct llama_context * ctx);

// with two-user scheduling engaged: reserve the live output slot and the prompt job's slot now (at load), each
// for the most outputs a decode can have, so a job's publish (it swaps its slot in) is not followed by a pinned-buffer
// growth at the next decode. returns false when the context cannot run jobs or LLAMA_TU_OUT_RESERVE=0 (nothing changes)
LLAMA_API bool llama_tu_out_reserve(struct llama_context * ctx);

// the effective LLAMA_TU_OUT_RESERVE and LLAMA_TU_INJ_DECODE, as the context reads them
LLAMA_API void llama_tu_switches(const struct llama_context * ctx, int32_t * out_reserve, int32_t * inj_decode);

// flags for the next decode: async submits it without waiting for the previous decode to finish (no graph-reuse or
// copy-slot synchronize); backup first copies the recurrent rollback snapshots of the previous decode aside, so that
// a later seq_rm can still roll back into it once this decode is discarded
LLAMA_API void llama_pipe_decode_flags(struct llama_context * ctx, bool async, bool backup);

// wait for every decode in flight
LLAMA_API void llama_pipe_drain(struct llama_context * ctx);

// the next decode (one ubatch) stops before its last GPU's half: that half and the outputs follow at
// llama_pipe_resume, or never with llama_pipe_abort (a chunk found wrong before its second half ran). nothing else
// may be decoded on the context in between
LLAMA_API void llama_pipe_hold_next(struct llama_context * ctx, bool hold);
LLAMA_API bool llama_pipe_resume   (struct llama_context * ctx);
LLAMA_API void llama_pipe_abort    (struct llama_context * ctx);

// [TAG_SPEC_PIPE_LOOSE] whether every row of the latest decode (one ubatch) depends only on its own
// prefix and the batch size, not on the other tokens of its batch: its graph did not gather the QSA union
// (LLAMA_QSA_UNION, qwen4exp-qsa-compact.h), the one place a verify row reads its batch's other tokens.
// with n_more > 0 the answer also covers every decode of the same shape while the cache view grows by up to
// n_more cells: true when the union is off, or when that view stays within the padded selection, the width below
// which the compact path, and so the union, never applies. a graph without QSA attention is independent
LLAMA_API bool llama_pipe_rows_independent(struct llama_context * ctx, int64_t n_more);

//
// [TAG_SPEC_REJECTION] rejection sampling for the MTP draft
//

// one uniform draw in [0, 1) from the random number generator of the chain's last sampler, when that is dist: the
// same generator, and the same kind of draw, its apply takes. false (u untouched) when the chain does not end in dist
LLAMA_API bool llama_sampler_chain_draw_uniform(struct llama_sampler * chain, double * u);

//
// the MTP draft's steps chained on the device
//

// whether this context can chain its draft steps in one graph: an MTP draft context of an architecture that builds the
// chain (qwen35), with flash attention and the draft vocabulary subset (llama_model_set_head_subset) and its embedding rows
LLAMA_API bool llama_mtp_chain_supported(const struct llama_context * ctx);

// the next decode chains n_steps draft steps (0: off) with k device candidates per step: its batch holds n_steps rows of
// each sequence at its next n_steps positions, row 0 with the token and hidden row to start from (the other rows' token
// and hidden row are not read). Every sequence of the batch needs its pick parameters (llama_mtp_chain_set_params). The
// setting holds for that one decode
LLAMA_API void llama_mtp_chain_set(struct llama_context * ctx, int32_t n_steps, int32_t k);

// a sequence's pick parameters for the chained steps: n_steps rows of GGML_DRAFT_PICK_NP floats (ggml-draft-pick.h)
LLAMA_API void llama_mtp_chain_set_params(struct llama_context * ctx, llama_seq_id seq_id, const float * prm, int32_t n);

// a sequence's pick at a step of the last chained decode (waits for it): GGML_DRAFT_PICK_HDR + 2*k ints
// (ggml-draft-pick.h), or nullptr. width: the ints of a pick
LLAMA_API const int32_t * llama_mtp_chain_get(struct llama_context * ctx, llama_seq_id seq_id, int32_t step, int32_t * width);

// (LLAMA_TU_DIAG) the recurrent rollback index that the first ubatch of seq_id after the last reset consumed (its
// s_copy source snapshot), or -1 if none ran since the reset; -1 without a recurrent memory
LLAMA_API void    llama_rs_idx_seen_reset(struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API int32_t llama_rs_idx_seen(const struct llama_context * ctx, llama_seq_id seq_id);
