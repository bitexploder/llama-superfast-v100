#pragma once

// MTP's boundary state, per sequence.
//
// MTP's drafter pairs the token at position p with the target's hidden row at p-1. Inside one target batch the row comes from
// the same sequence's previous token in that batch; the first token of a sequence in a batch takes the CARRY: the target row of
// the last position the drafter has already consumed for that sequence. The carry is valid only for the sequence's current
// lineage, so it holds its position, and anything that replaces or clears what the sequence holds (a cleared slot, a restored
// checkpoint or prompt-cache entry, a copy from another slot, a shift) resets it or restores the saved one:
//
//   - position 0 takes the fresh boundary: the zero row a newly created context starts with;
//   - position p > 0 takes the carry when it holds position p-1;
//   - otherwise the carry is invalid for p: there is no correct row (row_for returns the zero row and says so, and the
//     server recovers before a prompt starts: see common_speculative_state_valid).
//
// The verification window (the rows of the latest target batch, by position) belongs to the same lineage; accept() moves the
// carry to the last accepted row, and a reset drops the window so that no obsolete window or row survives a boundary.
//
// get_state/set_state serialize (position, row); an empty, malformed or foreign record resets the sequence instead of leaving
// the previous carry in place.

#include "llama.h"

#include <cstdint>
#include <vector>

struct common_mtp_carry {
    // where the carried row came from
    enum src_t : int32_t {
        SRC_NONE    = 0, // nothing carried (fresh or reset)
        SRC_PROCESS = 1, // the last row of a target batch (prompt or verify) seen by process()
        SRC_ACCEPT  = 2, // the last accepted row of a verify window
        SRC_PIPE    = 3, // the last row of a pipelined chunk's catch-up
        SRC_RESTORE = 4, // set_state: a checkpoint, a prompt-cache entry or another slot's state
    };

    // what the first input row at a position took
    enum use_t : int32_t {
        USE_FRESH   = 0, // position 0: the zero row
        USE_CARRY   = 1, // the carry, which holds position p-1
        USE_INVALID = 2, // p > 0 and no carry for p-1: the zero row, counted
    };

    int32_t n_embd = 0;

    std::vector<std::vector<float>> h;   // [n_seq][n_embd] the carried row
    std::vector<llama_pos>          pos; // [n_seq] its position, -1: none
    std::vector<int32_t>            src; // [n_seq] src_t

    // the verification window of the latest target batch per sequence: its rows and their positions, in batch order
    std::vector<std::vector<float>>     win_h;
    std::vector<std::vector<llama_pos>> win_pos;

    std::vector<float> zero; // the fresh boundary row

    // counters (also GGML_RT_COUNT "mtp.carry_*" under LLAMA_ROUND_TIMERS)
    uint64_t n_fresh   = 0;
    uint64_t n_carry   = 0;
    uint64_t n_invalid = 0;

    void init(uint32_t n_seq, int32_t n_embd);

    uint32_t n_seq() const { return (uint32_t) h.size(); }

    // a lineage boundary: nothing carried, the row zeroed as in a new context, no window
    void reset(llama_seq_id seq_id);

    // whether a sequence continuing at position p (its next input) has the row it needs: p == 0, or the carry holds p-1
    bool valid_for(llama_seq_id seq_id, llama_pos p) const;

    // the row the input at position p (the sequence's first in a batch) pairs with, and how it was chosen
    const float * row_for(llama_seq_id seq_id, llama_pos p, use_t & use);

    // the carry becomes row at position p
    void set(llama_seq_id seq_id, llama_pos p, const float * row, src_t s);

    // the window becomes n rows at positions p0, p0 + 1, ... (rows contiguous, n_embd floats each); the carry its last row
    void set_window(llama_seq_id seq_id, const float * rows, llama_pos p0, int32_t n, src_t s);

    // the window of the sequence's rows in a target batch (rows in batch order, h_tgt row k = batch token k); the carry its
    // last row. Sequences absent from the batch keep their state
    void set_windows(const llama_batch & batch, const float * h_tgt);

    // the carry moves to the window row of the n_accepted-th accepted draft (row 0: the round's first token)
    void accept(llama_seq_id seq_id, int32_t n_accepted);

    // the drafter's catch-up inputs for a target batch: dst row k (n_embd floats) pairs batch token k with the target row of its
    // sequence's previous token, the previous batch row of the same sequence or, for its first row, row_for. h_tgt rows are in
    // batch order. Sequences need not be contiguous in the batch
    void gather(const llama_batch & batch, const float * h_tgt, float * dst);

    // (position, row) of a valid carry; false, with data empty, when there is none
    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const;

    // restore a record from get_state; anything else (empty, malformed, another width) resets the sequence. true: restored
    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data);

    // bytes of one record (cache accounting)
    size_t state_size() const;

    // FNV-1a of a row's bytes, for the trace
    uint64_t row_hash(const float * row) const;

    // LLAMA_MTP_CARRY_TRACE=1: one line per consumption, write, reset, save and restore (sequence, position, provenance, hash)
    static bool trace_on();
};
