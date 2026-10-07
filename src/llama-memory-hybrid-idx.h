#pragma once

#include "llama-memory-hybrid.h"
#include "llama-kv-cells.h"

#include "ggml-cpp.h"

#include <memory>
#include <vector>

struct llama_ubatch;

// the host data of set_input_qsa's tensors and their shapes (see llama_memory_hybrid_idx::set_input_qsa)
struct llama_qsa_input {
    int32_t * cell_blk  = nullptr;   // cell ranking only
    int32_t * blk_cells = nullptr;
    int32_t * blk_pos   = nullptr;
    float   * bias      = nullptr;
    int32_t * tail      = nullptr;   // block selection only

    // block selection with the packed bias (LLAMA_QSA_PACK), in place of bias: (n_blocks + 31)/32 words per
    // query, bit b of word b>>5 (bit b&31) set where bias would hold 0.0f, clear where it would hold -inf
    uint32_t * bias_bits = nullptr;

    int64_t n_kv     = 0;
    int64_t n_ns     = 0;            // streams in the ubatch
    int64_t n_blocks = 0;
    int64_t n_tail   = 0;
};

// builds the inputs with one scan of every cell of each stream, O(n_kv) per call
// cells[s] is the cells array of stream s
void llama_qsa_input_scan(const llama_kv_cells * const * cells, const llama_qsa_input & dst,
                          const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias);

// [TAG_QSA_HOST_META]
// the block selection inputs from a block table kept in step with each cells array through its change journal
// (llama_kv_cells::jrnl_*), so a call costs the changed position buckets plus one pass over the blocks
// invariant: once synced, each bucket (pos/ratio) holds the groups llama_qsa_input_scan would build from it, in
// its order, and the table holds every cell that has a sequence
class llama_qsa_blk_table {
public:
    // fills dst as llama_qsa_input_scan would and returns true, or writes nothing and returns false when the
    // table cannot promise that for this call (not block selection, a repeated position in a block, a cell
    // with no sequence or past the window): the caller then runs the scan
    bool build(const llama_kv_cells * const * cells, const llama_qsa_input & dst,
               const llama_ubatch * ubatch, uint32_t ratio);

    uint64_t n_built    = 0;   // calls served from the table
    uint64_t n_rebuilt  = 0;   // full rebuilds, one scan each
    uint64_t n_declined = 0;   // calls left to the scan

private:
    using seq_set_t = llama_kv_cells::seq_set_t;

    struct tab {
        const llama_kv_cells * cells = nullptr;
        uint32_t r = 0;

        // journal position this table is synced to; epoch 0 never matches
        uint64_t epoch = 0;
        uint64_t seen  = 0;

        bool valid = false;   // false: a position lies past the table, so only the scan can serve

        std::vector<int32_t> head;       // [n_pb] first group of each bucket's chain, -1 if none

        // groups: one per (bucket, sequence set), chained newest first as the scan chains them
        std::vector<int32_t>  g_next;
        std::vector<int32_t>  g_first;   // lowest cell
        std::vector<uint64_t> g_slots;   // pos%r seen
        std::vector<int32_t>  g_set;     // interned sequence set
        std::vector<int32_t>  g_n;       // cells
        std::vector<uint8_t>  g_dup;     // a slot seen twice
        std::vector<int32_t>  g_cells;   // [r] per group: the highest cell in each slot
        std::vector<int32_t>  g_free;

        int64_t n_cells = 0;
        int64_t n_dup   = 0;
        int64_t pb_max  = -1;            // highest bucket with a group

        std::vector<seq_set_t> sets;
        int32_t set_last = -1;

        std::vector<uint32_t> stamp;     // [n_pb] dirty-bucket marks
        uint32_t stamp_cur = 0;
    };

    std::vector<std::unique_ptr<tab>> tabs;

    // scratch
    std::vector<int32_t>      dirty;
    std::vector<int32_t>      gather;
    std::vector<llama_seq_id> present;
    std::vector<int32_t>      bid_pb;
    std::vector<int32_t>      bid_set;
    std::vector<uint8_t>      inset;

    tab & get(const llama_kv_cells * cells, uint32_t r);

    // false: the cells keep no journal
    bool sync(tab & t);

    void rebuild(tab & t);
    void rebuild_bucket(tab & t, int64_t pb);
    void add_cell(tab & t, int64_t pb, int32_t j, llama_pos p);
    int32_t intern(tab & t, const seq_set_t & set);
};

// [TAG_QSA_SEL_KEEP] LLAMA_MTP_QSA_REUSE: a QSA block selection kept on the device across graph builds, so the
// qwen4exp MTP draft steps attend with the selection the catch-up computed for the last verified row instead of
// running the indexer again. graph_mtp writes and reads sel; this only allocates it and records, on the host,
// the sequence and position of each kept row and the cells the draft steps have added since.
struct llama_qsa_sel_keep {
    static constexpr int64_t n_rows = 64; // rows one ubatch may keep: a catch-up is n_seq*(n_draft + 1) tokens
    static constexpr int64_t n_app  = 8;  // cells a step may add to its kept row: the earlier steps' and its own

    struct row {
        llama_seq_id seq;
        llama_pos    pos;
        int32_t      idx;   // row of sel
    };

    struct add {
        llama_seq_id seq;
        llama_pos    pos;
        int32_t      cell;
    };

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;

    // I32 [w_max, 2, n_rows]: per row the cells, then the mask plane of ggml_qsa_select (0, or INT_MIN for unused)
    ggml_tensor * sel = nullptr;

    int64_t w_max = 0;
    int64_t w     = 0;      // width of the kept rows

    std::vector<row> rows;  // from the last ubatch that kept its selection
    std::vector<add> adds;  // cells the draft steps wrote since, each above a kept row of its sequence

    // allocates sel in buft; false if that fails
    bool alloc(ggml_backend_buffer_type_t buft, int64_t w_max);

    // a draft step: every token an output and the only one of its sequence, above a kept row of that sequence
    // with every position in between added, at most n_app positions above it.
    // base[t] receives the kept row and cells[t*n_app ...] the added cells in position order (n_add[t] of them,
    // the last slot left for the token's own cell); any of them may be null
    bool match(const llama_ubatch & ub, int32_t * base, int32_t * cells, int32_t * n_add) const;

    void forget();
    void forget(llama_seq_id seq_id, llama_pos p0);   // what a change at p0 or above may touch; seq_id < 0: every sequence
};

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // tail (blk_bias only) asks for block selection instead of cell ranking (ggml_qsa_select):
    //   bias      F32 [n_blocks, n_tokens/ns, ns] 0 for a block the query sees whole below its tail, else -inf
    //             or, packed, I32 [(n_blocks + 31)/32, n_tokens/ns, ns]: one bit per block, 1 for 0 and 0 for -inf
    //   tail      I32 [n_tail, n_tokens/ns, ns]   the query's cells from its tail start up to itself, -1 after
    //   cell_blk  unused, may be null
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * tail, const llama_ubatch * ubatch,
                       uint32_t ratio, bool blk_bias, int64_t n_kv) const;

    // [TAG_QSA_SEL_KEEP] every change to the cells below forgets what it touches
    llama_qsa_sel_keep * get_sel_keep() const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // mutable: set_input_qsa, its only user, runs from set_input on a const memory
    mutable llama_qsa_blk_table qsa_tab;

    // mutable: graph_mtp allocates it and its inputs' set_input keep the host side
    mutable llama_qsa_sel_keep sel_keep;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * tail, const llama_ubatch * ubatch,
                       uint32_t ratio, bool blk_bias) const;

    llama_qsa_sel_keep * get_sel_keep() const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // an update context: applying it may move or shift cells, so it forgets the kept selection
    const bool is_update = false;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
