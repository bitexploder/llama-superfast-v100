#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <utility>

//
// llama_memory_hybrid_idx
//

// [TAG_QSA_HOST_META] LLAMA_QSA_HOST_META: 1 (default) serves block selection from llama_qsa_blk_table,
// 0 always runs the full scan, 2 serves from the table and compares every call with the scan, aborting on any difference
static int llama_qsa_host_meta_mode() {
    static const int mode = [] {
        const char * e = getenv("LLAMA_QSA_HOST_META");
        return e == nullptr ? 1 : atoi(e);
    }();

    return mode;
}

//
// llama_qsa_sel_keep [TAG_QSA_SEL_KEEP]
//

bool llama_qsa_sel_keep::alloc(ggml_backend_buffer_type_t buft, int64_t w_max) {
    GGML_ASSERT(sel == nullptr && w_max > 0);

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr c(ggml_init(params));
    if (!c) {
        return false;
    }

    ggml_tensor * t = ggml_new_tensor_3d(c.get(), GGML_TYPE_I32, w_max, 2, n_rows);
    ggml_set_name(t, "qsa_sel_keep");

    ggml_backend_buffer_ptr b(ggml_backend_alloc_ctx_tensors_from_buft(c.get(), buft));
    if (!b) {
        return false;
    }

    // a row read before any was written must still name valid cells: zero is cell 0, masked by nothing,
    // but no reader looks at a row the host has not recorded
    ggml_backend_buffer_clear(b.get(), 0);

    LLAMA_LOG_INFO("%s: %s kept-selection buffer for the MTP draft steps, %.2f MiB\n", __func__,
            ggml_backend_buffer_name(b.get()), ggml_backend_buffer_get_size(b.get())/1024.0/1024.0);

    ctx         = std::move(c);
    buf         = std::move(b);
    sel         = t;
    this->w_max = w_max;

    return true;
}

bool llama_qsa_sel_keep::match(const llama_ubatch & ub, int32_t * base, int32_t * cells, int32_t * n_add) const {
    if (sel == nullptr || rows.empty() || w <= 0 || ub.n_tokens == 0 || ub.output == nullptr) {
        return false;
    }

    for (uint32_t t = 0; t < ub.n_tokens; ++t) {
        if (ub.output[t] == 0 || ub.n_seq_id[t] != 1) {
            return false;
        }

        const llama_seq_id s = ub.seq_id[t][0];
        const llama_pos    p = ub.pos[t];

        for (uint32_t u = 0; u < t; ++u) {
            if (ub.seq_id[u][0] == s) {
                return false;
            }
        }

        // the kept row of s just below p: a catch-up keeps each of its rows, acceptance drops the rejected ones
        const row * r = nullptr;
        for (const auto & k : rows) {
            if (k.seq == s && k.pos < p && (r == nullptr || k.pos > r->pos)) {
                r = &k;
            }
        }

        if (r == nullptr || p - r->pos > n_app) {
            return false;
        }

        // every position between the kept row and p must have been added by an earlier step
        const int32_t n = p - r->pos - 1;

        for (int32_t j = 0; j < n; ++j) {
            const llama_pos q = r->pos + 1 + j;

            const add * a = nullptr;
            for (const auto & d : adds) {
                if (d.seq == s && d.pos == q) {
                    a = &d;
                    break;
                }
            }

            if (a == nullptr) {
                return false;
            }

            if (cells) {
                cells[t*n_app + j] = a->cell;
            }
        }

        if (base) {
            base[t] = r->idx;
        }
        if (n_add) {
            n_add[t] = n;
        }
    }

    return true;
}

void llama_qsa_sel_keep::forget() {
    rows.clear();
    adds.clear();
}

void llama_qsa_sel_keep::forget(llama_seq_id seq_id, llama_pos p0) {
    if (p0 < 0) {
        p0 = 0;
    }

    // a row names cells at or below its own position, so removing any cell at p0 or above can touch it
    auto hit = [&](llama_seq_id s, llama_pos p) {
        return (seq_id < 0 || s == seq_id) && p >= p0;
    };

    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const row & r) { return hit(r.seq, r.pos); }), rows.end());
    adds.erase(std::remove_if(adds.begin(), adds.end(), [&](const add & a) { return hit(a.seq, a.pos); }), adds.end());
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // set_input_qsa's block table follows the indexer cells through their change journal [TAG_QSA_HOST_META]
    if (mem_idx && llama_qsa_host_meta_mode() != 0) {
        mem_idx->cells_jrnl_enable();
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    sel_keep.forget();

    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    sel_keep.forget(seq_id, p0);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    sel_keep.forget(seq_id_dst, -1);

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    sel_keep.forget();

    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    sel_keep.forget(seq_id, -1);

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    sel_keep.forget(seq_id, -1);

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    if (sel_keep.buf) {
        mb[ggml_backend_buffer_get_type(sel_keep.buf.get())] += ggml_backend_buffer_get_size(sel_keep.buf.get());
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    sel_keep.forget();

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

llama_qsa_sel_keep * llama_memory_hybrid_idx::get_sel_keep() const {
    return &sel_keep;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * tail,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        int64_t n_kv) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    // block selection needs the per-block bias and takes no per-cell block index
    GGML_ASSERT(tail == nullptr || blk_bias);
    GGML_ASSERT(tail != nullptr || cell_blk != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(blk_cells->buffer));
    GGML_ASSERT(cell_blk == nullptr || cell_blk->ne[0] == n_kv);

    // an I32 bias is the packed form, block selection only
    const bool packed = bias->type == GGML_TYPE_I32;
    GGML_ASSERT(!packed || tail != nullptr);

    llama_qsa_input dst;

    dst.cell_blk  = cell_blk ? (int32_t *) cell_blk->data : nullptr;
    dst.blk_cells = (int32_t *) blk_cells->data;
    dst.blk_pos   = (int32_t *) blk_pos->data;
    dst.bias      = packed ? nullptr : (float *) bias->data;
    dst.tail      = tail ? (int32_t *) tail->data : nullptr;
    dst.bias_bits = packed ? (uint32_t *) bias->data : nullptr;

    dst.n_kv     = n_kv;
    dst.n_ns     = blk_cells->ne[1];                 // streams in this ubatch
    dst.n_blocks = blk_pos->ne[0]/(4*dst.n_ns);
    dst.n_tail   = tail ? tail->ne[0] : 0;

    GGML_ASSERT(ubatch->n_tokens % dst.n_ns == 0);
    const int64_t n_tps = ubatch->n_tokens/dst.n_ns; // tokens per stream

    GGML_ASSERT(bias->ne[0] == (packed ? (dst.n_blocks + 31)/32 : (blk_bias ? dst.n_blocks : n_kv)));

    // ubatch index s*n_tps belongs to stream s; ask which cells array it uses
    std::vector<const llama_kv_cells *> strm_cells(dst.n_ns);

    for (int64_t s = 0; s < dst.n_ns; ++s) {
        strm_cells[s] = &get_mem_idx()->get_cells(ubatch->seq_id[s*n_tps][0]);
    }

    const int mode = llama_qsa_host_meta_mode();

    if (mode == 0 || !qsa_tab.build(strm_cells.data(), dst, ubatch, ratio)) {
        llama_qsa_input_scan(strm_cells.data(), dst, ubatch, ratio, blk_bias);
    } else if (mode == 2) {
        // the scan into separate buffers, then byte for byte against what the table wrote
        ggml_tensor * outs[4] = { blk_cells, blk_pos, bias, tail };

        std::vector<uint8_t> ref[4];
        for (int k = 0; k < 4; ++k) {
            ref[k].resize(ggml_nbytes(outs[k]));
        }

        llama_qsa_input chk = dst;

        chk.blk_cells = (int32_t *) ref[0].data();
        chk.blk_pos   = (int32_t *) ref[1].data();
        chk.bias      = packed ? nullptr : (float *) ref[2].data();
        chk.tail      = (int32_t *) ref[3].data();
        chk.bias_bits = packed ? (uint32_t *) ref[2].data() : nullptr;

        llama_qsa_input_scan(strm_cells.data(), chk, ubatch, ratio, blk_bias);

        for (int k = 0; k < 4; ++k) {
            if (memcmp(ref[k].data(), outs[k]->data, ref[k].size()) != 0) {
                LLAMA_LOG_ERROR("%s: %s from the block table differs from the scan (n_tokens %u, n_kv %lld, n_blocks %lld)\n",
                        __func__, ggml_get_name(outs[k]), ubatch->n_tokens, (long long) dst.n_kv, (long long) dst.n_blocks);
                GGML_ABORT("qsa host meta: block table differs from the scan");
            }
        }

        // warn level: the check mode is a diagnostic, and the server drops info lines
        if (qsa_tab.n_built % 256 == 1) {
            LLAMA_LOG_WARN("%s: check: %llu calls from the block table identical to the scan, %llu full rebuilds, %llu calls declined\n",
                    __func__, (unsigned long long) qsa_tab.n_built, (unsigned long long) qsa_tab.n_rebuilt,
                    (unsigned long long) qsa_tab.n_declined);
        }
    }

    // LLAMA_QSA_PACK_CHECK=N: the first N packed calls also run the scan with today's F32 bias into separate
    // buffers, and every block's bit must be set exactly where that bias is +0 (and clear where it is -inf)
    static int check_left = [] { const char * e = getenv("LLAMA_QSA_PACK_CHECK"); return e ? atoi(e) : 0; }();
    static int check_call = 0;
    if (!packed || check_left <= 0) {
        return;
    }
    --check_left;
    ++check_call;

    const int64_t n_rows  = n_tps*dst.n_ns;
    const int64_t n_words = bias->ne[0];

    std::vector<int32_t> ref_cells(ggml_nelements(blk_cells));
    std::vector<int32_t> ref_pos  (ggml_nelements(blk_pos));
    std::vector<int32_t> ref_tail (ggml_nelements(tail));
    std::vector<float>   ref_bias (dst.n_blocks*n_rows);

    llama_qsa_input chk = dst;

    chk.blk_cells = ref_cells.data();
    chk.blk_pos   = ref_pos.data();
    chk.bias      = ref_bias.data();
    chk.tail      = ref_tail.data();
    chk.bias_bits = nullptr;

    llama_qsa_input_scan(strm_cells.data(), chk, ubatch, ratio, blk_bias);

    int64_t n_agree = 0;
    int64_t n_pad   = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        const uint32_t * row = dst.bias_bits + n_words*i;
        for (int64_t b = 0; b < dst.n_blocks; ++b) {
            const bool bit = (row[b >> 5] >> (b & 31)) & 1;
            uint32_t v;
            memcpy(&v, &ref_bias[dst.n_blocks*i + b], sizeof(v));
            n_agree += (v == 0x00000000u && bit) || (v == 0xff800000u && !bit);
        }
        for (int64_t b = dst.n_blocks; b < 32*n_words; ++b) {
            n_pad += (row[b >> 5] >> (b & 31)) & 1;
        }
    }

    LLAMA_LOG_WARN("qsa pack check: bias call %d: %lld of %lld bits agree (%lld rows x %lld blocks, ratio %u, %lld padding bits set)\n",
            check_call, (long long) n_agree, (long long) (dst.n_blocks*n_rows), (long long) n_rows, (long long) dst.n_blocks,
            ratio, (long long) n_pad);
    GGML_ASSERT(n_agree == dst.n_blocks*n_rows && n_pad == 0);
}

void llama_qsa_input_scan(
        const llama_kv_cells * const * strm_cells,
        const llama_qsa_input & dst,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) {
    const int64_t n_kv     = dst.n_kv;
    const int64_t n_ns     = dst.n_ns;
    const int64_t n_blocks = dst.n_blocks;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    const int64_t n_tail   = dst.n_tail;
    const int64_t n_words  = (n_blocks + 31)/32;     // packed bias only

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    GGML_ASSERT(dst.bias_bits == nullptr || dst.tail != nullptr);

    int32_t * dst_cell_blk  = dst.cell_blk;
    int32_t * dst_blk_cells = dst.blk_cells;
    int32_t * dst_blk_pos   = dst.blk_pos;
    float   * dst_bias      = dst.bias;
    int32_t * dst_tail      = dst.tail;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // this runs per ubatch and is O(n_kv) per stream, about 1.1 ms at 64k context. the cost is the
    // per-cell scan rather than these allocations; llama_qsa_blk_table serves block selection without it
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::vector<int64_t>                      q_of(n_tps);   // last visible index of each query
    std::vector<std::pair<int64_t, int32_t>>  tail_cells;    // (index, cell) of the cells in any query's tail

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        const auto & cells = *strm_cells[s];

        int32_t * cur_cell_blk  = dst_cell_blk ? dst_cell_blk + s*n_kv : nullptr;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        // the last index each query sees: its position, or its rank under mrope
        int64_t tail_lo = INT64_MAX;
        int64_t tail_hi = INT64_MIN;

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            q_of[ii] = q;

            tail_lo = std::min(tail_lo, (q + 1)/r*r);
            tail_hi = std::max(tail_hi, q);
        }

        tail_cells.clear();

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            if (cur_cell_blk) {
                cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
            }

            // g >= 0: the cell is occupied and inside the cell window
            if (dst_tail && g >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                if (idx >= tail_lo && idx <= tail_hi) {
                    tail_cells.emplace_back(idx, (int32_t) j);
                }
            }
        }

        // position order, then cell order: the order the tail is listed in
        std::sort(tail_cells.begin(), tail_cells.end());

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const int64_t q = q_of[ii];

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (dst_tail) {
                // the reference ranks the blocks the query sees whole and appends the rest: a full
                // block below tail_start is either all visible to this sequence or not at all
                if (dst.bias_bits) {
                    uint32_t * cur_bits = dst.bias_bits + i*n_words;

                    std::fill(cur_bits, cur_bits + n_words, 0u);

                    for (int64_t b = 0; b < n_blocks; ++b) {
                        const bool whole = b < n_bid && bid_idx[b] < tail_start && cells.seq_has((uint32_t) bid_cell[b], seq_id);

                        cur_bits[b >> 5] |= (uint32_t) whole << (b & 31);
                    }
                } else {
                    float * cur_blk_bias = dst_bias + i*n_blocks;

                    for (int64_t b = 0; b < n_blocks; ++b) {
                        const bool whole = b < n_bid && bid_idx[b] < tail_start && cells.seq_has((uint32_t) bid_cell[b], seq_id);

                        cur_blk_bias[b] = whole ? 0.0f : -INFINITY;
                    }
                }

                int32_t * cur_tail = dst_tail + i*n_tail;
                int64_t   n_t      = 0;

                auto it = std::lower_bound(tail_cells.begin(), tail_cells.end(), std::make_pair(tail_start, (int32_t) -1));
                for (; it != tail_cells.end() && it->first <= q && n_t < n_tail; ++it) {
                    if (cells.seq_has((uint32_t) it->second, seq_id)) {
                        cur_tail[n_t++] = it->second;
                    }
                }

                for (; n_t < n_tail; ++n_t) {
                    cur_tail[n_t] = -1;
                }

                continue;
            }

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_qsa_blk_table [TAG_QSA_HOST_META]
//

llama_qsa_blk_table::tab & llama_qsa_blk_table::get(const llama_kv_cells * cells, uint32_t r) {
    for (auto & t : tabs) {
        if (t->cells == cells && t->r == r) {
            return *t;
        }
    }

    tabs.push_back(std::make_unique<tab>());

    tab & t = *tabs.back();

    t.cells = cells;
    t.r     = r;

    return t;
}

int32_t llama_qsa_blk_table::intern(tab & t, const seq_set_t & set) {
    // consecutive cells almost always share a set
    if (t.set_last >= 0 && t.sets[t.set_last] == set) {
        return t.set_last;
    }

    for (size_t k = 0; k < t.sets.size(); ++k) {
        if (t.sets[k] == set) {
            return t.set_last = (int32_t) k;
        }
    }

    t.sets.push_back(set);

    return t.set_last = (int32_t) t.sets.size() - 1;
}

// the scan's group_cells step for one cell, cells visited in ascending order within a bucket
void llama_qsa_blk_table::add_cell(tab & t, int64_t pb, int32_t j, llama_pos p) {
    const int64_t r   = t.r;
    const int32_t set = intern(t, t.cells->seq_get_all(j));

    int32_t g = -1;

    for (int32_t c = t.head[pb]; c >= 0; c = t.g_next[c]) {
        if (t.g_set[c] == set) {
            g = c;
            break;
        }
    }

    if (g < 0) {
        if (!t.g_free.empty()) {
            g = t.g_free.back();
            t.g_free.pop_back();
        } else {
            g = (int32_t) t.g_next.size();

            t.g_next .push_back(0);
            t.g_first.push_back(0);
            t.g_slots.push_back(0);
            t.g_set  .push_back(0);
            t.g_n    .push_back(0);
            t.g_dup  .push_back(0);
            t.g_cells.resize(t.g_cells.size() + r);
        }

        t.g_next [g] = t.head[pb];
        t.g_first[g] = j;
        t.g_slots[g] = 0;
        t.g_set  [g] = set;
        t.g_n    [g] = 0;
        t.g_dup  [g] = 0;

        std::fill(t.g_cells.begin() + g*r, t.g_cells.begin() + (g + 1)*r, 0);

        t.head[pb] = g;
        t.pb_max   = std::max(t.pb_max, pb);
    }

    const uint64_t bit = uint64_t(1) << (p%r);

    if ((t.g_slots[g] & bit) != 0 && !t.g_dup[g]) {
        t.g_dup[g] = 1;
        t.n_dup++;
    }

    t.g_slots[g] |= bit;
    t.g_cells[g*r + p%r] = j;
    t.g_n[g]++;

    t.n_cells++;
}

void llama_qsa_blk_table::rebuild(tab & t) {
    const llama_kv_cells & cells = *t.cells;

    const int64_t r    = t.r;
    const int64_t n_pb = ((int64_t) cells.size() + r - 1)/r;

    t.head .assign(n_pb, -1);
    t.stamp.assign(n_pb, 0);
    t.stamp_cur = 0;

    t.g_next .clear();
    t.g_first.clear();
    t.g_slots.clear();
    t.g_set  .clear();
    t.g_n    .clear();
    t.g_dup  .clear();
    t.g_cells.clear();
    t.g_free .clear();

    t.sets.clear();
    t.set_last = -1;

    t.n_cells = 0;
    t.n_dup   = 0;
    t.pb_max  = -1;
    t.valid   = true;

    const uint32_t n = cells.used_max_p1();

    for (uint32_t j = 0; j < n; ++j) {
        // a cell with no sequence is left out, so the cell count tells the table it cannot serve
        if (cells.is_empty(j) || cells.seq_get_all(j).none()) {
            continue;
        }

        const llama_pos p  = cells.pos_get(j);
        const int64_t   pb = p/r;

        if (pb >= n_pb) {
            t.valid = false;
            break;
        }

        add_cell(t, pb, (int32_t) j, p);
    }

    t.epoch = cells.jrnl_epoch();
    t.seen  = cells.jrnl_end();

    n_rebuilt++;
}

void llama_qsa_blk_table::rebuild_bucket(tab & t, int64_t pb) {
    const llama_kv_cells & cells = *t.cells;

    const int64_t r = t.r;

    for (int32_t g = t.head[pb]; g >= 0; ) {
        const int32_t next = t.g_next[g];

        t.n_cells -= t.g_n[g];
        t.n_dup   -= t.g_dup[g];
        t.g_free.push_back(g);

        g = next;
    }

    t.head[pb] = -1;

    // the cells in this bucket that have a sequence, from each sequence's position index, in cell order
    gather.clear();

    for (const llama_seq_id s : present) {
        cells.seq_cells_in(s, (llama_pos) (pb*r), (llama_pos) (pb*r + r - 1), gather);
    }

    std::sort(gather.begin(), gather.end());
    gather.erase(std::unique(gather.begin(), gather.end()), gather.end());

    for (const int32_t j : gather) {
        add_cell(t, pb, j, cells.pos_get(j));
    }

    while (t.pb_max >= 0 && t.head[t.pb_max] < 0) {
        t.pb_max--;
    }
}

bool llama_qsa_blk_table::sync(tab & t) {
    const llama_kv_cells & cells = *t.cells;

    if (!cells.jrnl_enabled()) {
        return false;
    }

    if (t.epoch != cells.jrnl_epoch() || t.seen < cells.jrnl_base() || !t.valid) {
        if (t.epoch != cells.jrnl_epoch() || t.seen != cells.jrnl_end()) {
            rebuild(t);
        }

        return true;
    }

    const uint64_t end = cells.jrnl_end();

    if (t.seen == end) {
        return true;
    }

    const int64_t r    = t.r;
    const int64_t n_pb = (int64_t) t.head.size();

    if (++t.stamp_cur == 0) {
        std::fill(t.stamp.begin(), t.stamp.end(), 0);
        t.stamp_cur = 1;
    }

    dirty.clear();

    for (uint64_t k = t.seen; k < end; ++k) {
        const llama_pos p = cells.jrnl_get(k);

        if (p < 0) {
            continue;
        }

        const int64_t pb = p/r;

        if (pb >= n_pb) {
            rebuild(t);
            return true;
        }

        if (t.stamp[pb] != t.stamp_cur) {
            t.stamp[pb] = t.stamp_cur;
            dirty.push_back((int32_t) pb);
        }
    }

    present.clear();

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (cells.seq_pos_min(s) >= 0) {
            present.push_back(s);
        }
    }

    // a bucket costs a lookup per sequence, a rebuild about 10 ns per cell
    if ((int64_t) dirty.size()*(int64_t) std::max<size_t>(present.size(), 1)*50 > (int64_t) cells.get_used()) {
        rebuild(t);
        return true;
    }

    for (const int32_t pb : dirty) {
        rebuild_bucket(t, pb);
    }

    t.seen = end;

    return true;
}

bool llama_qsa_blk_table::build(
        const llama_kv_cells * const * strm_cells,
        const llama_qsa_input & dst,
        const llama_ubatch * ubatch,
        uint32_t ratio) {
    const int64_t n_ns     = dst.n_ns;
    const int64_t n_blocks = dst.n_blocks;
    const int64_t n_tail   = dst.n_tail;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_tps    = n_tokens/n_ns;
    const int64_t r        = ratio;
    const int64_t n_words  = (n_blocks + 31)/32;   // packed bias only

    // block selection only; the scan asserts on a ratio past 64
    if (dst.tail == nullptr || r > 64) {
        n_declined++;
        return false;
    }

    // every stream must be servable before anything is written
    std::vector<tab *> ts(n_ns);

    for (int64_t s = 0; s < n_ns; ++s) {
        tab & t = get(strm_cells[s], ratio);

        // the scan asserts on a cell past the window, and ranks cells instead of using positions
        // only when a slot repeats: both are its to handle
        if (!sync(t) || !t.valid || t.n_dup > 0 || t.pb_max >= n_blocks ||
                t.n_cells != (int64_t) strm_cells[s]->get_used()) {
            n_declined++;
            return false;
        }

        ts[s] = &t;
    }

    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    if ((int64_t) bid_pb.size() < n_blocks) {
        bid_pb .resize(n_blocks);
        bid_set.resize(n_blocks);
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        const tab & t = *ts[s];

        const llama_kv_cells & cells = *t.cells;

        int32_t * cur_blk_cells = dst.blk_cells + s*(r*n_blocks);

        int32_t * cur_blk_pos[4];
        for (int64_t sec = 0; sec < 4; ++sec) {
            cur_blk_pos[sec] = dst.blk_pos + sec*(n_blocks*n_ns) + s*n_blocks;
        }

        // the full blocks in the scan's order: by bucket, then along the bucket's chain
        int64_t n_bid = 0;

        const int64_t pb_end = std::min<int64_t>(n_blocks, t.pb_max + 1);

        const int32_t  * head    = t.head.data();
        const int32_t  * g_next  = t.g_next.data();
        const uint64_t * g_slots = t.g_slots.data();
        const int32_t  * g_set   = t.g_set.data();
        const int32_t  * g_cells = t.g_cells.data();

        for (int64_t pb = 0; pb < pb_end; ++pb) {
            for (int32_t g = head[pb]; g >= 0; g = g_next[g]) {
                if (g_slots[g] != slots_full) {
                    continue;
                }

                GGML_ASSERT(n_bid < n_blocks);

                const int32_t * src = g_cells + (int64_t) g*r;
                int32_t       * out = cur_blk_cells + n_bid*r;

                if (r == 4) {
                    out[0] = src[0]; out[1] = src[1]; out[2] = src[2]; out[3] = src[3];
                } else {
                    for (int64_t k = 0; k < r; ++k) {
                        out[k] = src[k];
                    }
                }

                const int32_t idx = (int32_t) (pb*r);

                cur_blk_pos[0][n_bid] = idx;
                cur_blk_pos[1][n_bid] = idx;
                cur_blk_pos[2][n_bid] = idx;
                cur_blk_pos[3][n_bid] = idx;

                bid_pb [n_bid] = (int32_t) pb;
                bid_set[n_bid] = g_set[g];

                n_bid++;
            }
        }

        std::fill(cur_blk_cells + n_bid*r, cur_blk_cells + r*n_blocks, 0);

        for (int64_t sec = 0; sec < 4; ++sec) {
            std::fill(cur_blk_pos[sec] + n_bid, cur_blk_pos[sec] + n_blocks, 0);
        }

        llama_seq_id seq_cur = -1;
        bool         all     = false;   // every sequence set holds seq_cur

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const int64_t q          = ubatch->pos[i];
            const int64_t tail_start = (q + 1)/r*r;

            if (seq_id != seq_cur) {
                seq_cur = seq_id;
                all     = true;

                inset.resize(t.sets.size());

                for (size_t k = 0; k < t.sets.size(); ++k) {
                    inset[k] = t.sets[k].test(seq_id);
                    all &= inset[k] != 0;
                }
            }

            // a block the query sees whole: below its tail, which starts on a block boundary, and in its sequence
            const int64_t n_lim = std::lower_bound(bid_pb.begin(), bid_pb.begin() + n_bid, (int32_t) (tail_start/r)) - bid_pb.begin();

            if (dst.bias_bits) {
                uint32_t * cur_bits = dst.bias_bits + i*n_words;

                std::fill(cur_bits, cur_bits + n_words, 0u);

                for (int64_t b = 0; b < n_lim; ++b) {
                    cur_bits[b >> 5] |= (uint32_t) (all || inset[bid_set[b]]) << (b & 31);
                }
            } else {
                float * cur_blk_bias = dst.bias + i*n_blocks;

                if (all) {
                    std::fill(cur_blk_bias, cur_blk_bias + n_lim, 0.0f);
                } else {
                    for (int64_t b = 0; b < n_lim; ++b) {
                        cur_blk_bias[b] = inset[bid_set[b]] ? 0.0f : -INFINITY;
                    }
                }

                std::fill(cur_blk_bias + n_lim, cur_blk_bias + n_blocks, -INFINITY);
            }

            // the query's own cells from its tail start up to itself, in position order
            gather.clear();
            cells.seq_cells_in(seq_id, (llama_pos) tail_start, (llama_pos) q, gather);

            int32_t * cur_tail = dst.tail + i*n_tail;

            const int64_t n_t = std::min<int64_t>((int64_t) gather.size(), n_tail);

            std::copy(gather.begin(), gather.begin() + n_t, cur_tail);
            std::fill(cur_tail + n_t, cur_tail + n_tail, -1);
        }
    }

    n_built++;

    return true;
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    is_update(true),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    // an update may shift or move cells, which the kept rows name
    if (is_update && mem) {
        mem->get_sel_keep()->forget();
    }

    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

llama_qsa_sel_keep * llama_memory_hybrid_idx_context::get_sel_keep() const {
    GGML_ASSERT(mem != nullptr);

    return mem->get_sel_keep();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * tail,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, tail, ubatch, ratio, blk_bias, get_idx()->get_n_kv());
}
