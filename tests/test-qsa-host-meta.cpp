// QSA block selection inputs from the incremental block table (llama_qsa_blk_table) against the full scan
// (llama_qsa_input_scan) [TAG_QSA_HOST_META]: random cache histories (appends, rollbacks, prefix reuse,
// sequence copies, keeps, shifts, divisions, restores, clears, copies of the cells), every call compared
// byte for byte. The table may decline a call (the scan then serves it) but must never differ.
//
// test-qsa-host-meta [bench]: with "bench", time both at 20k, 64k and 128k instead

#include "../src/llama-batch.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

struct qsa_out {
    std::vector<int32_t> blk_cells, blk_pos, tail;
    std::vector<float>   bias;
    llama_qsa_input      dst;

    qsa_out(int64_t n_kv, int64_t n_ns, int64_t n_blocks, int64_t n_tail, int64_t n_tokens, uint32_t r) {
        // poisoned, so a value left unwritten shows
        blk_cells.assign(r*n_blocks*n_ns, 0x7b7b7b7b);
        blk_pos  .assign(4*n_blocks*n_ns, 0x7b7b7b7b);
        tail     .assign(n_tail*n_tokens, 0x7b7b7b7b);
        bias     .assign(n_blocks*n_tokens, 12345.0f);

        dst.blk_cells = blk_cells.data();
        dst.blk_pos   = blk_pos.data();
        dst.bias      = bias.data();
        dst.tail      = tail.data();
        dst.n_kv      = n_kv;
        dst.n_ns      = n_ns;
        dst.n_blocks  = n_blocks;
        dst.n_tail    = n_tail;
    }

    bool same(const qsa_out & o) const {
        return blk_cells == o.blk_cells && blk_pos == o.blk_pos && tail == o.tail &&
            memcmp(bias.data(), o.bias.data(), bias.size()*sizeof(float)) == 0;
    }
};

// a query batch: n_tps tokens per stream, each (seq, pos)
struct qsa_ubatch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq;
    std::vector<llama_seq_id *> seq_ptr;
    llama_ubatch ub = {};

    void finish() {
        const size_t n = pos.size();
        n_seq_id.assign(n, 1);
        seq_ptr.resize(n);
        for (size_t i = 0; i < n; ++i) {
            seq_ptr[i] = &seq[i];
        }
        ub = {};
        ub.n_tokens     = (uint32_t) n;
        ub.n_seq_tokens = (uint32_t) n;
        ub.n_seqs       = 1;
        ub.n_seqs_unq   = 1;
        ub.n_pos        = 1;
        ub.pos          = pos.data();
        ub.n_seq_id     = n_seq_id.data();
        ub.seq_id       = seq_ptr.data();
    }
};

static int64_t qsa_n_kv(const std::vector<llama_kv_cells *> & cells) {
    int64_t n = 0;
    for (auto * c : cells) {
        n = std::max<int64_t>(n, std::min<int64_t>(c->size(), std::max<int64_t>(256, GGML_PAD(c->used_max_p1(), 256))));
    }
    return n;
}

// both builds on the same cells; returns whether the table served the call
static bool qsa_compare(llama_qsa_blk_table & tab, const std::vector<llama_kv_cells *> & cells, const qsa_ubatch & qb,
                        uint32_t r, const char * what) {
    const int64_t n_ns     = (int64_t) cells.size();
    const int64_t n_kv     = qsa_n_kv(cells);
    const int64_t n_blocks = (n_kv + r - 1)/r;
    const int64_t n_tail   = std::max<int64_t>(r - 1, 1);
    const int64_t n_tokens = qb.ub.n_tokens;

    std::vector<const llama_kv_cells *> cc(cells.begin(), cells.end());

    qsa_out ref(n_kv, n_ns, n_blocks, n_tail, n_tokens, r);
    qsa_out got(n_kv, n_ns, n_blocks, n_tail, n_tokens, r);

    // a position past the window makes the scan assert; the table must leave that state to it
    bool oor = false;
    for (auto * c : cells) {
        for (uint32_t i = 0; i < c->size(); ++i) {
            oor |= !c->is_empty(i) && c->pos_get(i)/(int64_t) r >= n_blocks;
        }
    }

    if (oor) {
        CHECK(!tab.build(cc.data(), got.dst, &qb.ub, r), "%s: served a position past the window", what);
        return false;
    }

    llama_qsa_input_scan(cc.data(), ref.dst, &qb.ub, r, true);

    const uint64_t n_declined = tab.n_declined;
    const bool served = tab.build(cc.data(), got.dst, &qb.ub, r);

    if (!served) {
        CHECK(tab.n_declined == n_declined + 1, "%s: declined without counting", what);
        CHECK(got.blk_cells[0] == 0x7b7b7b7b && got.bias[0] == 12345.0f, "%s: declined after writing", what);
        return false;
    }

    if (!ref.same(got)) {
        CHECK(false, "%s: table differs from the scan (r %u, n_kv %lld, n_tokens %lld)", what, r, (long long) n_kv, (long long) n_tokens);
        for (size_t k = 0; k < ref.blk_cells.size(); ++k) {
            if (ref.blk_cells[k] != got.blk_cells[k]) { fprintf(stderr, "  blk_cells[%zu] scan %d table %d\n", k, ref.blk_cells[k], got.blk_cells[k]); break; }
        }
        for (size_t k = 0; k < ref.blk_pos.size(); ++k) {
            if (ref.blk_pos[k] != got.blk_pos[k]) { fprintf(stderr, "  blk_pos[%zu] scan %d table %d\n", k, ref.blk_pos[k], got.blk_pos[k]); break; }
        }
        for (size_t k = 0; k < ref.bias.size(); ++k) {
            if (memcmp(&ref.bias[k], &got.bias[k], 4) != 0) { fprintf(stderr, "  bias[%zu] scan %g table %g\n", k, ref.bias[k], got.bias[k]); break; }
        }
        for (size_t k = 0; k < ref.tail.size(); ++k) {
            if (ref.tail[k] != got.tail[k]) { fprintf(stderr, "  tail[%zu] scan %d table %d\n", k, ref.tail[k], got.tail[k]); break; }
        }
    }

    return true;
}

// a small model of llama_kv_cache's use of the cells: every mutation through llama_kv_cells' own methods
struct qsa_sim {
    std::mt19937 rng;
    std::vector<llama_kv_cells> strm;   // one per stream
    uint32_t n_seq;

    qsa_sim(uint32_t size, uint32_t n_strm, uint32_t n_seq, uint32_t seed) : rng(seed), strm(n_strm), n_seq(n_seq) {
        for (auto & c : strm) {
            c.resize(size);
            c.jrnl_enable();
        }
    }

    // stream of seq s: round robin, as a non-unified cache maps one sequence per stream
    llama_kv_cells & cells_of(llama_seq_id s) { return strm[s % strm.size()]; }

    int rnd(int n) { return (int) (rng() % (uint32_t) n); }

    // place n tokens of seq s at its next positions in free cells: first fit, or scattered
    bool append(llama_seq_id s, int n, bool scatter) {
        auto & c = cells_of(s);
        llama_pos p = c.seq_pos_max(s) + 1;
        std::vector<uint32_t> fr;
        for (uint32_t i = 0; i < c.size() && (int) fr.size() < (scatter ? 4*n : n); ++i) {
            if (c.is_empty(i)) {
                fr.push_back(i);
            }
        }
        if ((int) fr.size() < n) {
            return false;
        }
        if (scatter) {
            std::shuffle(fr.begin(), fr.end(), rng);
            fr.resize(n);
        }
        for (int k = 0; k < n; ++k) {
            c.pos_set(fr[k], p + k);
            c.seq_add(fr[k], s);
        }
        return true;
    }

    // llama_kv_cache::seq_rm over [p0, p1)
    void seq_rm(llama_seq_id s, llama_pos p0, llama_pos p1) {
        auto & c = cells_of(s);
        for (uint32_t i = 0; i < c.size(); ++i) {
            if (!c.is_empty(i) && c.seq_has(i, s) && c.pos_get(i) >= p0 && (p1 < 0 || c.pos_get(i) < p1)) {
                c.seq_rm(i, s);
            }
        }
    }

    // same-stream seq_cp: the destination joins the source's cells in [p0, p1)
    void seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
        if (&cells_of(src) != &cells_of(dst)) {
            return;
        }
        auto & c = cells_of(src);
        for (uint32_t i = 0; i < c.size(); ++i) {
            if (!c.is_empty(i) && c.seq_has(i, src) && !c.seq_has(i, dst) && c.pos_get(i) >= p0 && c.pos_get(i) < p1) {
                c.seq_add(i, dst);
            }
        }
    }

    void seq_keep(llama_seq_id s) {
        auto & c = cells_of(s);
        for (uint32_t i = 0; i < c.size(); ++i) {
            c.seq_keep(i, s);
        }
    }

    // llama_kv_cache::seq_add: shift [p0, p1) of seq s by d (d < 0 discards what falls below zero)
    void seq_shift(llama_seq_id s, llama_pos p0, llama_pos p1, llama_pos d) {
        auto & c = cells_of(s);
        for (uint32_t i = 0; i < c.size(); ++i) {
            if (!c.is_empty(i) && c.seq_has(i, s) && c.pos_get(i) >= p0 && c.pos_get(i) < p1) {
                c.pos_add(i, d);
            }
        }
        c.reset_shift();
    }

    void seq_div(llama_seq_id s, llama_pos p0, llama_pos p1, int d) {
        auto & c = cells_of(s);
        for (uint32_t i = 0; i < c.size(); ++i) {
            if (!c.is_empty(i) && c.seq_has(i, s) && c.pos_get(i) >= p0 && c.pos_get(i) < p1) {
                c.pos_div(i, d);
            }
        }
        c.reset_shift();
    }

    // a verify batch of seq s: its last n positions (the tokens just applied)
    void queries(qsa_ubatch & qb, llama_seq_id s, int n) {
        const llama_pos pmax = cells_of(s).seq_pos_max(s);
        for (int k = 0; k < n; ++k) {
            qb.pos.push_back(std::max<llama_pos>(0, pmax - n + 1 + k));
            qb.seq.push_back(s);
        }
    }

    // a batch the scan accepts: n tokens of seq s with one stream, n tokens of each stream's sequence with several
    void batch(qsa_ubatch & qb, llama_seq_id s, int n) {
        if (strm.size() == 1) {
            queries(qb, s, n);
        } else {
            for (uint32_t st = 0; st < strm.size(); ++st) {
                queries(qb, (llama_seq_id) st, n);
            }
        }
        qb.finish();
    }
};

static void test_random(uint32_t seed, uint32_t size, uint32_t n_strm, uint32_t n_seq, uint32_t r, int n_steps,
                        uint64_t & n_served, uint64_t & n_calls) {
    qsa_sim sim(size, n_strm, n_seq, seed);
    llama_qsa_blk_table tab;

    std::vector<llama_kv_cells *> cells;
    for (auto & c : sim.strm) {
        cells.push_back(&c);
    }

    for (int step = 0; step < n_steps; ++step) {
        const llama_seq_id s = sim.rnd(n_seq);
        auto & c = sim.cells_of(s);
        const llama_pos pmax = c.seq_pos_max(s);

        const int op = sim.rnd(100);
        char what[128];
        snprintf(what, sizeof(what), "seed %u step %d op %d", seed, step, op);

        if (op < 30) {
            // prefill or decode ubatch
            sim.append(s, op < 6 ? 1 + sim.rnd(3*r + 40) : 1 + sim.rnd(3), op % 7 == 0);
        } else if (op < 45) {
            // draft rollback: drop the last 0..2 positions
            if (pmax >= 0) {
                sim.seq_rm(s, pmax - sim.rnd(3), -1);
            }
        } else if (op < 50) {
            // prefix reuse with a divergent tail
            if (pmax > 0) {
                sim.seq_rm(s, sim.rnd(pmax + 1), -1);
            }
        } else if (op < 53) {
            // a hole in the middle
            if (pmax > 4) {
                const llama_pos p0 = sim.rnd(pmax);
                sim.seq_rm(s, p0, p0 + 1 + sim.rnd(2*r));
            }
        } else if (op < 57) {
            // a slot taking another's prefix: clear it, then share the cells (usually), or join a live one (rarely)
            const llama_seq_id d = sim.rnd(n_seq);
            if (pmax >= 0 && d != s) {
                if (sim.rnd(8) != 0) {
                    sim.seq_rm(d, -1, -1);
                }
                sim.seq_cp(s, d, 0, sim.rnd(pmax + 2));
            }
        } else if (op < 58) {
            sim.seq_keep(s);
        } else if (op < 60) {
            // context shift: discard [p0 - d, p0), move the rest down by d
            if (pmax > 0) {
                const llama_pos p0 = 1 + sim.rnd(pmax);
                const llama_pos d  = 1 + sim.rnd(p0);
                sim.seq_rm(s, p0 - d, p0);
                sim.seq_shift(s, p0, pmax + 1, -d);
            }
        } else if (op < 61) {
            // grouped positions repeat within a block: whatever serves it must match, then start over
            if (pmax > 0) {
                sim.seq_div(s, sim.rnd(pmax + 1), pmax + 1, 2);
                qsa_ubatch qb;
                sim.batch(qb, s, 2);
                char w2[160];
                snprintf(w2, sizeof(w2), "%s (divided)", what);
                n_served += qsa_compare(tab, cells, qb, r, w2);
                n_calls++;
                c.reset();
            }
        } else if (op < 64) {
            // what llama_kv_cache::prepare does: apply, then restore the cells it took
            std::vector<uint32_t> idxs;
            for (uint32_t i = 0; i < c.size() && idxs.size() < 5; ++i) {
                if (c.is_empty(i)) {
                    idxs.push_back(i);
                }
            }
            if (!idxs.empty() && pmax >= 0) {
                const llama_kv_cells backup = c.cp(idxs);
                for (size_t k = 0; k < idxs.size(); ++k) {
                    c.pos_set(idxs[k], pmax + 1 + (llama_pos) k);
                    c.seq_add(idxs[k], s);
                }
                char w2[160];
                snprintf(w2, sizeof(w2), "%s (prepared)", what);
                qsa_ubatch qb;
                sim.batch(qb, s, 3);
                n_served += qsa_compare(tab, cells, qb, r, w2);
                n_calls++;
                c.set(idxs, backup);
            }
        } else if (op < 65) {
            // state restore of a whole sequence: drop it, then rebuild it cell by cell elsewhere
            if (pmax >= 0) {
                const llama_pos n = pmax + 1;
                sim.seq_rm(s, -1, -1);
                sim.append(s, (int) std::min<llama_pos>(n, 64), true);
            }
        } else if (op < 66) {
            if (sim.rnd(4) == 0) {
                c.reset();
            }
        } else if (op < 67) {
            // the cells replaced by a copy: the journal must not be trusted across it
            const llama_kv_cells tmp = c;
            c = tmp;
        } else if (op < 69) {
            // a cell with a position and no sequence yet (apply_ubatch between pos_set and seq_add)
            for (uint32_t i = 0; i < c.size(); ++i) {
                if (c.is_empty(i)) {
                    c.pos_set(i, pmax + 1);
                    qsa_ubatch qb;
                    sim.batch(qb, s, 1);
                    char w2[160];
                    snprintf(w2, sizeof(w2), "%s (no seq)", what);
                    const bool served = qsa_compare(tab, cells, qb, r, w2);
                    CHECK(!served, "%s: served a cell with no sequence", w2);
                    n_calls++;
                    c.seq_add(i, s);
                    break;
                }
            }
        } else if (op < 70) {
            // a repeated position in a block (cache reuse via rm + add): served only by the scan, then gone again
            for (uint32_t i = 0; i < c.size() && pmax >= 0; ++i) {
                if (c.is_empty(i)) {
                    c.pos_set(i, pmax);
                    c.seq_add(i, s);
                    qsa_ubatch qb;
                    sim.batch(qb, s, 2);
                    char w2[160];
                    snprintf(w2, sizeof(w2), "%s (repeated position)", what);
                    // a repeat in another sequence set is no repeat to the scan either, so it may be served
                    n_served += qsa_compare(tab, cells, qb, r, w2);
                    n_calls++;
                    c.rm(i);
                    break;
                }
            }
        }

        // a query batch: one sequence, or two sequences of one stream (the unified cache batches several slots)
        qsa_ubatch qb;
        const int n_q = 1 + sim.rnd(r == 1 ? 3 : 4);
        if (n_strm == 1) {
            for (int k = 0; k < 1 + (sim.rnd(3) == 0); ++k) {
                const llama_seq_id sq = sim.rnd(n_seq);
                if (sim.cells_of(sq).seq_pos_max(sq) >= 0) {
                    sim.queries(qb, sq, n_q);
                }
            }
        } else {
            // n_tps tokens per stream, each stream's own sequence
            for (uint32_t st = 0; st < n_strm; ++st) {
                llama_seq_id sq = st;
                if (sim.cells_of(sq).seq_pos_max(sq) < 0) {
                    sim.append(sq, 1, false);
                }
                sim.queries(qb, sq, n_q);
            }
        }
        if (qb.pos.empty()) {
            continue;
        }
        qb.finish();

        // positions past the window make the scan assert: only query states it accepts
        bool oor = false;
        for (auto * cc : cells) {
            for (llama_seq_id sq = 0; sq < (llama_seq_id) n_seq; ++sq) {
                oor |= cc->seq_pos_max(sq) >= (qsa_n_kv(cells) + r - 1)/r*r;
            }
        }
        if (oor) {
            for (auto & cc : sim.strm) {
                cc.reset();
            }
            continue;
        }

        n_served += qsa_compare(tab, cells, qb, r, what);
        n_calls++;
    }
}

static double qsa_now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// seq 0 at depth, n_other more sequences of other_len; a verify of 3 tokens or a 512-token prefill ubatch.
// per call: the journal replay of a verify's apply + rollback, then the build
static void bench(int64_t depth, int n_tok, int n_other, int64_t other_len) {
    const uint32_t r = 4;
    llama_kv_cells c;
    c.resize(131072 + 4096);
    c.jrnl_enable();

    uint32_t i = 0;
    for (int64_t p = 0; p < depth; ++p, ++i) { c.pos_set(i, p); c.seq_add(i, 0); }
    for (int k = 1; k <= n_other; ++k) {
        for (int64_t p = 0; p < other_len; ++p, ++i) { c.pos_set(i, p); c.seq_add(i, k); }
    }
    const uint32_t i_q = i;
    for (int k = 0; k < n_tok; ++k) { c.pos_set(i_q + k, depth + k); c.seq_add(i_q + k, 0); }

    std::vector<llama_kv_cells *> cells = { &c };
    std::vector<const llama_kv_cells *> cc = { &c };

    qsa_ubatch qb;
    for (int k = 0; k < n_tok; ++k) { qb.pos.push_back(depth + k); qb.seq.push_back(0); }
    qb.finish();

    const int64_t n_kv     = qsa_n_kv(cells);
    const int64_t n_blocks = (n_kv + r - 1)/r;
    qsa_out out(n_kv, 1, n_blocks, r - 1, n_tok, r);

    llama_qsa_blk_table tab;
    tab.build(cc.data(), out.dst, &qb.ub, r);

    const int iters = n_tok > 16 ? 10 : 200;
    std::vector<double> t_scan, t_tab, t_alone;
    // the table alone, its data warm
    for (int it = 0; it < iters; ++it) {
        c.seq_rm(i_q + n_tok - 1, 0);
        c.pos_set(i_q + n_tok - 1, depth + n_tok - 1);
        c.seq_add(i_q + n_tok - 1, 0);
        const double t0 = qsa_now_us();
        tab.build(cc.data(), out.dst, &qb.ub, r);
        t_alone.push_back(qsa_now_us() - t0);
    }
    for (int it = 0; it < iters; ++it) {
        // a verify's cell traffic: the rejected draft token leaves, the ubatch's tokens are applied again
        c.seq_rm(i_q + n_tok - 1, 0);
        c.pos_set(i_q + n_tok - 1, depth + n_tok - 1);
        c.seq_add(i_q + n_tok - 1, 0);

        double t0 = qsa_now_us();
        llama_qsa_input_scan(cc.data(), out.dst, &qb.ub, r, true);
        double t1 = qsa_now_us();
        const bool ok = tab.build(cc.data(), out.dst, &qb.ub, r);
        double t2 = qsa_now_us();
        CHECK(ok, "bench: table declined");
        t_scan.push_back(t1 - t0);
        t_tab.push_back(t2 - t1);
    }
    std::sort(t_scan.begin(), t_scan.end());
    std::sort(t_tab.begin(), t_tab.end());
    std::sort(t_alone.begin(), t_alone.end());
    printf("bench depth %6lld n_tok %3d others %d x %lld  n_kv %6lld n_blocks %5lld: scan %8.1f us  table %7.1f us after the scan, %7.1f us alone (medians of %d)\n",
           (long long) depth, n_tok, n_other, (long long) other_len, (long long) n_kv, (long long) n_blocks,
           t_scan[iters/2], t_tab[iters/2], t_alone[iters/2], iters);
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "bench") {
        for (int64_t d : { 2048, 20480, 65536, 130048 }) {
            bench(d, 3, 0, 0);
        }
        bench(20480, 3, 3, 8192);
        bench(65536, 3, 3, 8192);
        for (int64_t d : { 20480, 65536, 130048 }) {
            bench(d, 512, 0, 0);
        }
        return n_fail == 0 ? 0 : 1;
    }

    uint64_t n_served = 0, n_calls = 0;

    struct cfg { uint32_t size, n_strm, n_seq, r; int steps; };
    const cfg cfgs[] = {
        {  512, 1, 1,  4, 3000 },
        { 2048, 1, 4,  4, 3000 },
        { 2048, 1, 4,  1, 1500 },
        { 2048, 1, 3,  3, 1500 },
        { 4096, 1, 6,  8, 1500 },
        { 4096, 1, 4, 64, 1500 },
        { 1024, 2, 2,  4, 1500 },
        { 1024, 4, 4,  4, 1500 },
    };

    for (uint32_t seed = 1; seed <= 6; ++seed) {
        for (const auto & c : cfgs) {
            test_random(seed*7919 + c.r*31 + c.n_seq, c.size, c.n_strm, c.n_seq, c.r, c.steps, n_served, n_calls);
        }
    }

    printf("test-qsa-host-meta: %llu calls, %llu served by the block table, all served ones identical to the scan: %s\n",
           (unsigned long long) n_calls, (unsigned long long) n_served, n_fail == 0 ? "yes" : "NO");

    CHECK(n_served*10 > n_calls*7, "the table served too few calls (%llu of %llu)", (unsigned long long) n_served, (unsigned long long) n_calls);

    return n_fail == 0 ? 0 : 1;
}
