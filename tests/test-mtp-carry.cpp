// MTP's boundary state across request and cache boundaries, on the production helpers:
//   - common_mtp_carry (common/mtp-carry.h), which the MTP drafter uses for every read and write of its carried row;
//   - llama_batch_allocr's ubatch order and llama_batch_row_swaps, which put the target's per-token rows in batch order;
//   - server_prompt_cache's accounting of the speculative bytes of an entry.
// Every target row is distinguishable: row(lineage, seq, pos) encodes all three, so a wrong pairing shows which row it took.
// The lifecycle steps follow the server's calls (process: gather then set_windows; a round: set_windows then accept; a
// checkpoint: get_state before its batch; a restore: set_state; a cleared slot: reset), without a model.

#include "testing.h"

#include "llama.h"
#include "common.h"
#include "mtp-carry.h"

#include "../src/llama-batch.h"
#include "../src/llama-vocab.h"

#include "server-task.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr int32_t N_EMBD = 6;

// the target's hidden row of token (seq, pos) in a request lineage: every field distinguishable
static std::vector<float> trow(int lineage, llama_seq_id seq, llama_pos pos) {
    std::vector<float> r(N_EMBD);
    r[0] = 1000.0f*lineage + 100.0f*seq + 0.5f; // lineage and sequence
    r[1] = (float) pos;
    r[2] = (float) lineage;
    r[3] = (float) seq;
    r[4] = -(float) pos;
    r[5] = 7.25f;
    return r;
}

static std::string desc(const float * r) {
    if (r[5] == 0.0f && r[0] == 0.0f && r[1] == 0.0f) {
        return "zero";
    }
    return "L" + std::to_string((int) r[2]) + "/s" + std::to_string((int) r[3]) + "/p" + std::to_string((int) r[1]);
}

static bool same(const float * a, const std::vector<float> & b) {
    return std::memcmp(a, b.data(), N_EMBD*sizeof(float)) == 0;
}

static bool is_zero(const float * a) {
    for (int k = 0; k < N_EMBD; ++k) {
        if (a[k] != 0.0f) {
            return false;
        }
    }
    return true;
}

// a target batch (tokens of one or more sequences) and the target's rows for it, in batch order
struct tbatch {
    llama_batch b;
    std::vector<float> h;   // [n][N_EMBD]
    std::vector<int>   lin; // lineage of each token

    explicit tbatch(int n_max) : b(llama_batch_init(n_max, 0, 1)) {}
    ~tbatch() { llama_batch_free(b); }

    void add(int lineage, llama_seq_id seq, llama_pos pos) {
        common_batch_add(b, 1, pos, { seq }, false);
        auto r = trow(lineage, seq, pos);
        h.insert(h.end(), r.begin(), r.end());
        lin.push_back(lineage);
    }

    // the server's process(): the drafter's inputs from the carry, then the windows and carries
    std::vector<float> process(common_mtp_carry & c) {
        std::vector<float> dst((size_t) b.n_tokens*N_EMBD, -1.0f);
        c.gather(b, h.data(), dst.data());
        c.set_windows(b, h.data());
        return dst;
    }
};

// a prompt (or its chunk) of seq, positions [p0, p1)
static std::vector<float> prompt(common_mtp_carry & c, int lineage, llama_seq_id seq, llama_pos p0, llama_pos p1) {
    tbatch tb(p1 - p0);
    for (llama_pos p = p0; p < p1; ++p) {
        tb.add(lineage, seq, p);
    }
    return tb.process(c);
}

// a speculative round of seq at pos0 with n_draft drafts, n_acc accepted: the verify batch, then accept
static std::vector<float> spec_round(common_mtp_carry & c, int lineage, llama_seq_id seq, llama_pos pos0, int n_draft, int n_acc) {
    tbatch tb(n_draft + 1);
    for (int i = 0; i <= n_draft; ++i) {
        tb.add(lineage, seq, pos0 + i);
    }
    auto dst = tb.process(c);
    c.accept(seq, n_acc);
    return dst;
}

static void test_carry(testing & t) {
    t.test("fresh_and_cleared", [](testing & t) {
        common_mtp_carry c;
        c.init(2, N_EMBD);

        // a new context: the fresh boundary at position 0, every other row its own predecessor
        auto in = prompt(c, 1, 1, 0, 5);
        t.assert_true("pos 0 takes the zero row", is_zero(&in[0]));
        for (int k = 1; k < 5; ++k) {
            t.assert_true("pos " + std::to_string(k) + " takes row p-1, got " + desc(&in[k*N_EMBD]), same(&in[k*N_EMBD], trow(1, 1, k - 1)));
        }
        t.assert_equal("carry pos", 4, c.pos[1]);
        t.assert_true("sequence 0 untouched", c.pos[0] == -1);

        // the slot is cleared, and an unrelated request of the same length comes: position 0 again takes the fresh row,
        // and a continuation at the old position finds no carry, although the old carry was for exactly that position
        c.reset(1);
        t.assert_true("cleared: not valid at 5", !c.valid_for(1, 5));
        t.assert_true("cleared: valid at 0", c.valid_for(1, 0));
        auto in2 = prompt(c, 2, 1, 0, 5);
        t.assert_true("new request pos 0: zero row, not the old carry", is_zero(&in2[0]));
        t.assert_true("new request pos 1: its own row", same(&in2[N_EMBD], trow(2, 1, 0)));

        c.reset(1);
        const uint64_t inv0 = c.n_invalid;
        auto in3 = prompt(c, 3, 1, 5, 7);
        t.assert_true("cleared, continued at 5: zero row", is_zero(&in3[0]));
        t.assert_equal("counted invalid", inv0 + 1, c.n_invalid);
        t.assert_true("then its own predecessor", same(&in3[N_EMBD], trow(3, 1, 5)));
        t.assert_true("valid again after the batch", c.valid_for(1, 7));
    });

    t.test("checkpoint_restore", [](testing & t) {
        common_mtp_carry c;
        c.init(2, N_EMBD);

        // the original request: chunks [0,4) and [4,8). The checkpoint at n_tokens 4 is taken before the second chunk
        prompt(c, 1, 0, 0, 4);
        t.assert_true("valid at the checkpoint's position", c.valid_for(0, 4));
        std::vector<uint8_t> ckpt;
        t.assert_true("get_state", c.get_state(0, ckpt));
        t.assert_equal("record size", c.state_size(), ckpt.size());
        prompt(c, 1, 0, 4, 8);
        // generation: rounds with partial acceptance move the carry well past the checkpoint
        spec_round(c, 1, 0, 8, 3, 1);   // accepts pos 8 (+1 draft): carry at 9
        t.assert_equal("carry after round 1", 9, c.pos[0]);
        spec_round(c, 1, 0, 10, 3, 3);  // all accepted: carry at 13
        t.assert_equal("carry after round 2", 13, c.pos[0]);

        // clean construction of the same prefix in another context: its carry at the checkpoint's position
        common_mtp_carry clean;
        clean.init(2, N_EMBD);
        prompt(clean, 1, 0, 0, 4);

        // restore and catch up one token
        t.assert_true("set_state", c.set_state(0, ckpt));
        t.assert_true("restored valid at 4", c.valid_for(0, 4));
        t.assert_true("restored row equals the clean construction's", std::memcmp(c.h[0].data(), clean.h[0].data(), N_EMBD*sizeof(float)) == 0);
        t.assert_equal("restored provenance", (int) common_mtp_carry::SRC_RESTORE, c.src[0]);
        auto one = prompt(c, 1, 0, 4, 5);
        t.assert_true("one-token catch-up takes row 3, got " + desc(&one[0]), same(&one[0], trow(1, 0, 3)));

        // restore again, catch up several tokens
        t.assert_true("set_state again", c.set_state(0, ckpt));
        auto multi = prompt(c, 1, 0, 4, 10);
        for (int k = 0; k < 6; ++k) {
            t.assert_true("multi-token catch-up row " + std::to_string(k) + ": got " + desc(&multi[k*N_EMBD]), same(&multi[k*N_EMBD], trow(1, 0, 3 + k)));
        }

        // a restore drops the verification window: an accept after it cannot bring back a row from before the restore
        spec_round(c, 1, 0, 10, 3, 0);
        t.assert_true("set_state", c.set_state(0, ckpt));
        c.accept(0, 2);
        t.assert_equal("accept after a restore keeps the restored row", 3, c.pos[0]);
    });

    t.test("cache_restore_into_another_slot", [](testing & t) {
        common_mtp_carry c;
        c.init(2, N_EMBD);

        // slot 0 serves request A to the end of its last round; slot 1 serves an unrelated request B of the same length
        prompt(c, 1, 0, 0, 6);
        spec_round(c, 1, 0, 6, 3, 2);   // carry at 8
        prompt(c, 2, 1, 0, 6);
        spec_round(c, 2, 1, 6, 3, 2);   // carry at 8 too: same position, another lineage
        t.assert_equal("A at 8", 8, c.pos[0]);
        t.assert_equal("B at 8", 8, c.pos[1]);

        // A's entry is saved at its endpoint, then loaded into slot 1
        std::vector<uint8_t> entry;
        t.assert_true("save A", c.get_state(0, entry));
        t.assert_true("load into slot 1", c.set_state(1, entry));
        auto in = prompt(c, 1, 1, 9, 12);
        t.assert_true("slot 1 continues A: row of A at 8, got " + desc(&in[0]), same(&in[0], trow(1, 0, 8)));
        t.assert_true("then its own rows", same(&in[N_EMBD], trow(1, 1, 9)));

        // an entry without the state (or a malformed one) resets the destination instead of keeping its row
        prompt(c, 2, 1, 0, 9);
        t.assert_true("B again valid at 9", c.valid_for(1, 9));
        t.assert_true("empty record", !c.set_state(1, {}));
        t.assert_true("empty record: not valid at 9", !c.valid_for(1, 9));
        t.assert_equal("empty record: reset", -1, c.pos[1]);

        prompt(c, 2, 1, 0, 9);
        auto bad = entry;
        bad[12] ^= 1; // the width
        t.assert_true("malformed record", !c.set_state(1, bad));
        t.assert_true("malformed record: not valid at 9", !c.valid_for(1, 9));
        auto bad2 = entry;
        bad2.pop_back();
        prompt(c, 2, 1, 0, 9);
        t.assert_true("short record", !c.set_state(1, bad2));
        t.assert_equal("short record: reset", -1, c.pos[1]);

        // a sequence without a valid carry saves nothing
        c.reset(1);
        std::vector<uint8_t> none = { 1, 2, 3 };
        t.assert_true("no state to save", !c.get_state(1, none));
        t.assert_true("no state: empty record", none.empty());
    });

    t.test("partial_acceptance_and_cancellation", [](testing & t) {
        common_mtp_carry c;
        c.init(1, N_EMBD);
        prompt(c, 1, 0, 0, 4);

        // a round at 4 with 3 drafts: accepted 0, 2, and more than the window holds
        spec_round(c, 1, 0, 4, 3, 0);
        t.assert_equal("0 accepted: the round's first token", 4, c.pos[0]);
        t.assert_true("its row", same(c.h[0].data(), trow(1, 0, 4)));
        auto in = spec_round(c, 1, 0, 5, 3, 2);
        t.assert_true("next round pairs with row 4", same(&in[0], trow(1, 0, 4)));
        t.assert_equal("2 accepted", 7, c.pos[0]);
        spec_round(c, 1, 0, 8, 2, 9);
        t.assert_equal("clamped to the window", 10, c.pos[0]);

        // cancelled between a verify and its accept: the slot is cleared, and the late accept cannot revive the window
        tbatch tb(4);
        for (int i = 0; i < 4; ++i) {
            tb.add(1, 0, 11 + i);
        }
        tb.process(c);
        c.reset(0);
        c.accept(0, 1);
        t.assert_equal("accept after a reset does nothing", -1, c.pos[0]);
    });

    t.test("interleaved_sequences", [](testing & t) {
        common_mtp_carry c;
        c.init(2, N_EMBD);
        prompt(c, 1, 0, 0, 3);   // seq 0 carries pos 2
        prompt(c, 2, 1, 0, 5);   // seq 1 carries pos 4

        // a batch whose sequences are not contiguous: each row still pairs with its own sequence's previous token
        tbatch tb(6);
        tb.add(1, 0, 3);
        tb.add(1, 0, 4);
        tb.add(2, 1, 5);
        tb.add(2, 1, 6);
        tb.add(1, 0, 5);
        tb.add(2, 1, 7);
        auto in = tb.process(c);
        const std::vector<std::vector<float>> want = {
            trow(1, 0, 2), trow(1, 0, 3), trow(2, 1, 4), trow(2, 1, 5), trow(1, 0, 4), trow(2, 1, 6) };
        for (int k = 0; k < 6; ++k) {
            t.assert_true("row " + std::to_string(k) + ": got " + desc(&in[k*N_EMBD]), same(&in[k*N_EMBD], want[k]));
        }
        t.assert_equal("seq 0 window", (size_t) 3, c.win_pos[0].size());
        t.assert_equal("seq 0 carry", 5, c.pos[0]);
        t.assert_equal("seq 1 carry", 7, c.pos[1]);
        c.accept(1, 1);
        t.assert_true("seq 1 accept 1: row at 6", c.pos[1] == 6 && same(c.h[1].data(), trow(2, 1, 6)));
    });
}

// the target's per-token rows land in ubatch order; the swaps must put them in batch order
static void test_row_order(testing & t) {
    llama_vocab vocab;

    struct pbatch {
        std::vector<float>       embd;
        std::vector<llama_pos>   pos;
        std::vector<int32_t>     n_seq_id;
        std::vector<int8_t>      logits;
        std::vector<llama_seq_id> seq;
        std::vector<llama_seq_id *> seq_ptr;
        void add(llama_seq_id s, llama_pos p, bool out = false) {
            embd.push_back(0.0f);
            pos.push_back(p);
            n_seq_id.push_back(1);
            seq.push_back(s);
            logits.push_back(out ? 1 : 0);
        }
        llama_batch make() {
            seq_ptr.clear();
            for (auto & s : seq) {
                seq_ptr.push_back(&s);
            }
            llama_batch b = {};
            b.n_tokens = (int32_t) seq.size();
            b.embd     = embd.data();
            b.pos      = pos.data();
            b.n_seq_id = n_seq_id.data();
            b.seq_id   = seq_ptr.data();
            b.logits   = logits.data();
            return b;
        }
    };

    // split as the hybrid memory does (split_equal, not sequential), then put the rows as the context stores them (ubatch
    // order) back into batch order with the swaps; returns whether the ubatch order differed
    auto check = [&](testing & t, pbatch & pb, uint32_t n_ubatch, uint32_t keep_tail, bool sequential = false) {
        llama_batch_allocr ba(1);
        t.assert_true("init", ba.init(pb.make(), vocab, nullptr, 1, 4, false));
        ba.split_reset();
        while (ba.split_equal(n_ubatch, sequential, keep_tail).n_tokens > 0) {}
        const auto & tok = ba.get_tok_ids();
        t.assert_equal("every token once", (size_t) pb.seq.size(), tok.size());

        std::vector<float> rows;
        for (int32_t b : tok) {
            auto r = trow(1, pb.seq[b], pb.pos[b]);
            rows.insert(rows.end(), r.begin(), r.end());
        }
        std::vector<llama_row_swap> swaps;
        t.assert_true("a permutation", llama_batch_row_swaps(tok, swaps));
        for (const auto & sw : swaps) {
            std::swap_ranges(rows.begin() + sw.i0*N_EMBD, rows.begin() + (sw.i0 + 1)*N_EMBD, rows.begin() + sw.i1*N_EMBD);
        }
        for (size_t k = 0; k < pb.seq.size(); ++k) {
            t.assert_true("batch row " + std::to_string(k) + " holds its own token, got " + desc(&rows[k*N_EMBD]),
                    same(&rows[k*N_EMBD], trow(1, pb.seq[k], pb.pos[k])));
        }
        bool identity = true;
        for (size_t r = 0; r < tok.size(); ++r) {
            identity = identity && tok[r] == (int32_t) r;
        }
        t.assert_equal("no swaps exactly when the order is kept", identity, swaps.empty());
        return !identity;
    };

    t.test("two_prompts_interleave", [&](testing & t) {
        // two slots reading prompts in one batch (Flash-Next, no TU): A's last chunk then B's first
        pbatch pb;
        for (int p = 0; p < 10; ++p) pb.add(0, 100 + p, p == 9);
        for (int p = 0; p < 4; ++p)  pb.add(1, p);
        t.assert_true("the ubatches interleave the sequences", check(t, pb, 8, 3));
    });

    t.test("two_prompts_outputs_out_of_order", [&](testing & t) {
        // both prompts end in the batch: B's output lands before A's in ubatch order
        pbatch pb;
        for (int p = 0; p < 12; ++p) pb.add(0, p, p == 11);
        for (int p = 0; p < 4; ++p)  pb.add(1, p, p == 3);
        t.assert_true("interleaved", check(t, pb, 8, 0));
    });

    t.test("verify_rows_keep_order", [&](testing & t) {
        // the 27B: two verify windows of unequal width (8 and 4 rows), keep_tail n_rs_seq + 1 = 8
        pbatch pb;
        for (int p = 0; p < 8; ++p) pb.add(0, 50 + p, true);
        for (int p = 0; p < 4; ++p) pb.add(1, 70 + p, true);
        t.assert_true("kept", !check(t, pb, 512, 8));
        pbatch pb2;
        for (int p = 0; p < 4; ++p) pb2.add(0, 50 + p, true);
        for (int p = 0; p < 8; ++p) pb2.add(1, 70 + p, true);
        t.assert_true("kept (wider second)", !check(t, pb2, 512, 8));
        // Flash-Next: widths 3 and 1, keep_tail 3; and a verify window then a prompt chunk
        pbatch pb3;
        for (int p = 0; p < 3; ++p) pb3.add(0, 50 + p, true);
        pb3.add(1, 70, true);
        t.assert_true("kept (3 and 1)", !check(t, pb3, 1024, 3));
        pbatch pb4;
        for (int p = 0; p < 3; ++p)  pb4.add(0, 50 + p, true);
        for (int p = 0; p < 40; ++p) pb4.add(1, p, p == 39);
        t.assert_true("kept (verify then prompt)", !check(t, pb4, 16, 3));
    });

    t.test("flashnext_observed_batches", [&](testing & t) {
        // (from a Flash-Next run, whose log shows these two co-batched prompt batches: slot 0's tokens 4096-4971
        // with slot 1's 0-971, then 4972-5995 with 972-1995; -ub 1024, non-unified cache, so split_equal is sequential,
        // keep_tail n_rs_seq + 1 = 3)
        pbatch pb;
        for (int p = 4096; p < 4972; ++p) pb.add(0, p);
        for (int p = 0; p < 972; ++p)    pb.add(1, p);
        t.assert_true("876 + 972: interleaved", check(t, pb, 1024, 3, true));
        pbatch pb2;
        for (int p = 4972; p < 5996; ++p) pb2.add(0, p);
        for (int p = 972; p < 1996; ++p)  pb2.add(1, p);
        t.assert_true("1024 + 1024: interleaved", check(t, pb2, 1024, 3, true));
    });

    t.test("one_sequence_keeps_order", [&](testing & t) {
        pbatch pb;
        for (int p = 0; p < 20; ++p) pb.add(0, p, p == 19);
        t.assert_true("kept", !check(t, pb, 8, 3));
    });

    t.test("not_a_permutation", [&](testing & t) {
        std::vector<llama_row_swap> swaps;
        t.assert_true("duplicate", !llama_batch_row_swaps({ 0, 0, 1 }, swaps));
        t.assert_true("out of range", !llama_batch_row_swaps({ 0, 3, 1 }, swaps));
        t.assert_true("empty swaps", swaps.empty());
    });
}

static void test_cache_accounting(testing & t) {
    t.test("entry_counts_speculative_bytes", [](testing & t) {
        server_prompt_cache cache(1, 0); // 1 MiB
        server_prompt prompt;
        prompt.tokens = server_tokens(llama_tokens { 1, 2, 3 }, false);

        const size_t main_size = 1024*1024 - 64;
        // without the speculative bytes the entry fits the limit; with them it does not, and is skipped
        t.assert_true("over the limit with its spec bytes", cache.alloc(prompt, main_size, 0, 128) == nullptr);
        auto * cur = cache.alloc(prompt, main_size, 0, 32);
        t.assert_true("within the limit", cur != nullptr);
        if (cur) {
            cur->data.spec.assign(32, 0x5a);
            t.assert_equal("entry size includes them", main_size + 32, cur->data.size());
            t.assert_equal("cache size includes them", main_size + 32, cache.size());
        }
    });
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("carry",     test_carry);
    t.test("row_order", test_row_order);
    t.test("cache",     test_cache_accounting);

    return t.summary();
}
