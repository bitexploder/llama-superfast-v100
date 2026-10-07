#include "mtp-carry.h"

#include "log.h"

#include "ggml-rtimer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

// a get_state record: magic, version, position, width, then the row
static constexpr uint32_t MTP_CARRY_MAGIC   = 0x4350544d; // "MTPC"
static constexpr uint32_t MTP_CARRY_VERSION = 1;
static constexpr size_t   MTP_CARRY_HEADER  = 4*sizeof(int32_t);

static const char * mtp_carry_src_str(int32_t s) {
    switch (s) {
        case common_mtp_carry::SRC_NONE:    return "none";
        case common_mtp_carry::SRC_PROCESS: return "process";
        case common_mtp_carry::SRC_ACCEPT:  return "accept";
        case common_mtp_carry::SRC_PIPE:    return "pipe";
        case common_mtp_carry::SRC_RESTORE: return "restore";
    }
    return "?";
}

bool common_mtp_carry::trace_on() {
    static const bool v = [] { const char * e = getenv("LLAMA_MTP_CARRY_TRACE"); return e != nullptr && atoi(e) != 0; }();
    return v;
}

void common_mtp_carry::init(uint32_t n_seq, int32_t n_embd) {
    this->n_embd = n_embd;
    h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
    pos.assign(n_seq, -1);
    src.assign(n_seq, SRC_NONE);
    win_h.assign(n_seq, {});
    win_pos.assign(n_seq, {});
    zero.assign(n_embd, 0.0f);
}

void common_mtp_carry::reset(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq()) {
        return;
    }
    std::fill(h[seq_id].begin(), h[seq_id].end(), 0.0f);
    pos[seq_id] = -1;
    src[seq_id] = SRC_NONE;
    win_h[seq_id].clear();
    win_pos[seq_id].clear();
    if (trace_on()) {
        LOG_INF("mtp-carry: reset seq %d\n", (int) seq_id);
    }
}

bool common_mtp_carry::valid_for(llama_seq_id seq_id, llama_pos p) const {
    if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq()) {
        return false;
    }
    return p == 0 || (p > 0 && pos[seq_id] == p - 1);
}

const float * common_mtp_carry::row_for(llama_seq_id seq_id, llama_pos p, use_t & use) {
    const bool in_range = seq_id >= 0 && seq_id < (llama_seq_id) n_seq();
    const float * row = zero.data();
    if (p == 0) {
        use = USE_FRESH;
        n_fresh++;
        GGML_RT_COUNT("mtp.carry_fresh", 1);
    } else if (in_range && pos[seq_id] == p - 1) {
        use = USE_CARRY;
        row = h[seq_id].data();
        n_carry++;
        GGML_RT_COUNT("mtp.carry_ok", 1);
    } else {
        use = USE_INVALID;
        n_invalid++;
        GGML_RT_COUNT("mtp.carry_invalid", 1);
        if (n_invalid <= 16 || n_invalid % 1000 == 0) {
            LOG_WRN("mtp-carry: seq %d at pos %d has no target row for pos %d (carry pos %d, src %s): the fresh-boundary row is used (%llu so far)\n",
                    (int) seq_id, (int) p, (int) p - 1, in_range ? (int) pos[seq_id] : -2, in_range ? mtp_carry_src_str(src[seq_id]) : "-",
                    (unsigned long long) n_invalid);
        }
    }
    if (trace_on()) {
        static const char * use_str[] = { "fresh", "carry", "INVALID" };
        LOG_INF("mtp-carry: use seq %d pos %d %s row_pos %d src %s hash %016llx\n", (int) seq_id, (int) p, use_str[use],
                use == USE_CARRY ? (int) pos[seq_id] : -1, use == USE_CARRY ? mtp_carry_src_str(src[seq_id]) : "-",
                (unsigned long long) row_hash(row));
    }
    return row;
}

void common_mtp_carry::set(llama_seq_id seq_id, llama_pos p, const float * row, src_t s) {
    if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq()) {
        return;
    }
    std::memcpy(h[seq_id].data(), row, (size_t) n_embd*sizeof(float));
    pos[seq_id] = p;
    src[seq_id] = s;
    if (trace_on()) {
        LOG_INF("mtp-carry: set seq %d pos %d src %s hash %016llx\n", (int) seq_id, (int) p, mtp_carry_src_str(s),
                (unsigned long long) row_hash(row));
    }
}

void common_mtp_carry::set_window(llama_seq_id seq_id, const float * rows, llama_pos p0, int32_t n, src_t s) {
    if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq() || n <= 0) {
        return;
    }
    win_h[seq_id].assign(rows, rows + (size_t) n*n_embd);
    win_pos[seq_id].resize(n);
    for (int32_t i = 0; i < n; ++i) {
        win_pos[seq_id][i] = p0 + i;
    }
    set(seq_id, p0 + n - 1, win_h[seq_id].data() + (size_t) (n - 1)*n_embd, s);
}

void common_mtp_carry::set_windows(const llama_batch & batch, const float * h_tgt) {
    const size_t row_bytes = (size_t) n_embd*sizeof(float);

    // the window of a sequence in this batch replaces its previous one; absent sequences keep theirs
    std::vector<int32_t> last(n_seq(), -1);
    std::vector<size_t>  cnt(n_seq(), 0);
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        const llama_seq_id s = batch.seq_id[k][0];
        if (s >= 0 && s < (llama_seq_id) n_seq()) {
            cnt[s]++;
        }
    }
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq(); ++s) {
        if (cnt[s] > 0) {
            win_h[s].clear();
            win_h[s].reserve(cnt[s]*n_embd);
            win_pos[s].clear();
            win_pos[s].reserve(cnt[s]);
        }
    }
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        const llama_seq_id s = batch.seq_id[k][0];
        if (s < 0 || s >= (llama_seq_id) n_seq()) {
            continue;
        }
        const size_t off = win_h[s].size();
        win_h[s].resize(off + n_embd);
        std::memcpy(win_h[s].data() + off, h_tgt + (size_t) k*n_embd, row_bytes);
        win_pos[s].push_back(batch.pos[k]);
        last[s] = k;
    }
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq(); ++s) {
        if (last[s] >= 0) {
            const int32_t n = (int32_t) win_pos[s].size();
            set(s, win_pos[s][n - 1], win_h[s].data() + (size_t) (n - 1)*n_embd, SRC_PROCESS);
        }
    }
}

void common_mtp_carry::accept(llama_seq_id seq_id, int32_t n_accepted) {
    if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq()) {
        return;
    }
    const int32_t n_rows = (int32_t) win_pos[seq_id].size();
    if (n_rows <= 0) {
        return;
    }
    const int32_t i = std::min<int32_t>(std::max<int32_t>(0, n_accepted), n_rows - 1);
    set(seq_id, win_pos[seq_id][i], win_h[seq_id].data() + (size_t) i*n_embd, SRC_ACCEPT);
}

void common_mtp_carry::gather(const llama_batch & batch, const float * h_tgt, float * dst) {
    const size_t row_bytes = (size_t) n_embd*sizeof(float);

    std::vector<int32_t> prev(n_seq(), -1);
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        const llama_seq_id s = batch.seq_id[k][0];
        float * d = dst + (size_t) k*n_embd;
        if (s < 0 || s >= (llama_seq_id) n_seq()) {
            std::memset(d, 0, row_bytes);
            continue;
        }
        if (prev[s] >= 0 && batch.pos[k] == batch.pos[prev[s]] + 1) {
            std::memcpy(d, h_tgt + (size_t) prev[s]*n_embd, row_bytes);
        } else {
            // the sequence's first row in the batch (or, never from the server, a row whose predecessor is not the previous
            // row of its sequence: the batch row at p-1 is not there, so it is invalid unless the carry has it)
            use_t use;
            std::memcpy(d, row_for(s, batch.pos[k], use), row_bytes);
        }
        prev[s] = k;
    }
}

size_t common_mtp_carry::state_size() const {
    return MTP_CARRY_HEADER + (size_t) n_embd*sizeof(float);
}

bool common_mtp_carry::get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const {
    data.clear();
    if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq() || pos[seq_id] < 0) {
        if (trace_on()) {
            LOG_INF("mtp-carry: save seq %d: none\n", (int) seq_id);
        }
        return false;
    }
    const int32_t hdr[4] = { (int32_t) MTP_CARRY_MAGIC, (int32_t) MTP_CARRY_VERSION, (int32_t) pos[seq_id], n_embd };
    data.resize(state_size());
    std::memcpy(data.data(), hdr, MTP_CARRY_HEADER);
    std::memcpy(data.data() + MTP_CARRY_HEADER, h[seq_id].data(), (size_t) n_embd*sizeof(float));
    if (trace_on()) {
        LOG_INF("mtp-carry: save seq %d pos %d src %s hash %016llx\n", (int) seq_id, (int) pos[seq_id], mtp_carry_src_str(src[seq_id]),
                (unsigned long long) row_hash(h[seq_id].data()));
    }
    return true;
}

bool common_mtp_carry::set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq()) {
        return false;
    }
    int32_t hdr[4] = { 0, 0, -1, 0 };
    if (data.size() >= MTP_CARRY_HEADER) {
        std::memcpy(hdr, data.data(), MTP_CARRY_HEADER);
    }
    const bool ok = data.size() == state_size() && hdr[0] == (int32_t) MTP_CARRY_MAGIC && hdr[1] == (int32_t) MTP_CARRY_VERSION &&
                    hdr[2] >= 0 && hdr[3] == n_embd;
    if (!ok) {
        if (trace_on()) {
            LOG_INF("mtp-carry: restore seq %d: %s record (%zu bytes), reset\n", (int) seq_id, data.empty() ? "no" : "invalid", data.size());
        }
        reset(seq_id);
        return false;
    }
    // a restored boundary has no verification window: the next round builds its own
    win_h[seq_id].clear();
    win_pos[seq_id].clear();
    set(seq_id, hdr[2], (const float *) (data.data() + MTP_CARRY_HEADER), SRC_RESTORE);
    return true;
}

uint64_t common_mtp_carry::row_hash(const float * row) const {
    uint64_t x = 1469598103934665603ull;
    const uint8_t * b = (const uint8_t *) row;
    for (size_t i = 0; i < (size_t) n_embd*sizeof(float); ++i) {
        x ^= b[i];
        x *= 1099511628211ull;
    }
    return x;
}
