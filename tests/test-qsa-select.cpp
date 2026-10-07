// Test for ggml_qsa_select, the qwen4exp QSA block selection.
//
// A brute-force reference written here (sort every selectable block by score desc, then block
// index asc), the CPU backend and every GPU backend must produce the same result bit for bit,
// and a GPU backend must produce the same result when run again. The score distributions are
// chosen so that ties straddle the k-th place: few distinct values, many zeros, -0 against +0,
// and rows with fewer selectable blocks than k.
//
// Every case runs again with the bias packed to one bit per block (I32), which must give
// the same output bytes as the F32 bias. ggml_qsa_vis_rows, the sparse prompt mask entries from
// one bit per cell, must give the same bytes on the CPU as get_rows of the F16 mask plus an add,
// and the same bytes on every GPU backend as on the CPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

enum score_mode {
    SCORE_RANDOM,  // continuous, ties unlikely
    SCORE_TIES,    // four values, half of them zero
    SCORE_ZERO,    // every score 0
    SCORE_SIGNED0, // -0, +0 and 1
    SCORE_NAN,     // ties plus some NaN (never selectable)
};

struct sel_case {
    const char * name;
    int64_t    n_blocks;
    int64_t    r;
    int64_t    n_rows;
    int64_t    n_stream;
    int64_t    n_tail;
    int        k;
    int        width;
    score_mode mode;
    double     p_valid; // chance a block is selectable (bias 0 rather than -inf)
};

struct sel_data {
    std::vector<float>   score, bias;
    std::vector<int32_t> cells, tail;
};

static sel_data make_data(const sel_case & c) {
    std::mt19937 rng(77 + (uint32_t) c.n_blocks*3 + (uint32_t) c.n_rows*5 + (uint32_t) c.mode*11);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::normal_distribution<float>       nd(0.0f, 1.0f);

    const int64_t n_el = c.n_blocks*c.n_rows*c.n_stream;

    sel_data d;
    d.score.resize(n_el);
    d.bias.resize(n_el);
    for (int64_t i = 0; i < n_el; ++i) {
        float v = 0.0f;
        switch (c.mode) {
            case SCORE_RANDOM:  v = fabsf(nd(rng))*3.0f; break;
            case SCORE_TIES:    v = ud(rng) < 0.5f ? 0.0f : 0.25f*(float) (1 + (int) (ud(rng)*3.0f)); break;
            case SCORE_ZERO:    v = 0.0f; break;
            case SCORE_SIGNED0: v = ud(rng) < 0.3f ? 1.0f : (ud(rng) < 0.5f ? -0.0f : 0.0f); break;
            case SCORE_NAN:     v = ud(rng) < 0.05f ? NAN : (ud(rng) < 0.5f ? 0.0f : 0.5f*(float) (1 + (int) (ud(rng)*2.0f))); break;
        }
        d.score[i] = v;
        d.bias[i]  = ud(rng) < c.p_valid ? 0.0f : -INFINITY;
    }

    // distinct cells per stream in a scrambled layout, as a ring-buffer cache gives
    d.cells.resize(c.r*c.n_blocks*c.n_stream);
    for (int64_t s = 0; s < c.n_stream; ++s) {
        std::vector<int32_t> perm(c.r*c.n_blocks);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::copy(perm.begin(), perm.end(), d.cells.begin() + s*c.r*c.n_blocks);
    }

    // 0..n_tail cells, then -1
    d.tail.assign(c.n_tail*c.n_rows*c.n_stream, -1);
    for (int64_t row = 0; row < c.n_rows*c.n_stream; ++row) {
        const int64_t nt = (int64_t) (ud(rng)*(float) (c.n_tail + 1)) % (c.n_tail + 1);
        for (int64_t j = 0; j < nt; ++j) {
            d.tail[row*c.n_tail + j] = (int32_t) (c.r*c.n_blocks + row*c.n_tail + j);
        }
    }

    return d;
}

// independent of the backends' key encoding: plain float comparison, which already treats -0 as +0
static std::vector<int32_t> reference(const sel_case & c, const sel_data & d) {
    const int64_t n_rows = c.n_rows*c.n_stream;
    const int64_t plane  = (int64_t) c.width*n_rows;

    std::vector<int32_t> out(2*plane);

    for (int64_t row = 0; row < n_rows; ++row) {
        const int64_t s = row/c.n_rows;

        std::vector<std::pair<float, int32_t>> cand;
        for (int64_t b = 0; b < c.n_blocks; ++b) {
            const float v = d.score[row*c.n_blocks + b] + d.bias[row*c.n_blocks + b];
            if (v > -INFINITY) {
                cand.emplace_back(v, (int32_t) b);
            }
        }
        std::sort(cand.begin(), cand.end(), [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        if ((int64_t) cand.size() > c.k) {
            cand.resize(c.k);
        }
        std::vector<int32_t> blocks;
        for (const auto & x : cand) {
            blocks.push_back(x.second);
        }
        std::sort(blocks.begin(), blocks.end());

        std::vector<int32_t> sel;
        for (const int32_t b : blocks) {
            for (int64_t j = 0; j < c.r; ++j) {
                sel.push_back(d.cells[s*c.r*c.n_blocks + b*c.r + j]);
            }
        }
        for (int64_t j = 0; j < c.n_tail && d.tail[row*c.n_tail + j] >= 0; ++j) {
            sel.push_back(d.tail[row*c.n_tail + j]);
        }
        if ((int64_t) sel.size() > c.width) {
            sel.resize(c.width);
        }

        const int32_t first = sel.empty() ? 0 : sel[0];
        for (int64_t j = 0; j < c.width; ++j) {
            const bool used = j < (int64_t) sel.size();
            out[row*c.width + j]         = used ? sel[j] : first;
            out[plane + row*c.width + j] = used ? 0 : INT32_MIN;
        }
    }

    return out;
}

// the bias as one bit per block: bit b&31 of word b>>5 set for 0, clear for -inf, padding clear
static std::vector<int32_t> pack_bias(const sel_case & c, const sel_data & d) {
    const int64_t n_words = (c.n_blocks + 31)/32;
    const int64_t n_rows  = c.n_rows*c.n_stream;

    std::vector<int32_t> bits(n_words*n_rows, 0);
    for (int64_t row = 0; row < n_rows; ++row) {
        for (int64_t b = 0; b < c.n_blocks; ++b) {
            const float v = d.bias[row*c.n_blocks + b];
            GGML_ASSERT(v == 0.0f || v == -INFINITY);
            if (v == 0.0f) {
                bits[row*n_words + (b >> 5)] |= (int32_t) (1u << (b & 31));
            }
        }
    }

    return bits;
}

// runs the op `reps` times on backend; returns false if the backend does not support it
// packed: the bias goes in as pack_bias's I32 words instead of F32
static bool run_backend(ggml_backend_t backend, const sel_case & c, const sel_data & d, int reps,
                        std::vector<std::vector<int32_t>> & outs, bool packed = false) {
    ggml_context * ctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });

    const std::vector<int32_t> bits = packed ? pack_bias(c, d) : std::vector<int32_t>();

    ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_blocks, c.n_rows, c.n_stream);
    ggml_tensor * bias  = packed ?
        ggml_new_tensor_3d(ctx, GGML_TYPE_I32, (c.n_blocks + 31)/32, c.n_rows, c.n_stream) :
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_blocks, c.n_rows, c.n_stream);
    ggml_tensor * cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.r*c.n_blocks, c.n_stream);
    ggml_tensor * tail  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, c.n_tail, c.n_rows, c.n_stream);
    ggml_tensor * out   = ggml_qsa_select(ctx, score, bias, cells, tail, c.k, c.width);

    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(score, d.score.data(), 0, ggml_nbytes(score));
    ggml_backend_tensor_set(bias,  packed ? (const void *) bits.data() : (const void *) d.bias.data(), 0, ggml_nbytes(bias));
    ggml_backend_tensor_set(cells, d.cells.data(), 0, ggml_nbytes(cells));
    ggml_backend_tensor_set(tail,  d.tail.data(),  0, ggml_nbytes(tail));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < reps; ++i) {
        // poison the output so that a slot the backend leaves unwritten shows up
        std::vector<int32_t> junk(ggml_nelements(out), 0x5a5a5a5a + i);
        ggml_backend_tensor_set(out, junk.data(), 0, ggml_nbytes(out));

        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        std::vector<int32_t> h(ggml_nelements(out));
        ggml_backend_tensor_get(out, h.data(), 0, ggml_nbytes(out));
        outs.push_back(std::move(h));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static int64_t first_diff(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (a[i] != b[i]) {
            return (int64_t) i;
        }
    }
    return a.size() == b.size() ? -1 : (int64_t) std::min(a.size(), b.size());
}

// ggml_qsa_vis_rows

struct vis_case {
    const char * name;
    int64_t n_kv;
    int64_t n_rows;
    int64_t width;
    double  p_vis;  // chance a cell is visible
    double  p_ninf; // chance a bias entry is -inf
};

struct vis_data {
    std::vector<int32_t>     vis;  // [(n_kv + 31)/32, n_rows]
    std::vector<ggml_fp16_t> mask; // [n_kv, n_rows], 0 or -inf: the same visibility as the F16 KQ mask
    std::vector<int32_t>     ids;  // [width, n_rows]
    std::vector<float>       bias; // [width, n_rows]
};

static vis_data make_vis_data(const vis_case & c) {
    std::mt19937 rng(911 + (uint32_t) c.n_kv*7 + (uint32_t) c.n_rows*13 + (uint32_t) c.width);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::normal_distribution<float>       nd(0.0f, 1.0f);

    const int64_t n_words = (c.n_kv + 31)/32;

    vis_data d;
    d.vis.assign(n_words*c.n_rows, 0);
    d.mask.resize(c.n_kv*c.n_rows);
    for (int64_t row = 0; row < c.n_rows; ++row) {
        for (int64_t j = 0; j < c.n_kv; ++j) {
            const bool v = ud(rng) < c.p_vis;
            if (v) {
                d.vis[row*n_words + (j >> 5)] |= (int32_t) (1u << (j & 31));
            }
            d.mask[row*c.n_kv + j] = ggml_fp32_to_fp16(v ? 0.0f : -INFINITY);
        }
    }

    d.ids.resize(c.width*c.n_rows);
    d.bias.resize(c.width*c.n_rows);
    std::uniform_int_distribution<int32_t> cell(0, (int32_t) c.n_kv - 1);
    for (int64_t i = 0; i < c.width*c.n_rows; ++i) {
        d.ids[i] = cell(rng);
        const float u = ud(rng);
        d.bias[i] = u < c.p_ninf ? -INFINITY : (u < c.p_ninf + 0.3f ? 0.0f : (u < c.p_ninf + 0.4f ? -0.0f : nd(rng)));
    }

    return d;
}

// ref_form: the graph's unpacked form, get_rows of the F16 mask's [1, n_kv, n_rows] view plus the add
static bool run_vis(ggml_backend_t backend, const vis_case & c, const vis_data & d, bool ref_form, std::vector<float> & out) {
    ggml_context * ctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });

    ggml_tensor * vis  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, (c.n_kv + 31)/32, c.n_rows);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, c.n_rows);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.width, c.n_rows);
    ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.width, c.n_rows);

    ggml_tensor * res = nullptr;
    if (ref_form) {
        ggml_tensor * rows = ggml_view_4d(ctx, mask, 1, c.n_kv, c.n_rows, 1, mask->nb[0], mask->nb[1], mask->nb[3], 0);
        res = ggml_get_rows(ctx, rows, ggml_reshape_3d(ctx, ids, c.width, c.n_rows, 1));
        res = ggml_reshape_2d(ctx, res, c.width, c.n_rows);
        res = ggml_add(ctx, res, bias);
    } else {
        res = ggml_qsa_vis_rows(ctx, vis, ids, bias);
    }

    if (!ggml_backend_supports_op(backend, res)) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(vis,  d.vis.data(),  0, ggml_nbytes(vis));
    ggml_backend_tensor_set(mask, d.mask.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(ids,  d.ids.data(),  0, ggml_nbytes(ids));
    ggml_backend_tensor_set(bias, d.bias.data(), 0, ggml_nbytes(bias));

    // poison the output so that an entry the backend leaves unwritten shows up
    std::vector<float> junk(ggml_nelements(res), 12345.0f);
    ggml_backend_tensor_set(res, junk.data(), 0, ggml_nbytes(res));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    out.resize(ggml_nelements(res));
    ggml_backend_tensor_get(res, out.data(), 0, ggml_nbytes(res));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static bool same_bytes(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    std::vector<std::string>    names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (be) {
                backends.push_back(be);
                names.push_back(ggml_backend_dev_name(dev));
            }
        }
    }

    const sel_case cases[] = {
        // name              n_blocks  r  rows ns tail  k    width mode           p_valid
        { "few-blocks",           100, 4,   1, 1,  3,  512,  400, SCORE_TIES,    1.00 }, // everything fits (width = n_kv)
        { "decode-16k-ties",     4096, 4,   1, 1,  3,  512, 2051, SCORE_TIES,    0.95 },
        { "verify-4-random",     4096, 4,   4, 1,  3,  512, 2051, SCORE_RANDOM,  0.90 },
        { "streams-ties",        8192, 4,   2, 2,  3,  512, 2051, SCORE_TIES,    0.90 },
        { "global-keys-ties",   20000, 4,   3, 1,  3,  512, 2051, SCORE_TIES,    0.97 }, // keys not staged in shared memory
        { "128k-ties",          32768, 4,   2, 1,  3,  512, 2051, SCORE_TIES,    0.99 },
        { "128k-random",        32768, 4,   1, 2,  3,  512, 2051, SCORE_RANDOM,  0.99 },
        { "few-valid",           4096, 4,   3, 1,  3,  512, 2051, SCORE_TIES,    0.05 }, // fewer selectable blocks than k
        { "none-valid",          1024, 4,   2, 1,  3,  512, 2051, SCORE_TIES,    0.00 },
        { "all-zero",            4096, 4,   2, 1,  3,  512, 2051, SCORE_ZERO,    0.90 }, // one tie group straddles the cut
        { "signed-zero",         4096, 4,   2, 1,  3,  512, 2051, SCORE_SIGNED0, 0.90 },
        { "nan",                 4096, 4,   2, 1,  3,  512, 2051, SCORE_NAN,     0.90 },
        { "prefill-rows",        2048, 4, 128, 1,  3,  512, 2051, SCORE_TIES,    0.90 },
        { "ratio-2",             5000, 2,   3, 1,  1, 1024, 2049, SCORE_TIES,    0.90 },
    };

    bool ok = true;
    for (const auto & c : cases) {
        const sel_data d = make_data(c);
        const std::vector<int32_t> ref = reference(c, d);

        int64_t n_fill = 0;
        for (size_t i = ref.size()/2; i < ref.size(); ++i) {
            n_fill += ref[i] != 0;
        }

        bool case_ok = true;
        std::string report;
        for (size_t bi = 0; bi < backends.size(); ++bi) {
            // F32 bias, then the packed bias, which must also give the F32 run's bytes
            for (const bool packed : { false, true }) {
                const char * form = packed ? "packed" : "f32";
                std::vector<std::vector<int32_t>> outs;
                if (!run_backend(backends[bi], c, d, 3, outs, packed)) {
                    report += " " + names[bi] + "/" + form + ":unsupported";
                    case_ok = false;
                    continue;
                }
                bool form_ok = true;
                for (size_t i = 0; i < outs.size(); ++i) {
                    const int64_t fd = first_diff(ref, outs[i]);
                    if (fd >= 0) {
                        fprintf(stderr, "[%s] %s %s run %zu differs from the reference at %lld: %d vs %d\n",
                                c.name, names[bi].c_str(), form, i, (long long) fd, fd < (int64_t) outs[i].size() ? outs[i][fd] : -1,
                                fd < (int64_t) ref.size() ? ref[fd] : -1);
                        form_ok = false;
                    }
                }
                report += " " + names[bi] + "/" + form + ":" + (form_ok ? "same" : "DIFF");
                case_ok &= form_ok;
            }
        }

        printf("[%s] n_blocks=%lld rows=%lld streams=%lld k=%d width=%d filler_slots=%lld%s %s\n",
               c.name, (long long) c.n_blocks, (long long) c.n_rows, (long long) c.n_stream, c.k, c.width,
               (long long) n_fill, report.c_str(), case_ok ? "OK" : "FAIL");
        ok &= case_ok;
    }

    const vis_case vcases[] = {
        // name                 n_kv  rows  width  p_vis p_ninf
        { "vis-tiny",             37,    5,    37, 0.50, 0.20 }, // one partial word
        { "vis-pad",            4100,   64,  2051, 0.70, 0.10 }, // n_kv not a multiple of 32
        { "vis-16k",           16384,  128,  2051, 0.95, 0.05 },
        { "vis-131k",         131072,   16,  2051, 0.99, 0.02 },
        { "vis-all-ninf",       1024,    8,   300, 0.00, 0.50 }, // nothing visible
        { "vis-bias-ninf",      2048,    8,   512, 1.00, 0.90 }, // mostly -inf bias
    };

    ggml_backend_t cpu = nullptr;
    for (size_t bi = 0; bi < backends.size(); ++bi) {
        if (ggml_backend_dev_type(ggml_backend_get_device(backends[bi])) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu = backends[bi];
            break;
        }
    }
    GGML_ASSERT(cpu != nullptr);

    for (const auto & c : vcases) {
        const vis_data d = make_vis_data(c);

        bool case_ok = true;
        std::string report;

        // on the CPU, against get_rows of the F16 mask plus the add
        std::vector<float> ref, cpu_out;
        if (!run_vis(cpu, c, d, true, ref) || !run_vis(cpu, c, d, false, cpu_out)) {
            report += " CPU:unsupported";
            case_ok = false;
        } else {
            const bool same = same_bytes(ref, cpu_out);
            report += std::string(" CPU-vs-get_rows+add:") + (same ? "same" : "DIFF");
            case_ok &= same;
        }

        // every other backend against the CPU's result
        for (size_t bi = 0; bi < backends.size() && case_ok; ++bi) {
            if (backends[bi] == cpu) {
                continue;
            }
            std::vector<float> out;
            if (!run_vis(backends[bi], c, d, false, out)) {
                report += " " + names[bi] + ":unsupported";
                case_ok = false;
                continue;
            }
            const bool same = same_bytes(cpu_out, out);
            report += " " + names[bi] + "-vs-CPU:" + (same ? "same" : "DIFF");
            case_ok &= same;
        }

        int64_t n_ninf = 0;
        for (const float v : ref) {
            n_ninf += v == -INFINITY;
        }

        printf("[%s] n_kv=%lld rows=%lld width=%lld ninf_entries=%lld%s %s\n",
               c.name, (long long) c.n_kv, (long long) c.n_rows, (long long) c.width, (long long) n_ninf,
               report.c_str(), case_ok ? "OK" : "FAIL");
        ok &= case_ok;
    }

    for (auto be : backends) {
        ggml_backend_free(be);
    }

    printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
