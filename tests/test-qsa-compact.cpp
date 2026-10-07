// Differential test for the qwen4exp QSA compact selected-KV attention.
//
// Reference: dense flash attention over the whole cache view with the QSA mask
// (the cell's own mask where top_k names it, -inf elsewhere), which is what
// build_attn_qsa computes on its original path. Candidate: llama_qsa_compact_attn,
// which gathers the selected rows and their mask entries and attends over those.
// Both run on the first GPU backend with the same BF16 cache contents.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../src/models/qwen4exp-qsa-compact.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

struct qsa_case {
    const char * name;
    int64_t n_kv;       // attended cache view length
    int64_t kv_size;    // allocated cells per stream (>= n_kv, gives the stream stride)
    int64_t n_stream;
    int64_t n_tps;      // queries per stream
    int64_t width;      // top-k width
    int64_t pos0;       // position of the first query; cells < pos are visible (causal)
    int64_t pad_to;
    int64_t n_rep = 0;  // trailing slots that repeat the first cell with a -inf top_k_bias, as ggml_qsa_select emits
    int64_t n_diff = -1; // >= 0: queries after the first keep all but n_diff of the first query's other cells (overlap)
};

static const int64_t D      = 256;
static const int64_t N_HEAD = 24;
static const int64_t N_HKV  = 2;

static bool run_case(ggml_backend_t backend, const qsa_case & c) {
    std::mt19937 rng(1234 + (uint32_t) c.n_kv + (uint32_t) c.n_tps*7 + (uint32_t) c.n_stream*13);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int64_t n_tokens = c.n_tps*c.n_stream;

    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead()*2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * cache_k = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, D, N_HKV, c.kv_size, c.n_stream);
    ggml_tensor * cache_v = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, D, N_HKV, c.kv_size, c.n_stream);
    ggml_tensor * q       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_HEAD, n_tokens);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tps, 1, c.n_stream);
    ggml_tensor * top_k   = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, c.width, c.n_tps, 1, c.n_stream);
    ggml_tensor * dmask   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tps, 1, c.n_stream);
    ggml_tensor * tk_bias = c.n_rep > 0 ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.width, c.n_tps, 1, c.n_stream) : nullptr;

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // cache contents (host copies kept for the exact reference)
    std::vector<ggml_bf16_t> hk, hv;
    std::vector<float> hq;
    {
        std::vector<float> f(ggml_nelements(cache_k));
        std::vector<ggml_bf16_t> b(f.size());
        for (auto & x : f) x = nd(rng);
        ggml_fp32_to_bf16_row(f.data(), b.data(), (int64_t) f.size());
        ggml_backend_tensor_set(cache_k, b.data(), 0, ggml_nbytes(cache_k));
        hk = b;
        for (auto & x : f) x = nd(rng);
        ggml_fp32_to_bf16_row(f.data(), b.data(), (int64_t) f.size());
        ggml_backend_tensor_set(cache_v, b.data(), 0, ggml_nbytes(cache_v));

        std::vector<float> qf(ggml_nelements(q));
        for (auto & x : qf) x = 0.25f*nd(rng);
        ggml_backend_tensor_set(q, qf.data(), 0, ggml_nbytes(q));
        hq = qf;
    }

    // causal mask, per-query selection (distinct cells, deliberately including some
    // invisible ones so the gathered mask must carry -inf), and the dense QSA mask
    std::vector<ggml_fp16_t> m(ggml_nelements(kq_mask)), dm(ggml_nelements(dmask));
    std::vector<int32_t> tk(ggml_nelements(top_k));
    std::vector<float>   tb(ggml_nelements(top_k), 0.0f);
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f), f16_ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t s = 0; s < c.n_stream; ++s) {
        std::vector<int32_t> sel0;
        for (int64_t t = 0; t < c.n_tps; ++t) {
            const int64_t pos = c.pos0 + t + s*17; // streams hold different histories
            const int64_t row = (s*c.n_tps + t)*c.n_kv;
            for (int64_t i = 0; i < c.n_kv; ++i) {
                m[row + i] = i <= pos ? f16_zero : f16_ninf;
            }
            std::vector<int32_t> cells(c.n_kv);
            std::iota(cells.begin(), cells.end(), 0);
            std::shuffle(cells.begin(), cells.end(), rng);
            // the query's own cell and the incomplete tail are always selected, as QSA biases them visible
            const int64_t tail0 = pos - pos % 4;
            std::vector<int32_t> sel;
            for (int64_t i = tail0; i <= pos; ++i) sel.push_back((int32_t) i);
            for (int32_t x : cells) {
                if ((int64_t) sel.size() >= c.width) break;
                if (x >= tail0 && x <= pos) continue;
                sel.push_back(x);
            }
            if (c.n_diff >= 0 && t > 0) {
                // adjacent verification queries select mostly the same cells: keep the first
                // query's cells outside this tail, swap n_diff of them for others
                std::vector<int32_t> keep;
                for (int32_t x : sel0) if (!(x >= tail0 && x <= pos)) keep.push_back(x);
                std::shuffle(keep.begin(), keep.end(), rng);
                keep.resize(std::max<int64_t>(0, (int64_t) keep.size() - c.n_diff));
                std::vector<char> in(c.n_kv, 0);
                std::vector<int32_t> nsel;
                for (int64_t i = tail0; i <= pos; ++i) { nsel.push_back((int32_t) i); in[i] = 1; }
                for (int32_t x : keep) if (!in[x] && (int64_t) nsel.size() < c.width) { nsel.push_back(x); in[x] = 1; }
                for (int32_t x : cells) {
                    if ((int64_t) nsel.size() >= c.width) break;
                    if (!in[x]) { nsel.push_back(x); in[x] = 1; }
                }
                sel = nsel;
            }
            if (t == 0) sel0 = sel;
            std::shuffle(sel.begin(), sel.end(), rng); // top-k output order is not sorted
            // repeats: the cell they displace is not selected, and the repeat itself must not count
            if (c.n_rep > 0) {
                std::stable_partition(sel.begin(), sel.end(), [&](int32_t x) { return x >= tail0 && x <= pos; });
                sel.resize(c.width - c.n_rep);
            }
            for (int64_t j = 0; j < c.width; ++j) {
                const bool r = j >= (int64_t) sel.size();
                tk[(s*c.n_tps + t)*c.width + j] = r ? sel[0] : sel[j];
                tb[(s*c.n_tps + t)*c.width + j] = r ? (float) INT32_MIN : 0.0f;
            }
            for (int64_t i = 0; i < c.n_kv; ++i) dm[row + i] = f16_ninf;
            for (int32_t x : sel) dm[row + x] = m[row + x];
        }
    }
    // poison V rows that no query of the stream may attend (unselected, or selected but
    // invisible), so any leak of such a cell moves the output far beyond the tolerance
    {
        std::vector<ggml_bf16_t> vb(ggml_nelements(cache_v));
        ggml_backend_tensor_get(cache_v, vb.data(), 0, ggml_nbytes(cache_v));
        const ggml_bf16_t poison = ggml_fp32_to_bf16(30.0f);
        int64_t n_poison = 0;
        for (int64_t s = 0; s < c.n_stream; ++s) {
            std::vector<char> ok(c.kv_size, 0);
            for (int64_t t = 0; t < c.n_tps; ++t) {
                const int64_t row = (s*c.n_tps + t)*c.n_kv;
                for (int64_t i = 0; i < c.n_kv; ++i) {
                    ok[i] |= dm[row + i] == f16_zero;
                }
            }
            for (int64_t i = 0; i < c.kv_size; ++i) {
                if (ok[i]) continue;
                ++n_poison;
                ggml_bf16_t * r = vb.data() + (s*c.kv_size + i)*N_HKV*D;
                std::fill(r, r + N_HKV*D, poison);
            }
        }
        ggml_backend_tensor_set(cache_v, vb.data(), 0, ggml_nbytes(cache_v));
        hv = vb;
        printf("[%s] poisoned %lld of %lld cells\n", c.name, (long long) n_poison, (long long) (c.kv_size*c.n_stream));
    }

    ggml_backend_tensor_set(kq_mask, m.data(),  0, ggml_nbytes(kq_mask));
    ggml_backend_tensor_set(top_k,   tk.data(), 0, ggml_nbytes(top_k));
    ggml_backend_tensor_set(dmask,   dm.data(), 0, ggml_nbytes(dmask));
    if (tk_bias) {
        ggml_backend_tensor_set(tk_bias, tb.data(), 0, ggml_nbytes(tk_bias));
    }

    // cache views as llama_kv_cache::get_k/get_v present them: [D, n_hkv, n_kv, n_stream]
    ggml_context * gctx = ggml_init({ 256*ggml_tensor_overhead() + ggml_graph_overhead()*2, nullptr, true });
    ggml_tensor * k = ggml_view_4d(gctx, cache_k, D, N_HKV, c.n_kv, c.n_stream, cache_k->nb[1], cache_k->nb[2], cache_k->nb[3], 0);
    ggml_tensor * v = ggml_view_4d(gctx, cache_v, D, N_HKV, c.n_kv, c.n_stream, cache_v->nb[1], cache_v->nb[2], cache_v->nb[3], 0);
    const float scale = 1.0f/sqrtf((float) D);

    // reference: dense, as build_attn_mha does it
    ggml_tensor * rq = ggml_view_4d(gctx, q, D, N_HEAD, c.n_tps, c.n_stream, q->nb[1], q->nb[2], q->nb[2]*c.n_tps, 0);
    rq = ggml_permute(gctx, rq, 0, 2, 1, 3);
    ggml_tensor * ref = ggml_flash_attn_ext(gctx, rq, ggml_permute(gctx, k, 0, 2, 1, 3), ggml_permute(gctx, v, 0, 2, 1, 3), dmask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(ref, GGML_PREC_F32);
    ref = ggml_reshape_2d(gctx, ref, ref->ne[0]*ref->ne[1], ref->ne[2]*ref->ne[3]);

    if (!llama_qsa_compact_applies(q, k, v, kq_mask, top_k, 8, c.pad_to)) {
        fprintf(stderr, "[%s] FAIL: compact path does not apply\n", c.name);
        return false;
    }
    ggml_tensor * fa = nullptr;
    ggml_tensor * out = llama_qsa_compact_attn(gctx, q, k, v, kq_mask, top_k, scale, 0.0f, 0.0f, c.pad_to,
            ggml_backend_get_device(backend), &fa, tk_bias);
    // on the GPU the K/V gathers must be the direct f16 form
    if (!(fa->src[1]->view_src && fa->src[1]->view_src->op == GGML_OP_GET_ROWS && fa->src[1]->view_src->type == GGML_TYPE_F16)) {
        fprintf(stderr, "[%s] FAIL: GPU build did not use the direct f16 gather\n", c.name);
        return false;
    }

    // the union form, for verification batches
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_tensor * fau  = nullptr;
    ggml_tensor * outu = nullptr;
    if (c.n_tps > 1) {
        if (!llama_qsa_union_applies(k, top_k, dev)) {
            fprintf(stderr, "[%s] FAIL: union form does not apply\n", c.name);
            return false;
        }
        outu = llama_qsa_union_attn(gctx, q, k, v, kq_mask, top_k, scale, 0.0f, 0.0f, c.pad_to, dev, &fau, tk_bias);
        ggml_set_output(fau->src[3]->src[0]->view_src); // keep the union readable after the graph
    }

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, out);
    if (outu) {
        ggml_build_forward_expand(gf, outu);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[%s] FAIL: compute\n", c.name);
        return false;
    }

    std::vector<float> a(ggml_nelements(ref)), b(ggml_nelements(out));
    GGML_ASSERT(a.size() == b.size());
    ggml_backend_tensor_get(ref, a.data(), 0, ggml_nbytes(ref));
    ggml_backend_tensor_get(out, b.data(), 0, ggml_nbytes(out));

    // exact reference in double over the selected visible cells, from the stored bf16 cache
    std::vector<double> ex(a.size(), 0.0);
    for (int64_t s = 0; s < c.n_stream; ++s) {
        for (int64_t t = 0; t < c.n_tps; ++t) {
            const int64_t tok = s*c.n_tps + t, row = tok*c.n_kv;
            for (int64_t h = 0; h < N_HEAD; ++h) {
                const int64_t hk_i = h/(N_HEAD/N_HKV);
                const float * qr = hq.data() + (tok*N_HEAD + h)*D;
                std::vector<double> w(c.n_kv, 0.0);
                double mx = -INFINITY;
                for (int64_t i = 0; i < c.n_kv; ++i) {
                    if (dm[row + i] != f16_zero) continue;
                    const ggml_bf16_t * kr = hk.data() + ((s*c.kv_size + i)*N_HKV + hk_i)*D;
                    double dot = 0.0;
                    for (int64_t d = 0; d < D; ++d) dot += (double) qr[d]*ggml_bf16_to_fp32(kr[d]);
                    w[i] = dot*scale; mx = std::max(mx, w[i]);
                }
                double den = 0.0;
                std::vector<double> acc(D, 0.0);
                for (int64_t i = 0; i < c.n_kv; ++i) {
                    if (dm[row + i] != f16_zero) continue;
                    const double e = exp(w[i] - mx); den += e;
                    const ggml_bf16_t * vr = hv.data() + ((s*c.kv_size + i)*N_HKV + hk_i)*D;
                    for (int64_t d = 0; d < D; ++d) acc[d] += e*ggml_bf16_to_fp32(vr[d]);
                }
                for (int64_t d = 0; d < D; ++d) ex[tok*N_HEAD*D + h*D + d] = acc[d]/den;
            }
        }
    }
    double dense_exact = 0.0, compact_exact = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dense_exact   = std::max(dense_exact,   fabs(a[i] - ex[i]));
        compact_exact = std::max(compact_exact, fabs(b[i] - ex[i]));
    }

    double max_err = 0.0, max_ref = 0.0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        finite &= std::isfinite(b[i]);
        max_err = std::max(max_err, (double) fabsf(a[i] - b[i]));
        max_ref = std::max(max_ref, (double) fabsf(a[i]));
    }
    // the compact path must stay within the dense path's own error envelope against exact
    bool ok = finite && max_err <= 2e-3*std::max(1.0, max_ref) && compact_exact <= 2.0*dense_exact + 5e-5;
    if (c.n_diff >= 0) {
        ok = true; // the union cases judge the union form below; the per-query form is reported only
    }

    if (outu) {
        std::vector<float> bu(ggml_nelements(outu));
        GGML_ASSERT(bu.size() == a.size());
        ggml_backend_tensor_get(outu, bu.data(), 0, ggml_nbytes(outu));
        // the union's cell count, from plane 0 of stream 0 (the first cell repeats or -1 follows it)
        const ggml_tensor * un = fau->src[3]->src[0]->view_src; // mask cast <- view <- union
        std::vector<float> p0(un->ne[0]);
        ggml_backend_tensor_get(un, p0.data(), 0, p0.size()*sizeof(float));
        int64_t n_u = 1;
        while (n_u < (int64_t) p0.size() && p0[n_u] > p0[n_u - 1]) n_u++;
        double union_exact = 0.0, uerr = 0.0;
        bool ufinite = true;
        for (size_t i = 0; i < a.size(); ++i) {
            ufinite &= std::isfinite(bu[i]);
            union_exact = std::max(union_exact, fabs(bu[i] - ex[i]));
            uerr = std::max(uerr, (double) fabsf(a[i] - bu[i]));
        }
        if (getenv("QSA_TEST_PER_TOKEN")) {
            for (int64_t tok = 0; tok < c.n_tps*c.n_stream; ++tok) {
                double eu = 0.0, ec = 0.0, ed = 0.0;
                for (int64_t i = tok*N_HEAD*D; i < (tok + 1)*N_HEAD*D; ++i) {
                    eu = std::max(eu, fabs(bu[i] - ex[i])); ec = std::max(ec, fabs(b[i] - ex[i])); ed = std::max(ed, fabs(a[i] - ex[i]));
                }
                printf("[%s] token %lld vs exact: dense %.3g compact %.3g union %.3g\n", c.name, (long long) tok, ed, ec, eu);
            }
        }
        // it replaces the per-query form, so it must stay within the envelope of the worse of the two
        // existing paths against exact
        const bool uok = ufinite && un->op == GGML_OP_QSA_UNION && uerr <= 2e-3*std::max(1.0, max_ref) &&
                         union_exact <= 2.0*std::max(dense_exact, compact_exact) + 5e-5;
        printf("[%s] union: cells %lld of %lld gathered rows, vs exact %.3g, vs dense %.3g %s\n", c.name,
               (long long) n_u, (long long) fau->src[1]->ne[1], union_exact, uerr, uok ? "OK" : "FAIL");
        ok &= uok;
    }
    printf("[%s] vs exact fp64: dense %.3g compact %.3g\n", c.name, dense_exact, compact_exact);
    printf("[%s] n_kv=%lld streams=%lld tps=%lld width=%lld rep=%lld kv=%lld fa_kv=%lld max_err=%.3g max_ref=%.3g %s\n",
           c.name, (long long) c.n_kv, (long long) c.n_stream, (long long) c.n_tps, (long long) c.width, (long long) c.n_rep,
           (long long) c.n_kv, (long long) fa->src[1]->ne[1], max_err, max_ref, ok ? "OK" : "FAIL");

    ggml_gallocr_free(ga);
    ggml_free(gctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        fprintf(stderr, "no GPU backend; skipping\n");
        return 0;
    }

    const qsa_case cases[] = {
        // name                 n_kv  kv_size ns tps width pos0  pad
        { "decode",             4096, 4096,   1, 1,  2051, 3000, 256 },
        { "decode-short-hist",  4096, 4096,   1, 1,  2051, 1500, 256 }, // fewer visible cells than width
        { "verify-3",           4096, 4096,   1, 3,  2051, 3500, 256 },
        { "verify-2-streams",   4096, 6144,   2, 2,  2051, 3200, 256 }, // stream stride > n_kv
        { "decode-no-pad",      8192, 8192,   1, 1,  2048, 7000, 256 }, // width already aligned
        { "tail-boundary",      4096, 4096,   1, 3,  2051, 2047, 256 }, // queries straddle a 4-token block
        { "verify-3-long",     16384, 16384,  1, 3,  2051, 16000, 256 }, // most cells unselected
        // ggml_qsa_select output: repeats of the first cell masked by top_k_bias
        { "decode-repeats",     4096, 4096,   1, 1,  2051, 3000, 256, 3 },   // tail slots unused
        { "verify-3-repeats",   4096, 4096,   1, 3,  2051, 1500, 256, 700 }, // few blocks visible
        { "verify-2-str-rep",   4096, 6144,   2, 2,  2051, 3200, 256, 2 },
        // the union form's own shapes: adjacent queries select mostly the same cells
        { "union-3-overlap",   16384, 16384,  1, 3,  2051, 16000, 256, 0, 64 },
        { "union-3-same",      16384, 16384,  1, 3,  2051, 16000, 256, 0, 0 },    // union = one selection + tails
        { "union-3-rep",       16384, 16384,  1, 3,  2051, 1500,  256, 700, 32 }, // repeats inside the union
        { "union-2-streams",    8192, 10240,  2, 2,  2051, 6000,  256, 0, 200 },
        { "union-4-ub4",        8192, 8192,   1, 4,  2051, 7000,  256, 0, 128 },  // the KL leg's -ub 4
        { "union-8",           16384, 16384,  1, 8,  2051, 12000, 256, 0, 300 },  // the largest verification batch
    };

    bool ok = true;
    for (const auto & c : cases) {
        ok &= run_case(backend, c);
    }

    // ggml_qsa_union: the GPU result equals the CPU backend's bit for bit, with repeats (bias
    // INT32_MIN), duplicate cells across queries, finite non-zero mask entries and two streams
    for (const int64_t n_kv : { (int64_t) 3000, (int64_t) 70000 }) {
        const int64_t width = 2051, n_tps = 3, n_stream = 2, n_rows = 8, pad = 256;
        const int64_t n_max = GGML_PAD(std::min(width*n_tps, n_kv), pad);
        ggml_context * ctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, width, n_tps, 1, n_stream);
        ggml_tensor * mk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 5, 1, n_stream); // more mask rows than queries
        ggml_tensor * bi = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, n_tps, 1, n_stream);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::mt19937 rng(11 + (uint32_t) n_kv);
        std::vector<int32_t> hk(ggml_nelements(tk));
        std::vector<float>   hb(hk.size());
        for (int64_t s = 0; s < n_stream; ++s) {
            std::vector<int32_t> base(width);
            for (auto & x : base) x = (int32_t) (rng() % n_kv);
            for (int64_t t = 0; t < n_tps; ++t) {
                for (int64_t j = 0; j < width; ++j) {
                    const int64_t i = (s*n_tps + t)*width + j;
                    const bool rep = j >= width - 40*(t + 1);
                    hk[i] = rep ? base[0] : (rng() % 8 == 0 ? (int32_t) (rng() % n_kv) : base[j]);
                    hb[i] = rep ? (float) INT32_MIN : 0.0f;
                }
            }
        }
        std::vector<ggml_fp16_t> hm(ggml_nelements(mk));
        for (auto & x : hm) {
            const uint32_t r = rng() % 4;
            x = ggml_fp32_to_fp16(r == 0 ? -INFINITY : r == 1 ? -0.5f*(float) (rng() % 7) : 0.0f);
        }
        ggml_backend_tensor_set(tk, hk.data(), 0, ggml_nbytes(tk));
        ggml_backend_tensor_set(mk, hm.data(), 0, ggml_nbytes(mk));
        ggml_backend_tensor_set(bi, hb.data(), 0, ggml_nbytes(bi));

        ggml_context * gctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * u = ggml_qsa_union(gctx, tk, mk, bi, (int) n_max, (int) n_rows, (int) pad);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, u);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        bool same = ggml_backend_supports_op(backend, u) && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        std::vector<float> hg(ggml_nelements(u));
        ggml_backend_tensor_get(u, hg.data(), 0, ggml_nbytes(u));

        // the same op on the CPU backend, from host copies of the inputs
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_context * cctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * ctk = ggml_dup_tensor(cctx, tk), * cmk = ggml_dup_tensor(cctx, mk), * cbi = ggml_dup_tensor(cctx, bi);
        ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, cpu);
        ggml_backend_tensor_set(ctk, hk.data(), 0, ggml_nbytes(ctk));
        ggml_backend_tensor_set(cmk, hm.data(), 0, ggml_nbytes(cmk));
        ggml_backend_tensor_set(cbi, hb.data(), 0, ggml_nbytes(cbi));
        ggml_context * cg = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * cu = ggml_qsa_union(cg, ctk, cmk, cbi, (int) n_max, (int) n_rows, (int) pad);
        ggml_cgraph * cgf = ggml_new_graph(cg);
        ggml_build_forward_expand(cgf, cu);
        ggml_gallocr_t cga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        ggml_gallocr_alloc_graph(cga, cgf);
        same = same && ggml_backend_graph_compute(cpu, cgf) == GGML_STATUS_SUCCESS;
        std::vector<float> hc(ggml_nelements(cu));
        ggml_backend_tensor_get(cu, hc.data(), 0, ggml_nbytes(cu));
        same = same && memcmp(hg.data(), hc.data(), hg.size()*sizeof(float)) == 0;

        // and against a direct reading of the contract, stream 0
        std::vector<int32_t> cells(hk.begin(), hk.begin() + n_tps*width);
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        const int64_t n = (int64_t) cells.size();
        bool contract = GGML_PAD(n, pad) <= n_max;
        for (int64_t i = 0; i < n_max && contract; ++i) {
            const float want = i < n ? (float) cells[i] : i < GGML_PAD(n, pad) ? (float) cells[0] : -1.0f;
            contract = hg[i] == want;
        }
        for (int64_t t = 0; t < n_rows && contract; ++t) {
            for (int64_t i = 0; i < n_max && contract; ++i) {
                float want = -INFINITY;
                for (int64_t j = 0; t < n_tps && i < n && j < width; ++j) {
                    if (hk[t*width + j] == cells[i]) {
                        want = std::max(want, ggml_fp16_to_fp32(hm[t*n_kv + cells[i]]) + hb[t*width + j]);
                    }
                }
                if (want == 0.0f) want = 0.0f;
                contract = memcmp(&hg[(1 + t)*n_max + i], &want, sizeof(float)) == 0;
            }
        }
        printf("[qsa_union] n_kv %lld: %lld cells of %lld slots, gpu == cpu: %d, contract: %d %s\n", (long long) n_kv,
               (long long) n, (long long) (n_tps*width), same, contract, same && contract ? "OK" : "FAIL");
        ok &= same && contract;

        ggml_gallocr_free(cga);
        ggml_free(cg);
        ggml_backend_buffer_free(cbuf);
        ggml_free(cctx);
        ggml_backend_free(cpu);
        ggml_gallocr_free(ga);
        ggml_free(gctx);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // ggml_get_rows_as(F16) on the GPU equals ggml_get_rows + cast bit for bit (bf16 -> f16 both via f32);
    // the CPU backend, whose kernels only write F32/I32, reports it unsupported
    {
        ggml_context * ctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * a  = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, 512, 300, 2);
        ggml_tensor * ix = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 700, 2);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::mt19937 rng(7);
        std::vector<float> f(ggml_nelements(a));
        for (auto & x : f) x = std::normal_distribution<float>(0.0f, 3.0f)(rng);
        std::vector<ggml_bf16_t> bb(f.size());
        ggml_fp32_to_bf16_row(f.data(), bb.data(), (int64_t) f.size());
        ggml_backend_tensor_set(a, bb.data(), 0, ggml_nbytes(a));
        std::vector<int32_t> iv(ggml_nelements(ix));
        for (auto & x : iv) x = (int32_t) (rng() % 300);
        ggml_backend_tensor_set(ix, iv.data(), 0, ggml_nbytes(ix));

        ggml_context * gctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * d = ggml_get_rows_as(gctx, a, ix, GGML_TYPE_F16);
        ggml_tensor * r = ggml_cast(gctx, ggml_get_rows(gctx, a, ix), GGML_TYPE_F16);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, d);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        bool same = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        std::vector<ggml_fp16_t> hd(ggml_nelements(d)), hr(ggml_nelements(r));
        ggml_backend_tensor_get(d, hd.data(), 0, ggml_nbytes(d));
        ggml_backend_tensor_get(r, hr.data(), 0, ggml_nbytes(r));
        same = same && d->type == GGML_TYPE_F16 && memcmp(hd.data(), hr.data(), hd.size()*sizeof(ggml_fp16_t)) == 0;

        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        const bool cpu_declines = cpu && !ggml_backend_supports_op(cpu, d) && ggml_backend_supports_op(cpu, ggml_get_rows(gctx, a, ix));
        printf("[get_rows_as] f16 direct == get_rows+cast: %d, cpu declines f16 result: %d %s\n", same, cpu_declines,
               same && cpu_declines ? "OK" : "FAIL");
        ok &= same && cpu_declines;
        if (cpu) ggml_backend_free(cpu);
        ggml_gallocr_free(ga);
        ggml_free(gctx);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // compatibility: for a CPU-held cache (or an unknown device) the builder must fall back to the
    // portable f32 gather + cast, so every GET_ROWS node it emits is one the CPU backend supports
    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        bool compat = cpu_dev != nullptr;
        for (ggml_backend_dev_t dev : { cpu_dev, (ggml_backend_dev_t) nullptr }) {
            ggml_context * ctx = ggml_init({ 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
            ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, D, N_HKV, 4096, 1);
            ggml_tensor * q1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_HEAD, 1);
            ggml_tensor * m1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4096, 1, 1, 1);
            ggml_tensor * t1 = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2051, 1, 1, 1);
            ggml_tensor * fa = nullptr;
            ggml_tensor * o  = llama_qsa_compact_attn(ctx, q1, k, k, m1, t1, 0.0625f, 0.0f, 0.0f, 256, dev, &fa);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op == GGML_OP_GET_ROWS) {
                    compat &= n->type == GGML_TYPE_F32 && (!cpu_dev || ggml_backend_dev_supports_op(cpu_dev, n));
                }
            }
            ggml_free(ctx);
        }
        printf("[cpu-compat] CPU/unknown device builds only CPU-supported f32 gathers: %s\n", compat ? "OK" : "FAIL");
        ok &= compat;
    }

    // the compact form must decline cases it cannot express, leaving the original path
    {
        ggml_context * ctx = ggml_init({ 16*ggml_tensor_overhead(), nullptr, true });
        ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, D, N_HKV, 2048, 1);
        ggml_tensor * k8 = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, D, N_HKV, 8192, 1);
        ggml_tensor * q1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_HEAD, 1);
        ggml_tensor * qp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_HEAD, 512);
        ggml_tensor * m1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2048, 1, 1, 1);
        ggml_tensor * mp = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8192, 512, 1, 1);
        ggml_tensor * t1 = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2048, 1, 1, 1);
        ggml_tensor * tp = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2048, 512, 1, 1);
        const bool full = llama_qsa_compact_applies(q1, k, k, m1, t1, 8, 256);   // selection covers the cache
        const bool pre  = llama_qsa_compact_applies(qp, k8, k8, mp, tp, 8, 256);   // prefill-sized batch
        printf("[fallback] full-selection=%d prefill=%d (both must be 0) %s\n", full, pre, (!full && !pre) ? "OK" : "FAIL");
        ok &= !full && !pre;
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    printf(ok ? "ALL OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
