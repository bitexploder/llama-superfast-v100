// Test for the Volta tensor-core path of the dense multi-token K-quant GEMVs (LLAMA_MMVQ_TC).
//
// Runs on the first CUDA device; on a device without the path every product takes dp4a and the checks still hold.
//  1. accuracy: a Q5_K product large enough for the path, 3 to 8 tokens, against an fp64 reference
//  2. range guard: activations with |x| up to 1e6 give finite outputs of the same relative accuracy
//  3. non-finite input: a token holding inf or NaN is computed as the dp4a path computes it (compared with a
//     child process run with LLAMA_MMVQ_TC=0), and the other tokens are unaffected
//  4. the q8_1 sharing hazard (LLAMA_Q8_SHARE): a tensor-core product reads the same activations as
//     two dp4a products and runs first. It quantizes nothing, so it must neither lead nor join their q8_1 group:
//     the dp4a outputs must equal their outputs run alone, bit for bit, and the plan must count 2 consumers.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t K_TC  = 512;
static const int64_t R_TC  = 65536; // at the path's minimum row count
static const int64_t R_DP  = 256;

struct weight {
    ggml_type            type;
    int64_t              K, R;
    std::vector<uint8_t> q;
    std::vector<float>   deq;
};

static weight make_weight(ggml_type type, int64_t K, int64_t R, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> f(K*R);
    for (auto & v : f) {
        v = nd(rng);
    }
    weight w { type, K, R, std::vector<uint8_t>(ggml_row_size(type, K)*R), std::vector<float>(K*R) };
    ggml_quantize_chunk(type, f.data(), w.q.data(), 0, R, K, nullptr);
    ggml_get_type_traits(type)->to_float(w.q.data(), w.deq.data(), K*R);
    return w;
}

static std::vector<float> make_x(int64_t K, int64_t C, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(K*C);
    for (auto & v : x) {
        v = nd(rng);
    }
    return x;
}

// y_i = mul_mat(W_i, x) for each weight, all in one graph and in this order; returns the outputs
static std::vector<std::vector<float>> run(ggml_backend_t be, const std::vector<const weight *> & ws, const std::vector<float> & x, int64_t C) {
    const int64_t K = ws[0]->K;
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, C);
    ggml_set_input(xt);
    std::vector<ggml_tensor *> wt, yt;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (const weight * w : ws) {
        wt.push_back(ggml_new_tensor_2d(ctx, w->type, w->K, w->R));
        yt.push_back(ggml_mul_mat(ctx, wt.back(), xt));
        ggml_set_output(yt.back());
        ggml_build_forward_expand(gf, yt.back());
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (size_t i = 0; i < ws.size(); ++i) {
        ggml_backend_tensor_set(wt[i], ws[i]->q.data(), 0, ws[i]->q.size());
    }
    ggml_backend_tensor_set(xt, x.data(), 0, x.size()*sizeof(float));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }
    std::vector<std::vector<float>> out;
    for (ggml_tensor * y : yt) {
        out.emplace_back(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.back().data(), 0, ggml_nbytes(y));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// max |y - ref| / rms(ref) over column c; nonfinite counts outputs that are inf or NaN
static double rel_err(const weight & w, const std::vector<float> & x, const std::vector<float> & y, int64_t c, int64_t * nonfinite) {
    double maxabs = 0.0, rms = 0.0;
    *nonfinite = 0;
    for (int64_t r = 0; r < w.R; ++r) {
        double ref = 0.0;
        for (int64_t k = 0; k < w.K; ++k) {
            ref += (double) w.deq[r*w.K + k] * (double) x[c*w.K + k];
        }
        rms += ref*ref;
        const float v = y[c*w.R + r];
        if (!std::isfinite(v)) {
            ++*nonfinite;
        } else {
            maxabs = std::max(maxabs, std::fabs(v - ref));
        }
    }
    return maxabs / std::sqrt(rms / w.R);
}

static std::vector<std::string> g_log;
static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    g_log.emplace_back(text);
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    g_fail += !ok;
}

static std::vector<float> nonfinite_x() {
    std::vector<float> x = make_x(K_TC, 3, 7);
    x[1*K_TC + 5] = INFINITY;
    x[2*K_TC + 7] = NAN;
    return x;
}

int main(int argc, char ** argv) {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) {
        printf("no CUDA0 device, skipping\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    const weight wtc = make_weight(GGML_TYPE_Q5_K, K_TC, R_TC, 1);
    const weight wd1 = make_weight(GGML_TYPE_Q4_0, K_TC, R_DP, 2);
    const weight wd2 = make_weight(GGML_TYPE_Q4_0, K_TC, R_DP, 3);
    const std::vector<float> xh1 = make_x(K_TC, 3, 11), xh2 = make_x(K_TC, 3, 12);

    // child: the dp4a results the parent compares with (run with LLAMA_MMVQ_TC=0)
    if (argc == 3 && strcmp(argv[1], "--child") == 0) {
        FILE * f = fopen(argv[2], "wb");
        const auto nf = run(be, { &wtc }, nonfinite_x(), 3);
        const auto hz = run(be, { &wtc, &wd1, &wd2 }, xh1, 3);
        fwrite(nf[0].data(), sizeof(float), nf[0].size(), f);
        fwrite(hz[0].data(), sizeof(float), hz[0].size(), f);
        fclose(f);
        ggml_backend_free(be);
        return 0;
    }
    const std::string child_out = std::string(argv[0]) + ".child.bin";
    const std::string cmd = "LLAMA_MMVQ_TC=0 '" + std::string(argv[0]) + "' --child '" + child_out + "' > /dev/null";
    if (system(cmd.c_str()) != 0) {
        fprintf(stderr, "child failed: %s\n", cmd.c_str());
        return 1;
    }
    std::vector<float> dp_nf(3*R_TC), dp_hz(3*R_TC);
    {
        FILE * f = fopen(child_out.c_str(), "rb");
        const bool ok = f && fread(dp_nf.data(), sizeof(float), dp_nf.size(), f) == dp_nf.size() &&
                             fread(dp_hz.data(), sizeof(float), dp_hz.size(), f) == dp_hz.size();
        if (f) {
            fclose(f);
        }
        remove(child_out.c_str());
        if (!ok) {
            fprintf(stderr, "cannot read %s\n", child_out.c_str());
            return 1;
        }
    }

    // 1. accuracy at 2 to 8 tokens
    for (int64_t C : { 3, 4, 5, 8 }) {
        const std::vector<float> x = make_x(K_TC, C, 100 + C);
        const auto y = run(be, { &wtc }, x, C);
        double worst = 0.0;
        int64_t bad = 0;
        for (int64_t c = 0; c < C; ++c) {
            int64_t nf;
            worst = std::max(worst, rel_err(wtc, x, y[0], c, &nf));
            bad += nf;
        }
        char what[128];
        snprintf(what, sizeof what, "Q5_K %lld tokens: max |err|/rms %.3e, %lld non-finite", (long long) C, worst, (long long) bad);
        check(bad == 0 && worst < 5e-2, what);
    }

    // 2. range guard: each token scaled so its largest |x| is 1e6
    {
        std::vector<float> x = make_x(K_TC, 3, 21);
        for (int64_t c = 0; c < 3; ++c) {
            float m = 0.0f;
            for (int64_t k = 0; k < K_TC; ++k) {
                m = std::max(m, std::fabs(x[c*K_TC + k]));
            }
            for (int64_t k = 0; k < K_TC; ++k) {
                x[c*K_TC + k] *= 1e6f / m;
            }
        }
        const auto y = run(be, { &wtc }, x, 3);
        double worst = 0.0;
        int64_t bad = 0;
        for (int64_t c = 0; c < 3; ++c) {
            int64_t nf;
            worst = std::max(worst, rel_err(wtc, x, y[0], c, &nf));
            bad += nf;
        }
        char what[128];
        snprintf(what, sizeof what, "|x| up to 1e6: max |err|/rms %.3e, %lld non-finite", worst, (long long) bad);
        check(bad == 0 && worst < 5e-2, what);
    }

    // 3. non-finite input: token 1 holds +inf, token 2 a NaN
    {
        const std::vector<float> x = nonfinite_x();
        const auto y = run(be, { &wtc }, x, 3);
        int64_t nf0;
        const double e0 = rel_err(wtc, x, y[0], 0, &nf0);
        int64_t mismatch = 0;
        for (int64_t i = R_TC; i < 3*R_TC; ++i) {
            const float a = y[0][i], b = dp_nf[i];
            const bool same = std::isnan(a) ? std::isnan(b) : std::isinf(a) ? a == b :
                              std::isfinite(b) && std::fabs(a - b) <= 1e-4f*std::max(1.0f, std::fabs(b));
            mismatch += !same;
        }
        char what[160];
        snprintf(what, sizeof what, "non-finite tokens as dp4a computes them: %lld mismatches; finite token max |err|/rms %.3e", (long long) mismatch, e0);
        check(mismatch == 0 && nf0 == 0 && e0 < 5e-2, what);
    }

    // 4. the q8_1 sharing hazard, twice with different activations so that stale slot contents cannot pass
    {
        bool tc_ran = false, same = true;
        g_log.clear();
        for (const auto * x : { &xh1, &xh2 }) {
            const auto yg = run(be, { &wtc, &wd1, &wd2 }, *x, 3);
            const auto y1 = run(be, { &wd1 }, *x, 3);
            const auto y2 = run(be, { &wd2 }, *x, 3);
            same = same && memcmp(yg[1].data(), y1[0].data(), y1[0].size()*sizeof(float)) == 0
                        && memcmp(yg[2].data(), y2[0].data(), y2[0].size()*sizeof(float)) == 0;
            if (x == &xh1) {
                tc_ran = memcmp(yg[0].data(), dp_hz.data(), dp_hz.size()*sizeof(float)) != 0;
            }
        }
        check(same, "dp4a products next to a tensor-core product equal their outputs run alone, bit for bit");
        // the plan is logged once per distinct outcome, so look for it over the whole run
        int consumers = -1;
        for (const std::string & l : g_log) {
            const size_t p = l.find(" shared q8_1 inputs, ");
            if (l.find("ggml_cuda_plan_q8_share") != std::string::npos && p != std::string::npos && l.find(": 1 shared") != std::string::npos) {
                consumers = atoi(l.c_str() + p + strlen(" shared q8_1 inputs, "));
            }
        }
        if (consumers < 0) {
            printf("skip: no q8_1 sharing plan logged (LLAMA_Q8_SHARE=0?)\n");
        } else {
            char what[160];
            snprintf(what, sizeof what, "q8_1 plan: %d consumers with the tensor-core path %s (expect %d)",
                consumers, tc_ran ? "taken" : "not taken", tc_ran ? 2 : 3);
            check(consumers == (tc_ran ? 2 : 3), what);
        }
    }

    ggml_backend_free(be);
    printf("%s\n", g_fail ? "FAILED" : "ALL OK");
    return g_fail ? 1 : 0;
}
