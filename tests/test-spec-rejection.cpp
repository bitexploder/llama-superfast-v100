// (also the adaptive draft length, run_adapt)
// [TAG_SPEC_REJECTION] the rejection step of speculative sampling keeps the target's distribution.
//
// For each synthetic pair (p, q): draw x from q, run common_sampler_rejection_step (accept x with probability
// min(1, p(x)/q(x)), else sample the normalized residual max(0, p - q)), 10^6 times. The emitted tokens must follow p:
// a chi-square test over p's support must give a p-value above 0.01, and no token outside p's support may appear.
// The acceptance rate is printed beside its expected value, 1 - TV(p, q) = sum min(p, q). CPU only, no model.

#include "sampling.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

// the regularized upper incomplete gamma function Q(a, x) (Numerical Recipes gammq): the chi-square p-value is
// Q(df / 2, chi2 / 2)
static double gamma_q(double a, double x) {
    if (x <= 0.0) {
        return 1.0;
    }
    const double gln = std::lgamma(a);
    if (x < a + 1.0) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 10000; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::fabs(del) < std::fabs(sum) * 1e-15) {
                break;
            }
        }
        return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
    }
    const double fpmin = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / fpmin, d = 1.0 / b, h = d;
    for (int i = 1; i < 10000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < fpmin) d = fpmin;
        c = b + an / c;
        if (std::fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < 1e-15) {
            break;
        }
    }
    return std::exp(-x + a * std::log(x) - gln) * h;
}

struct test_case {
    std::string name;
    std::vector<llama_token_data> p; // the target's distribution
    std::vector<llama_token_data> q; // the draft's
};

static std::vector<llama_token_data> dist(const std::vector<std::pair<llama_token, double>> & w) {
    double sum = 0.0;
    for (const auto & e : w) {
        sum += e.second;
    }
    std::vector<llama_token_data> res;
    for (const auto & e : w) {
        res.push_back({ e.first, 0.0f, (float) (e.second / sum) });
    }
    return res;
}

// a softmax over logits, for rows shaped like the sampler's
static std::vector<llama_token_data> softmax(const std::vector<std::pair<llama_token, double>> & l, double temp) {
    std::vector<std::pair<llama_token, double>> w;
    for (const auto & e : l) {
        w.push_back({ e.first, std::exp(e.second / temp) });
    }
    return dist(w);
}

static bool run(const test_case & tc, uint64_t seed, int n) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::function<double()> draw = [&]() { return uni(rng); };

    std::map<llama_token, double> p_of;
    for (const auto & e : tc.p) {
        p_of[e.id] += e.p;
    }
    double acc_expected = 0.0;
    for (const auto & e : tc.q) {
        acc_expected += std::min((double) e.p, p_of.count(e.id) ? p_of[e.id] : 0.0);
    }

    std::map<llama_token, long> hist;
    long n_acc = 0;
    for (int i = 0; i < n; ++i) {
        // x ~ q, by the inverse CDF
        const double u = draw();
        double run = 0.0;
        llama_token x = tc.q.back().id;
        for (const auto & e : tc.q) {
            run += e.p;
            if (run >= u && e.p > 0.0f) {
                x = e.id;
                break;
            }
        }
        bool accepted = false;
        const llama_token t = common_sampler_rejection_step(tc.p.data(), tc.p.size(), tc.q, x, draw, &accepted);
        hist[t]++;
        n_acc += accepted;
    }

    double chi2 = 0.0;
    int n_bins = 0;
    long n_outside = 0;
    for (const auto & h : hist) {
        if (!p_of.count(h.first) || p_of[h.first] <= 0.0) {
            n_outside += h.second;
        }
    }
    for (const auto & e : p_of) {
        if (e.second <= 0.0) {
            continue;
        }
        const double expct = e.second * n;
        const double obs    = hist.count(e.first) ? (double) hist[e.first] : 0.0;
        chi2 += (obs - expct) * (obs - expct) / expct;
        n_bins++;
    }
    const int df = n_bins - 1;
    const double pval = df > 0 ? gamma_q(df / 2.0, chi2 / 2.0) : (n_outside == 0 ? 1.0 : 0.0);
    const bool ok = n_outside == 0 && pval > 0.01;

    printf("%-34s bins %3d  chi2 %9.3f  df %3d  p-value %.4f  outside %ld  acceptance %.5f (1 - TV %.5f)  %s\n",
            tc.name.c_str(), n_bins, chi2, df, pval, n_outside, (double) n_acc / n, acc_expected, ok ? "ok" : "FAIL");
    return ok;
}

// [TAG_SPEC_REJECTION_PIPE] the pipelined protocol with rejection sampling, on a Markov chain over V
// tokens: p(y | prev) the target, q(y | prev) the draft. A round verifies a chunk C = [t, x1, x2]; the guess g of its
// bonus token is drawn from q(. | x2) and, when the gate q(x1) q(x2) q(g) >= thr passes, a chunk N = [g, n1, n2] is
// drafted behind C. C's verify is the rejection step on x1, x2 and g against p (no bonus row); N is kept exactly when
// all three are accepted, and is then the next round's chunk. Otherwise the next chunk is drafted from the last
// emitted token. The draft draws from its own generator, the verify from another. The first K emitted tokens must
// follow the chain p exactly: a chi-square over the V^K sequences (cells expected below 5 pooled). With g_always
// false (the control), g is tested only when N was submitted, and the bonus is sampled from p otherwise: the gate
// then reads g, and the output is expected to be biased
struct pipe_case {
    std::string name;
    std::vector<std::vector<llama_token_data>> p; // [prev] the target's row
    std::vector<std::vector<llama_token_data>> q; // [prev] the draft's row
    double thr;
    bool   g_always;
};

static double row_p(const std::vector<llama_token_data> & r, llama_token y) {
    for (const auto & e : r) {
        if (e.id == y) {
            return e.p;
        }
    }
    return 0.0;
}

static llama_token row_sample(const std::vector<llama_token_data> & r, double u) {
    double run = 0.0;
    llama_token x = LLAMA_TOKEN_NULL;
    for (const auto & e : r) {
        if (e.p <= 0.0f) {
            continue;
        }
        x = e.id;
        run += e.p;
        if (run >= u) {
            break;
        }
    }
    return x;
}

static bool run_pipe(const pipe_case & pc, uint64_t seed, int n, int K) {
    const int V = (int) pc.p.size();
    std::mt19937 rng_v(seed), rng_d(seed ^ 0x70u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::function<double()> draw_v = [&]() { return uni(rng_v); };
    const auto draw_d = [&]() { return uni(rng_d); };

    long n_cells = 1;
    for (int k = 0; k < K; ++k) {
        n_cells *= V;
    }
    std::vector<long> hist(n_cells, 0);
    long n_rounds = 0, n_behind = 0, n_keep = 0, n_g = 0, n_g_acc = 0, n_emit = 0;

    for (int it = 0; it < n; ++it) {
        std::vector<llama_token> out;
        llama_token t = 0;                // the round's first token: sampled, or a kept guess
        llama_token x1 = LLAMA_TOKEN_NULL, x2 = LLAMA_TOKEN_NULL;
        bool have_drafts = false;         // x1, x2 carried over from a kept N
        while ((int) out.size() < K) {
            if (!have_drafts) {
                x1 = row_sample(pc.q[t],  draw_d());
                x2 = row_sample(pc.q[x1], draw_d());
            }
            have_drafts = false;
            // the guess, and N behind C when the gate passes
            const llama_token g = row_sample(pc.q[x2], draw_d());
            const bool behind = row_p(pc.q[t], x1) * row_p(pc.q[x1], x2) * row_p(pc.q[x2], g) >= pc.thr;
            llama_token n1 = LLAMA_TOKEN_NULL, n2 = LLAMA_TOKEN_NULL;
            if (behind) {
                n1 = row_sample(pc.q[g],  draw_d());
                n2 = row_sample(pc.q[n1], draw_d());
                n_behind++;
            }
            n_rounds++;
            // C's verify
            bool acc = false;
            llama_token y = common_sampler_rejection_step(pc.p[t].data(), pc.p[t].size(), pc.q[t], x1, draw_v, &acc);
            out.push_back(y);
            if (acc) {
                y = common_sampler_rejection_step(pc.p[x1].data(), pc.p[x1].size(), pc.q[x1], x2, draw_v, &acc);
                out.push_back(y);
            }
            bool keep = false;
            if (acc) {
                if (pc.g_always || behind) {
                    y = common_sampler_rejection_step(pc.p[x2].data(), pc.p[x2].size(), pc.q[x2], g, draw_v, &acc);
                    n_g++;
                    n_g_acc += acc;
                    keep = acc && behind;
                } else {
                    y = row_sample(pc.p[x2], draw_v());
                }
                out.push_back(y);
            }
            t = out.back();
            if (keep) {
                n_keep++;
                x1 = n1;
                x2 = n2;
                have_drafts = true;
            }
        }
        n_emit += (long) out.size();
        long cell = 0;
        for (int k = 0; k < K; ++k) {
            cell = cell * V + out[k];
        }
        hist[cell]++;
    }

    // the exact distribution of the first K tokens, from token 0
    double chi2 = 0.0, pool_e = 0.0, pool_o = 0.0;
    int n_bins = 0;
    for (long cell = 0; cell < n_cells; ++cell) {
        std::vector<llama_token> seq(K);
        long c = cell;
        for (int k = K - 1; k >= 0; --k) {
            seq[k] = (llama_token) (c % V);
            c /= V;
        }
        double pr = 1.0;
        llama_token prev = 0;
        for (int k = 0; k < K; ++k) {
            pr *= row_p(pc.p[prev], seq[k]);
            prev = seq[k];
        }
        const double expct = pr * n;
        if (expct < 5.0) {
            pool_e += expct;
            pool_o += (double) hist[cell];
            continue;
        }
        chi2 += (hist[cell] - expct) * (hist[cell] - expct) / expct;
        n_bins++;
    }
    if (pool_e > 0.0) {
        chi2 += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        n_bins++;
    }
    const int df = n_bins - 1;
    const double pval = gamma_q(df / 2.0, chi2 / 2.0);

    printf("%-44s cells %4d  chi2 %9.2f  df %4d  p-value %.4f  rounds %ld  N behind %ld  kept %ld  g accepted %.4f  tok/round %.3f  %s\n",
            pc.name.c_str(), n_bins, chi2, df, pval, n_rounds, n_behind, n_keep, n_g ? (double) n_g_acc / n_g : 0.0,
            (double) n_emit / n_rounds, pc.g_always ? (pval > 0.01 ? "ok" : "FAIL") : (pval < 0.01 ? "biased, as expected" : "not detected"));
    return pc.g_always ? pval > 0.01 : true;
}

// [TAG_SPEC_REJECTION_ADAPT] the adaptive draft length, on the same kind of Markov chain: a round from
// the last emitted token t drafts x1 ~ q(. | t), and drafts x_{k+1} ~ q(. | x_k) only while k < nmax and the product
// q(x1) ... q(x_k) is at least the cutoff, so the length depends on q. The verify is the rejection step on each draft
// in turn, then a bonus token from p when every draft is accepted. The draws come from one generator, the drafts'
// before the verify's, as in the server. The first K emitted tokens must follow the chain p exactly. With
// peek (the control), x_{k+1} is drawn first and kept only when the product including its own q stays at least the
// cutoff: the decision then reads the token it keeps or drops, and the output is expected to be biased
struct adapt_case {
    std::string name;
    std::vector<std::vector<llama_token_data>> p;
    std::vector<std::vector<llama_token_data>> q;
    double cutoff;
    int    nmax;
    bool   peek;
};

static bool run_adapt(const adapt_case & ac, uint64_t seed, int n, int K) {
    const int V = (int) ac.p.size();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::function<double()> draw = [&]() { return uni(rng); };

    long n_cells = 1;
    for (int k = 0; k < K; ++k) {
        n_cells *= V;
    }
    std::vector<long> hist(n_cells, 0);
    std::vector<long> n_len(ac.nmax + 1, 0);
    long n_rounds = 0, n_emit = 0;

    for (int it = 0; it < n; ++it) {
        std::vector<llama_token> out;
        llama_token t = 0;
        while ((int) out.size() < K) {
            std::vector<llama_token> d;
            double prod = 1.0;
            llama_token prev = t;
            while ((int) d.size() < ac.nmax) {
                const llama_token x = row_sample(ac.q[prev], draw());
                const double qx = row_p(ac.q[prev], x);
                if (ac.peek && !d.empty() && prod * qx < ac.cutoff) {
                    break;
                }
                d.push_back(x);
                prod *= qx;
                prev = x;
                if (!ac.peek && prod < ac.cutoff) {
                    break;
                }
            }
            n_len[d.size()]++;
            n_rounds++;
            prev = t;
            bool acc = true;
            for (const llama_token x : d) {
                const llama_token y = common_sampler_rejection_step(ac.p[prev].data(), ac.p[prev].size(), ac.q[prev], x, draw, &acc);
                out.push_back(y);
                if (!acc) {
                    break;
                }
                prev = x;
            }
            if (acc) {
                out.push_back(row_sample(ac.p[prev], draw()));
            }
            t = out.back();
        }
        n_emit += (long) out.size();
        long cell = 0;
        for (int k = 0; k < K; ++k) {
            cell = cell * V + out[k];
        }
        hist[cell]++;
    }

    double chi2 = 0.0, pool_e = 0.0, pool_o = 0.0;
    int n_bins = 0;
    for (long cell = 0; cell < n_cells; ++cell) {
        std::vector<llama_token> sq(K);
        long c = cell;
        for (int k = K - 1; k >= 0; --k) {
            sq[k] = (llama_token) (c % V);
            c /= V;
        }
        double pr = 1.0;
        llama_token prev = 0;
        for (int k = 0; k < K; ++k) {
            pr *= row_p(ac.p[prev], sq[k]);
            prev = sq[k];
        }
        const double expct = pr * n;
        if (expct < 5.0) {
            pool_e += expct;
            pool_o += (double) hist[cell];
            continue;
        }
        chi2 += (hist[cell] - expct) * (hist[cell] - expct) / expct;
        n_bins++;
    }
    if (pool_e > 0.0) {
        chi2 += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        n_bins++;
    }
    const int df = n_bins - 1;
    const double pval = gamma_q(df / 2.0, chi2 / 2.0);

    std::string lens;
    for (int k = 1; k <= ac.nmax; ++k) {
        lens += (k > 1 ? " " : "") + std::to_string((double) n_len[k] / n_rounds).substr(0, 5);
    }
    printf("%-44s cells %4d  chi2 %9.2f  df %4d  p-value %.4f  rounds %ld  draft length 1..%d: %s  tok/round %.3f  %s\n",
            ac.name.c_str(), n_bins, chi2, df, pval, n_rounds, ac.nmax, lens.c_str(), (double) n_emit / n_rounds,
            !ac.peek ? (pval > 0.01 ? "ok" : "FAIL") : (pval < 0.01 ? "biased, as expected" : "not detected"));
    return !ac.peek ? pval > 0.01 : true;
}

// [TAG_SPEC_BLOCK] block verification, on the same kind of Markov chain: a round from the last emitted
// token t drafts up to gamma tokens x_1 ~ q(. | t), x_{k+1} ~ q(. | x_k) from the draft's own generator (with cutoff >= 0
// it stops once the product q(x_1) ... q(x_k) falls below the cutoff, as the adaptive length does), then verifies them
// from another: mode "block" calls common_spec_block_verify (tau kept, then the residual's token, or the bonus from p
// when tau = gamma); mode "token" is today's verify, common_sampler_rejection_step on each draft in turn, then the bonus.
// The first K emitted tokens must follow the chain p exactly. Two controls must come out biased, or the test cannot see
// the two likeliest porting errors: "first fail" stops at the first eta_i > h_i as the token loop does, and "token
// residual" draws the residual at tau without the factor p_tau. Tokens a round are printed for both modes: block
// verification is never below token verification in expectation (the paper's Theorem 2)
enum block_mode { BLOCK_TOKEN, BLOCK_BLOCK, BLOCK_CTRL_FIRST_FAIL, BLOCK_CTRL_TOKEN_RESIDUAL };

struct block_case {
    std::string name;
    std::vector<std::vector<llama_token_data>> p;
    std::vector<std::vector<llama_token_data>> q;
    int        gamma;
    double     cutoff; // < 0: always gamma drafts
    block_mode mode;
};

// the controls: common_spec_block_verify with one error each
static int block_control(const std::vector<std::vector<llama_token_data>> & P, const std::vector<std::vector<llama_token_data>> & Q,
        const std::vector<llama_token> & d, const std::function<double()> & draw, block_mode mode, llama_token * y) {
    const int gamma = (int) d.size();
    std::vector<double> eta(gamma);
    for (int i = 0; i < gamma; ++i) {
        eta[i] = draw();
    }
    int tau = 0;
    double w = 1.0, w_tau = 1.0;
    for (int i = 1; i <= gamma; ++i) {
        const double qx = row_p(Q[i - 1], d[i - 1]);
        w = qx > 0.0 ? std::min(w * row_p(P[i - 1], d[i - 1]) / qx, 1.0) : 0.0;
        double h = w;
        if (i < gamma) {
            double s = 0.0;
            for (const auto & e : P[i]) {
                s += std::max(0.0, w * e.p - row_p(Q[i], e.id));
            }
            h = s + 1.0 - w > 0.0 ? s / (s + 1.0 - w) : 1.0;
        }
        if (eta[i - 1] < h) {
            tau = i;
            w_tau = w;
        } else if (mode == BLOCK_CTRL_FIRST_FAIL) {
            break;
        }
    }
    *y = LLAMA_TOKEN_NULL;
    if (tau < gamma) {
        const double wr = mode == BLOCK_CTRL_TOKEN_RESIDUAL ? 1.0 : w_tau;
        std::vector<llama_token_data> r;
        double sum = 0.0;
        for (const auto & e : P[tau]) {
            const double v = std::max(0.0, wr * e.p - row_p(Q[tau], e.id));
            r.push_back({ e.id, 0.0f, (float) v });
            sum += v;
        }
        if (sum <= 0.0) {
            *y = row_sample(P[tau], draw());
        } else {
            for (auto & e : r) {
                e.p = (float) (e.p / sum);
            }
            *y = row_sample(r, draw());
        }
    }
    return tau;
}

static bool run_block(const block_case & bc, uint64_t seed, int n, int K) {
    const int V = (int) bc.p.size();
    std::mt19937 rng_v(seed), rng_d(seed ^ 0xb1u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::function<double()> draw = [&]() { return uni(rng_v); };

    long n_cells = 1;
    for (int k = 0; k < K; ++k) {
        n_cells *= V;
    }
    std::vector<long> hist(n_cells, 0);
    long n_rounds = 0, n_emit = 0;

    for (int it = 0; it < n; ++it) {
        std::vector<llama_token> out;
        llama_token t = 0;
        while ((int) out.size() < K) {
            std::vector<llama_token> d;
            std::vector<std::vector<llama_token_data>> P, Q; // rows 0 .. gamma - 1, then P's bonus row gamma
            llama_token prev = t;
            double prod = 1.0;
            for (int k = 0; k < bc.gamma; ++k) {
                P.push_back(bc.p[prev]);
                Q.push_back(bc.q[prev]);
                d.push_back(row_sample(bc.q[prev], uni(rng_d)));
                prod *= row_p(bc.q[prev], d.back());
                prev = d.back();
                if (bc.cutoff >= 0.0 && prod < bc.cutoff) {
                    break;
                }
            }
            P.push_back(bc.p[prev]);
            const int G = (int) d.size();
            n_rounds++;
            if (bc.mode == BLOCK_TOKEN) {
                bool acc = true;
                for (int k = 0; k < G && acc; ++k) {
                    out.push_back(common_sampler_rejection_step(P[k].data(), P[k].size(), Q[k], d[k], draw, &acc));
                }
                if (acc) {
                    out.push_back(row_sample(P[G], draw()));
                }
            } else {
                llama_token y = LLAMA_TOKEN_NULL;
                int tau = 0;
                if (bc.mode == BLOCK_BLOCK) {
                    std::vector<const llama_token_data *> pp;
                    std::vector<size_t> np;
                    for (int k = 0; k < G; ++k) {
                        pp.push_back(P[k].data());
                        np.push_back(P[k].size());
                    }
                    tau = common_spec_block_verify(pp, np, Q, d, draw, &y);
                } else {
                    tau = block_control(P, Q, d, draw, bc.mode, &y);
                }
                for (int k = 0; k < tau; ++k) {
                    out.push_back(d[k]);
                }
                out.push_back(tau == G ? row_sample(P[G], draw()) : y);
            }
            t = out.back();
        }
        n_emit += (long) out.size();
        long cell = 0;
        for (int k = 0; k < K; ++k) {
            cell = cell * V + out[k];
        }
        hist[cell]++;
    }

    double chi2 = 0.0, pool_e = 0.0, pool_o = 0.0;
    int n_bins = 0;
    for (long cell = 0; cell < n_cells; ++cell) {
        std::vector<llama_token> sq(K);
        long c = cell;
        for (int k = K - 1; k >= 0; --k) {
            sq[k] = (llama_token) (c % V);
            c /= V;
        }
        double pr = 1.0;
        llama_token prev = 0;
        for (int k = 0; k < K; ++k) {
            pr *= row_p(bc.p[prev], sq[k]);
            prev = sq[k];
        }
        const double expct = pr * n;
        if (expct < 5.0) {
            pool_e += expct;
            pool_o += (double) hist[cell];
            continue;
        }
        chi2 += (hist[cell] - expct) * (hist[cell] - expct) / expct;
        n_bins++;
    }
    if (pool_e > 0.0) {
        chi2 += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        n_bins++;
    }
    const int df = n_bins - 1;
    const double pval = gamma_q(df / 2.0, chi2 / 2.0);

    const bool control = bc.mode == BLOCK_CTRL_FIRST_FAIL || bc.mode == BLOCK_CTRL_TOKEN_RESIDUAL;
    printf("%-44s cells %4d  chi2 %9.2f  df %4d  p-value %.4f  rounds %ld  tok/round %.3f  %s\n",
            bc.name.c_str(), n_bins, chi2, df, pval, n_rounds, (double) n_emit / n_rounds,
            !control ? (pval > 0.01 ? "ok" : "FAIL") : (pval < 0.01 ? "biased, as expected" : "not detected"));
    return !control ? pval > 0.01 : pval < 0.01;
}

int main(void) {
    const int n = 1000000;

    std::vector<test_case> cases;

    cases.push_back({ "disjoint supports",
            dist({ {0, 0.5}, {1, 0.3}, {2, 0.2} }),
            dist({ {3, 0.6}, {4, 0.4} }) });

    cases.push_back({ "partly disjoint",
            dist({ {0, 0.4}, {1, 0.3}, {2, 0.2}, {3, 0.1} }),
            dist({ {2, 0.5}, {3, 0.3}, {7, 0.2} }) });

    cases.push_back({ "near-ties",
            dist({ {0, 0.33340}, {1, 0.33330}, {2, 0.33330} }),
            dist({ {0, 0.33330}, {1, 0.33340}, {2, 0.33330} }) });

    cases.push_back({ "near-tie top two, swapped",
            dist({ {0, 0.4001}, {1, 0.3999}, {2, 0.1}, {3, 0.1} }),
            dist({ {1, 0.4001}, {0, 0.3999}, {2, 0.15}, {3, 0.05} }) });

    cases.push_back({ "overconfident draft",
            dist({ {0, 0.30}, {1, 0.20}, {2, 0.15}, {3, 0.10}, {4, 0.08}, {5, 0.06}, {6, 0.05}, {7, 0.03}, {8, 0.02}, {9, 0.01} }),
            dist({ {0, 0.95}, {1, 0.05} }) });

    cases.push_back({ "overconfident draft, wrong token",
            dist({ {0, 0.6}, {1, 0.3}, {2, 0.1} }),
            dist({ {1, 0.97}, {0, 0.02}, {2, 0.01} }) });

    cases.push_back({ "underconfident draft",
            dist({ {0, 0.90}, {1, 0.05}, {2, 0.05} }),
            dist({ {0, 0.1}, {1, 0.1}, {2, 0.1}, {3, 0.1}, {4, 0.1}, {5, 0.1}, {6, 0.1}, {7, 0.1}, {8, 0.1}, {9, 0.1} }) });

    cases.push_back({ "identical",
            dist({ {5, 0.5}, {6, 0.25}, {7, 0.25} }),
            dist({ {5, 0.5}, {6, 0.25}, {7, 0.25} }) });

    cases.push_back({ "one-hot, same token (T=0)",
            dist({ {3, 1.0} }),
            dist({ {3, 1.0} }) });

    cases.push_back({ "one-hot, different token (T=0)",
            dist({ {3, 1.0} }),
            dist({ {4, 1.0} }) });

    // rows shaped like the sampler's: 20 target candidates (top-k 20) at T 1.0, and 26 draft candidates whose logits
    // are the target's with noise and a shift, at the draft temperatures 1.0 and 0.7
    {
        std::mt19937 g(1234);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<std::pair<llama_token, double>> lp, lq;
        double l = 0.0;
        for (int i = 0; i < 26; ++i) {
            l -= std::fabs(nd(g)) * 0.6;
            if (i < 20) {
                lp.push_back({ 100 + i, l });
            }
            lq.push_back({ 100 + i, l + 0.8 * nd(g) });
        }
        cases.push_back({ "sampler-shaped rows, draft T 1.0", softmax(lp, 1.0), softmax(lq, 1.0) });
        cases.push_back({ "sampler-shaped rows, draft T 0.7", softmax(lp, 1.0), softmax(lq, 0.7) });
    }

    bool ok = true;
    uint64_t seed = 69;
    for (const auto & tc : cases) {
        ok = run(tc, seed++, n) && ok;
    }

    // [TAG_SPEC_REJECTION_PIPE] the pipelined protocol
    {
        const std::vector<std::vector<llama_token_data>> p = {
            dist({ {0, 0.50}, {1, 0.30}, {2, 0.20} }),
            dist({ {0, 0.20}, {1, 0.10}, {2, 0.70} }),
            dist({ {0, 0.35}, {1, 0.60}, {2, 0.05} }),
        };
        const std::vector<std::vector<llama_token_data>> q = {
            dist({ {0, 0.80}, {1, 0.15}, {2, 0.05} }),
            dist({ {1, 0.50}, {2, 0.50} }),           // no mass on 0: its residual must supply it
            dist({ {0, 0.10}, {1, 0.85}, {2, 0.05} }),
        };
        const std::vector<std::vector<llama_token_data>> q_same = p;
        std::vector<pipe_case> pcs = {
            { "pipeline: gate 0.1, g always tested",        p, q,      0.10, true  },
            { "pipeline: gate 0.3, g always tested",        p, q,      0.30, true  },
            { "pipeline: no gate (every N submitted)",      p, q,      0.00, true  },
            { "pipeline: gate 1.1 (no N submitted)",        p, q,      1.10, true  },
            { "pipeline: q = p, gate 0.1",                  p, q_same, 0.10, true  },
            { "control: g tested only when N submitted",    p, q,      0.30, false },
        };
        for (const auto & pc : pcs) {
            ok = run_pipe(pc, seed++, n, 6) && ok;
        }
    }

    // [TAG_SPEC_REJECTION_ADAPT] the adaptive draft length
    {
        const std::vector<std::vector<llama_token_data>> p = {
            dist({ {0, 0.50}, {1, 0.30}, {2, 0.20} }),
            dist({ {0, 0.20}, {1, 0.10}, {2, 0.70} }),
            dist({ {0, 0.35}, {1, 0.60}, {2, 0.05} }),
        };
        const std::vector<std::vector<llama_token_data>> q = {
            dist({ {0, 0.80}, {1, 0.15}, {2, 0.05} }),
            dist({ {1, 0.50}, {2, 0.50} }),           // no mass on 0: its residual must supply it
            dist({ {0, 0.10}, {1, 0.85}, {2, 0.05} }),
        };
        const std::vector<std::vector<llama_token_data>> q_same = p;
        std::vector<adapt_case> acs = {
            { "adaptive: cutoff 0.5, up to 3",              p, q,      0.50, 3, false },
            { "adaptive: cutoff 0.3, up to 3",              p, q,      0.30, 3, false },
            { "adaptive: cutoff 0.7, up to 4",              p, q,      0.70, 4, false },
            { "adaptive: cutoff 0 (always 3)",              p, q,      0.00, 3, false },
            { "adaptive: cutoff 1.1 (always 1)",            p, q,      1.10, 3, false },
            { "adaptive: q = p, cutoff 0.5, up to 3",       p, q_same, 0.50, 3, false },
            { "control: keep x_k+1 by its own q (peek)",    p, q,      0.50, 3, true  },
        };
        for (const auto & ac : acs) {
            ok = run_adapt(ac, seed++, n, 6) && ok;
        }
    }

    // [TAG_SPEC_BLOCK] block verification
    {
        const std::vector<std::vector<llama_token_data>> p = {
            dist({ {0, 0.50}, {1, 0.30}, {2, 0.20} }),
            dist({ {0, 0.20}, {1, 0.10}, {2, 0.70} }),
            dist({ {0, 0.35}, {1, 0.60}, {2, 0.05} }),
        };
        const std::vector<std::vector<llama_token_data>> q = {
            dist({ {0, 0.80}, {1, 0.15}, {2, 0.05} }),
            dist({ {1, 0.50}, {2, 0.50} }),           // no mass on 0: its residual must supply it
            dist({ {0, 0.10}, {1, 0.85}, {2, 0.05} }),
        };
        const std::vector<std::vector<llama_token_data>> q_same = p;
        // a peaked target and a close draft, as an MTP head's rows: four tokens, the target's top token at 0.75-0.9
        const std::vector<std::vector<llama_token_data>> pk = {
            dist({ {0, 0.85}, {1, 0.08}, {2, 0.05}, {3, 0.02} }),
            dist({ {0, 0.05}, {1, 0.75}, {2, 0.15}, {3, 0.05} }),
            dist({ {0, 0.03}, {1, 0.04}, {2, 0.90}, {3, 0.03} }),
            dist({ {0, 0.10}, {1, 0.05}, {2, 0.05}, {3, 0.80} }),
        };
        const std::vector<std::vector<llama_token_data>> qk = {
            dist({ {0, 0.70}, {1, 0.20}, {2, 0.10} }),
            dist({ {0, 0.10}, {1, 0.85}, {2, 0.05} }),
            dist({ {1, 0.15}, {2, 0.80}, {3, 0.05} }),
            dist({ {0, 0.25}, {3, 0.75} }),
        };
        std::vector<block_case> bcs = {
            { "block: gamma 1",                             p,  q,      1, -1.0, BLOCK_BLOCK },
            { "block: gamma 2",                             p,  q,      2, -1.0, BLOCK_BLOCK },
            { "block: gamma 3",                             p,  q,      3, -1.0, BLOCK_BLOCK },
            { "token: gamma 3 (today's verify)",            p,  q,      3, -1.0, BLOCK_TOKEN },
            { "block: gamma 7",                             p,  q,      7, -1.0, BLOCK_BLOCK },
            { "token: gamma 7 (today's verify)",            p,  q,      7, -1.0, BLOCK_TOKEN },
            { "block: q = p, gamma 3",                      p,  q_same, 3, -1.0, BLOCK_BLOCK },
            { "block: peaked, gamma 7",                     pk, qk,     7, -1.0, BLOCK_BLOCK },
            { "token: peaked, gamma 7 (today's verify)",    pk, qk,     7, -1.0, BLOCK_TOKEN },
            { "block: adaptive, cutoff 0.3, up to 4",       p,  q,      4,  0.3, BLOCK_BLOCK },
            { "token: adaptive, cutoff 0.3, up to 4",       p,  q,      4,  0.3, BLOCK_TOKEN },
            { "control: stop at the first fail, gamma 3",   p,  q,      3, -1.0, BLOCK_CTRL_FIRST_FAIL },
            { "control: residual without p_tau, gamma 3",   p,  q,      3, -1.0, BLOCK_CTRL_TOKEN_RESIDUAL },
        };
        for (const auto & bc : bcs) {
            ok = run_block(bc, seed++, n, bc.p.size() == 4 ? 5 : 6) && ok;
        }
    }

    printf("%s\n", ok ? "all cases match p" : "FAILED");
    return ok ? 0 : 1;
}
