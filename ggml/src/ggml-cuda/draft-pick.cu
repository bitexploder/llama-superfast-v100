#include "draft-pick.cuh"
#include "../ggml-draft-pick.h"

// GGML_OP_DRAFT_PICK: the MTP draft's pick of its next token, one warp per row (one sequence's draft step).
// The semantics and the arithmetic are ggml_draft_pick_row's (ggml-draft-pick.h, also the CPU backend's): the stable sorts
// are by rank, the exps run on the lanes, and every sum the host forms in order is formed in the same order by lane 0.

struct draft_pick_args {
    const float   * a[3];
    const int32_t * b[3];
    const int32_t * c[3];
    int             src_mode;
    int             n_a;
    const float   * params;
    size_t          params_nb1;
    int32_t       * dst;
    int             n;
    int             k;
};

// stable sort of l[0, n) descending by rank (ties: earlier first), carrying e and p
static __device__ __forceinline__ void draft_pick_sort(float * l, int * e, float * p, const int n, const int lane) {
    float lv[2];
    int   ev[2];
    float pv[2];
    int   rk[2];
    for (int t = 0; t < 2; ++t) {
        const int i = lane + 32*t;
        if (i < n) {
            lv[t] = l[i];
            ev[t] = e[i];
            pv[t] = p[i];
            int r = 0;
#pragma unroll 4
            for (int j = 0; j < n; ++j) { // unrolled: the shared loads overlap (the count is the same)
                r += l[j] > lv[t] || (l[j] == lv[t] && j < i);
            }
            rk[t] = r;
        }
    }
    __syncwarp();
    for (int t = 0; t < 2; ++t) {
        const int i = lane + 32*t;
        if (i < n) {
            l[rk[t]] = lv[t];
            e[rk[t]] = ev[t];
            p[rk[t]] = pv[t];
        }
    }
    __syncwarp();
}

static __global__ void draft_pick_kernel(const draft_pick_args a) {
    const int r    = blockIdx.x;
    const int lane = threadIdx.x;
    const int k    = a.k;

    const float   * prm    = (const float *) ((const char *) a.params + r*a.params_nb1);
    int32_t       * out    = a.dst + a.n + r*(GGML_DRAFT_PICK_HDR + 2*k);

    // the row's k candidates from their source (ggml_draft_pick_cands)
    __shared__ float   s_logits[GGML_DRAFT_PICK_K_MAX];
    __shared__ int32_t s_ids[GGML_DRAFT_PICK_K_MAX];
    __shared__ int32_t s_rows[GGML_DRAFT_PICK_K_MAX];
    {
        const float   * src_a = a.a[r];
        const int32_t * src_b = a.b[r];
        const int32_t * src_c = a.c[r];
        if (a.src_mode == 0) {
            for (int i = lane; i < k; i += 32) {
                s_logits[i] = src_a[i];
                s_ids[i]    = src_b[i];
                s_rows[i]   = src_c[i];
            }
        } else if (a.src_mode == 1) {
            const int n_c = a.n_a/2;
            for (int i = lane; i < n_c; i += 32) {
                const float v = src_a[i];
                int rk = 0;
#pragma unroll 4
                for (int j = 0; j < n_c; ++j) {
                    const float vj = src_a[j];
                    rk += vj > v || (vj == v && j < i);
                }
                if (rk < k) {
                    const int32_t row = (int32_t) src_a[n_c + i];
                    s_logits[rk] = v;
                    s_rows[rk]   = row;
                    s_ids[rk]    = src_c[row];
                }
            }
        } else {
            for (int i = lane; i < k; i += 32) {
                const int32_t row = src_b[i];
                s_logits[i] = src_a[row];
                s_rows[i]   = row;
                s_ids[i]    = src_c[row];
            }
        }
        __syncwarp();
    }
    const float   * logits = s_logits;
    const int32_t * ids    = s_ids;
    const int32_t * rows   = s_rows;

    __shared__ float l[GGML_DRAFT_PICK_K_MAX];
    __shared__ float p[GGML_DRAFT_PICK_K_MAX];
    __shared__ int   e[GGML_DRAFT_PICK_K_MAX];
    __shared__ float s_f;     // a broadcast float
    __shared__ int   s_n;     // the candidates left
    __shared__ int   s_sorted;

    for (int i = lane; i < GGML_DRAFT_PICK_HDR + 2*k; i += 32) {
        out[i] = 0;
    }
    for (int i = lane; i < k; i += 32) {
        l[i] = logits[i];
        e[i] = i;
        p[i] = 0.0f;
    }
    __syncwarp();

    // the draft's own order and p0
    draft_pick_sort(l, e, p, k, lane);
    // whether l is in the sort's order now (s_sorted is the host chain's notion, which decides the semantics). A sort
    // of an array already in that order is the identity (stable, ties by position), so the sorts below run only when it is not:
    // a logit bias or the argmax marking of temp <= 0 may break the order; truncation, compaction and a division by t > 0 keep it
    bool phys_sorted = true;
    for (int i = lane; i < k; i += 32) {
        p[i] = GGML_DRAFT_PICK_EXPF(l[i] - l[0]);
    }
    __syncwarp();
    if (lane == 0) {
        float cum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cum += p[i];
        }
        out[3] = ggml_draft_pick_f2i(GGML_DRAFT_PICK_DIVF(GGML_DRAFT_PICK_EXPF(l[0] - l[0]), cum));
        s_n = k;
        s_sorted = 0;
    }
    __syncwarp();
    for (int i = lane; i < k; i += 32) {
        p[i] = 0.0f;
    }
    __syncwarp();

    const int mode = (int) prm[0];
    if (mode == 1) {
        // logit bias
        const int n_bias = (int) prm[7];
        phys_sorted = phys_sorted && n_bias <= 0;
        for (int i = lane; i < k; i += 32) {
            for (int b = 0; b < n_bias && b < GGML_DRAFT_PICK_NBIAS; ++b) {
                if (ids[e[i]] == (int32_t) prm[8 + 2*b]) {
                    l[i] += prm[8 + 2*b + 1];
                    break;
                }
            }
        }
        __syncwarp();

        const float min_keep = prm[5];
        for (int o = 0; o < 4; ++o) {
            const int code = (int) prm[24 + o];
            if (code == 0) {
                break;
            }
            const int n = s_n;
            if (code == 1) {
                int kk = (int) prm[1];
                if (kk <= 0) {
                    continue;
                }
                kk = kk < n ? kk : n;
                if (!s_sorted && !phys_sorted) {
                    draft_pick_sort(l, e, p, n, lane);
                }
                __syncwarp();
                if (lane == 0) {
                    s_sorted = 1;
                    s_n = kk;
                }
                __syncwarp();
            } else if (code == 2) {
                const float tp = prm[2];
                if (tp >= 1.0f) {
                    continue;
                }
                if (lane == 0) {
                    float max_l = l[0];
                    if (!s_sorted) {
                        for (int i = 1; i < n; ++i) {
                            max_l = l[i] > max_l ? l[i] : max_l;
                        }
                    }
                    s_f = max_l;
                }
                __syncwarp();
                for (int i = lane; i < n; i += 32) {
                    p[i] = GGML_DRAFT_PICK_EXPF(l[i] - s_f);
                }
                __syncwarp();
                if (lane == 0) {
                    float cum = 0.0f;
                    for (int i = 0; i < n; ++i) {
                        cum += p[i];
                    }
                    s_f = cum;
                }
                __syncwarp();
                for (int i = lane; i < n; i += 32) {
                    p[i] = GGML_DRAFT_PICK_DIVF(p[i], s_f);
                }
                __syncwarp();
                if (!s_sorted && !phys_sorted) {
                    draft_pick_sort(l, e, p, n, lane);
                }
                __syncwarp();
                if (lane == 0) {
                    s_sorted = 1;
                    float cs = 0.0f;
                    int last = n;
                    for (int i = 0; i < n; ++i) {
                        cs += p[i];
                        if (cs >= tp && (float) (i + 1) >= min_keep) {
                            last = i + 1;
                            break;
                        }
                    }
                    s_n = last;
                }
                __syncwarp();
            } else if (code == 3) {
                const float mp = prm[3];
                if (mp <= 0.0f || n == 0) {
                    continue;
                }
                if (lane == 0) {
                    bool applied = false;
                    const float lmp = GGML_DRAFT_PICK_LOGF(mp);
                    if (!s_sorted) {
                        float max_l = -3.402823466e+38f;
                        for (int i = 0; i < n; ++i) {
                            max_l = l[i] > max_l ? l[i] : max_l;
                        }
                        const float min_l = max_l + lmp;
                        int nf = 0;
                        for (int i = 0; i < n; ++i) {
                            nf += l[i] >= min_l;
                        }
                        if (nf > 0 && (float) nf >= min_keep) {
                            nf = 0;
                            for (int i = 0; i < n; ++i) {
                                if (l[i] >= min_l) {
                                    l[nf] = l[i];
                                    e[nf] = e[i];
                                    p[nf] = p[i];
                                    nf++;
                                }
                            }
                            s_n = nf;
                            applied = true;
                        }
                    }
                    s_f = applied ? 1.0f : 0.0f;
                }
                __syncwarp();
                if (s_f == 0.0f) {
                    if (!s_sorted && !phys_sorted) {
                        draft_pick_sort(l, e, p, n, lane);
                    }
                    __syncwarp();
                    if (lane == 0) {
                        s_sorted = 1;
                        const float min_l = l[0] + GGML_DRAFT_PICK_LOGF(mp);
                        int i = 1;
                        for (; i < n; ++i) {
                            if (l[i] < min_l && (float) i >= min_keep) {
                                break;
                            }
                        }
                        s_n = i;
                    }
                    __syncwarp();
                }
            } else if (code == 4) {
                const float t = prm[4];
                if (n == 0) {
                    continue;
                }
                if (t <= 0.0f) {
                    phys_sorted = false;
                    if (lane == 0) {
                        int   max_i = 0;
                        float max_l = l[0];
                        for (int i = 1; i < n; ++i) {
                            if (l[i] > max_l) {
                                l[max_i] = -INFINITY;
                                max_i = i;
                                max_l = l[i];
                            } else {
                                l[i] = -INFINITY;
                            }
                        }
                    }
                } else {
                    for (int i = lane; i < n; i += 32) {
                        l[i] = GGML_DRAFT_PICK_DIVF(l[i], t);
                    }
                }
                __syncwarp();
            }
        }

        const int n = s_n;
        bool picked = false;
        if (n > 0) {
            const float tr = prm[28];
            if (tr > 0.0f && prm[4] > 0.0f && tr != prm[4]) {
                const float s = GGML_DRAFT_PICK_DIVF(prm[4], tr);
                for (int i = lane; i < n; i += 32) {
                    if (ggml_draft_pick_isfinite(l[i])) {
                        l[i] *= s;
                    }
                }
                __syncwarp();
            }
            if (lane == 0) {
                float max_l = -INFINITY;
                for (int i = 0; i < n; ++i) {
                    max_l = l[i] > max_l ? l[i] : max_l;
                }
                s_f = max_l;
            }
            __syncwarp();
            const float max_l = s_f;
            if (ggml_draft_pick_isfinite(max_l)) {
                for (int i = lane; i < n; i += 32) {
                    p[i] = ggml_draft_pick_isfinite(l[i]) ? GGML_DRAFT_PICK_EXPF(l[i] - max_l) : 0.0f;
                }
                __syncwarp();
                __shared__ double s_d;
                if (lane == 0) {
                    double sum = 0.0;
                    for (int i = 0; i < n; ++i) {
                        sum += p[i];
                    }
                    s_d = sum;
                }
                __syncwarp();
                for (int i = lane; i < n; i += 32) {
                    p[i] = (float) (p[i] / s_d);
                }
                __syncwarp();
                if (lane == 0) {
                    double tot = 0.0;
                    int last = -1;
                    for (int i = 0; i < n; ++i) {
                        if ((double) p[i] > 0.0) {
                            tot += (double) p[i];
                            last = i;
                        }
                    }
                    if (last >= 0) {
                        const double tgt = ((double) prm[6] + (double) prm[29]) * tot;
                        double run = 0.0;
                        int sel = last;
                        for (int i = 0; i < n; ++i) {
                            if ((double) p[i] > 0.0) {
                                run += (double) p[i];
                                if (run >= tgt) {
                                    sel = i;
                                    break;
                                }
                            }
                        }
                        int nq = 0;
                        for (int i = 0; i < n; ++i) {
                            if (p[i] > 0.0f) {
                                out[GGML_DRAFT_PICK_HDR + nq]     = ids[e[i]];
                                out[GGML_DRAFT_PICK_HDR + k + nq] = ggml_draft_pick_f2i(p[i]);
                                nq++;
                            }
                        }
                        out[0] = ids[e[sel]];
                        out[1] = rows[e[sel]];
                        a.dst[r] = rows[e[sel]];
                        out[2] = ggml_draft_pick_f2i(p[sel]);
                        out[4] = nq;
                        out[5] = e[sel];
                        s_n = -1;
                    }
                }
                __syncwarp();
                picked = s_n == -1;
            }
        }
        if (picked) {
            return;
        }
    }

    // the argmax: the highest raw logit, ties earlier first, with p0 and no q row
    if (lane == 0) {
        int best = 0;
        for (int i = 1; i < k; ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        out[0] = ids[best];
        out[1] = rows[best];
        a.dst[r] = rows[best];
        out[2] = out[3];
        out[4] = 0;
        out[5] = best;
        out[6] = mode == 1 ? 1 : 0;
    }
}

void ggml_cuda_op_draft_pick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int n = ggml_get_op_params_i32(dst, 0);
    const int k = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n >= 1 && n <= 3 && k >= 1 && k <= GGML_DRAFT_PICK_K_MAX);

    draft_pick_args a = {};
    a.src_mode = ggml_get_op_params_i32(dst, 2);
    a.n_a      = ggml_get_op_params_i32(dst, 3);
    GGML_ASSERT(a.src_mode != 1 || a.n_a/2 <= GGML_DRAFT_PICK_K_MAX);
    for (int r = 0; r < n; ++r) {
        a.a[r] = (const float *)   dst->src[3*r + 0]->data;
        a.b[r] = dst->src[3*r + 1] ? (const int32_t *) dst->src[3*r + 1]->data : nullptr;
        a.c[r] = (const int32_t *) dst->src[3*r + 2]->data;
    }
    a.params     = (const float *) dst->src[9]->data;
    a.params_nb1 = dst->src[9]->nb[1];
    a.dst        = (int32_t *) dst->data;
    a.n          = n;
    a.k          = k;

    draft_pick_kernel<<<n, 32, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
