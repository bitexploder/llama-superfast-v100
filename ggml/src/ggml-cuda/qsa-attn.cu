// Sparse attention over a BF16/F16 K/V cache for QSA prompt batches (ported from
// PentaCoxian's Q8_0 kernel in his llama-v0.5.0 patch): every query attends to its own list of
// selected cells, read straight from the cache. No gathered K/V copy and no cache-sized mask, so
// the cost follows the selected budget instead of the context length.
//
// Expressed as a GGML_OP_FLASH_ATTN_EXT node with src[5] = cell ids (see ggml_cuda_qsa_attn_is_sparse):
//   src[0] q     F32       [D, n_head, n_tok]
//   src[1] k     BF16/F16  [D, n_head_kv, n_cells]   (a cache view)
//   src[2] v     BF16/F16  [D, n_head_kv, n_cells]   (same type as k, not transposed)
//   src[3] mask  F32       [n_sel, n_tok]            per selected slot: the KQ mask entry plus the selection's own bias
//   src[5] ids   I32       [n_sel, n_tok]            cell of each slot (masked slots repeat a valid cell)
//   dst          F32       [D, n_head, n_tok]
//   op_params: [0] scale, [6] QSA_ATTN_R (marks the node)
//
// The slots are taken QSA_ATTN_R at a time (the last group padded with masked slots); each lane
// holds 8 of the 256 dims, so one cell row of one KV head is one 16-byte load per lane.

#include "qsa-attn.cuh"
#include "qsa-attn-mma.cuh"
#include "common.cuh"

#include <atomic>

#define QSA_ATTN_D     256
#define QSA_ATTN_R     4

static constexpr int qsa_dpl = QSA_ATTN_D / WARP_SIZE; // dims per lane: 8

// raw cache data of one lane's 8 dims (dims 8*lane .. 8*lane+7) for the R cells of a group,
// all issued before the math
struct qsa_raw {
    uint4 q[QSA_ATTN_R];
};

static __device__ __forceinline__ void qsa_load_raw(const char * base, const int32_t * cells, const size_t nb_cell, const int lane, qsa_raw & r) {
#pragma unroll
    for (int c = 0; c < QSA_ATTN_R; ++c) {
        r.q[c] = *((const uint4 *) (base + (int64_t) cells[c]*nb_cell) + lane);
    }
}

template <bool is_bf16>
static __device__ __forceinline__ void qsa_dequant(const qsa_raw & r, const int c, float (&x)[qsa_dpl]) {
    const uint32_t w[4] = { r.q[c].x, r.q[c].y, r.q[c].z, r.q[c].w };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (is_bf16) {
            x[2*i + 0] = __uint_as_float(w[i] << 16);
            x[2*i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
        } else {
            const float2 f = __half22float2(*(const half2 *) &w[i]);
            x[2*i + 0] = f.x;
            x[2*i + 1] = f.y;
        }
    }
}

template <bool is_bf16, int HPW, int NGRP>
__launch_bounds__(NGRP*WARP_SIZE)
static __global__ void qsa_attn_kernel(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const float * __restrict__ mask, const int32_t * __restrict__ ids, float * __restrict__ dst,
        float * __restrict__ part_acc, float2 * __restrict__ part_ml,
        const int n_head, const int gqa, const int n_sel, const int n_split,
        const size_t nbq1, const size_t nbq2, const size_t nbk1, const size_t nbk2,
        const size_t nbv1, const size_t nbv2, const size_t nbm_tok, const size_t nbi_tok,
        const float scale) {
    static_assert(HPW*QSA_ATTN_R <= WARP_SIZE, "one lane per (head, cell) score");
    // one CUDA block per (split, kv head, token); its warps split the kv head's query heads, so they
    // read the same K/V rows (shared through L1)
    const int lane  = threadIdx.x;
    const int split = blockIdx.x;
    const int h_kv  = blockIdx.y;
    const int tok   = blockIdx.z;
    const int h0    = h_kv*gqa + threadIdx.y*HPW;   // first query head of this warp

    __shared__ float red[NGRP][WARP_SIZE][HPW*QSA_ATTN_R + 1];

    // this block's share of the query's groups of R slots
    const int n_grp_sel = (n_sel + QSA_ATTN_R - 1) / QSA_ATTN_R;
    const int per = (n_grp_sel + n_split - 1) / n_split;
    const int j0  = split*per;
    const int j1  = min(n_grp_sel, j0 + per);

    // the share's cell ids and masks, loaded once for all warps of the block; slots past n_sel are off
    extern __shared__ char qsa_smem[];
    int32_t * s_ids  = (int32_t *) qsa_smem;
    float   * s_mask = (float *) (s_ids + per*QSA_ATTN_R);
    const int32_t * id_t = (const int32_t *) ((const char *) ids + tok*nbi_tok);
    const float   * m_t  = (const float *) ((const char *) mask + tok*nbm_tok);
    for (int i = threadIdx.y*WARP_SIZE + lane; i < (j1 - j0)*QSA_ATTN_R; i += NGRP*WARP_SIZE) {
        const int slot = j0*QSA_ATTN_R + i;
        const bool in  = slot < n_sel;
        s_ids[i]  = id_t[in ? slot : 0];
        s_mask[i] = in ? m_t[slot] : -INFINITY;
    }
    __syncthreads();

    float qr[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const float4 * qh = (const float4 *) (q + tok*nbq2 + (h0 + h)*nbq1) + 2*lane;
        const float4 a = qh[0];
        const float4 b = qh[1];
        qr[h][0] = a.x*scale; qr[h][1] = a.y*scale; qr[h][2] = a.z*scale; qr[h][3] = a.w*scale;
        qr[h][4] = b.x*scale; qr[h][5] = b.y*scale; qr[h][6] = b.z*scale; qr[h][7] = b.w*scale;
    }

    float acc[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            acc[h][d] = 0.0f;
        }
    }
    // running max and sum of the head this lane scores (lanes h*R .. h*R + R-1 agree)
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int  my_c   = lane % QSA_ATTN_R;
    const bool scorer = lane < HPW*QSA_ATTN_R;

    // next group with a visible cell (padding groups are off entirely)
    auto next_active = [&](int j) {
        while (j < j1) {
            const float * mj = s_mask + (j - j0)*QSA_ATTN_R;
            bool any = false;
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                any |= mj[c] != -INFINITY;
            }
            if (any) {
                break;
            }
            ++j;
        }
        return j;
    };

    const char * k_h = k + h_kv*nbk1;
    const char * v_h = v + h_kv*nbv1;

    for (int j = next_active(j0); j < j1; j = next_active(j + 1)) {
        const float mv = s_mask[(j - j0)*QSA_ATTN_R + my_c];
        // K and V reads both issued before the math
        const int32_t * cells = s_ids + (j - j0)*QSA_ATTN_R;
        qsa_raw kr, vr;
        qsa_load_raw(k_h, cells, nbk2, lane, kr);
        qsa_load_raw(v_h, cells, nbv2, lane, vr);

        // q.k partial sums over this lane's dims, for every (head, cell)
        float part[HPW][QSA_ATTN_R];
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
            float kx[qsa_dpl];
            qsa_dequant<is_bf16>(kr, c, kx);
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                float sacc = 0.0f;
#pragma unroll
                for (int d = 0; d < qsa_dpl; ++d) {
                    sacc += qr[h][d] * kx[d];
                }
                part[h][c] = sacc;
            }
        }
        // transpose through shared memory: lane h*R + c sums (h, c) over all lanes
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                red[threadIdx.y][lane][h*QSA_ATTN_R + c] = part[h][c];
            }
        }
        __syncwarp();
        float s = -INFINITY;
        if (scorer) {
            float sum = 0.0f;
#pragma unroll 8
            for (int i = 0; i < WARP_SIZE; ++i) {
                sum += red[threadIdx.y][i][lane];
            }
            s = mv == -INFINITY ? -INFINITY : sum + mv;
        }
        __syncwarp();

        // online softmax per head over the group's R cells (groups of R lanes)
        float bmax = s;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFF, bmax, off));
        }
        const float m_new = fmaxf(m_run, bmax);
        const float alpha = m_new == -INFINITY ? 1.0f : expf(m_run - m_new);
        const float p     = s == -INFINITY ? 0.0f : expf(s - m_new);
        float psum = p;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            psum += __shfl_xor_sync(0xFFFFFFFF, psum, off);
        }
        l_run = l_run*alpha + psum;
        m_run = m_new;

        // acc = acc*alpha + sum_c p(h, c) * v_c
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float al = __shfl_sync(0xFFFFFFFF, alpha, h*QSA_ATTN_R);
#pragma unroll
            for (int d = 0; d < qsa_dpl; ++d) {
                acc[h][d] *= al;
            }
        }
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
            float vx[qsa_dpl];
            qsa_dequant<is_bf16>(vr, c, vx);
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                const float pc = __shfl_sync(0xFFFFFFFF, p, h*QSA_ATTN_R + c);
#pragma unroll
                for (int d = 0; d < qsa_dpl; ++d) {
                    acc[h][d] += pc * vx[d];
                }
            }
        }
    }

    // per head (m, l) from its scoring lanes
    float mh[HPW], lh[HPW];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        mh[h] = __shfl_sync(0xFFFFFFFF, m_run, h*QSA_ATTN_R);
        lh[h] = __shfl_sync(0xFFFFFFFF, l_run, h*QSA_ATTN_R);
    }

    if (n_split == 1) {
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float inv = lh[h] > 0.0f ? 1.0f/lh[h] : 0.0f;
            float4 * o = (float4 *) (dst + ((int64_t) tok*n_head + h0 + h)*QSA_ATTN_D) + 2*lane;
            o[0] = make_float4(acc[h][0]*inv, acc[h][1]*inv, acc[h][2]*inv, acc[h][3]*inv);
            o[1] = make_float4(acc[h][4]*inv, acc[h][5]*inv, acc[h][6]*inv, acc[h][7]*inv);
        }
        return;
    }

#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const int64_t row = ((int64_t) tok*n_head + h0 + h)*n_split + split;
        float * o = part_acc + row*QSA_ATTN_D + lane*qsa_dpl;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            o[d] = acc[h][d];
        }
        if (lane == 0) {
            part_ml[row] = make_float2(mh[h], lh[h]);
        }
    }
}

// one block per (head, token): merge the splits' running softmax states
static __global__ void qsa_attn_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml,
        float * __restrict__ dst, const int n_split) {
    const int64_t row = (int64_t) blockIdx.y*gridDim.x + blockIdx.x;   // tok*n_head + head
    const float2 * ml = part_ml + row*n_split;
    float m = -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        m = fmaxf(m, ml[s].x);
    }
    float l = 0.0f;
    float o = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float w = ml[s].x == -INFINITY ? 0.0f : expf(ml[s].x - m);
        l += w*ml[s].y;
        o += w*part_acc[(row*n_split + s)*QSA_ATTN_D + threadIdx.x];
    }
    dst[row*QSA_ATTN_D + threadIdx.x] = l > 0.0f ? o/l : 0.0f;
}

bool ggml_cuda_qsa_attn_is_sparse(const ggml_tensor * dst) {
    return dst->op == GGML_OP_FLASH_ATTN_EXT && dst->src[5] != nullptr && dst->src[5]->type == GGML_TYPE_I32;
}

// 3 heads per warp for gqa 12 (4 warps per block); other group sizes pick a divisor
static int qsa_attn_hpw(const int gqa) {
    return gqa % 3 == 0 && gqa/3 <= 8 ? 3 : gqa % 4 == 0 && gqa/4 <= 8 ? 4 : gqa % 2 == 0 && gqa/2 <= 8 ? 2 : 1;
}

bool ggml_cuda_qsa_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * ids = dst->src[5];
    if (!q || !k || !v || !m || q->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F32 || k->type != v->type ||
            (k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F16)) {
        return false;
    }
    if (q->ne[0] != QSA_ATTN_D || k->ne[0] != QSA_ATTN_D || v->ne[0] != QSA_ATTN_D || ggml_get_op_params_i32(dst, 6) != QSA_ATTN_R) {
        return false;
    }
    if (k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type) || k->nb[1] % 16 != 0 || v->nb[1] % 16 != 0 ||
            k->nb[2] % 16 != 0 || v->nb[2] % 16 != 0 || k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2]) {
        return false;
    }
    if (q->ne[1] % k->ne[1] != 0) {
        return false;
    }
    const int gqa = q->ne[1] / k->ne[1];
    const int hpw = qsa_attn_hpw(gqa);
    const int ngrp = gqa / hpw;
    if (!((hpw == 3 && ngrp == 4) || (hpw == 4 && ngrp == 2) || (hpw == 4 && ngrp == 4) ||
          (hpw == 2 && ngrp == 4) || (hpw == 2 && ngrp == 1) || (hpw == 1 && ngrp == 1))) {
        return false;
    }
    return q->nb[0] == sizeof(float) && q->nb[1] % 16 == 0 && q->nb[2] % 16 == 0 &&
           ggml_is_contiguous(m) && ggml_is_contiguous(ids) &&
           ids->ne[1] == q->ne[2] && m->ne[0] == ids->ne[0] && m->ne[1] == q->ne[2] && ids->ne[0] > 0 &&
           dst->ne[0] == QSA_ATTN_D && dst->ne[1] == q->ne[1] && dst->ne[2] == q->ne[2] && ggml_is_contiguous(dst);
}

// prompt chunks (one split) of 12 query heads per KV head go to Strata's m8n8k4 kernel on sm_70
// (qsa-attn-mma.cu); LLAMA_QSA_ATTN_MMA=0 keeps this file's kernel. Decode and verify (several splits) keep it always.
static bool qsa_attn_mma_route(const int n_tok, const int n_sel, const int gqa, const int n_split) {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_QSA_ATTN_MMA");
        return e != nullptr && atoi(e) != 0;
    }();
    if (!enabled || gqa != 12 || n_split != 1) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc != GGML_CUDA_CC_VOLTA || !volta_mma_available(cc)) {
        return false;
    }
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("qsa attention: Strata's m8n8k4 prompt kernel (LLAMA_QSA_ATTN_MMA): taken, tokens %d, slots %d\n",
                      n_tok, n_sel);
    }
    return true;
}

template <bool is_bf16>
static void qsa_attn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * ids = dst->src[5];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_head    = q->ne[1];
    const int n_tok     = q->ne[2];
    const int gqa       = n_head / k->ne[1];
    const int n_sel     = ids->ne[0];
    const int n_grp_sel = (n_sel + QSA_ATTN_R - 1) / QSA_ATTN_R;

    const int hpw   = qsa_attn_hpw(gqa);
    const int n_grp = gqa / hpw;

    // enough blocks to fill the GPU: small batches split each query's slot list
    const int n_sm    = ggml_cuda_info().devices[ctx.device].nsm;
    const int blocks0 = n_tok * (int) k->ne[1];
    int n_split = std::max(1, std::min(64, (n_sm*8 + blocks0 - 1) / blocks0));
    n_split = std::min(n_split, n_grp_sel);

    if (qsa_attn_mma_route(n_tok, n_sel, gqa, n_split)) {
        ggml_cuda_qsa_attn_mma(ctx, dst);
        return;
    }

    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool());
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool());
    if (n_split > 1) {
        part_acc.alloc((size_t) n_tok*n_head*n_split*QSA_ATTN_D);
        part_ml.alloc((size_t) n_tok*n_head*n_split);
    }

    const dim3 grid(n_split, k->ne[1], n_tok);
    const dim3 block(WARP_SIZE, n_grp);
    cudaStream_t stream = ctx.stream();

    const int    per  = (n_grp_sel + n_split - 1) / n_split;
    const size_t smem = (size_t) per*QSA_ATTN_R*(sizeof(int32_t) + sizeof(float));
#define QSA_LAUNCH(HPW, NGRP) qsa_attn_kernel<is_bf16, HPW, NGRP><<<grid, block, smem, stream>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, (const float *) m->data, \
        (const int32_t *) ids->data, (float *) dst->data, part_acc.ptr, part_ml.ptr, \
        n_head, gqa, n_sel, n_split, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], \
        m->nb[1], ids->nb[1], scale)
    if      (hpw == 3 && n_grp == 4) { QSA_LAUNCH(3, 4); }
    else if (hpw == 4 && n_grp == 2) { QSA_LAUNCH(4, 2); }
    else if (hpw == 4 && n_grp == 4) { QSA_LAUNCH(4, 4); }
    else if (hpw == 2 && n_grp == 4) { QSA_LAUNCH(2, 4); }
    else if (hpw == 2 && n_grp == 1) { QSA_LAUNCH(2, 1); }
    else if (hpw == 1 && n_grp == 1) { QSA_LAUNCH(1, 1); }
    else { GGML_ABORT("qsa attention: unsupported head grouping %d x %d", hpw, n_grp); }
#undef QSA_LAUNCH
    CUDA_CHECK(cudaGetLastError());

    if (n_split > 1) {
        qsa_attn_combine<<<dim3(n_head, n_tok), QSA_ATTN_D, 0, stream>>>(part_acc.ptr, part_ml.ptr, (float *) dst->data, n_split);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_qsa_attn_supported(dst));
    if (dst->src[1]->type == GGML_TYPE_BF16) {
        qsa_attn_launch<true>(ctx, dst);
    } else {
        qsa_attn_launch<false>(ctx, dst);
    }
}
