#include "gated_delta_net.cuh"
#include "allreduce-p2p.cuh"
#include "gated_delta_net_chunked.cuh"
#include "ggml-cuda/common.cuh"

// rms_norm_f32<256> (ncols < 1024) then scale_f32 on one row of S_v floats held as x[r] = row[r*32 + lane]:
// rms_norm's warp w sums row[32w .. 32w+31]^2 and warp 0 then sums the 8 warp totals, so the same two
// butterfly levels over the same operands give the same bits (LLAMA_GDN_FUSE_QKNORM)
template <int rows_per_lane>
static __device__ __forceinline__ void gdn_fold_rms_scale(float (&x)[rows_per_lane], const int lane, const int ncols,
                                                          const float eps, const float scale, const float bias) {
    static_assert(rows_per_lane <= 256 / WARP_SIZE, "one element per rms_norm thread");
    float part = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const float w = warp_reduce_sum(x[r] * x[r]);
        part = lane == r ? w : part;
    }
    const float tmp = warp_reduce_sum(part);

    const float mean = tmp / ncols;
    const float rs   = rsqrtf(mean + eps);
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const float n = rs * x[r];
        x[r] = scale * n + bias;
    }
}

template <int S_v, bool KDA, bool keep_rs_t, bool fold = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const ggml_cuda_gdn_fold fd) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    if (fold && fd.states != nullptr && !gdn_state_hazard(fd)) {
        // each warp reads its column before it writes any snapshot of it, so the row may be one it overwrites
        curr_state = fd.states + fd.ids[sequence] * fd.state_row + h_idx * S_v * S_v + col * S_v;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val;
        if (fold && fd.beta != nullptr) {
            // unary.cu op_sigmoid
            const float x = fd.beta[sequence * fd.sr3 + t * fd.sr2 + h_idx * fd.sr1];
            beta_val = 1.0f / (1.0f + expf(-x));
        } else {
            beta_val = *beta_t;
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        if (fold && fd.q != nullptr) {
            const float * qr_t = fd.q + iq3 * fd.sq3 + t * fd.sq2 + iq1 * fd.sq1;
            const float * kr_t = fd.k + iq3 * fd.sq3 + t * fd.sq2 + iq1 * fd.sq1;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = kr_t[i];
                q_reg[r] = qr_t[i];
            }
            gdn_fold_rms_scale<rows_per_lane>(q_reg, lane, fd.norm_ncols, fd.eps_q, fd.scale_q, fd.bias_q);
            gdn_fold_rms_scale<rows_per_lane>(k_reg, lane, fd.norm_ncols, fd.eps_k, fd.scale_k, fd.bias_k);
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            }
        }

        if constexpr (!KDA) {
            float g_val;
            if (fold && fd.alpha != nullptr) {
                // op_add, op_softplus, op_mul, each rounded as its own kernel stores it
                const float x  = fd.alpha[sequence * fd.sa3 + t * fd.sa2 + h_idx * fd.sa1] + fd.dt[h_idx];
                const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
                const float gg = sp * fd.a[h_idx];
                g_val = expf(gg);
            } else {
                g_val = expf(*g_t);
            }

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Decode and verify batches (LLAMA_GDN_STAGED, default on): the inputs of every token do not depend on
// the state, yet gated_delta_net_cuda prepares them inside its token loop, so each warp's serial chain holds, per
// token, the loads of k and q, their folded L2 norms (about fifty dependent shuffles), the gates and the loads of v.
// Here the block's warps first stage every token's k and q (normalized when folded), decay and beta in shared memory,
// each token prepared once per block by one warp with the same expressions, and the recurrence then reads them.
// Every value is computed by the same operations in the same order as there, so outputs and states are bit-identical.
// Each warp owns cpw adjacent columns (LLAMA_GDN_COLS_PER_WARP 1, 2 or 4, default 4): lane l still holds
// rows r*32 + l of each of them, so every kv[col] and attn[col] is the same four terms per lane summed in the same
// order and reduced by the same butterfly as at one column; the cpw columns' chains are independent and interleave,
// and a head's inputs are staged by cpw times fewer blocks. cpw = 1 is the kernel as it was.
#define GGML_CUDA_GDN_STAGED_MAX_TOKENS 8

template <int S_v, bool keep_rs_t, bool fold, int n_warps, int cpw>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * n_warps, 2)
gated_delta_net_staged_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const ggml_cuda_gdn_fold fd) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;
    constexpr int T_max         = GGML_CUDA_GDN_STAGED_MAX_TOKENS;

    __shared__ float k_s[T_max][S_v];
    __shared__ float q_s[T_max][S_v];
    __shared__ float g_s[T_max];
    __shared__ float b_s[T_max];
    // cpw > 1: the block's columns of v for every token, so they do not hold T_max * cpw registers per lane
    __shared__ float v_s[cpw > 1 ? T_max : 1][n_warps * cpw];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    // the warp's first column; it owns columns col .. col + cpw - 1
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * cpw;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float s_shard[cpw][rows_per_lane];

    ggml_cuda_pdl_sync();
    if (fold && fd.states != nullptr && !gdn_state_hazard(fd)) {
        // each warp reads its columns before it writes any snapshot of them, so the row may be one it overwrites
        curr_state = fd.states + fd.ids[sequence] * fd.state_row + h_idx * S_v * S_v + col * S_v;
    }
#pragma unroll
    for (int c = 0; c < cpw; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i   = r * warp_size + lane;
            s_shard[c][r] = curr_state[c * S_v + i];
        }
    }

    // stage: warp w prepares tokens w, w + n_warps, ... as gated_delta_net_cuda prepares each token
    for (int t = threadIdx.y; t < n_tokens; t += n_warps) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset;

        float beta_val;
        if (fold && fd.beta != nullptr) {
            // unary.cu op_sigmoid
            const float x = fd.beta[sequence * fd.sr3 + t * fd.sr2 + h_idx * fd.sr1];
            beta_val = 1.0f / (1.0f + expf(-x));
        } else {
            beta_val = *beta_t;
        }

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        if (fold && fd.q != nullptr) {
            const float * qr_t = fd.q + iq3 * fd.sq3 + t * fd.sq2 + iq1 * fd.sq1;
            const float * kr_t = fd.k + iq3 * fd.sq3 + t * fd.sq2 + iq1 * fd.sq1;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = kr_t[i];
                q_reg[r] = qr_t[i];
            }
            gdn_fold_rms_scale<rows_per_lane>(q_reg, lane, fd.norm_ncols, fd.eps_q, fd.scale_q, fd.bias_q);
            gdn_fold_rms_scale<rows_per_lane>(k_reg, lane, fd.norm_ncols, fd.eps_k, fd.scale_k, fd.bias_k);
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            }
        }

        float g_val;
        if (fold && fd.alpha != nullptr) {
            // op_add, op_softplus, op_mul, each rounded as its own kernel stores it
            const float x  = fd.alpha[sequence * fd.sa3 + t * fd.sa2 + h_idx * fd.sa1] + fd.dt[h_idx];
            const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
            const float gg = sp * fd.a[h_idx];
            g_val = expf(gg);
        } else {
            g_val = expf(*g_t);
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_s[t][i] = k_reg[r];
            q_s[t][i] = q_reg[r];
        }
        if (lane == 0) {
            g_s[t] = g_val;
            b_s[t] = beta_val;
        }
    }

    // each warp's own column of v for every token (cpw = 1), or the block's columns staged in shared memory
    float v_col[cpw > 1 ? 1 : T_max];
    if constexpr (cpw == 1) {
#pragma unroll
        for (int t = 0; t < T_max; t++) {
            v_col[t] = t < n_tokens ? v[sequence * sv3 + t * sv2 + h_idx * sv1 + col] : 0.0f;
        }
    } else {
        const int col_b = blockIdx.z * blockDim.y * cpw;
        for (int j = threadIdx.y * warp_size + lane; j < n_tokens * n_warps * cpw; j += n_warps * warp_size) {
            const int t = j / (n_warps * cpw);
            const int c = j % (n_warps * cpw);
            v_s[t][c] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col_b + c];
        }
    }

    __syncthreads();

#pragma unroll
    for (int t = 0; t < T_max; t++) {
        if (t >= n_tokens) {
            break;
        }
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_s[t][i];
            q_reg[r] = q_s[t][i];
        }
        const float g_val    = g_s[t];
        const float beta_val = b_s[t];

        // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
        float kv_shard[cpw];
#pragma unroll
        for (int c = 0; c < cpw; c++) {
            kv_shard[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard[c] += s_shard[c][r] * k_reg[r];
            }
        }
        float kv_col[cpw];
#pragma unroll
        for (int c = 0; c < cpw; c++) {
            kv_col[c] = warp_reduce_sum<warp_size>(kv_shard[c]);
        }

        // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
        // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
        float attn_partial[cpw];
#pragma unroll
        for (int c = 0; c < cpw; c++) {
            // delta[col] = (v[col] - g * kv[col]) * beta
            const float v_val     = cpw == 1 ? v_col[t] : v_s[t][threadIdx.y * cpw + c];
            const float delta_col = (v_val - g_val * kv_col[c]) * beta_val;

            attn_partial[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r]    = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                attn_partial[c] += s_shard[c][r] * q_reg[r];
            }
        }

        float attn_col[cpw];
#pragma unroll
        for (int c = 0; c < cpw; c++) {
            attn_col[c] = warp_reduce_sum<warp_size>(attn_partial[c]);
        }

        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < cpw; c++) {
                attn_data[col + c] = attn_col[c] * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < cpw; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[(col + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < cpw; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i                = r * warp_size + lane;
                state[(col + c) * S_v + i] = s_shard[c][r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t, bool fold>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const ggml_cuda_gdn_fold & fd, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    // the folds assume full 32-lane warps (the host only plans them for S_v >= 32)
    GGML_ASSERT(!fold || S_v >= 32);

    static const bool staged = [] {
        const char * e = getenv("LLAMA_GDN_STAGED");
        return e == nullptr || atoi(e) != 0;
    }();
    if constexpr (!KDA) {
        if (staged && n_tokens <= GGML_CUDA_GDN_STAGED_MAX_TOKENS && warp_size == 32 && (S_v == 64 || S_v == 128)) {
            // LLAMA_GDN_COLS_PER_WARP: columns per warp (1, 2 or 4, default 4); 1 is the one-column kernel.
            // S_v (64 or 128) is a multiple of every n_warps * cpw here, 16 * 4 = 64 at most
            static const int cpw_env = [] {
                const char * e = getenv("LLAMA_GDN_COLS_PER_WARP");
                const int n = e ? atoi(e) : 0;
                return n == 1 || n == 2 || n == 4 ? n : 0;
            }();
            // under --split-mode tensor each card holds half the heads (24 of the 27B's 48), so 4 columns a warp leave
            // 96 blocks for 72 SMs; there, a grid under two blocks per SM takes 2 columns a warp (one card's 192 blocks). Bitwise
            // the same (every column's sums are the same at any cpw). One card keeps 4 (Flash-Next's head count would cross)
            const bool tp_fill = cpw_env == 0 && ggml_cuda_p2p_ar_active() && H*n_seqs*(S_v/(8*4)) < 2*72;
            const int cpw_staged = cpw_env != 0 ? cpw_env : tp_fill ? 2 : 4;
            // LLAMA_GDN_STAGED_WARPS: warps per block (4, 8 or 16), so fewer blocks prepare each head's inputs; default 8
            // at several columns per warp (at 8 tokens each warp then stages one token), 4 at one column
            static const int n_warps_env = [] {
                const char * e = getenv("LLAMA_GDN_STAGED_WARPS");
                const int n = e ? atoi(e) : 0;
                return n == 4 || n == 8 || n == 16 ? n : 0;
            }();
            const int n_warps_staged = n_warps_env != 0 ? n_warps_env : cpw_staged > 1 ? 8 : 4;
            const dim3 grid_s(H, n_seqs, S_v / (n_warps_staged * cpw_staged));
            const dim3 block_s(warp_size, n_warps_staged, 1);
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_s, block_s, 0, stream);
#define GDN_STAGED_LAUNCH(SV, NW, CPW) \
            ggml_cuda_kernel_launch(gated_delta_net_staged_cuda<SV, keep_rs_t, fold, NW, CPW>, lp, \
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, \
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, fd)
#define GDN_STAGED_CPW(SV, NW) \
            switch (cpw_staged) { \
                case 1:  GDN_STAGED_LAUNCH(SV, NW, 1); break; \
                case 2:  GDN_STAGED_LAUNCH(SV, NW, 2); break; \
                default: GDN_STAGED_LAUNCH(SV, NW, 4); break; \
            }
            if (S_v == 64) {
                switch (n_warps_staged) {
                    case 8:  GDN_STAGED_CPW(64, 8);  break;
                    case 16: GDN_STAGED_CPW(64, 16); break;
                    default: GDN_STAGED_CPW(64, 4);  break;
                }
            } else {
                switch (n_warps_staged) {
                    case 8:  GDN_STAGED_CPW(128, 8);  break;
                    case 16: GDN_STAGED_CPW(128, 16); break;
                    default: GDN_STAGED_CPW(128, 4);  break;
                }
            }
#undef GDN_STAGED_CPW
#undef GDN_STAGED_LAUNCH
            return;
        }
    }

    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, fd);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, fold>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, fd);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, fold>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, fd);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, fold>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, fd);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// GET_ROWS of the sequences' state rows (an exact copy), only when gdn_state_hazard holds; otherwise every block returns
static __global__ void gdn_stage_states(const ggml_cuda_gdn_fold fd, float * __restrict__ dst, const int64_t D) {
    if (!gdn_state_hazard(fd)) {
        return;
    }
    const float * src = fd.states + (int64_t) fd.ids[blockIdx.y] * fd.state_row;
    float       * d   = dst + (int64_t) blockIdx.y * D;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < D; i += (int64_t) gridDim.x * blockDim.x) {
        d[i] = src[i];
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_cuda_gdn_fold * fold_in) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // chains folded into this launch
    static const ggml_cuda_gdn_fold no_fold;
    const bool fold = fold_in != nullptr;
    ggml_cuda_gdn_fold fd = fold ? *fold_in : no_fold;

    // the state folded for several sequences while the snapshots go into the cache: the rows each sequence writes, and
    // the conditional gather into the skipped GET_ROWS's buffer (s_d), which runs only when a source row is another sequence's
    if (fold && fd.states != nullptr && n_seqs > 1 && cache != nullptr) {
        const int64_t off = cache->data - fd.states;
        GGML_ASSERT(off >= 0 && off % fd.state_row == 0 && cache->slot_stride % fd.state_row == 0 && fd.n_ids == n_seqs);
        fd.head_row  = off / fd.state_row;
        fd.slot_rows = cache->slot_stride / fd.state_row;
        fd.n_written = (int) (keep_rs ? std::min<int64_t>(K, n_tokens) : 1);
        gdn_stage_states<<<dim3(64, n_seqs, 1), 256, 0, stream>>>(fd, (float *) src_state->data, H * S_v * S_v);
        CUDA_CHECK(cudaGetLastError());
    }
    GGML_ASSERT(!fold || !kda);

    // LLAMA_GDN_CHUNKED (default on): prefill batches run chunked on the tensor cores (gated_delta_net_chunked.cuh)
    // for all tokens but the last K, whose snapshots the rollback reads; those K (none when K == 1) run token-serial
    // below from the chunked state, so every snapshot is a state the recurrence reaches token by token
    static const bool chunked_on = [] {
        const char * e = getenv("LLAMA_GDN_CHUNKED");
        return e == nullptr || atoi(e) != 0;
    }();
    const int64_t n_tail = keep_rs ? std::min<int64_t>(K, n_tokens) : 0;
    const int     cc     = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    auto al16 = [](const void * ptr, int64_t s1, int64_t s2, int64_t s3) {
        return (uintptr_t) ptr % 16 == 0 && s1 % 4 == 0 && s2 % 4 == 0 && s3 % 4 == 0;
    };
    if (chunked_on && !kda && S_v == GDN_CHUNK_D && n_tokens - n_tail >= GDN_CHUNK && H % neqk1 == 0 &&
        GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA &&
        fd.q == nullptr && fd.alpha == nullptr && fd.beta == nullptr && fd.states == nullptr &&
        al16(q_d, sq1, sq2, sq3) && al16(k_d, sq1, sq2, sq3) && al16(v_d, sv1, sv2, sv3)) {
        const int64_t n_c      = n_tokens - n_tail;
        const int64_t n_chunks = (n_c + GDN_CHUNK - 1) / GDN_CHUNK;
        ggml_cuda_pool_alloc<char>  rec(ctx.pool(), gdn_chunk_scratch_bytes(n_seqs, H, n_chunks));
        ggml_cuda_pool_alloc<float> mid(ctx.pool());
        const size_t nrec = (size_t) n_seqs * H * n_chunks;

        gdn_chunk_args p;
        p.q = q_d; p.k = k_d; p.v = v_d; p.g = g_d; p.beta = b_d;
        p.state_in  = s_d;
        p.state_out = state_d;
        if (n_tail > 0) {
            p.state_out = mid.alloc(n_seqs * H * S_v * S_v);
        }
        p.dst   = dst_d;
        p.rec_h = (__half *) rec.get();
        p.rec_f = (float *) (rec.get() + nrec * GDN_CHUNK_REC_H * sizeof(__half));
        p.rec_g = p.rec_f + nrec * GDN_CHUNK_REC_F;
        p.sq1 = sq1; p.sq2 = sq2; p.sq3 = sq3; p.sv1 = sv1; p.sv2 = sv2; p.sv3 = sv3; p.sb1 = sb1; p.sb2 = sb2; p.sb3 = sb3;
        p.H = H; p.neqk1 = neqk1; p.rq3 = rq3; p.n_tokens = n_tokens; p.n_c = n_c; p.n_chunks = n_chunks;
        p.scale = scale;
        gdn_chunk_launch(p, n_seqs, stream);
        CUDA_CHECK(cudaGetLastError());

        // the last n_tail tokens of each sequence, from its chunked state, writing snapshots 0 .. n_tail - 1
        for (int64_t seq = 0; n_tail > 0 && seq < n_seqs; seq++) {
            const int64_t iq3 = seq / rq3;
            launch_gated_delta_net<false, true, false>(
                q_d + iq3 * sq3 + n_c * sq2, k_d + iq3 * sq3 + n_c * sq2, v_d + seq * sv3 + n_c * sv2,
                g_d + seq * sb3 + n_c * sb2, b_d + seq * sb3 + n_c * sb2, p.state_out + seq * H * S_v * S_v,
                dst_d + (seq * n_tokens + n_c) * H * S_v, state_d + seq * H * S_v * S_v,
                S_v, H, n_tail, 1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, 1,
                scale, state_slot_stride, K, no_fold, stream);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, fd, stream);
        } else {
            launch_gated_delta_net<true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, fd, stream);
        }
    } else if (fold) {
        if (keep_rs) {
            launch_gated_delta_net<false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, fd, stream);
        } else {
            launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, fd, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, fd, stream);
        } else {
            launch_gated_delta_net<false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, fd, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gdn_fold * fold) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr, fold);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache,
        const ggml_cuda_gdn_fold * fold) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache, fold);
}

// op_add, op_softplus and op_mul (k_bin_bcast, unary_op_kernel), each rounded as its own kernel stores it
static __global__ void gdn_gate_f32(const float * alpha, const float * dt, const float * a, float * g,
                                    const int64_t n, const int64_t ne0) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t h  = i % ne0;
    const float   x  = alpha[i] + dt[h];
    const float   sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    g[i] = sp * a[h];
}

void ggml_cuda_op_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * alpha, const ggml_tensor * dt,
                           const ggml_tensor * a, ggml_tensor * g) {
    const int64_t n = ggml_nelements(g);
    const int     block = 256;
    gdn_gate_f32<<<(n + block - 1) / block, block, 0, ctx.stream()>>>(
        (const float *) alpha->data, (const float *) dt->data, (const float *) a->data, (float *) g->data, n, g->ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
