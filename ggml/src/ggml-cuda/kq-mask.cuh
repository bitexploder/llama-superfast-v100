#pragma once

#include "common.cuh"

// [TAG_KQ_MASK_DEVICE] the KQ mask of one causal sequence built on the device from a mirror of the KV
// cells (llama_kv_cache::set_input_kq_mask_device has the rule and the mirror's layout)

#define KQM_MAX_TOKENS 64  // small decode rows carried in kernel arguments
#define KQM_MAX_UPD    256 // changed cells carried in the kernel's arguments; more go as one copy first

// mask: F16 [n_kv, n_tokens], contiguous; cells: I32 [3*kv_size] (position or -1, ext.y, ext.x per cell);
// p/py/px: the rows' positions; upd: [3][upd_n] new values of cells [upd_lo, upd_lo + upd_n)
void ggml_cuda_kq_mask(ggml_backend_cuda_context & ctx, half * mask, int64_t n_kv, int32_t * cells, int64_t kv_size,
        const int32_t * p, const int32_t * py, const int32_t * px, int32_t n_tokens, bool use_2d,
        const int32_t * upd, int32_t upd_lo, int32_t upd_n);
