#include "llama-kv-cache.h"
#include "ggml-rtimer.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
             const char *   name_tag) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u*(1 + n_stream)*n_layer + 1)*ggml_tensor_overhead()), // + 1: [TAG_KQ_MASK_DEVICE]
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    // [TAG_KQ_MASK_DEVICE] the one device buffer type that holds every layer's K/V, for the cells' device mirror
    ggml_backend_buffer_type_t kqm_buft = nullptr;
    bool kqm_one = true;
    auto kqm_track = [&](ggml_backend_buffer_type_t b) {
        if (b == nullptr || (kqm_buft != nullptr && kqm_buft != b)) {
            kqm_one = false;
        }
        kqm_buft = b;
    };

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                kqm_track(layer_share.k && layer_share.k->buffer ? ggml_backend_buffer_get_type(layer_share.k->buffer) : nullptr);

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        kqm_track(buft);

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }


    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    // [TAG_KQ_MASK_DEVICE] Allocate plain mirrors after the KV buffers: a meta
    // tensor's simple views identify the participating devices in rank order.
    const char * e = getenv("LLAMA_KQ_MASK_DEVICE");
    const char * ens = getenv("LLAMA_KQ_MASK_DEVICE_NS");
    if ((e == nullptr || atoi(e) != 0) && offload && (n_stream == 1 || ens == nullptr || atoi(ens) != 0) &&
            !hparams.no_alloc && !layers.empty() && kqm_one && kqm_buft && !ggml_backend_buft_is_host(kqm_buft)) {
        ggml_tensor * kv = layers.front().k;
        const bool split = ggml_backend_buft_is_meta(kqm_buft);
        const size_t n = split ? ggml_backend_meta_buffer_n_simple(kv->buffer) : 1;
        ggml_tensor * first = split ? ggml_backend_meta_tensor_simple(kv, 0) : kv;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(first->buffer));
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        kqm_fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kq_mask");
        if (kqm_fn) {
            this->kqm_buft = kqm_buft;
            for (size_t j = 0; j < n; ++j) {
                ggml_tensor * simple = split ? ggml_backend_meta_tensor_simple(kv, j) : kv;
                auto buft = ggml_backend_buffer_get_type(simple->buffer);
                ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
                if (!ctx) {
                    throw std::runtime_error("failed to allocate KQ mirror context");
                }
                auto mirror = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3*(int64_t) kv_size*n_stream);
                ggml_format_name(mirror, "cache_%skqm_cells_%zu", name_tag, j);
                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
                if (!buf) {
                    throw std::runtime_error("failed to allocate KQ cell mirror");
                }
                kqm_cells.push_back(mirror);
                ctxs_bufs.emplace_back(std::move(ctx), std::move(buf));
                LLAMA_LOG_INFO("%s: KQ mask built on the device %s from a %.2f MiB cell mirror (LLAMA_KQ_MASK_DEVICE=0 to disable)\n",
                        __func__, ggml_backend_buft_name(buft), ggml_nbytes(mirror)/1024.0/1024.0);
            }
            kqm_epoch.assign(n_stream, 0);
            kqm_cursor.assign(n_stream, 0);
            for (auto & cells : v_cells) {
                cells.cjrnl_enable();
            }
        }
    }

    // decode graphs over the whole cache. From whole_min used keys on, a ubatch of at most whole_max_tok tokens attends
    // over every cell of the stream, with the used count (today's n_kv) passed as data to the streaming attention kernel, so the
    // graph keeps one shape instead of changing at every multiple of 256 (LLAMA_KV_WHOLE_GRAPH=0: off)
    {
        const char * ew = getenv("LLAMA_KV_WHOLE_GRAPH");
        bool ok = (ew == nullptr || atoi(ew) != 0) && model.arch == LLM_ARCH_QWEN35 && kqm_fn != nullptr && !kqm_cells.empty() &&
                n_stream == 1 && swa_type == LLAMA_SWA_TYPE_NONE && !hparams.use_alibi && !hparams.attn_soft_cap && !layers.empty();
        for (const auto & layer : layers) {
            if (!ok) {
                break;
            }
            const uint32_t il = layer.il;
            ok = layer.k && layer.v && layer.k->type == GGML_TYPE_F16 && layer.v->type == GGML_TYPE_F16 &&
                hparams.n_embd_head_k(il) == 256 && hparams.n_embd_head_v(il) == 256 &&
                hparams.n_head(il) == 6*hparams.n_head_kv(il) && !hparams.is_swa(il);
        }
        int min_keys = 0;
        int max_tok  = 0;
        if (ok) {
            ggml_tensor * kv = layers.front().k;
            ggml_tensor * first = ggml_backend_buft_is_meta(kqm_buft) ? ggml_backend_meta_tensor_simple(kv, 0) : kv;
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(first->buffer));
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto limits = (void (*)(int *, int *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fattn_stream_limits");
            if (limits) {
                limits(&min_keys, &max_tok);
            }
        }
        if (ok && min_keys > 0) {
            whole_min     = min_keys;
            whole_max_tok = max_tok;
            LLAMA_LOG_WARN("%s: decode graphs over the whole cache from %d keys (LLAMA_KV_WHOLE_GRAPH=0 turns it off)\n", __func__, whole_min);
        } else {
            whole_min     = 0;
            whole_max_tok = 0;
        }
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_RT_SCOPE("kv.seq_rm");
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        // [TAG_KV_SEQ_RM_INDEX] the cells that hold seq_id at a position in [p0, p1) come from the
        // sequence's position index instead of a scan of every cell (131072 at the full context, three calls per
        // speculative round); visited in ascending cell order, as the scan did, so the cells, the head and the
        // journal end the same. LLAMA_KV_SEQ_RM_SCAN=1 restores the scan
        static const bool scan = [] { const char * e = getenv("LLAMA_KV_SEQ_RM_SCAN"); return e != nullptr && atoi(e) != 0; }();
        if (!scan) {
            static thread_local std::vector<int32_t> idx;
            idx.clear();
            if (p0 < p1) {
                cells.seq_cells_in(seq_id, p0, p1 - 1, idx);
            }
            std::sort(idx.begin(), idx.end());
            for (const int32_t i : idx) {
                if (cells.seq_rm(i, seq_id)) {
                    if (new_head == cells.size()) {
                        new_head = i;
                    }
                }
            }
        } else
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                if (ubatch.token) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                    // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                    ext.tok = hparams.ple_image_token_id != 0
                        ? (llama_token) hparams.ple_image_token_id
                        : (llama_token) hparams.ple_eos_token_id;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

void llama_kv_cache::set_kpool_dirty() {
    kpool_dirty = true;
}

bool llama_kv_cache::get_kpool_dirty() const {
    return kpool_dirty;
}

void llama_kv_cache::clear_kpool_dirty() const {
    kpool_dirty = false;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

uint64_t llama_kv_cache::tu_seq_hash(llama_seq_id seq_id) const {
    const auto & cells = get_cells(seq_id);
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
    mix(cells.get_used());
    for (uint32_t i = cells.used_min(); i < cells.used_max_p1(); ++i) {
        if (cells.is_empty(i)) {
            continue;
        }
        mix(i);
        mix((uint64_t) (int64_t) cells.pos_get(i));
        for (llama_seq_id s = 0; s < (llama_seq_id) seq_to_stream.size(); ++s) {
            if (cells.seq_has(i, s)) {
                mix((uint64_t) s + 0x100000000ULL);
            }
        }
    }
    return h;
}

void llama_kv_cache::cells_jrnl_enable() {
    for (auto & cells : v_cells) {
        cells.jrnl_enable();
    }
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

uint32_t llama_kv_cache::get_n_kv_pos_contiguous(const slot_info & sinfo, const llama_ubatch & ubatch) const {
    if (sinfo.n_stream() != 1 || ubatch.n_seqs_unq != 1 || ubatch.n_tokens == 0 ||
        ubatch.pos == nullptr || ubatch.n_seq_id == nullptr ||
        ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr) {
        return 0;
    }

    const llama_seq_id seq_id = ubatch.seq_id[0][0];
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return 0;
    }

    const uint32_t stream = seq_to_stream[seq_id];
    if (sinfo.strm[0] < 0 || (uint32_t) sinfo.strm[0] != stream || stream >= v_cells.size()) {
        return 0;
    }

    const auto & cells = v_cells[stream];
    const llama_pos pos_max = cells.seq_pos_max(seq_id);

    if (pos_max < 0 || pos_max >= (llama_pos) cells.size()) {
        return 0;
    }

    // the banded op aligns Q to the tail of K: the ubatch must be that monotonic tail, else dense bias
    if ((uint32_t) pos_max + 1 < ubatch.n_tokens) {
        return 0;
    }
    const llama_pos pos_start = pos_max + 1 - ubatch.n_tokens;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] != pos_start + (llama_pos) i ||
            ubatch.n_seq_id[i] < 1 || ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != seq_id) {
            return 0;
        }
    }

    for (llama_pos pos = 0; pos <= pos_max; ++pos) {
        if (cells.is_empty(pos) || cells.pos_get(pos) != pos || !cells.seq_has(pos, seq_id)) {
            return 0;
        }
    }

    return pos_max + 1;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_k_part(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il,
        int64_t n_embd, int64_t i_off) const {
    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_gqa = k->ne[0];
    const int64_t kv_size    = get_size();
    const int64_t n_stream   = k->ne[2];

    GGML_ASSERT(i_off >= 0 && i_off + n_embd <= n_embd_gqa);
    GGML_ASSERT(k_cur->ne[0] == n_embd);

    // merge the streams: k_idxs are global, exactly as in cpy_k
    ggml_tensor * k2 = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);

    // a row-slice view of every cell. ggml_set_rows needs contiguous rows in the DEST,
    // which ggml_is_contiguous_rows() grants for a view whose ne[0] slice is contiguous.
    ggml_tensor * dst = ggml_view_2d(ctx, k2, n_embd, kv_size*n_stream,
            k2->nb[1], ggml_row_size(k2->type, i_off));

    return ggml_set_rows(ctx, dst, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::get_cell_idxs(const llama_ubatch * ubatch, const slot_info & sinfo, std::vector<int64_t> & idxs) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    idxs.resize(n_tokens);
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            idxs[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

// the KQ mask entries of a visible and a dropped cell. the uint8_t form is set_input_kq_vis's byte per cell,
// which it then packs to one bit; llama_cast has no uint8_t form, so it is never asked for one
template<typename T> static T kq_mask_keep() { return llama_cast<T>(0.0f); }
template<typename T> static T kq_mask_drop() { return llama_cast<T>(-INFINITY); }
template<> uint8_t kq_mask_keep<uint8_t>() { return 1; }
template<> uint8_t kq_mask_drop<uint8_t>() { return 0; }

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = kq_mask_keep<T>();
    const T mask_drop = kq_mask_drop<T>();

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    // see llama_non_causal_type
                    const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
                    if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if constexpr (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    // [TAG_KQ_MASK_DEVICE] a mask placed on the device (kq_mask_device_backend) is built there when the ubatch allows it
    const bool on_host = ggml_backend_buffer_is_host(dst->buffer);
    bool check = false;
    if (!on_host && set_input_kq_mask_device(dst, ubatch, causal_attn)) {
        // Compare the first N masks, including prompt ubatches and every TP rank.
        static int check_left = [] { const char * e = getenv("LLAMA_KQ_MASK_DEVICE_CHECK"); return e ? atoi(e) : 0; }();
        if (check_left <= 0) {
            return;
        }
        --check_left;
        check = true;
    }

    // see llama_non_causal_type
    // only the SWA cache (or the SWA layers of a single cache) become non-causal
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    // [TAG_KQ_MASK_DEVICE] a device mask this ubatch cannot build there is written on the host and uploaded
    std::vector<uint8_t> staging;
    void * data = dst->data;
    if (!on_host) {
        staging.resize(ggml_nbytes(dst));
        data = staging.data();
    }

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) data, causal_attn);
    }

    if (check) {
        ggml_backend_synchronize(kqm_backend);
        std::vector<uint8_t> dev(ggml_nbytes(dst));
        const size_t n = ggml_backend_is_meta(kqm_backend) ? ggml_backend_meta_n_backends(kqm_backend) : 1;
        for (size_t j = 0; j < n; ++j) {
            auto simple = ggml_backend_is_meta(kqm_backend) ? ggml_backend_meta_tensor_simple(dst, j) : dst;
            ggml_backend_tensor_get(simple, dev.data(), 0, dev.size());
            const ggml_fp16_t * a = (const ggml_fp16_t *) dev.data();
            const ggml_fp16_t * b = (const ggml_fp16_t *) staging.data();
            size_t ndiff = 0;
            for (size_t i = 0; i < dev.size()/sizeof(ggml_fp16_t); ++i) {
                ndiff += a[i] != b[i];
            }
            LLAMA_LOG_WARN("%s: kq mask device check: card %zu, %lld streams x %lld rows x %lld cells: %zu of %zu values differ\n",
                    __func__, j, (long long) n_stream, (long long) n_tps, (long long) n_kv, ndiff, dev.size()/sizeof(ggml_fp16_t));
            GGML_ASSERT(ndiff == 0);
        }
        return;
    }

    if (!on_host) {
        // the previous graph on the backend's stream may still read this compute-buffer region
        if (kqm_backend != nullptr) {
            ggml_backend_synchronize(kqm_backend);
        }
        ggml_backend_tensor_set(dst, staging.data(), 0, staging.size());
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

// The KQ mask as one bit per cell for the sparse prompt path. set_input_kq_mask_impl writes a byte per cell
// (1 visible, 0 dropped) with ALiBi off, then each row is packed: bit j of a row is word j>>5, bit j&31
template<bool causal, bool swa>
static void set_input_kq_vis_impl(const args_set_input_kq_mask & args, uint8_t * data) {
    if (args.ubatch->is_pos_2d()) {
        set_input_kq_mask_impl<uint8_t, causal, swa, true,  /*alibi*/ false>(args, data);
    } else {
        set_input_kq_mask_impl<uint8_t, causal, swa, false, /*alibi*/ false>(args, data);
    }
}

template<bool causal>
static void set_input_kq_vis_impl(const args_set_input_kq_mask & args, uint8_t * data) {
    if (args.swa_type != LLAMA_SWA_TYPE_NONE) {
        set_input_kq_vis_impl<causal, true> (args, data);
    } else {
        set_input_kq_vis_impl<causal, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_vis(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, int64_t n_kv) const {
    GGML_ASSERT(!hparams.use_alibi);
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && ggml_is_contiguous(dst) && ggml_backend_buffer_is_host(dst->buffer));

    const uint32_t n_tokens = ubatch->n_tokens;

    const int64_t n_words  = dst->ne[0];
    const int64_t n_stream = dst->ne[2];

    GGML_ASSERT(n_words == (n_kv + 31)/32 && dst->ne[3] == 1);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const int64_t n_tps = n_tokens/n_stream;

    GGML_ASSERT(dst->ne[1] == n_tps);

    // as set_input_kq_mask
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    const size_t n_bytes = (size_t) n_kv*n_tokens;
    if (kqv_bytes.size() < n_bytes) {
        kqv_bytes.resize(n_bytes);
    }

    uint8_t * bytes = kqv_bytes.data();

    if (causal_attn) {
        set_input_kq_vis_impl<true> (args, bytes);
    } else {
        set_input_kq_vis_impl<false>(args, bytes);
    }

    // eight 0/1 bytes to eight bits: the product puts byte k's bit at bit 56 + k, with no carries below
    const int64_t n_full = n_kv/32;

    for (int64_t i = 0; i < n_tokens; ++i) {
        const uint8_t * src = bytes + n_kv*i;
        uint32_t      * row = (uint32_t *) dst->data + n_words*i;

        for (int64_t w = 0; w < n_full; ++w) {
            uint32_t word = 0;
            for (int q = 0; q < 4; ++q) {
                uint64_t x;
                memcpy(&x, src + 32*w + 8*q, sizeof(x));
                word |= (uint32_t) ((x*0x0102040810204080ull) >> 56) << (8*q);
            }
            row[w] = word;
        }

        if (n_full < n_words) {
            uint32_t word = 0;
            for (int64_t j = 32*n_full; j < n_kv; ++j) {
                word |= (uint32_t) src[j] << (j & 31);
            }
            row[n_full] = word;
        }
    }

    // LLAMA_QSA_PACK_CHECK=N: the first N calls also build today's F16 mask and compare it cell for cell
    static int check_left = [] { const char * e = getenv("LLAMA_QSA_PACK_CHECK"); return e ? atoi(e) : 0; }();
    static int check_call = 0;
    if (check_left <= 0) {
        return;
    }
    --check_left;
    ++check_call;

    std::vector<ggml_fp16_t> ref(n_bytes);
    set_input_kq_mask_impl<ggml_fp16_t>(args, ref.data(), causal_attn);

    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);

    int64_t n_agree = 0;
    int64_t n_pad   = 0;
    for (int64_t i = 0; i < n_tokens; ++i) {
        const uint32_t * row = (const uint32_t *) dst->data + n_words*i;
        for (int64_t j = 0; j < n_kv; ++j) {
            const bool        bit = (row[j >> 5] >> (j & 31)) & 1;
            const ggml_fp16_t v   = ref[n_kv*i + j];
            n_agree += (v == f16_zero && bit) || (v == f16_ninf && !bit);
        }
        for (int64_t j = n_kv; j < 32*n_words; ++j) {
            n_pad += (row[j >> 5] >> (j & 31)) & 1;
        }
    }

    LLAMA_LOG_WARN("qsa pack check: mask call %d: %lld of %lld bits agree (%lld rows x %lld cells, %lld padding bits set)\n",
            check_call, (long long) n_agree, (long long) n_bytes, (long long) n_tokens, (long long) n_kv, (long long) n_pad);
    GGML_ASSERT(n_agree == (int64_t) n_bytes && n_pad == 0);
}

// [TAG_KQ_MASK_DEVICE]
//
// For one sequence (0), one stream, a causal mask with no SWA and no ALiBi, the host loop above writes, for query row i
// (position p1, M-RoPE y1, x1) and cell j < n_kv: 0 when the cell is not empty, holds sequence 0 and has position
// p0 <= p1, except that with 2D positions a cell at p0 == p1 whose ext is "2D-greater" (y > y1, or y == y1 and x > x1)
// is dropped; -inf otherwise (an empty cell, a cell of another sequence, a later position). Its row-reuse shortcut
// only skips cells whose value cannot differ between rows, so this rule is exact for every row.
//
// The device mirror kqm_cells holds, per cell, the position if the cell holds sequence 0 (else -1), ext.y and ext.x.
// The cells' change journal says which cells changed since this cache last built a mask on the device; their new
// values go with the kernel's arguments (or as one copy when many changed), and the kernel writes the mirror and the
// mask on the backend's stream, before the graph that reads it.
typedef bool (*kqm_fn_t)(ggml_backend_t backend, ggml_tensor * mask, ggml_tensor * cells,
        const int32_t * p, const int32_t * py, const int32_t * px, int32_t n_tokens, bool use_2d,
        const int32_t * upd, int32_t upd_lo, int32_t upd_n);

ggml_backend_t llama_kv_cache::kq_mask_device_backend(ggml_backend_sched_t sched, const ggml_tensor * mask, bool causal_attn) const {
    if (kqm_cells.empty() || mask == nullptr || sched == nullptr) {
        return nullptr;
    }

    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    if (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] < 1 || mask->ne[3] > n_stream ||
            !causal_attn || swa_type != LLAMA_SWA_TYPE_NONE || hparams.use_alibi) {
        return nullptr;
    }

    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && ggml_backend_supports_buft(backend, kqm_buft)) {
            kqm_backend = backend;
            return backend;
        }
    }

    return nullptr;
}

bool llama_kv_cache::set_input_kq_mask_device(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    if (kqm_cells.empty() || kqm_backend == nullptr || kqm_fn == nullptr) {
        return false;
    }

    const uint32_t n_tokens = ubatch->n_tokens;

    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    // the ubatch's tokens in ns groups of n_tps (one per stream of the mask, [n_kv, n_tps, 1, ns]), each group one
    // sequence, the one its stream's mirror holds (sequence 0 with one stream, sequence s in stream s)
    const int64_t ns = dst->ne[3];
    if (dst->type != GGML_TYPE_F16 || n_tokens == 0 || ns < 1 || ns > n_stream || n_tokens % ns != 0 || dst->ne[1]*ns != n_tokens ||
            dst->ne[2] != 1 || !causal_attn || swa_type != LLAMA_SWA_TYPE_NONE || hparams.use_alibi) {
        return false;
    }
    const uint32_t n_tps = n_tokens/ns;

    uint32_t strm[LLAMA_MAX_SEQ];
    for (int64_t g = 0; g < ns; ++g) {
        const llama_seq_id seq = ubatch->seq_id[g*n_tps][0];
        if (seq < 0 || seq >= (llama_seq_id) seq_to_stream.size()) {
            return false;
        }
        const uint32_t s = seq_to_stream[seq];
        if (seq != (n_stream == 1 ? 0 : (llama_seq_id) s)) {
            return false;
        }
        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            if (ubatch->seq_id[g*n_tps + ii][0] != seq) {
                return false;
            }
        }
        if (!v_cells[s].cjrnl_enabled() || (uint32_t) dst->ne[0] > v_cells[s].size()) {
            return false;
        }
        strm[g] = s;
    }

    const bool is_2d = ubatch->is_pos_2d();

    for (int64_t g = 0; g < ns; ++g) {
        const uint32_t s     = strm[g];
        const auto   & cells = v_cells[s];
        const llama_seq_id mseq = n_stream == 1 ? 0 : (llama_seq_id) s;

        const int32_t * p  = ubatch->pos + g*n_tps;
        const int32_t * py = is_2d ? p + n_tokens : nullptr;
        const int32_t * px = is_2d ? p + 2*n_tokens : nullptr;

        // the cells of stream s that changed since its mirror was last written
        const uint64_t epoch = cells.cjrnl_epoch();
        const uint64_t end   = cells.cjrnl_end();

        uint32_t lo = 0;
        uint32_t hi = 0;
        if (epoch != kqm_epoch[s] || kqm_cursor[s] < cells.cjrnl_base()) {
            hi = cells.size();
        } else if (kqm_cursor[s] < end) {
            lo = UINT32_MAX;
            for (uint64_t k = kqm_cursor[s]; k < end; ++k) {
                const uint32_t j = cells.cjrnl_get(k);
                lo = std::min(lo, j);
                hi = std::max(hi, j + 1);
            }
        }

        const uint32_t n_upd = hi - lo;

        std::vector<int32_t> upd(3*(size_t) n_upd);
        for (uint32_t u = 0; u < n_upd; ++u) {
            const uint32_t j = lo + u;
            const bool empty = cells.is_empty(j);
            upd[u]           = !empty && cells.seq_has(j, mseq) ? cells.pos_get(j) : -1;
            upd[n_upd + u]   = !empty ? cells.ext_get(j).y : 0;
            upd[2*n_upd + u] = !empty ? cells.ext_get(j).x : 0;
        }

        const bool split = ggml_backend_is_meta(kqm_backend);
        GGML_ASSERT(!split || ggml_backend_meta_n_backends(kqm_backend) == kqm_cells.size());
        for (size_t j = 0; j < kqm_cells.size(); ++j) {
            auto simple = split ? ggml_backend_meta_tensor_simple(dst, j) : dst;
            auto backend = split ? ggml_backend_meta_simple_backend(kqm_backend, j) : kqm_backend;
            GGML_ASSERT(simple != nullptr);
            ggml_tensor mask_g = *simple;
            mask_g.data  = (char *) simple->data + g*simple->nb[3];
            mask_g.ne[3] = 1;
            ggml_tensor cells_g = *kqm_cells[j];
            cells_g.ne[0] = 3*(int64_t) cells.size();
            cells_g.data  = (char *) kqm_cells[j]->data + (size_t) s*3*cells.size()*sizeof(int32_t);

            // Same simple backend/stream as meta graph_compute; no host fence.
            const bool ok = ((kqm_fn_t) kqm_fn)(backend, &mask_g, &cells_g, p, py, px, (int32_t) n_tps, is_2d,
                    upd.data(), (int32_t) lo, (int32_t) n_upd);
            GGML_ASSERT(ok || (g == 0 && j == 0));
            if (!ok) {
                return false;
            }
        }

        kqm_epoch [s] = epoch;
        kqm_cursor[s] = end;
    }

    return true;
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    const int64_t n_kv = dst->ne[0];
    GGML_ASSERT(dst->ne[1] == n_tokens);

    // [n_kv, n_tokens] in GLOBAL token order (stream-major, same as the KQ mask)
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        const llama_pos p1 = ubatch->pos[i];

        for (int64_t j = 0; j < n_kv; ++j) {
            // use the ACTUAL absolute position in the KV cell; physical slot order is not monotonic
            int32_t rel = (int32_t) extent; // zero-bias column
            if (!cells.is_empty(j)) {
                const llama_pos d = p1 - cells.pos_get(j);
                if (d >= 0 && d < (llama_pos) extent) {
                    rel = (int32_t) d;
                }
            }

            data[i*n_kv + j] = (int32_t) (i*(extent + 1)) + rel;
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
    //       of this very ubatch as well, which is what we want
    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
    // share the same temporal pos

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (!ubatch.token) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (!ubatch.token) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot && k_rot->buffer) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // a whole-context restore replaces every stream, so the cache is emptied once here
    // clear() resets all streams at once, so doing it per stream below would keep only the last one
    if (seq_id == -1) {
        clear(true);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
            if (sinfos_in && !(*sinfos_in)[s].empty()) {
                throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
            }
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, sinfos_in ? &(*sinfos_in)[s] : nullptr);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }

        if (sinfos_out) {
            (*sinfos_out)[s] = sinfo;
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        // the ext as it was saved, to put back after apply_ubatch()
        std::vector<llama_kv_cell_ext> exts;
        if (has_cell_ext()) {
            exts.resize(cell_count);
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;

                exts[i] = ext;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // the layout is cell indices, so it means the same in both caches only while their streams line up
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.strm[0] = strm;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[0][i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
        // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
        // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
        // is what the whole-context path below already does.
        for (uint32_t i = 0; i < (uint32_t) exts.size(); ++i) {
            cells.ext_set(sinfo.idxs[0][i], exts[i]);
        }

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count)) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    // batch the scatter reads per contiguous run of destination indices
    // from inclusive, to exclusive - same convention as cell_ranges_t
    // contiguous cells yield a single run covering the whole block
    struct cell_run { uint32_t from; uint32_t to; };
    std::vector<cell_run> runs;
    if (cell_count > 0) {
        const auto & idxs = sinfo.idxs[0];
        uint32_t i0 = 0;
        while (i0 < cell_count) {
            uint32_t i1 = i0 + 1;
            while (i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        for (const auto & r : runs) {
            io.read_tensor(k, (size_t) r.from * k_size_row, (size_t) (r.to - r.from) * k_size_row);
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            for (const auto & r : runs) {
                io.read_tensor(v, (size_t) r.from * v_size_row, (size_t) (r.to - r.from) * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & r : runs) {
                    const size_t dst_offset = ((size_t) r.from + j * cells.size()) * v_size_el;
                    io.read_tensor(v, dst_offset, (size_t) (r.to - r.from) * v_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv_eff = kv->get_n_kv(sinfos[i_cur]);

    // a decode graph over the whole cache (see the cache's constructor); the used count n_kv_eff goes in as data
    const auto & sinfo = sinfos[i_cur];
    whole = kv->get_whole_min() > 0 && sinfo.n_stream() == 1 && ubatches[i_cur].n_tokens <= (uint32_t) kv->get_whole_max_tok() &&
        n_kv_eff >= (uint32_t) kv->get_whole_min() && n_kv_eff % 256 == 0;
    n_kv = whole ? kv->get_size() : n_kv_eff;

    return true;
}

uint32_t llama_kv_cache_context::get_n_kv_eff() const {
    return n_kv_eff;
}

bool llama_kv_cache_context::is_whole() const {
    return whole;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_context::get_n_kv_pos_contiguous() const {
    // Full-cache and update contexts do not carry a concrete ubatch/slot pair.
    if (ubatches.empty() || sinfos.empty() || i_cur >= ubatches.size() || i_cur >= sinfos.size()) {
        // reserve context: report the whole cache as position-contiguous so the worst-case graph
        // is the banded path; reserving the dense fallback is unallocatable at large n_ctx
        if (kv != nullptr && lctx == nullptr && kv->get_n_stream() == 1) {
            return n_kv;
        }
        return 0;
    }

    const uint32_t result = kv->get_n_kv_pos_contiguous(sinfos[i_cur], ubatches[i_cur]);
    return result <= (uint32_t) n_kv ? result : 0;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

uint32_t llama_kv_cache_context::get_n_stream() const {
    return sinfos[i_cur].s1 - sinfos[i_cur].s0 + 1;
}

uint32_t llama_kv_cache_context::get_strm(uint32_t s) const {
    const auto & sinfo = sinfos[i_cur];

    GGML_ASSERT(s < sinfo.strm.size());

    return sinfo.strm[s];
}

const llama_kv_cache * llama_kv_cache_context::get_kv() const {
    return kv;
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_part(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il,
        int64_t n_embd, int64_t i_off) const {
    return kv->cpy_k_part(ctx, k_cur, k_idxs, il, n_embd, i_off);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

uint32_t llama_kv_cache_context::get_cell(uint32_t s, uint32_t i) const {
    GGML_ASSERT(i_cur < sinfos.size());
    GGML_ASSERT(s < sinfos[i_cur].n_stream() && i < sinfos[i_cur].size());

    return sinfos[i_cur].idxs[s][i];
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::get_cell_idxs(const llama_ubatch * ubatch, std::vector<int64_t> & idxs) const {
    kv->get_cell_idxs(ubatch, sinfos[i_cur], idxs);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_kq_vis(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_vis(dst, ubatch, causal_attn, n_kv);
}

ggml_backend_t llama_kv_cache_context::kq_mask_device_backend(ggml_backend_sched_t sched, const ggml_tensor * mask, bool causal_attn) const {
    return kv->kq_mask_device_backend(sched, mask, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    kv->set_input_pos_rel_flat(dst, ubatch, extent);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
