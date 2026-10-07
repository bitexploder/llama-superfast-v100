#include "llama-context.h"

#include "ggml.h"
#include "ggml-rtimer.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;

    // an MTP draft's schedulers have meta arenas of their own (llama_context::meta_arena_main)
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        meta_arena_main = 1;
        meta_arena_inj  = 2;
    }
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    {
        // the fused kernel sums heads in a different order, so a near-tied top-k can differ
        const char * LLAMA_FUSED_LID_DISABLE = getenv("LLAMA_FUSED_LID_DISABLE");
        if (LLAMA_FUSED_LID_DISABLE && atoi(LLAMA_FUSED_LID_DISABLE) != 0) {
            cparams.fused_lid = false;
            cparams.auto_flid = false;
        }
    }

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // the devices of a --split-mode tensor model, for the backend top-k samplers
    cparams.n_tensor_split = model.split_mode() == LLAMA_SPLIT_MODE_TENSOR ? (uint32_t) model.get_split_state_ud.n_devices : 0;

    // [TAG_LOGITS_DEFER]
    {
        const char * e = getenv("LLAMA_BACKEND_TOPK");
        logits_defer = e == nullptr || atoi(e) != 0;
    }

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // the saved residual of a prompt job's ubatch, mirrored on every card, only for a context that can run jobs
    if (job_supported()) {
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        tu_resid_ctx.reset(ggml_init(ip));
        tu_resid = ggml_new_tensor_2d(tu_resid_ctx.get(), GGML_TYPE_F32, hparams.n_embd, cparams.n_ubatch);
        ggml_set_name(tu_resid, "tu_resid");
        auto * dev = model.dev_layer(0);
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        tu_resid_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(tu_resid_ctx.get(), buft));
        if (!tu_resid_buf) {
            throw std::runtime_error("failed to allocate the saved residual (LLAMA_TU)");
        }
        ggml_backend_buffer_clear(tu_resid_buf.get(), 0);
        LLAMA_LOG_INFO("%s: %10s saved residual for prompt jobs (LLAMA_TU) = %8.2f MiB per device\n", __func__,
                ggml_backend_buffer_name(tu_resid_buf.get()), ggml_nbytes(tu_resid) / 1024.0 / 1024.0);
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        // A context whose layers and output share one device gains no overlap from pipeline copies, which only
        // duplicate its inputs (the MTP draft on card 1). LLAMA_PIPE_ONE_DEVICE=1 keeps the copies
        if (pipeline_parallel) {
            static const bool pipe_one_device = [] {
                const char * e = getenv("LLAMA_PIPE_ONE_DEVICE");
                return e != nullptr && atoi(e) != 0;
            }();

            // where the weights sit, not the layer map: the draft is loaded with the target's tensor split, so its empty
            // nominal layers map to card 0. memory_breakdown also answers under no_alloc (the memory-fit probe)
            bool multi_dev = false;
            ggml_backend_dev_t first = nullptr;
            for (const auto & [buft, size] : model.memory_breakdown()) {
                if (size == 0 || ggml_backend_buft_is_host(buft)) continue;
                ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
                if (first == nullptr) first = dev; else if (dev != first) multi_dev = true;
            }

            if (!multi_dev && !pipe_one_device) {
                pipeline_parallel = false;
                LLAMA_LOG_WARN("%s: pipeline copies off: every layer and the output on one device (LLAMA_PIPE_ONE_DEVICE)\n", __func__);
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    pipe_drain();
    synchronize();

    // the second scheduler's rounds over the context's life, decodes with no job active included
    if (tu_peer_total.n > 0) {
        LLAMA_LOG_WARN("TU second-scheduler rounds at exit: %lld, of them %lld with no job active (LLAMA_TU_INJ_DECODE), graphs reused %lld, rebuilt %lld\n",
                (long long) tu_peer_total.n, (long long) tu_peer_total.n_outside, (long long) tu_peer_total.n_reused, (long long) tu_peer_total.n_rebuilt);
    }

    // the asynchronous state copies
    state_seq_wait();
    for (auto & ev : state_async_ev) {
        if (ev) {
            ggml_backend_event_free(ev);
            ev = nullptr;
        }
    }

    for (auto & ev : pipe_ev) {
        if (ev) {
            ggml_backend_event_free(ev);
            ev = nullptr;
        }
    }

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    // the injection's scheduler and arena are made again on the next injection
    GGML_ASSERT(!inj_arena);
    gf_res_inj.reset();
    gf_res_inj_active = nullptr;
    sched_inj.reset();

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    sched_own_events(sched.get());

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                sched_own_events(sched.get());
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.hparams.no_alloc);
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }
    GGML_RT_SCOPE("llm.synchronize");

    // [TAG_SPEC_PIPELINE] the live slot is older than the latest decode: wait only for its own outputs
    if (pipe_live != pipe_newest && pipe_ev_set[pipe_live]) {
        ggml_backend_event_synchronize(pipe_ev[pipe_live]);
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // [TAG_LOGITS_DEFER] the memory update runs its own graphs
        if (logits_dev.t != nullptr) {
            synchronize();
            logits_dev_fetch(-1);
            logits_dev = {};
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;
        if (gf_res_inj) {
            gf_res_inj->reset();
        }
        gf_res_inj_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    logits_dev_fetch(-1);

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);

        logits_dev_fetch(j);

        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_logits_full_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);

        logits_dev_fetch(j);

        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

void llama_context::logits_dev_fetch(int64_t row) {
    if (logits_dev.t == nullptr) {
        return;
    }

    // a meta tensor is read through the views of the arena its graph was allocated in (a peer round's is not 0)
    const meta_arena_guard arena_g(logits_dev.arena);

    const int64_t n_vocab = model.vocab.n_tokens();
    const size_t  nbrow   = n_vocab*sizeof(float);

    const int64_t r0 = row < 0 ? 0                             : row;
    const int64_t r1 = row < 0 ? (int64_t) logits_dev_row.size() : row + 1;

    for (int64_t r = r0; r < r1 && r < (int64_t) logits_dev_row.size(); ++r) {
        const int32_t dev_row = logits_dev_row[r];
        if (dev_row < 0) {
            continue;
        }

        GGML_ASSERT(dev_row < logits_dev.n);
        ggml_backend_tensor_get(logits_dev.t, logits.data + r*n_vocab, dev_row*nbrow, nbrow);

        logits_dev_row[r] = -1;
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp_dev(int32_t n_rows_max) {
    cparams.layer_inp_dev_rows = std::max<int32_t>(0, n_rows_max);
}

bool llama_context::copy_layer_inp_dev(uint32_t slot, const ggml_tensor * dst, size_t dst_offs, ggml_backend_t backend_dst) const {
    if (layer_inp_dev_n <= 0 || slot >= layer_inp_dev_t.size() || layer_inp_dev_backend == nullptr || dst == nullptr || dst->data == nullptr) {
        return false;
    }
    const ggml_tensor * src = layer_inp_dev_t[slot];
    // dst's slice for this slot: the same shape as src, contiguous, at dst_offs
    ggml_tensor d = *dst;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        d.ne[i] = src->ne[i];
        d.nb[i] = src->nb[i];
    }
    if (d.type != src->type || !ggml_is_contiguous(src) || dst_offs + ggml_nbytes(src) > ggml_nbytes(dst)) {
        return false;
    }
    d.data     = (char *) dst->data + dst_offs;
    d.view_src = dst->view_src ? dst->view_src : const_cast<ggml_tensor *>(dst);
    ggml_backend_tensor_copy_async(layer_inp_dev_backend, backend_dst, src, &d);
    return true;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    // under --split-mode tensor a chain of backend top-k samplers only runs on the device (ggml_top_k_split);
    //     any other chain samples on the CPU as before
    bool topk_only = sampler != nullptr && llama_sampler_chain_get(sampler, -1) != nullptr && llama_sampler_chain_n(sampler) > 0;
    for (int i = 0; topk_only && i < llama_sampler_chain_n(sampler); ++i) {
        const char * name = llama_sampler_name(llama_sampler_chain_get(sampler, i));
        if (name[0] == '+' || name[0] == '-') {
            name++;
        }
        topk_only = strcmp(name, "top-k") == 0;
    }
    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR && !topk_only) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

static bool sched_copy0() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_SCHED_COPY0");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

static bool prompt_reserve() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_PROMPT_RESERVE");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

static bool prompt_src_wait() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_PROMPT_SRC_WAIT");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    // a peer round's host time; a decode round on the second scheduler, not another prompt
    // and, after the first job, a decode-sized batch there with no job active (LLAMA_TU_INJ_DECODE)
    const bool tu_peer_round = (tu_job.active || (tu_job_seen && tu_inj_decode())) && inj_arena;
    const int64_t t_peer0 = tu_peer_round ? ggml_time_us() : 0;
    {
        GGML_RT_SCOPE("llm.mctx_apply");
        if (mctx && !mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
    }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    bool rt_reuse;
    {
        GGML_RT_SCOPE("llm.can_reuse");
        rt_reuse = !graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams);
    }
    if (rt_reuse) {
        GGML_RT_COUNT("llm.graph_reused", 1);
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        // [TAG_SPEC_PIPELINE] an async decode keeps the copy slot of the decode in flight; the scheduler orders the
        // input copies itself (ggml_backend_sched_set_pipe), and the outputs go to the other output slot
        if (cparams.pipeline_parallel && !pipe_async) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        // [TAG_SCHED_COPY0] with pipeline parallelism every alloc takes the next input copy slot, so a decode-sized
        // graph rebuilt each round (the draft's catch-up and step 0) gets different input-copy pointers every time,
        // and ggml-cuda re-captures its graph instead of launching the cached one. a synchronize while not
        // allocated puts the scheduler back on copy 0, as it already does between generation steps; the host reads
        // these graphs' outputs before the next round, so the wait costs nothing there. LLAMA_SCHED_COPY0=0 restores
        // the rotation
        if (cparams.pipeline_parallel && ubatch.n_tokens <= 32 && sched_copy0() && !pipe_async) {
            ggml_backend_sched_synchronize(sched.get());
        }

        //const auto t_start_us = ggml_time_us();

        GGML_RT_COUNT("llm.graph_rebuilt", 1);
        {
            GGML_RT_SCOPE("llm.build");
            gf = model.build_graph(gparams);
        }

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        bool rt_alloc_ok;
        {
            GGML_RT_SCOPE("llm.sched_alloc");
            rt_alloc_ok = ggml_backend_sched_alloc_graph(sched.get(), gf);
        }
        if (!rt_alloc_ok) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    // [TAG_SPEC_PIPE_LOOSE] a reused graph keeps the record of its build
    qsa_last = res->qsa;

    const int64_t t_peer_alloc = tu_peer_round ? ggml_time_us() - t_peer0 : 0;

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        GGML_RT_SCOPE("llm.set_inputs");
        ggml_backend_sched_wait_uploads(sched.get()); // the previous uploads from these inputs are done
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    if (tu_peer_round) {
        const int64_t t_peer_inputs = ggml_time_us() - t_peer0;
        tu_peer.n++;
        (rt_reuse ? tu_peer.n_reused : tu_peer.n_rebuilt)++;
        if (tu_inj_decode()) {
            tu_peer.n_outside += !tu_job.active;
            tu_peer_total.n++;
            tu_peer_total.n_outside += !tu_job.active;
            (rt_reuse ? tu_peer_total.n_reused : tu_peer_total.n_rebuilt)++;
        }
        tu_peer.t_alloc_sum_us  += t_peer_alloc;
        tu_peer.t_alloc_max_us   = std::max(tu_peer.t_alloc_max_us, t_peer_alloc);
        tu_peer.t_inputs_sum_us += t_peer_inputs;
        tu_peer.t_inputs_max_us  = std::max(tu_peer.t_inputs_max_us, t_peer_inputs);
    }

    ggml_status status;
    {
        GGML_RT_SCOPE("llm.graph_compute");
        status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // this context's meta arena (--split-mode tensor), restored on return
    const meta_arena_guard arena_g(meta_arena_main);

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // Memoryless models (e.g. canvas diffusion) route decode() here but often need logits for only a few rows
    // (the canvas). When the caller caps n_outputs_max below the batch size, honor the per-token output flags
    // instead of forcing every row, avoiding an [n_tokens, n_vocab] reserve (huge at a 262k vocab). The default
    // n_outputs_max == n_batch >= n_tokens, so this is a no-op for encoder/embedding models.
    const bool output_all = cparams.embeddings || cparams.n_outputs_max >= (uint32_t) batch_inp.n_tokens;

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // [TAG_LOGITS_DEFER] the previous batch's outputs are replaced
    logits_dev = {};
    logits_dev_row.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer (capped: only the flagged rows when output_all is false)
    const uint32_t n_outputs_enc = output_all ? n_tokens : balloc->get_n_outputs();
    if (output_reserve(n_outputs_enc) < n_outputs_enc) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs_enc);
        return -2;
    };

    std::fill(output_ids.begin(), output_ids.end(), -1);
    if (output_all) {
        for (uint32_t i = 0; i < n_tokens; ++i) {
            output_ids[i] = i;
        }
    } else {
        // map each flagged token index -> its output-buffer row (out_ids is sorted for canvas models)
        const auto & out_ids = balloc->get_out_ids();
        for (uint32_t i = 0; i < n_outputs_enc; ++i) {
            output_ids[out_ids[i]] = i;
        }
    }

    n_outputs = n_outputs_enc;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_outputs_enc*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // between the steps of a prompt job, only other sequences' batches
    if (tu_job.active) {
        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;
            for (int j = 0; j < ns; ++j) {
                const llama_seq_id sid = batch_inp.seq_id ? batch_inp.seq_id[i][j] : 0;
                GGML_ASSERT(!job_has_seq(sid) && "a decode of the active prompt job's sequence");
            }
        }
    }

    // this context's meta arena (--split-mode tensor) for the graphs it allocates and computes, restored on return
    const meta_arena_guard arena_g(meta_arena_main);

    // a chained draft request (llama_mtp_chain_set) holds for this decode alone
    struct chain_guard {
        llm_graph_mtp_chain_state & st;
        ~chain_guard() { st.n_steps = 0; }
    } chain_g { mtp_chain };

    // [TAG_ROUND_TIMERS] the graphs and host phases from here on belong to this context and batch size
    if (ggml_rt_on()) {
        ggml_rt_label(this, batch_inp.n_tokens);
    }
    GGML_RT_SCOPE("llm.decode");
    static const int rt_pre_id = ggml_rt_on() ? ggml_rt_id("llm.decode_pre_ubatch") : -1;
    const auto rt_t0 = rt_pre_id >= 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const bool    mtp_embd = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && batch_inp.embd;
    // DFlash embd batches carry the fused target features at the encoder input width
    const bool    dflash_embd = model.arch == LLM_ARCH_DFLASH && batch_inp.embd;
    // [TAG_DFLASH2_FEAT_DEV] a device-fed injection reads no embd rows: the batch carries one float per token
    const int64_t n_embd  = mtp_embd ? hparams.n_embd_out() : dflash_embd ? (cparams.inject_dev ? 1 : hparams.n_embd_inp_enc()) : hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // embedding contexts output every token even when batch.logits is not set
    if (has_samplers && (output_all || batch_inp.logits)) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (!output_all && batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    token_swaps.clear();

    // [TAG_LOGITS_DEFER] the previous batch's outputs are replaced
    logits_dev = {};
    logits_dev_row.clear();

    // [TAG_SPEC_PIPELINE] an async decode: the other slot's deferred logits are about to be overwritten by this
    // decode's graph output, so move them aside first; the scheduler orders the copies into the reused input slot
    if (sched) {
        ggml_backend_sched_set_pipe(sched.get(), pipe_async);
    }
    if (pipe_held.active) {
        pipe_abort(); // a held decode that was neither resumed nor aborted is dropped by the next one
    }
    if (pipe_async) {
        pipe_save_alt_logits();
    }
    pipe_ev_set[pipe_live] = false;

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    // none may run while a prompt job lives: it would move the job's cells
    const bool mem_updated = memory_update(false);
    GGML_ASSERT(!(mem_updated && tu_job.active) && "a cache update during a prompt job");

    // a DFlash2 draft's injection runs on its own scheduler and graph arena (inj_arena_swap), restored on every return
    static const bool inj_arena_on = [] { const char * e = getenv("LLAMA_DFLASH2_INJ_ARENA"); return e == nullptr || atoi(e) != 0; }();
    // so does an MTP draft's step (every row an output, at most one per sequence) or chain of steps
    // (llama_mtp_chain_set): the catch-up and the step keep their graphs allocated side by side, and each is reused round after
    // round instead of both being rebuilt, re-split (--split-mode tensor) and re-allocated twice a round.
    // LLAMA_MTP_STEP_ARENA=0 turns it off
    static const bool mtp_arena_on = [] { const char * e = getenv("LLAMA_MTP_STEP_ARENA"); return e == nullptr || atoi(e) != 0; }();
    const bool mtp_step = mtp_arena_on && mtp_embd && (mtp_chain.n_steps > 0 ||
            (n_tokens_all <= cparams.n_seq_max && balloc->get_n_outputs() == n_tokens_all));
    // a chained draft decode is one ubatch of an MTP batch
    if (mtp_chain.n_steps > 0) {
        mtp_chain.out.clear();
        mtp_chain.seqs.clear();
        mtp_chain.width  = 0;
        mtp_chain.synced = true;
        if (!mtp_embd || n_tokens_all > cparams.n_ubatch) {
            LLAMA_LOG_ERROR("%s: a chained MTP draft decode needs an MTP batch of one ubatch\n", __func__);
            return -1;
        }
    }
    struct inj_guard {
        llama_context * ctx;
        bool on;
        ~inj_guard() { if (on) { ctx->inj_arena_swap(); } }
    } inj_g { this, (((inj_arena_on && dflash_embd && model.hparams.dflash_selector_rank > 0) || mtp_step) && !pipe_async) ||
                    ((tu_job.active || (tu_job_seen && tu_inj_decode())) && n_tokens_all <= (uint32_t) job_peer_rows()) };
    // above: after the context's first job, a decode-sized batch runs there with no job active too, so a job's
    // begin and end do not move it between the schedulers (each move builds, allocates and instantiates its graphs again)
    // a decode while a prompt job is active (a peer round; its sequences are disjoint from the
    // job's, asserted above) runs on the second scheduler, graph result and meta arena, so the job's group graphs on the main
    // scheduler do not evict its graph and it is reused round after round. With no job active, nothing changes
    // only a decode-sized one (LLAMA_TU_PEER_ROWS), so sched_inj stays decode-sized. A larger one, another
    // sequence's prompt, runs on the main scheduler: its buffer is prompt-sized, and nothing live sits in it between the job's
    // groups (the job's outputs are in its own slot, swapped out between its steps)
    if (tu_job.active && !inj_g.on) {
        GGML_ASSERT(!tu_job.out_live && "a prompt decode on the main scheduler while the job's outputs are live");
    }

    // with pipeline parallelism, a prompt ubatch whose tensors outgrow the scheduler's plan (the cache view
    // grows by a ubatch each time) makes ggml_backend_sched_alloc_splits drain every device and re-plan before it is
    // issued, so the devices take turns. reserve the worst-case prompt graph once per decode instead: the later prompt
    // ubatches of the same topology then fit the plan. only when the decode runs more than one ubatch: a prompt that fits
    // one ubatch has no later ubatch to keep in the plan, so it runs as without the reserve and pays no extra graph build.
    // one output row, as a prompt ubatch has (none mid-prompt, one at the end): the backend samplers build a chain per
    // output row. not for the MTP draft, whose catch-up ubatches have no outputs at all (a different graph).
    // LLAMA_PROMPT_RESERVE=0 turns it off
    if (cparams.pipeline_parallel && !pipe_async && !inj_g.on && memory && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
            n_tokens_all > cparams.n_ubatch && prompt_reserve()) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: prompt reserve (LLAMA_PROMPT_RESERVE): prompt batches of more than one ubatch reserve the worst-case prompt graph under pipeline parallelism (%u tokens, ubatch %u)\n", __func__, n_tokens_all, cparams.n_ubatch);
            warned = true;
        }
        const auto mctx_full = memory->init_full();
        if (!mctx_full || !graph_reserve(std::min(cparams.n_ctx, cparams.n_ubatch), cparams.n_seq_max, 1, mctx_full.get())) {
            LLAMA_LOG_ERROR("%s: failed to reserve the prompt graph\n", __func__);
        }
    }

    // a prompt decode of more ubatches than the scheduler has input copies (-b above 4x -ub) writes a copy
    // slot again while the next device may still read it: the activation copy between devices runs on the source
    // device's stream, which the scheduler's wait on the destination's stream does not order. the source stream waits for
    // the destination's last use of the slot too, as the spec pipeline does. ordering only; set_pipe(pipe_async) above
    // resets it for the next decode. same scope as the reserve. LLAMA_PROMPT_SRC_WAIT=0 turns it off
    if (sched && cparams.pipeline_parallel && !pipe_async && !inj_g.on && memory && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
            n_tokens_all > cparams.n_ubatch && prompt_src_wait()) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: prompt source wait (LLAMA_PROMPT_SRC_WAIT): copies between devices in prompt batches of more than one ubatch wait for the copy slot's last reader (%u tokens, ubatch %u)\n", __func__, n_tokens_all, cparams.n_ubatch);
            warned = true;
        }
        ggml_backend_sched_set_pipe(sched.get(), true);
    }

    if (inj_g.on) {
        inj_arena_swap();
    }

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    // no cache optimization while a prompt job lives: the decode waits for the job instead
                    if (!did_optimize && !tu_job.active) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // [TAG_SPEC_PIPELINE] keep the previous decode's recurrent snapshots, which this decode overwrites
    if (pipe_backup) {
        pipe_rs_backup(batch_inp);
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    // not the active prompt job's: it began its own
    for (const auto & entry : sampling.samplers) {
        if (job_has_seq(entry.first)) {
            continue;
        }
        llama_sampler_backend_begin(entry.second);
    }

    if (logits_defer) {
        logits_dev_row.assign(n_outputs_all, -1);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    if (rt_pre_id >= 0) {
        ggml_rt_add(rt_pre_id, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - rt_t0).count());
    }

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        // [TAG_LOGITS_DEFER] the next ubatch reuses the graph output: copy the pending raw logits first
        if (logits_dev.t != nullptr) {
            const meta_arena_guard arena_dev(logits_dev.arena);
            const int64_t n_vocab_dev = vocab.n_tokens();
            ggml_backend_tensor_get_async(logits_dev.backend, logits_dev.t, logits.data + logits_dev.row0*n_vocab_dev,
                    0, logits_dev.n*n_vocab_dev*sizeof(float));
            std::fill_n(logits_dev_row.begin() + logits_dev.row0, logits_dev.n, -1);
            logits_dev = {};
        }

        ggml_status status;

        // [TAG_SPEC_PIPELINE] a held decode stops before its last GPU's splits (one ubatch only)
        ggml_backend_t hold_be = nullptr;
        if (pipe_hold && sched) {
            GGML_ASSERT(n_tokens_all <= cparams.n_ubatch && !cparams.embeddings && "a held decode is one ubatch");
            for (auto & b : backends) {
                if (ggml_backend_dev_type(ggml_backend_get_device(b.get())) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    hold_be = b.get();
                }
            }
            ggml_backend_sched_set_hold(sched.get(), hold_be);
        }

        if (mtp_chain.n_steps > 0 && ubatch.n_tokens != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: a chained MTP draft decode split into ubatches\n", __func__);
            return -1;
        }

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (hold_be) {
            ggml_backend_sched_set_hold(sched.get(), nullptr);
        }

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        // [TAG_SPEC_PIPELINE] held: the outputs do not exist yet, pipe_resume extracts them
        if (hold_be && ggml_backend_sched_is_held(sched.get())) {
            pipe_held.active = true;
            pipe_held.slot   = pipe_live;
            pipe_held.res    = const_cast<llm_graph_result *>(res);
            pipe_held.n_out  = n_outputs;
            pipe_held.n_tok  = ubatch.n_tokens;
            pipe_held.raw    = needs_raw_logits(ubatch, sampling.samplers);
            n_outputs_prev += n_outputs;
            n_tokens_prev  += ubatch.n_tokens;
            continue;
        }

        // a chained draft decode's picks: every step's, read back once, after the whole chain
        if (!res->t_mtp_chain.empty()) {
            const size_t n_per = (size_t) ggml_nelements(res->t_mtp_chain[0]);
            const size_t n_seq = mtp_chain.seqs.size();
            GGML_ASSERT(n_seq > 0 && n_per % n_seq == 0);
            mtp_chain.width = (int32_t) (n_per/n_seq - 1);
            mtp_chain.out.resize(n_per*res->t_mtp_chain.size());
            if (ggml_tensor * t = res->t_mtp_chain_all) {
                // every step's picks in one tensor, one copy
                GGML_ASSERT((size_t) ggml_nelements(t) == mtp_chain.out.size() && ggml_is_contiguous(t));
                ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), t);
                GGML_ASSERT(be != nullptr);
                ggml_backend_tensor_get_async(be, t, mtp_chain.out.data(), 0, mtp_chain.out.size()*sizeof(int32_t));
            } else
            for (size_t i = 0; i < res->t_mtp_chain.size(); ++i) {
                ggml_tensor * t = res->t_mtp_chain[i];
                ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), t);
                GGML_ASSERT(be != nullptr);
                ggml_backend_tensor_get_async(be, t, mtp_chain.out.data() + i*n_per, 0, n_per*sizeof(int32_t));
            }
            mtp_chain.synced = false;
        }

        extract_ubatch_outputs(res, ubatch, n_outputs_prev, n_tokens_prev, n_outputs_all, n_tokens_all);

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    output_map(*balloc);

    // [TAG_SPEC_PIPELINE] mark the end of this slot's output copies
    pipe_newest = pipe_live;
    if (pipe_async && !(pipe_held.active && pipe_held.slot == pipe_live)) {
        ggml_backend_t be = nullptr;
        for (auto & b : backends) {
            if (ggml_backend_dev_type(ggml_backend_get_device(b.get())) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                be = b.get(); // the last GPU holds the output layer with a layer split
            }
        }
        if (be) {
            if (pipe_ev[pipe_live] == nullptr || pipe_ev_be[pipe_live] != be) {
                if (pipe_ev[pipe_live]) {
                    ggml_backend_event_free(pipe_ev[pipe_live]);
                }
                pipe_ev[pipe_live]    = ggml_backend_event_new(ggml_backend_get_device(be));
                pipe_ev_be[pipe_live] = be;
            }
            if (pipe_ev[pipe_live]) {
                ggml_backend_event_record(pipe_ev[pipe_live], be);
                pipe_ev_set[pipe_live] = true;
            }
        }
    }
    pipe_async  = false;
    pipe_backup = false;
    pipe_hold   = false;

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

// the outputs of a computed ubatch into the live output slot: decode() and a prompt job's last group of a ubatch
void llama_context::extract_ubatch_outputs(const llm_graph_result * res, const llama_ubatch & ubatch,
        int64_t n_outputs_prev, int64_t n_tokens_prev, int64_t n_outputs_all, int64_t n_tokens_all) {
    const auto & hparams = model.hparams;
    const int64_t n_vocab = model.vocab.n_tokens();
    const bool has_samplers = !sampling.samplers.empty();

    auto * t_logits  = res->get_logits();
    auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

    if (t_embd && res->get_embd_pooled()) {
        t_embd = res->get_embd_pooled();
    }

    // [TAG_DRAFT_VOCAB] logits over a vocabulary subset exist only for the backend samplers: they are never copied
    // out, so every output row must have one
    if (t_logits && t_logits->ne[0] != n_vocab) {
        // a DFlash2 draft never reads its logits (its selector's lattice is its output), e.g. a warmup's
        GGML_ASSERT((model.arch == LLM_ARCH_DFLASH || !needs_raw_logits(ubatch, sampling.samplers)) && "a head subset needs a backend sampler on every output");
    } else
    // extract logits
    if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        float * logits_out = logits.data + n_outputs_prev*n_vocab;

        if (n_outputs) {
            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
            GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
            ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
        }
    } else if (logits_defer && logits.data && t_logits && n_outputs > 0) {
        // [TAG_LOGITS_DEFER] every output has a backend sampler: leave the raw logits on the device
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
        GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);

        logits_dev = { backend_res, t_logits, n_outputs_prev, n_outputs, ggml_backend_meta_get_arena() };
        for (int64_t j = 0; j < n_outputs; ++j) {
            logits_dev_row[n_outputs_prev + j] = (int32_t) j;
        }
    }

    // extract embeddings
    if (embd.data && t_embd && n_outputs > 0) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();
                    float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                    if (n_outputs) {
                        GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                        GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings (cleared before processing each batch)
                    auto & embd_seq_out = embd_seq;

                    // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                    // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // [TAG_DFLASH2_FEAT_DEV] a one-ubatch decode of few tokens (a verify) keeps its layer inputs on the device
    if (cparams.layer_inp_dev_rows > 0 && n_tokens_prev == 0 && ubatch.n_tokens == n_tokens_all &&
            (int32_t) ubatch.n_tokens <= cparams.layer_inp_dev_rows) {
        layer_inp_dev_t.clear();
        layer_inp_dev_backend = nullptr;
        for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
            if (cparams.embeddings_layer_inp[il]) {
                ggml_tensor * t = res->get_layer_inp((int) il);
                GGML_ASSERT(t != nullptr && "layer input tensor not found");
                ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), t);
                GGML_ASSERT(be != nullptr && (layer_inp_dev_backend == nullptr || layer_inp_dev_backend == be));
                layer_inp_dev_backend = be;
                layer_inp_dev_t.push_back(t);
            }
        }
        layer_inp_dev_n = (int32_t) ubatch.n_tokens;
    } else {
        layer_inp_dev_n = 0;
        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);
    }

    // extract nextn embeddings before
    // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
    {
        const bool masked    = cparams.embeddings_nextn_masked;
        const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
        const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

        if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
            ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
            GGML_ASSERT(backend_h != nullptr);

            const uint32_t n_embd  = hparams.n_embd_out();
            float * embd_nextn_out = embd_nextn.data + offset*n_embd;

            GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
            ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
        }
    }

    if (has_samplers) {
        const auto stride = n_vocab;

        // async copy the sampling data from the backend to the host
        copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
        copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
        copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
        copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
    }
}

// the output mappings of a logical batch (n_outputs is its outputs), from its batch allocator
void llama_context::output_map(llama_batch_allocr & balloc) {
    // every token has its per-token rows (an unmasked nextn row, layer inputs), outputs or not, at its place in
    // ubatch order: the swaps that put them in batch order
    token_swaps.clear();
    bool token_rows = embd_nextn.data != nullptr && cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;
    for (const auto & li : embd_layer_inp) {
        token_rows = token_rows || li.size > 0;
    }
    if (token_rows && !llama_batch_row_swaps(balloc.get_tok_ids(), token_swaps)) {
        GGML_ABORT("the ubatches do not hold every token of the batch once");
    }
    if (!token_swaps.empty()) {
        static const bool trace = [] { const char * e = getenv("LLAMA_MTP_CARRY_TRACE"); return e != nullptr && atoi(e) != 0; }();
        if (trace) {
            LLAMA_LOG_INFO("%s: the per-token rows of a %zu-token batch were in ubatch order: %zu row swaps put them in batch order\n",
                    __func__, balloc.get_tok_ids().size(), token_swaps.size());
        }
    }

    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc.get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }
}

//
// a resumable prompt job: one logical batch, each ubatch run as groups of layers with other sequences' decodes
// between them. Commit once, restore only: a ubatch is applied to the memory once, at its first group, and the bindings its
// graphs read are captured there (the KV slot info and n_kv stay in the job's memory context, which is never applied again;
// the recurrent head, n, rs_z and copy sources are captured by llama_memory_recurrent_context::tu_capture; the positions are
// the ubatch's own). Every group builds and sets its inputs from that capture. The job's cells (its KV stream) and recurrent
// rows are hashed at the capture and checked before each later group: nothing may move them while the ubatch is open
//

bool llama_context::job_supported() const {
    static const bool env_on = [] {
        const char * e = getenv("LLAMA_TU");
        return e == nullptr || atoi(e) != 0;
    }();
    return env_on && memory && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && cparams.n_seq_max > 1 && !cparams.kv_unified &&
        model.supports_layer_groups() && dynamic_cast<const llama_memory_hybrid *>(memory.get()) != nullptr;
}

void llama_context::job_set_layers(int32_t n_layers) {
    const int32_t p = std::max<int32_t>(1, model.layer_group_period());
    // clamped to the layer count, a multiple of the period: a group of more layers runs the whole ubatch anyway.
    // the request is clamped to [1, n_layer] before it is rounded up, so the sum below stays within n_layer + p - 1
    // (no int32 overflow); the period divides n_layer, so the rounded value stays at most n_layer
    const int32_t n_layer = std::max<int32_t>(p, (int32_t) model.hparams.n_layer());
    const int32_t c = std::min<int32_t>(std::max<int32_t>(1, n_layers), n_layer);
    const int32_t n = ((c + p - 1)/p)*p;
    if (n != n_layers) {
        const char * how = c != n_layers && n != c ? "clamped to the layer count and rounded up to the period" :
                           c != n_layers           ? "clamped to the layer count" : "rounded up to the period";
        LLAMA_LOG_WARN("%s: %d layers per group %s: %d (layer count %d, period of layer types %d)\n", __func__, n_layers, how, n, n_layer, p);
    }
    tu_layers = n;
}

// a batch decoded while a job is active runs on the second scheduler only if it is decode-sized
int32_t llama_context::job_peer_rows() const {
    static const int32_t v = [] {
        const char * e = getenv("LLAMA_TU_PEER_ROWS");
        return e ? std::max(1, atoi(e)) : 64;
    }();
    return v;
}

// LLAMA_TU_INJ_DECODE=0: a decode runs on the second scheduler only while a job is active, as before
bool llama_context::tu_inj_decode() const {
    static const bool v = [] {
        const char * e = getenv("LLAMA_TU_INJ_DECODE");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

bool llama_context::job_has_seq(llama_seq_id seq_id) const {
    return tu_job.active && std::find(tu_job.seqs.begin(), tu_job.seqs.end(), seq_id) != tu_job.seqs.end();
}

void llama_context::job_hash(std::vector<uint64_t> & kv, std::vector<uint64_t> & rs) const {
    const auto * mem = static_cast<const llama_memory_hybrid *>(memory.get());
    kv.clear();
    rs.clear();
    for (llama_seq_id s : tu_job.seqs) {
        kv.push_back(mem->get_mem_attn()->tu_seq_hash(s));
        rs.push_back(mem->get_mem_recr()->tu_seq_hash(s));
    }
}

void llama_context::job_end() {
    auto & J = tu_job;
    J.mctx.reset();
    J.active = false;
    J.cancel = false;
    J.out_live = false; // published, the outputs are the context's
    J.info.t_host_ms = J.t_host_us/1000.0;

    // the job's peer rounds, on the second scheduler
    if (tu_peer.n > 0) {
        std::string bufs;
        for (size_t i = 0; sched_inj && i < backend_ptrs.size(); ++i) {
            const size_t sz = ggml_backend_sched_get_buffer_size(sched_inj.get(), backend_ptrs[i]);
            if (sz > 0) {
                bufs += format(" %s %.2f MiB", ggml_backend_buft_name(backend_buft[i]), sz/1024.0/1024.0);
            }
        }
        // with LLAMA_TU_INJ_DECODE the counters run from the previous job's line, so they include the decodes between
        std::string outside;
        if (tu_inj_decode()) {
            outside = format("; of the rounds, %lld with no job active (since the previous job's line, LLAMA_TU_INJ_DECODE)", (long long) tu_peer.n_outside);
        }
        LLAMA_LOG_WARN("TU peer rounds: %lld, graphs reused %lld, rebuilt %lld, host ms to allocation mean %.3f max %.3f, to inputs mean %.3f max %.3f; second scheduler compute buffers:%s%s\n",
                (long long) tu_peer.n, (long long) tu_peer.n_reused, (long long) tu_peer.n_rebuilt,
                tu_peer.t_alloc_sum_us/1000.0/tu_peer.n, tu_peer.t_alloc_max_us/1000.0,
                tu_peer.t_inputs_sum_us/1000.0/tu_peer.n, tu_peer.t_inputs_max_us/1000.0, bufs.c_str(), outside.c_str());
    }
    if (tu_inj_decode()) {
        tu_peer = {};
    }
}

int32_t llama_context::job_begin(const llama_batch & batch_inp) {
    auto & J = tu_job;

    const char * why = nullptr;
    if (J.active) {
        why = "a job is active";
    } else if (!job_supported() || tu_resid == nullptr) {
        why = "the context cannot run jobs";
    } else if (cparams.pipeline_parallel || cparams.embeddings || !cparams.causal_attn || pipe_async || pipe_hold || pipe_held.active ||
            mtp_chain.n_steps > 0 || inj_arena) {
        why = "the context's mode";
    } else if (batch_inp.n_tokens <= 0 || (uint32_t) batch_inp.n_tokens > cparams.n_batch || batch_inp.token == nullptr ||
            batch_inp.embd != nullptr || batch_inp.pos == nullptr || batch_inp.n_seq_id == nullptr || batch_inp.seq_id == nullptr) {
        why = "the batch";
    }
    for (bool b : cparams.embeddings_layer_inp) {
        if (b && why == nullptr) {
            why = "layer-input outputs";
        }
    }
    if (why != nullptr) {
        LLAMA_LOG_ERROR("%s: cannot begin a prompt job: %s\n", __func__, why);
        return -1;
    }

    // the batch, owned by the job
    const int32_t n = batch_inp.n_tokens;
    J.tokens.assign(batch_inp.token, batch_inp.token + n);
    J.pos.assign(batch_inp.pos, batch_inp.pos + n);
    J.n_seq_id.assign(batch_inp.n_seq_id, batch_inp.n_seq_id + n);
    J.seq_id_data.clear();
    J.seqs.clear();
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < batch_inp.n_seq_id[i]; ++j) {
            const llama_seq_id sid = batch_inp.seq_id[i][j];
            J.seq_id_data.push_back(sid);
            if (std::find(J.seqs.begin(), J.seqs.end(), sid) == J.seqs.end()) {
                J.seqs.push_back(sid);
            }
        }
    }
    if (J.seqs.size() != 1) {
        LLAMA_LOG_ERROR("%s: cannot begin a prompt job: a job is one sequence (%zu)\n", __func__, J.seqs.size());
        return -1;
    }
    J.seq_id.resize(n);
    for (int32_t i = 0, k = 0; i < n; k += J.n_seq_id[i], ++i) {
        J.seq_id[i] = J.seq_id_data.data() + k;
    }
    if (batch_inp.logits) {
        J.logits.assign(batch_inp.logits, batch_inp.logits + n);
    } else {
        J.logits.clear();
    }
    llama_batch batch = {
        /*.n_tokens =*/ n,
        /*.token    =*/ J.tokens.data(),
        /*.embd     =*/ nullptr,
        /*.pos      =*/ J.pos.data(),
        /*.n_seq_id =*/ J.n_seq_id.data(),
        /*.seq_id   =*/ J.seq_id.data(),
        /*.logits   =*/ J.logits.empty() ? nullptr : J.logits.data(),
    };

    const meta_arena_guard arena_g(meta_arena_main);

    if (!J.balloc) {
        J.balloc = std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd());
    }
    if (!J.balloc->init(batch, model.vocab, memory.get(), model.hparams.n_embd_inp(), cparams.n_seq_max, false)) {
        LLAMA_LOG_ERROR("%s: failed to initialize the batch\n", __func__);
        return -1;
    }

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n;

    sched_reserve();

    // pending shifts/copies run before the job, as in decode(); none may run while it lives
    memory_update(false);

    llama_memory_context_ptr mctx;
    bool did_optimize = false;
    while (true) {
        mctx = memory->init_batch(*J.balloc, cparams.n_ubatch, false);
        if (!mctx) {
            return -2;
        }
        if (mctx->get_status() == LLAMA_MEMORY_STATUS_FAILED_PREPARE) {
            if (!did_optimize) {
                did_optimize = true;
                if (memory_update(true)) {
                    continue;
                }
            }
            LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, n);
            return 1;
        }
        if (mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());
            return -2;
        }
        break;
    }

    // the job's outputs: its own slot (pipe_swap's fields), live only during its steps
    output_slot_swap(J.out);
    J.out_live = true;
    embd_seq.clear();
    output_swaps.clear();
    token_swaps.clear();
    logits_dev = {};
    logits_dev_row.clear();
    const uint32_t n_outputs_all = J.balloc->get_n_outputs();
    const bool ok = output_reserve(n_outputs_all) >= n_outputs_all;
    if (ok) {
        for (const auto & entry : sampling.samplers) {
            if (entry.first == J.seqs[0]) {
                llama_sampler_backend_begin(entry.second);
            }
        }
        if (logits_defer) {
            logits_dev_row.assign(n_outputs_all, -1);
        }
    }
    output_slot_swap(J.out);
    J.out_live = false;
    if (!ok) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs_all);
        return -2;
    }

    J.mctx           = std::move(mctx);
    J.n_tokens_all   = J.balloc->get_n_tokens();
    J.n_outputs_all  = n_outputs_all;
    J.n_outputs_prev = 0;
    J.n_tokens_prev  = 0;
    J.n_outputs_ub   = 0;
    J.il_next        = 0;
    J.layers         = tu_layers;
    J.t_host_us      = 0;
    J.info           = {};
    J.info.n_tokens  = (int32_t) J.n_tokens_all;
    J.info.n_layers  = J.layers;
    J.info.rs_idx    = -1;
    J.cancel         = false;
    J.active         = true;
    J.valid          = true;
    if (!tu_inj_decode()) {
        tu_peer = {};
    }
    tu_job_seen      = true;

    return 0;
}

llm_graph_result * llama_context::job_process_group(const llama_ubatch & ubatch, int32_t il0, int32_t il1, ggml_status & ret) {
    auto & J = tu_job;
    const int32_t n_layer = (int32_t) model.hparams.n_layer();

    tu_graph.mode = il0 == 0 ? (il1 == n_layer ? LLM_GRAPH_TU_NONE : LLM_GRAPH_TU_FIRST)
                             : (il1 == n_layer ? LLM_GRAPH_TU_LAST : LLM_GRAPH_TU_MID);
    tu_graph.il0  = il0;
    tu_graph.il1  = il1;
    struct tu_graph_reset {
        llama_context * c;
        ~tu_graph_reset() { c->tu_graph = {}; }
    } tg { this };

    const int64_t t0 = ggml_time_us();

    if (!J.res) {
        J.res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    auto * res = J.res.get();

    const auto gparams = graph_params(res, ubatch, J.mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

    // the job's graphs have their own result, so a decode's built graph is not reset by them (the shared scheduler still has
    // to allocate it again)
    if (graph_reuse_disable || gf_res_prev_active != res || !res->can_reuse(gparams)) {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        ggml_cgraph * gf = model.build_graph(gparams);
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        gf_res_prev_active = res;
    }

    ggml_backend_sched_wait_uploads(sched.get());
    res->set_inputs(&ubatch);

    J.t_host_us += ggml_time_us() - t0;

    const ggml_status status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;
    return res;
}

int32_t llama_context::job_step(int32_t n_groups) {
    auto & J = tu_job;
    if (!J.active) {
        LLAMA_LOG_ERROR("%s: no prompt job is active\n", __func__);
        return LLAMA_DECODE_JOB_ERROR;
    }

    const meta_arena_guard arena_g(meta_arena_main);

    sched_reserve();

    const int32_t n_layer = (int32_t) model.hparams.n_layer();
    const int64_t n_vocab = model.vocab.n_tokens();

    auto * mctx_h = static_cast<llama_memory_hybrid_context *>(J.mctx.get());
    auto * recr   = const_cast<llama_memory_recurrent_context *>(mctx_h->get_recr());

    // the job's outputs are live during the step
    output_slot_swap(J.out);
    J.out_live = true;
    bool publish = false;

    int32_t ret = LLAMA_DECODE_JOB_YIELDED;
    for (int32_t g = 0; g < std::max<int32_t>(1, n_groups); ++g) {
        const auto & ubatch = J.mctx->get_ubatch();

        const int64_t t0 = ggml_time_us();
        bool ok = true;
        if (J.il_next == 0) {
            int32_t n_outputs_new = 0;
            if (J.n_outputs_all == J.n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }
            n_outputs = n_outputs_new;
            J.n_outputs_ub = n_outputs_new;

            // commit once: the ubatch is applied here, at its first group, and never again
            if (!J.mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
                ok = false;
            } else {
                int32_t idx = -1;
                recr->tu_capture(J.seqs[0], &idx);
                if (J.info.n_ubatches == 0) {
                    J.info.rs_idx = idx;
                }
                job_hash(J.hash_kv, J.hash_rs);
            }
        } else {
            n_outputs = J.n_outputs_ub;

            std::vector<uint64_t> hash_kv;
            std::vector<uint64_t> hash_rs;
            job_hash(hash_kv, hash_rs);
            GGML_ASSERT(hash_kv == J.hash_kv && "the prompt job's cache cells moved between its groups");
            GGML_ASSERT(hash_rs == J.hash_rs && "the prompt job's recurrent rows moved between its groups");
        }
        J.t_host_us += ggml_time_us() - t0;

        const int32_t il0 = J.il_next;
        const int32_t il1 = std::min(n_layer, il0 + J.layers);

        ggml_status status = GGML_STATUS_FAILED;
        const llm_graph_result * res = ok ? job_process_group(ubatch, il0, il1, status) : nullptr;
        if (!res) {
            // a failed group: the job's sequence is dropped from the memory, the job fails
            for (llama_seq_id s : J.seqs) {
                LLAMA_LOG_WARN("%s: a group of the prompt job failed (status %d): removing memory module entries for seq_id = %d\n", __func__, status, s);
                memory->seq_rm(s, -1, -1);
            }
            ret = LLAMA_DECODE_JOB_ERROR;
            break;
        }

        J.info.n_groups++;
        J.il_next = il1;

        if (il1 < n_layer) {
            continue;
        }

        // the ubatch completed: its outputs go to the job's slot
        extract_ubatch_outputs(res, ubatch, J.n_outputs_prev, J.n_tokens_prev, J.n_outputs_all, J.n_tokens_all);

        // [TAG_LOGITS_DEFER] the graph outputs are reused by other decodes before the job's outputs are read: copy them now
        if (logits_dev.t != nullptr) {
            const meta_arena_guard arena_dev(logits_dev.arena);
            ggml_backend_tensor_get_async(logits_dev.backend, logits_dev.t, logits.data + logits_dev.row0*n_vocab,
                    0, logits_dev.n*n_vocab*sizeof(float));
            std::fill_n(logits_dev_row.begin() + logits_dev.row0, logits_dev.n, -1);
            logits_dev = {};
        }

        J.n_outputs_prev += n_outputs;
        J.n_tokens_prev  += ubatch.n_tokens;
        J.il_next = 0;
        J.info.n_ubatches++;
        J.info.n_tokens_done = (int32_t) J.n_tokens_prev;

        const bool more = J.mctx->next(); // releases the capture

        if (J.cancel) {
            ret = LLAMA_DECODE_JOB_CANCELLED;
        } else if (!more) {
            n_outputs = J.n_outputs_all;
            output_map(*J.balloc);
            publish = true;
            ret = LLAMA_DECODE_JOB_BATCH_DONE;
        } else {
            ret = LLAMA_DECODE_JOB_UBATCH_DONE;
        }
        break;
    }

    // the slice ends with a synchronize: its copies are done, and the GPU is free for another decode
    synchronize();

    // the published outputs stay live; else the job's slot goes back aside
    if (!publish) {
        output_slot_swap(J.out);
        J.out_live = false;
    }
    if (ret != LLAMA_DECODE_JOB_YIELDED && ret != LLAMA_DECODE_JOB_UBATCH_DONE) {
        job_end();
    }

    return ret;
}

void llama_context::job_cancel() {
    if (tu_job.active) {
        tu_job.cancel = true;
    }
}

bool llama_context::job_get_info(llama_decode_job_info * info) const {
    if (!tu_job.valid || info == nullptr) {
        return false;
    }
    *info = tu_job.info;
    info->t_host_ms = tu_job.t_host_us/1000.0;
    return true;
}

void llama_context::rs_idx_seen_reset(llama_seq_id seq_id) {
    auto * mem = dynamic_cast<llama_memory_hybrid *>(memory.get());
    auto * recr = mem ? mem->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
    if (recr && seq_id >= 0 && (size_t) seq_id < recr->rs_idx_seen.size()) {
        recr->rs_idx_seen[seq_id] = -1;
    }
}

int32_t llama_context::rs_idx_seen(llama_seq_id seq_id) const {
    const auto * mem = dynamic_cast<const llama_memory_hybrid *>(memory.get());
    const auto * recr = mem ? mem->get_mem_recr() : dynamic_cast<const llama_memory_recurrent *>(memory.get());
    if (recr && seq_id >= 0 && (size_t) seq_id < recr->rs_idx_seen.size()) {
        return recr->rs_idx_seen[seq_id];
    }
    return -1;
}

//
// [TAG_SPEC_PIPELINE] pipelined speculative decoding
//

void llama_context::pipe_swap() {
    output_slot_swap(pipe_alt);
}

void llama_context::output_slot_swap(pipe_output_slot & a) {
    std::swap(buf_output,                a.buf_output);
    std::swap(buf_output_rows,           a.buf_output_rows);
    std::swap(logits,                    a.logits);
    std::swap(embd,                      a.embd);
    std::swap(embd_nextn,                a.embd_nextn);
    if (a.embd_layer_inp.size() != embd_layer_inp.size()) {
        a.embd_layer_inp.assign(embd_layer_inp.size(), buffer_view<float>{nullptr, 0});
    }
    std::swap(embd_layer_inp,            a.embd_layer_inp);
    std::swap(sampling.logits,           a.s_logits);
    std::swap(sampling.sampled,          a.s_sampled);
    std::swap(sampling.probs,            a.s_probs);
    std::swap(sampling.candidates,       a.s_candidates);
    std::swap(sampling.logits_count,     a.s_logits_count);
    std::swap(sampling.probs_count,      a.s_probs_count);
    std::swap(sampling.candidates_count, a.s_candidates_count);
    std::swap(logits_dev,                a.logits_dev);
    std::swap(logits_dev_row,            a.logits_dev_row);
    std::swap(embd_seq,                  a.embd_seq);
    std::swap(n_outputs,                 a.n_outputs);
    std::swap(output_ids,                a.output_ids);
    std::swap(output_swaps,              a.output_swaps);
    std::swap(token_swaps,               a.token_swaps);
}

void llama_context::pipe_reserve() {
    // the live slot, then the other one, each for the most outputs a decode can have; pipe_live unchanged
    output_reserve((int32_t) cparams.n_outputs_max);
    pipe_swap();
    output_reserve((int32_t) cparams.n_outputs_max);
    pipe_swap();
}

// as pipe_reserve, for the live slot and the job's slot: a job publishes by swapping its slot in, and output_reserve
// only grows a buffer, so without this the next decode after a job pins a larger one. LLAMA_TU_OUT_RESERVE=0 turns it off
bool llama_context::tu_out_reserve_on() const {
    static const bool on = [] { const char * e = getenv("LLAMA_TU_OUT_RESERVE"); return e == nullptr || atoi(e) != 0; }();
    return on;
}

bool llama_context::tu_out_reserve() {
    if (!job_supported() || tu_job.active) {
        return false;
    }
    if (!tu_out_reserve_on()) {
        LLAMA_LOG_WARN("%s: output slots reserved once (LLAMA_TU_OUT_RESERVE): off; decode-sized batches stay on sched_inj after the first job (LLAMA_TU_INJ_DECODE): %s\n",
                __func__, tu_inj_decode() ? "on" : "off");
        return false;
    }
    auto pinned = [this] {
        return (buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0) +
               (buf_output_rows ? ggml_backend_buffer_get_size(buf_output_rows.get()) : 0);
    };
    tu_out_sizing = true;
    const size_t live0 = pinned();
    const bool ok_live = output_reserve((int32_t) cparams.n_outputs_max) >= cparams.n_outputs_max;
    const size_t live1 = pinned();
    output_slot_swap(tu_job.out);
    const size_t job0 = pinned();
    const bool ok_job = output_reserve((int32_t) cparams.n_outputs_max) >= cparams.n_outputs_max;
    const size_t job1 = pinned();
    output_slot_swap(tu_job.out);
    tu_out_sizing = false;
    tu_out_reserved = ok_live && ok_job;
    LLAMA_LOG_WARN("%s: output slots reserved once for %u outputs (LLAMA_TU_OUT_RESERVE): %s; live slot %.2f -> %.2f MiB, job slot %.2f -> %.2f MiB pinned; decode-sized batches stay on sched_inj after the first job (LLAMA_TU_INJ_DECODE): %s\n",
            __func__, cparams.n_outputs_max, tu_out_reserved ? "on" : "off, a reserve did not succeed",
            live0/1024.0/1024.0, live1/1024.0/1024.0, job0/1024.0/1024.0, job1/1024.0/1024.0, tu_inj_decode() ? "on" : "off");
    return tu_out_reserved;
}

void llama_context::pipe_select(int32_t slot) {
    GGML_ASSERT(slot == 0 || slot == 1);
    if (slot != pipe_live) {
        pipe_swap();
        pipe_live = slot;
    }
}

void llama_context::pipe_decode_flags(bool async, bool backup) {
    pipe_async  = async;
    pipe_backup = backup;
}

void llama_context::pipe_hold_next(bool hold) {
    pipe_hold = hold;
}

bool llama_context::pipe_resume() {
    if (!pipe_held.active) {
        return false;
    }
    pipe_held.active = false;

    const int32_t prev = pipe_live;
    pipe_select(pipe_held.slot);

    ggml_backend_sched_set_pipe(sched.get(), true);
    if (ggml_backend_sched_resume(sched.get()) != GGML_STATUS_SUCCESS) {
        pipe_select(prev);
        return false;
    }

    // the output extraction decode() skipped for this ubatch (row offset 0)
    llm_graph_result * res = pipe_held.res;
    const int64_t n_out   = pipe_held.n_out;
    const int64_t n_vocab = model.vocab.n_tokens();

    ggml_tensor * t_logits  = res->get_logits();
    ggml_tensor * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    ggml_backend_t be = nullptr;
    if (logits.data && t_logits && n_out > 0 && t_logits->ne[0] == n_vocab) {
        be = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        if (pipe_held.raw) {
            ggml_backend_tensor_get_async(be, t_logits, logits.data, 0, n_out*n_vocab*sizeof(float));
        } else if (logits_defer) {
            logits_dev = { be, t_logits, 0, n_out, ggml_backend_meta_get_arena() };
            for (int64_t j = 0; j < n_out; ++j) {
                logits_dev_row[j] = (int32_t) j;
            }
        }
    }
    {
        const bool    masked = cparams.embeddings_nextn_masked;
        const int64_t n_rows = masked ? n_out : pipe_held.n_tok;
        if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
            ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
            const uint32_t n_embd = model.hparams.n_embd_out();
            GGML_ASSERT(n_rows*n_embd <= (int64_t) embd_nextn.size);
            ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_rows*n_embd*sizeof(float));
            be = be ? be : backend_h;
        }
    }
    if (!sampling.samplers.empty()) {
        copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,       0, sched.get());
        copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     n_vocab, 0, sched.get(), &sampling.logits_count);
        copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      n_vocab, 0, sched.get(), &sampling.probs_count);
        copy_tensor_async_rows(res->t_candidates,     sampling.candidates, n_vocab, 0, sched.get(), &sampling.candidates_count);
    }

    if (be) {
        if (pipe_ev[pipe_live] == nullptr || pipe_ev_be[pipe_live] != be) {
            if (pipe_ev[pipe_live]) {
                ggml_backend_event_free(pipe_ev[pipe_live]);
            }
            pipe_ev[pipe_live]    = ggml_backend_event_new(ggml_backend_get_device(be));
            pipe_ev_be[pipe_live] = be;
        }
        if (pipe_ev[pipe_live]) {
            ggml_backend_event_record(pipe_ev[pipe_live], be);
            pipe_ev_set[pipe_live] = true;
        }
    }

    pipe_select(prev);

    return true;
}

void llama_context::pipe_abort() {
    if (pipe_held.active) {
        ggml_backend_sched_abort_held(sched.get());
        pipe_held.active = false;
    }
}

bool llama_context::pipe_rows_independent(int64_t n_more) const {
    const auto & q = qsa_last;
    if (!q.has) {
        return true;
    }
    if (q.union_used) {
        return false;
    }
    if (n_more <= 0 || !q.union_env) {
        return true;
    }
    // the cache view is its used cells padded to 256 (llama_kv_cache::get_n_kv, n_pad 1 here): n_more new cells
    // widen it to at most this. the compact path applies only to a view wider than wp (llama_qsa_compact_applies)
    return GGML_PAD(q.n_kv + n_more, 256) <= q.wp;
}

void llama_context::pipe_drain() {
    if (sched) {
        ggml_backend_sched_synchronize(sched.get());
    }
    pipe_newest    = pipe_live;
    pipe_ev_set[0] = false;
    pipe_ev_set[1] = false;
}

void llama_context::pipe_save_alt_logits() {
    auto & ld = pipe_alt.logits_dev;
    if (ld.t == nullptr || ld.backend == nullptr) {
        return;
    }

    const int32_t  alt     = 1 - pipe_live;
    const int64_t  n_vocab = model.vocab.n_tokens();
    const int64_t  cap     = 16;
    ggml_backend_t be      = ld.backend;

    ggml_backend_dev_t dev = ggml_backend_get_device(be);

    if (ld.n <= cap && ld.t->type == GGML_TYPE_F32 && ld.t->ne[0] == n_vocab && ggml_is_contiguous(ld.t)) {
        if (pipe_side == nullptr || ggml_backend_buft_get_device(ggml_backend_buffer_get_type(pipe_side_buf.get())) != dev) {
            ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
            pipe_side_ctx.reset(ggml_init(ip));
            pipe_side = ggml_new_tensor_2d(pipe_side_ctx.get(), GGML_TYPE_F32, n_vocab, cap);
            pipe_side_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(pipe_side_ctx.get(), ggml_backend_dev_buffer_type(dev)));
            GGML_ASSERT(pipe_side_buf && "failed to allocate the pipelined logits side buffer");
        }

        ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
        pipe_view_ctx[alt].reset(ggml_init(ip));
        ggml_tensor * v = ggml_view_2d(pipe_view_ctx[alt].get(), pipe_side, ld.t->ne[0], ld.t->ne[1], ld.t->nb[1], 0);
        ggml_backend_view_init(v);

        ggml_backend_tensor_copy_async(be, be, ld.t, v);
        ld.t = v;
    } else {
        // does not fit the side buffer: copy the rows to the host now
        GGML_ASSERT(pipe_alt.logits.data != nullptr);
        ggml_backend_tensor_get_async(be, ld.t, pipe_alt.logits.data + ld.row0*n_vocab, 0, ld.n*n_vocab*sizeof(float));
        std::fill_n(pipe_alt.logits_dev_row.begin() + ld.row0, ld.n, -1);
        ld = {};
    }

    // the older slot's reads must also wait for this copy
    if (pipe_ev[alt] && pipe_ev_be[alt] == be) {
        ggml_backend_event_record(pipe_ev[alt], be);
    } else {
        // the copy ran on another stream than the slot's event (not expected with a layer split)
        ggml_backend_synchronize(be);
    }
}

void llama_context::pipe_rs_backup(const llama_batch & batch_inp) {
    llama_memory_recurrent * mr = nullptr;
    if (auto * mh = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        mr = mh->get_mem_recr();
    } else {
        mr = dynamic_cast<llama_memory_recurrent *>(memory.get());
    }
    if (mr == nullptr) {
        return;
    }

    auto copy = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(src->buffer));
        for (auto & b : backends) {
            if (ggml_backend_get_device(b.get()) == dev) {
                ggml_backend_tensor_copy_async(b.get(), b.get(), src, dst);
                return;
            }
        }
        GGML_ABORT("no backend for a recurrent state buffer");
    };

    std::vector<bool> done(LLAMA_MAX_SEQ, false);
    for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
        const llama_seq_id s = batch_inp.seq_id ? batch_inp.seq_id[i][0] : 0;
        if (s >= 0 && s < LLAMA_MAX_SEQ && !done[s]) {
            done[s] = true;
            mr->rs_backup(s, copy);
        }
    }
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    // unmasked nextn rows live in buf_output_rows
    const bool nextn_rows = has_embd_nextn && !cparams.embeddings_nextn_masked;
    if (nextn_rows) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    auto * buft = ggml_backend_cpu_buffer_type();
    // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
    auto * output_dev = model.dev_output();
    auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
    if (output_dev_host_buft) {
        buft = output_dev_host_buft;
    }

    // the rows sized by n_batch alone, in their own buffer: its size depends on no request, so it is
    // pinned and cleared once (at load with llama_pipe_reserve) instead of with every growth of buf_output
    const size_t rows_size = ((nextn_rows ? embd_nextn.size : 0) + embd_layer_inp_float_count) * sizeof(float);
    // after tu_out_reserve, no slot grows: reported once, not fatal
    auto tu_out_grow = [this](const char * what, size_t from, size_t to) {
        if (tu_out_reserved && !tu_out_grew) {
            tu_out_grew = true;
            LLAMA_LOG_WARN("%s: TU invariant broken: %s grew after the reserve at load, %.2f -> %.2f MiB (reported once)\n",
                    __func__, what, from/1024.0/1024.0, to/1024.0/1024.0);
        }
    };

    if (rows_size > 0 && (!buf_output_rows || ggml_backend_buffer_get_size(buf_output_rows.get()) < rows_size)) {
        tu_out_grow("buf_output_rows", buf_output_rows ? ggml_backend_buffer_get_size(buf_output_rows.get()) : 0, rows_size);
        if (buf_output_rows) {
            // only when the enabled rows change after the first allocation
            synchronize();

            buf_output_rows = nullptr;
            if (nextn_rows) {
                embd_nextn.data = nullptr;
            }
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        buf_output_rows.reset(ggml_backend_buft_alloc_buffer(buft, rows_size));
        if (buf_output_rows == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output rows buffer of size %.2f MiB\n", __func__, rows_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output_rows.get(), 0);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    size_t new_size  =
        (logits.size + embd.size + (nextn_rows ? 0 : embd_nextn.size) + backend_float_count) * sizeof(float) +
        (                                                                backend_token_count) * sizeof(llama_token);

    // the reserve at load also makes room for backend samplers, which a slot sets with its first request
    if (tu_out_sizing && !has_sampling) {
        new_size += 2*n_vocab*n_outputs_max*sizeof(float) + (1 + n_vocab)*n_outputs_max*sizeof(llama_token);
    }

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        tu_out_grow("buf_output", prev_size, new_size);
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            if (!nextn_rows) {
                embd_nextn.data = nullptr;
            }
        }

        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    uint8_t * rows_base   = buf_output_rows ? (uint8_t *) ggml_backend_buffer_get_base(buf_output_rows.get()) : nullptr;
    size_t    rows_offset = 0;

    if (nextn_rows) {
        embd_nextn = buffer_view<float>{(float *) rows_base, embd_nextn.size};
        rows_offset += embd_nextn.size * sizeof(float);
    } else {
        embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
        offset += embd_nextn.size * sizeof(float);
    }

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (rows_base + rows_offset), (size_t) n_embd * n_batch};
            rows_offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        // [TAG_LOGITS_DEFER] a row still on the device moves with its mapping; two such rows need no copy
        const bool dev0 = i0 < logits_dev_row.size() && logits_dev_row[i0] >= 0;
        const bool dev1 = i1 < logits_dev_row.size() && logits_dev_row[i1] >= 0;
        if (dev0 || dev1) {
            std::swap(logits_dev_row[i0], logits_dev_row[i1]);
        }

        if (logits.size > 0 && !(dev0 && dev1)) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        // masked nextn rows are per output; unmasked ones are per token and follow token_swaps below
        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        // the layer inputs are per token too: they follow token_swaps below

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            // [TAG_LOGITS_DEFER] only the first *_count entries of a row are ever read
            const uint64_t n_l = logits_defer ? std::max(sampling.logits_count[i0],     sampling.logits_count[i1])     : n_vocab;
            const uint64_t n_p = logits_defer ? std::max(sampling.probs_count[i0],      sampling.probs_count[i1])      : n_vocab;
            const uint64_t n_c = logits_defer ? std::max(sampling.candidates_count[i0], sampling.candidates_count[i1]) : n_vocab;

            for (uint64_t k = 0; k < n_l; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_p; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_c; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    // the per-token rows (unmasked nextn, layer inputs) into batch order, so that row i (bulk or _ith) is batch
    // token i's
    for (const auto & sw : token_swaps) {
        if (embd_nextn.size > 0 && !cparams.embeddings_nextn_masked) {
            std::swap_ranges(embd_nextn.data + (size_t) sw.i0*n_embd_out, embd_nextn.data + (size_t) (sw.i0 + 1)*n_embd_out,
                             embd_nextn.data + (size_t) sw.i1*n_embd_out);
        }
        for (auto & li : embd_layer_inp) {
            if (li.size > 0) {
                std::swap_ranges(li.data + (size_t) sw.i0*n_embd, li.data + (size_t) (sw.i0 + 1)*n_embd, li.data + (size_t) sw.i1*n_embd);
            }
        }
    }

    output_swaps.clear();
    token_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5NEXT) {
        // the n_tokens*40 budget below runs out by ubatch 3840: KDA costs 182 nodes + ~16/token
        // per layer, so 34 KDA layers alone need 6.2k + 31.9*n_tokens before DSA or the MoE
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

// (LLAMA_SCHED_OWN_EVENTS, default on) an MTP draft context's schedulers (the catch-up's and the step's, which share the
// backends) keep events of their own: the step's uploads then wait for the step's own previous graph and not for the catch-up still
// running, and under --split-mode tensor the uploads' completion is waited for only before the inputs are written again
void llama_context::sched_own_events(ggml_backend_sched_t s) const {
    static const bool on = [] { const char * e = getenv("LLAMA_SCHED_OWN_EVENTS"); return e == nullptr || atoi(e) != 0; }();
    if (on && s != nullptr && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        ggml_backend_sched_set_own_events(s);
    }
}

void llama_context::inj_arena_swap() {
    if (!inj_arena && !sched_inj) {
        const size_t max_nodes = this->graph_max_nodes(std::min(cparams.n_ctx, cparams.n_ubatch));
        sched_inj.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
        sched_own_events(sched_inj.get());
        if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
            LLAMA_LOG_WARN("%s: the second scheduler (sched_inj, meta arena %d) is created for the peer rounds of a prompt job\n",
                    __func__, meta_arena_inj);
        }
    }
    std::swap(sched, sched_inj);
    std::swap(gf_res_prev_active, gf_res_inj_active);
    inj_arena = !inj_arena;
    // the scheduler's own meta arena (--split-mode tensor): its graph's views stay valid while the other one runs
    ggml_backend_meta_set_arena(inj_arena ? meta_arena_inj : meta_arena_main);
}

llm_graph_result * llama_context::get_gf_res_prev() {
    if (inj_arena) {
        if (!gf_res_inj) {
            gf_res_inj.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
        }
        return gf_res_inj.get();
    }
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    // [TAG_LOGITS_DEFER] the reserve reuses the graph outputs
    if (logits_dev.t != nullptr) {
        synchronize();
        logits_dev_fetch(-1);
        logits_dev = {};
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.mtp_chain       =*/ (uint32_t) mtp_chain.n_steps,
        /*.mtp_chain_k     =*/ (uint32_t) (mtp_chain.n_steps > 0 ? mtp_chain.k : 0),
        /*.mtp_chain_state =*/ mtp_chain.n_steps > 0 ? const_cast<llm_graph_mtp_chain_state *>(&mtp_chain) : nullptr,
        /*.tu_mode         =*/ tu_graph.mode,
        /*.tu_il_begin     =*/ tu_graph.il0,
        /*.tu_il_end       =*/ tu_graph.il1,
        /*.tu_resid        =*/ tu_graph.mode != LLM_GRAPH_TU_NONE ? tu_resid : nullptr,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

// llama_io_write_host, except that the destructor queues each recorded write on the stream of the first non-CPU backend
// that accepts the tensor's buffer type (a tensor in a host buffer is copied directly), and reports the backends it used
class llama_io_write_host_async : public llama_io_write_i {
public:
    llama_io_write_host_async(
            uint8_t * p, size_t len, const std::vector<ggml_backend_t> & cands, std::vector<ggml_backend_t> & used)
        : ptr(p), buf_size(len), cands(cands), used(used) {}

    ~llama_io_write_host_async() {
        for (const auto & winfo : winfos) {
            ggml_backend_buffer_t buf = winfo.tensor->view_src ? winfo.tensor->view_src->buffer : winfo.tensor->buffer;
            ggml_backend_t be = nullptr;
            if (!ggml_backend_buffer_is_host(buf)) {
                for (ggml_backend_t b : cands) {
                    if (ggml_backend_supports_buft(b, ggml_backend_buffer_get_type(buf))) {
                        be = b;
                        break;
                    }
                }
            }
            if (be == nullptr) {
                ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
                continue;
            }
            ggml_backend_tensor_get_async(be, winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
            if (std::find(used.begin(), used.end(), be) == used.end()) {
                used.push_back(be);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    const std::vector<ggml_backend_t> & cands;
    std::vector<ggml_backend_t> & used;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags & ~LLAMA_STATE_SEQ_FLAGS_ASYNC);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    // the copies are queued on the backends' streams; state_seq_wait waits for them
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ASYNC) && !(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        std::vector<ggml_backend_t> cands;
        for (const auto & b : backends) {
            if (b.get() != backend_cpu) {
                cands.push_back(b.get());
            }
        }
        std::vector<ggml_backend_t> used;
        size_t n = 0;
        {
            llama_io_write_host_async io(dst, size, cands, used);
            try {
                io.write(&io_magic, sizeof(io_magic));
                io.write(&seq_id, sizeof(seq_id));

                n = state_seq_write_data(io, seq_id, flags & ~LLAMA_STATE_SEQ_FLAGS_ASYNC);
            } catch (const std::exception & err) {
                LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
                n = 0;
            }
        }
        for (ggml_backend_t be : used) {
            size_t i = 0;
            while (i < state_async_be.size() && state_async_be[i] != be) {
                i++;
            }
            if (i == state_async_be.size()) {
                state_async_be.push_back(be);
                state_async_ev.push_back(ggml_backend_event_new(ggml_backend_get_device(be)));
                state_async_set.push_back(false);
            }
            if (state_async_ev[i]) {
                ggml_backend_event_record(state_async_ev[i], be);
                state_async_set[i] = true;
            } else {
                ggml_backend_synchronize(be); // no events on this device: wait now
            }
        }
        return n;
    }

    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags & ~LLAMA_STATE_SEQ_FLAGS_ASYNC);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

void llama_context::state_seq_wait() {
    for (size_t i = 0; i < state_async_ev.size(); ++i) {
        if (state_async_set[i]) {
            ggml_backend_event_synchronize(state_async_ev[i]);
            state_async_set[i] = false;
        }
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;

        // the graph changes without flash attention, need to reserve again
        sched_need_reserve = true;
        sched_reserve();
    }

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    // quantized V forces the dense path, which transposes V - invalid for a block-quantized tensor
    if (model->arch == LLM_ARCH_INKLING && ggml_is_quantized(params.type_v)) {
        LLAMA_LOG_ERROR("%s: model does not support a quantized V cache (%s)\n",
            __func__, ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_logits_full_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_full_ith(i);
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

bool llama_model_supports_layer_groups(const llama_model * model) {
    return model->supports_layer_groups();
}

bool llama_decode_job_supported(const llama_context * ctx) {
    return ctx->job_supported();
}

void llama_decode_job_set_layers(llama_context * ctx, int32_t n_layers) {
    ctx->job_set_layers(n_layers);
}

int32_t llama_decode_job_get_layers(const llama_context * ctx) {
    return ctx->job_get_layers();
}

int32_t llama_decode_job_peer_rows(const llama_context * ctx) {
    return ctx->job_peer_rows();
}

int32_t llama_decode_job_begin(llama_context * ctx, llama_batch batch) {
    return ctx->job_begin(batch);
}

int32_t llama_decode_job_step(llama_context * ctx, int32_t n_groups) {
    return ctx->job_step(n_groups);
}

bool llama_decode_job_active(const llama_context * ctx) {
    return ctx->job_active();
}

void llama_decode_job_cancel(llama_context * ctx) {
    ctx->job_cancel();
}

bool llama_decode_job_get_info(const llama_context * ctx, llama_decode_job_info * info) {
    return ctx->job_get_info(info);
}

void llama_rs_idx_seen_reset(llama_context * ctx, llama_seq_id seq_id) {
    ctx->rs_idx_seen_reset(seq_id);
}

int32_t llama_rs_idx_seen(const llama_context * ctx, llama_seq_id seq_id) {
    return ctx->rs_idx_seen(seq_id);
}

void llama_pipe_select(llama_context * ctx, int32_t slot) {
    ctx->pipe_select(slot);
}

void llama_pipe_reserve(llama_context * ctx) {
    ctx->pipe_reserve();
}

bool llama_tu_out_reserve(llama_context * ctx) {
    return ctx->tu_out_reserve();
}

void llama_tu_switches(const llama_context * ctx, int32_t * out_reserve, int32_t * inj_decode) {
    *out_reserve = ctx->tu_out_reserve_on();
    *inj_decode  = ctx->tu_inj_decode();
}

void llama_pipe_decode_flags(llama_context * ctx, bool async, bool backup) {
    ctx->pipe_decode_flags(async, backup);
}

void llama_pipe_drain(llama_context * ctx) {
    ctx->pipe_drain();
}

void llama_pipe_hold_next(llama_context * ctx, bool hold) {
    ctx->pipe_hold_next(hold);
}

bool llama_pipe_resume(llama_context * ctx) {
    return ctx->pipe_resume();
}

void llama_pipe_abort(llama_context * ctx) {
    ctx->pipe_abort();
}

bool llama_pipe_rows_independent(llama_context * ctx, int64_t n_more) {
    return ctx->pipe_rows_independent(n_more);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_embeddings_layer_inp_dev(llama_context * ctx, int32_t n_rows_max) {
    ctx->set_embeddings_layer_inp_dev(n_rows_max);
}

int32_t llama_get_embeddings_layer_inp_dev_rows(llama_context * ctx) {
    return ctx->get_embeddings_layer_inp_dev_rows();
}

void llama_set_inject_dev(llama_context * ctx, bool value) {
    ctx->set_inject_dev(value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

bool llama_mtp_chain_supported(const llama_context * ctx) {
    const auto & model = ctx->get_model();
    const auto & cp    = ctx->get_cparams();
    // as graph_mtp: the subset of the target's head, not an MTP head of its own
    const uint32_t il = model.hparams.n_layer();
    const bool own_head = il < model.layers.size() && model.layers[il].nextn.shared_head_head != nullptr;
    return cp.ctx_type == LLAMA_CONTEXT_TYPE_MTP && cp.flash_attn && model.arch == LLM_ARCH_QWEN35 && model.hparams.n_layer_nextn == 1 &&
        !own_head && model.output_s == nullptr &&
        model.head_subset_w != nullptr && model.head_subset_ids != nullptr && model.head_subset_embd != nullptr;
}

void llama_mtp_chain_set(llama_context * ctx, int32_t n_steps, int32_t k) {
    ctx->mtp_chain.n_steps = std::max(0, n_steps);
    ctx->mtp_chain.k       = k;
}

void llama_mtp_chain_set_params(llama_context * ctx, llama_seq_id seq_id, const float * prm, int32_t n) {
    ctx->mtp_chain.params[seq_id].assign(prm, prm + n);
}

const int32_t * llama_mtp_chain_get(llama_context * ctx, llama_seq_id seq_id, int32_t step, int32_t * width) {
    auto & st = ctx->mtp_chain;
    if (!st.synced) {
        ctx->synchronize();
        st.synced = true;
    }
    const size_t n_seq = st.seqs.size();
    if (width) {
        *width = st.width;
    }
    if (n_seq == 0 || st.width <= 0) {
        return nullptr;
    }
    const size_t n_per   = n_seq*(1 + (size_t) st.width);
    const size_t n_steps = st.out.size()/n_per;
    for (size_t j = 0; j < n_seq; ++j) {
        if (st.seqs[j] == seq_id && step >= 0 && (size_t) step < n_steps) {
            return st.out.data() + (size_t) step*n_per + n_seq + j*(size_t) st.width;
        }
    }
    return nullptr;
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
void llama_state_seq_wait(llama_context * ctx) {
    ctx->state_seq_wait();
}

size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
