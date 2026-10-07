// Harness for the two-batch overlap of the tensor split's prefill (qwen35 under --split-mode tensor), no real weights:
//
//   test-tp-tbo gen <out.gguf> [n_layer]
//       a random qwen35 with the 27B's per-layer shapes (n_embd 5120, 24 / 4 heads of 256, FFN 17408, GDN 6144 inner,
//       128 state, 16 groups, 48 value heads, conv 4, an attention layer every 4th), a small vocabulary; F32 weights,
//       to be quantized with llama-quantize so the products take the real path (QPN repack, fp16 through cuBLAS)
//   test-tp-tbo run <model.gguf> <out.bin> <tp|one> <n_prompt> [n_rs_seq] [n_ubatch] [kv type: f16|bf16] [n_decode]
//       one prompt of n_prompt random tokens with every row's logits, then n_decode single-token steps; writes every
//       logit (float32) to out.bin. tp: both CUDA devices under LLAMA_SPLIT_MODE_TENSOR; one: CUDA0 alone.
//
//   test-tp-tbo state <model.gguf> <tp|one> <n_prompt>
// the target state taken synchronously (A) and as an asynchronous checkpoint (B, LLAMA_STATE_SEQ_FLAGS_ASYNC) after
//       the prompt; a decode is queued before waiting for B. Checks that A and B hold the same bytes, and that restoring B as the
//       server does gives the same logits for the same token again.
//
// Compare two runs' files bitwise (cmp), e.g. LLAMA_TP_TBO=0 against the default, or GGML_META_AR_DEFER=0 against 1.

#include "common.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static uint64_t splitmix64(uint64_t & x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// uniform in [-1, 1)
static float urand(uint64_t & x) {
    return (float) ((splitmix64(x) >> 40) * (1.0 / 8388608.0) - 1.0);
}

static void set_tensor_data(struct ggml_tensor * tensor, void * /*userdata*/) {
    const std::string name = tensor->name;
    uint64_t seed = std::hash<std::string>{}(name);
    const int64_t ne = ggml_nelements(tensor);
    std::vector<float> v(ne);
    const bool is_norm = name.find("norm") != std::string::npos;
    if (is_norm) {
        for (int64_t i = 0; i < ne; i++) {
            v[i] = 1.0f + 0.1f*urand(seed);
        }
    } else if (name.find("ssm_a") != std::string::npos) {
        // a decay: negative, so the state does not grow over thousands of tokens
        for (int64_t i = 0; i < ne; i++) {
            v[i] = -(0.05f + 0.45f*(urand(seed) + 1.0f));
        }
    } else if (name.find("ssm_dt") != std::string::npos) {
        for (int64_t i = 0; i < ne; i++) {
            v[i] = urand(seed) - 1.0f;
        }
    } else if (name.find("conv1d") != std::string::npos) {
        for (int64_t i = 0; i < ne; i++) {
            v[i] = 0.5f*urand(seed);
        }
    } else {
        // uniform with variance 1 / ne[0]: a product's outputs stay near unit scale
        const float a = std::sqrt(3.0f / (float) tensor->ne[0]);
        for (int64_t i = 0; i < ne; i++) {
            v[i] = a*urand(seed);
        }
    }
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, v.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(ne);
        for (int64_t i = 0; i < ne; i++) {
            h[i] = ggml_fp32_to_fp16(v[i]);
        }
        ggml_backend_tensor_set(tensor, h.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> h(ne);
        for (int64_t i = 0; i < ne; i++) {
            h[i] = ggml_fp32_to_bf16(v[i]);
        }
        ggml_backend_tensor_set(tensor, h.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected type %s for %s", ggml_type_name(tensor->type), tensor->name);
    }
}

static int gen(const char * path, const uint32_t n_layer) {
    const uint32_t n_vocab   = 4096;
    const uint32_t n_embd    = 5120;
    const uint32_t n_head    = 24;
    const uint32_t n_head_kv = 4;
    const uint32_t n_head_d  = 256;
    const uint32_t n_ff      = 17408;

    gguf_context_ptr gctx(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN35, gctx.get());
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,          llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                    n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,                uint32_t(262144));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,              n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                   n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,           n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,          n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,       n_head_kv);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,          n_head_d);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,        n_head_d);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,          uint32_t(64));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,       std::vector<uint32_t>({11, 11, 10, 0}));
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,                10000000.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,   1e-6f);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,       uint32_t(4));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,               uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,                uint32_t(6144));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,                uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,            uint32_t(48));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,               uint32_t(16));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,               "no_vocab");

    llama_model_params mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs = { nullptr }; // CPU
    mp.devices = devs.data();
    llama_model_ptr model(llama_model_init_from_user(gctx.get(), set_tensor_data, nullptr, mp));
    if (!model) {
        fprintf(stderr, "failed to create the model\n");
        return 1;
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        fprintf(stderr, "can not open %s\n", path);
        return 1;
    }
    // init_from_user creates optional attention biases too; the real 27B has none.
    // Omit them from the fixture rather than silently disabling TBO eligibility.
    for (auto & layer : model->layers) {
        layer.wq_b = layer.wk_b = layer.wv_b = layer.wqkv_b = layer.wo_b = nullptr;
    }
    llama_model_saver sv(model.get());
    sv.add_kv_from_model();
    sv.add_tensors_from_model();
    sv.save(f);
    fclose(f);
    printf("wrote %s\n", path);
    return 0;
}

static int run(int argc, char ** argv) {
    const char * path     = argv[2];
    const char * out      = argv[3];
    const bool   tp       = strcmp(argv[4], "tp") == 0;
    const int    n_prompt = atoi(argv[5]);
    const int    n_rs_seq = argc > 6 ? atoi(argv[6]) : 7;
    const int    n_ubatch = argc > 7 ? atoi(argv[7]) : 2048;
    const char * kv       = argc > 8 ? argv[8] : "f16";
    const int    n_decode = argc > 9 ? atoi(argv[9]) : 4;

    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(dev);
        }
    }
    if (devs.empty() || (tp && devs.size() < 2)) {
        fprintf(stderr, "need %d CUDA devices\n", tp ? 2 : 1);
        return 1;
    }
    devs.resize(tp ? 2 : 1);
    devs.push_back(nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.devices      = devs.data();
    mp.split_mode   = tp ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
    mp.n_gpu_layers = 999;
    llama_model_ptr model(llama_model_load_from_file(path, mp));
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) (n_prompt + n_decode + 256);
    cp.n_batch         = (uint32_t) std::max(n_ubatch, n_prompt);
    cp.n_ubatch        = (uint32_t) n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq        = (uint32_t) n_rs_seq;
    const ggml_type kvt = strcmp(kv, "bf16") == 0 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
    cp.type_k = kvt;
    cp.type_v = kvt;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    uint64_t seed = 1234;
    std::vector<llama_token> tokens(n_prompt);
    for (auto & t : tokens) {
        t = (llama_token) (splitmix64(seed) % (uint64_t) n_vocab);
    }

    std::vector<float> logits;
    logits.reserve((size_t) (n_prompt + n_decode) * n_vocab);

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, tokens[i], i, {0}, true);
    }
    // the prompt n_rep times from an empty cache (LLAMA_TBO_REP, default 2): the first run of each graph shape does one-time work,
    // so the last run is the timed one and the one whose logits are kept
    const int n_rep = getenv("LLAMA_TBO_REP") ? std::max(1, atoi(getenv("LLAMA_TBO_REP"))) : 2;
    double ms = 0.0;
    for (int r = 0; r < n_rep; r++) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        const auto t0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx.get(), batch)) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        llama_synchronize(ctx.get());
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("rep %d: prompt %.1f ms (%.1f tok/s)\n", r, ms, 1000.0*n_prompt/ms);
    }
    for (int i = 0; i < n_prompt; i++) {
        const float * l = llama_get_logits_ith(ctx.get(), i);
        logits.insert(logits.end(), l, l + n_vocab);
    }
    llama_batch_free(batch);

    // a few single-token steps on the state the prefill left
    llama_token next = 0;
    {
        const float * l = llama_get_logits_ith(ctx.get(), n_prompt - 1);
        for (int j = 1; j < n_vocab; j++) {
            if (l[j] > l[next]) {
                next = j;
            }
        }
    }
    for (int s = 0; s < n_decode; s++) {
        llama_batch b1 = llama_batch_init(1, 0, 1);
        common_batch_add(b1, next, n_prompt + s, {0}, true);
        if (llama_decode(ctx.get(), b1)) {
            fprintf(stderr, "decode step failed\n");
            return 1;
        }
        const float * l = llama_get_logits_ith(ctx.get(), 0);
        logits.insert(logits.end(), l, l + n_vocab);
        next = 0;
        for (int j = 1; j < n_vocab; j++) {
            if (l[j] > l[next]) {
                next = j;
            }
        }
        llama_batch_free(b1);
    }

    size_t n_bad = 0;
    double sum = 0.0;
    for (float x : logits) {
        n_bad += !std::isfinite(x);
        sum   += std::isfinite(x) ? std::fabs(x) : 0.0;
    }
    FILE * f = fopen(out, "wb");
    if (f == nullptr) {
        fprintf(stderr, "can not open %s\n", out);
        return 1;
    }
    fwrite(logits.data(), sizeof(float), logits.size(), f);
    fclose(f);
    printf("run %s n_prompt=%d n_ubatch=%d n_rs_seq=%d kv=%s: prompt %.1f ms (%.1f tok/s), %zu logits, %zu non-finite, mean |logit| %.4f -> %s\n",
        tp ? "tp" : "one", n_prompt, n_ubatch, n_rs_seq, kv, ms, 1000.0*n_prompt/ms, logits.size(), n_bad, sum/logits.size(), out);
    return 0;
}

// see the header
static int state(int argc, char ** argv) {
    (void) argc;
    const char * path     = argv[2];
    const bool   tp       = strcmp(argv[3], "tp") == 0;
    const int    n_prompt = atoi(argv[4]);
    const int    n_rs_seq = 7;
    const int    n_ubatch = 2048;
    const int    n_decode = 4;

    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(dev);
        }
    }
    if (devs.empty() || (tp && devs.size() < 2)) {
        fprintf(stderr, "need %d CUDA devices\n", tp ? 2 : 1);
        return 1;
    }
    devs.resize(tp ? 2 : 1);
    devs.push_back(nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.devices      = devs.data();
    mp.split_mode   = tp ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
    mp.n_gpu_layers = 999;
    llama_model_ptr model(llama_model_load_from_file(path, mp));
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) (n_prompt + n_decode + 256);
    cp.n_batch         = (uint32_t) std::max(n_ubatch, n_prompt);
    cp.n_ubatch        = (uint32_t) n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq        = (uint32_t) n_rs_seq;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    uint64_t seed = 1234;
    std::vector<llama_token> tokens(n_prompt);
    for (auto & t : tokens) {
        t = (llama_token) (splitmix64(seed) % (uint64_t) n_vocab);
    }
    auto argmax = [n_vocab](const float * l) {
        llama_token best = 0;
        for (int j = 1; j < n_vocab; j++) {
            if (l[j] > l[best]) {
                best = j;
            }
        }
        return best;
    };
    // one token at pos; its logits into out
    auto step = [&](llama_token t, llama_pos pos, std::vector<float> & out) {
        llama_batch b1 = llama_batch_init(1, 0, 1);
        common_batch_add(b1, t, pos, {0}, true);
        const int ret = llama_decode(ctx.get(), b1);
        llama_batch_free(b1);
        if (ret) {
            return false;
        }
        const float * l = llama_get_logits_ith(ctx.get(), 0);
        out.assign(l, l + n_vocab);
        return true;
    };

    // 1. the prompt
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            common_batch_add(batch, tokens[i], i, {0}, true);
        }
        if (llama_decode(ctx.get(), batch)) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        llama_batch_free(batch);
    }
    const llama_token t1 = argmax(llama_get_logits_ith(ctx.get(), n_prompt - 1));

    // 2. the target state, synchronously (A) and as an asynchronous checkpoint (B)
    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> A(llama_state_seq_get_size_ext(ctx.get(), 0, flags));
    if (llama_state_seq_get_data_ext(ctx.get(), A.data(), A.size(), 0, flags) != A.size()) {
        fprintf(stderr, "state size mismatch\n");
        return 1;
    }
    common_ckpt_pool_reserve(A.size(), 2); // B in pinned memory, as the server's checkpoints
    common_prompt_checkpoint B;
    B.update_tgt(ctx.get(), 0, flags | LLAMA_STATE_SEQ_FLAGS_ASYNC);

    // 3. t1 at n_prompt, queued with no wait for B
    std::vector<float> L1;
    if (!step(t1, n_prompt, L1)) {
        fprintf(stderr, "decode step failed\n");
        return 1;
    }

    // 4. check one: the queued copy read the state from before the decode queued after it
    llama_state_seq_wait(ctx.get());
    const bool bytes_equal = A.size() == B.data_tgt.size() && memcmp(A.data(), B.data_tgt.data(), A.size()) == 0;

    // 5. t2 at n_prompt + 1
    std::vector<float> L2;
    if (!step(argmax(L1.data()), n_prompt + 1, L2)) {
        fprintf(stderr, "decode step failed\n");
        return 1;
    }

    // 6. restore B as the server does at its "restored context checkpoint" site, then truncate the cache past it as the server does
    B.load_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    B.load_dft(nullptr, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    common_speculative_set_state(nullptr, 0, B.data_spec);
    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, n_prompt, -1)) {
        fprintf(stderr, "seq_rm failed\n");
        return 1;
    }

    // 7. check two: t1 at n_prompt again
    std::vector<float> L1b;
    if (!step(t1, n_prompt, L1b)) {
        fprintf(stderr, "decode step failed\n");
        return 1;
    }
    const bool restore_equal = L1.size() == L1b.size() && memcmp(L1.data(), L1b.data(), L1.size()*sizeof(float)) == 0;

    printf("state %s: bytes equal %d, restore equal %d (%zu bytes)\n", tp ? "tp" : "one", bytes_equal ? 1 : 0, restore_equal ? 1 : 0, B.data_tgt.size());
    return bytes_equal && restore_equal ? 0 : 1;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    if (argc >= 3 && strcmp(argv[1], "gen") == 0) {
        return gen(argv[2], argc > 3 ? (uint32_t) atoi(argv[3]) : 8);
    }
    if (argc >= 6 && strcmp(argv[1], "run") == 0) {
        return run(argc, argv);
    }
    if (argc >= 5 && strcmp(argv[1], "state") == 0) {
        return state(argc, argv);
    }
    fprintf(stderr, "usage: %s gen <out.gguf> [n_layer] | run <model.gguf> <out.bin> <tp|one> <n_prompt> [n_rs_seq] [n_ubatch] [f16|bf16] [n_decode]"
        " | state <model.gguf> <tp|one> <n_prompt>\n", argv[0]);
    return 1;
}
