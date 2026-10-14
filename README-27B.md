# Qwen3.8-27B on the Tesla V100

This file covers how to build and run Qwen3.8-27B (Unsloth's `UD-Q4_K_XL`) on this fork, on one or two Tesla V100 32 GB cards. It also gives the figures behind the headline numbers in [`README.md`](README.md). The server decodes speculatively: Unsloth's multi-token-prediction (MTP) head drafts up to 7 tokens and the 27B checks them in one pass.

## What this repository is

A fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) at upstream commit `6ba30d0`, imported from the source tree of Unsloth's `b11030-mix-5ff778e` build rather than cloned, so there is no upstream history behind the root commit. Diff against upstream `6ba30d0` to see our work (plus whatever Unsloth's build carried). The published tree is the one we build and run, less comments and internal documentation.

## Hardware and software

- One or two Tesla V100 32 GB PCIe. No NVLink is needed. The CUDA kernels target `sm_70` only; other GPUs are untested.
- NVIDIA driver 580, CUDA 12.8, gcc/g++ 14, Ubuntu 26.04.
- Two cards in tensor parallel need the IOMMU in passthrough mode (`iommu=pt`); see `README.md`.

## Build

`ninja`, `cmake`, `gcc-14` and CUDA 12.8 must be installed.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

The binary is `build/bin/llama-server`; the build takes about 9 minutes on 12 threads with ccache. It fetches the web UI bundle from Hugging Face and falls back to the latest bundle if the pinned checksum fails.

## Models

Put them flat in `./models`, beside the draft-vocabulary file that ships there. MTP needs the first two; the third is only for the optional DFlash2 draft.

| File | Repository | Size |
|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `unsloth/Qwen3.8-27B-GGUF` | 16.4 GiB |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | `unsloth/Qwen3.8-27B-GGUF`, in `MTP/` | 1.3 GiB |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | `z-lab/Qwen3.8-27B-DFlash2-GGUF` | 1.1 GiB |

```
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir models
mv models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf models/ && rmdir models/MTP
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models
```

## Run

**One card, one user** (131,072 tokens, about 29.5 GB of the card):

```
scripts/serve-27b.sh mtp
```

The script takes an optional model directory (default `./models`) and reads `LLAMA_HOST` (default `127.0.0.1`), `LLAMA_PORT` (default `8080`) and `LLAMA_BIN`. Pick a card with `CUDA_VISIBLE_DEVICES`. It runs this line:

```
LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

**Two cards, two users** (tensor parallel, 131,072 tokens each):

```
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_P2P=1 LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 262144 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --parallel 2 --split-mode tensor --no-cache-idle-slots --temp 1.0 --top-p 0.95 --top-k 20
```

**Two cards, one user:** the same line with `--parallel 1 -c 131072`. A single user also fits a longer window: one user at 256K peaked at about 22.8 GiB a card on an earlier build.

**DFlash2 instead of MTP:** `scripts/serve-27b.sh dflash`. It needs the MTP file too, because its output head reads that file's draft-vocabulary rows. All the figures below use MTP.

### Settings

- **Sampling:** temperature 1.0, top-p 0.95, top-k 20.
- **Thinking:** on, at the chat template's default. Clients must not send a reasoning-effort field.
- **Drafting:** a 7-token window. Rejection sampling and the 98,304-id draft vocabulary are engine defaults; `LLAMA_SPEC_REJECTION=0` and `LLAMA_SPEC_DRAFT_VOCAB=-1` restore upstream behaviour.
- **Two users:** two-user scheduling is on by default. A user's long prompt is read in slices that yield to the other user's decoding. `LLAMA_TU=0` turns it off.
- **Cache:** f16 KV for the target and the draft. The fast attention paths read only f16 and bf16 caches.

### Using it from a coding agent

This is the OpenAI-compatible provider block we use. Keep `supportsReasoningEffort` false. With two users, set each agent's compaction to about 85% of its 131,072-token window.

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B (V100)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
            "supportsStore": false,
            "supportsUsageInStreaming": true,
            "maxTokensField": "max_tokens"
          }
        }
      ]
    }
  }
}
```

## Measured speed

All figures use the launch lines above at temperature 1.0, one sample per run.
- **Agentic figures** come from DeepSWE 1.1 tasks (`datacurve-ai/deep-swe`) run by the Pi coding agent, and from replaying 62 of those agents' real requests (1K to 98K tokens of context).
- **Builds changed between measurements.** Each later build was checked for drift from the original model (KL divergence at noise level), and was not slower where re-measured.

### One card, one user

- **Prefill, cold prompt:** 951 tok/s at 16K and 770 tok/s at 64K.
- **Agentic coding:**
  - **Seven hours of DeepSWE, 17 tasks:** 99.4 tok/s mean decode (86.6 to 112.3 by task), 4.65 tokens a round. It survived 52 context compactions with no restart and resolved 7 of the 17 tasks.
  - **The 62 replayed requests:** 113 tok/s decode, 44.8 ms a round.
- **Short coding tasks:**
  - **Four feature tasks on a small Python web app:** all passed. Decode averaged 126.6 tok/s from an empty context and 111.9 tok/s with a 72K-token repository preloaded.
  - **HumanEval and HumanEval+, problems 0 to 31:** 32 of 32 passed, at 107.7 tok/s steady decode.
- **Decode by depth** (ms a round on the replayed requests): 39.2 ms near 0, 40.2 at 16K, 43.2 at 32K, 46.0 at 64K and 48.3 at 80K.

### Two cards, one user

- **Prefill, cold prompt, average speed reading up to each depth:**
  - 1,383 tok/s up to 65,536;
  - 969 tok/s up to 131,072;
  - 686 tok/s up to 245,760.
- **Prefill on agents' real requests:** 1.73× one card.
- **Agentic coding:**
  - **A live DeepSWE task:** 155.9 tok/s decode, 4.95 tokens a round.
  - **The 62 replayed requests:** 152 tok/s, 32.9 ms a round, 1.35× one card.
- **Decode by depth** (ms a round on a fixed benchmark): 25.9 ms at 512 tokens, 29.7 at 65,536, 33.4 at 131,072 and 40.2 at 245,760.
- **One DeepSWE task, repeated:** 4 of 6 runs resolved it, the fastest in 22.4 minutes, against 48 minutes on one card.

### Two cards, two users

These figures come from two DeepSWE agents running at once. Each rate is measured only over the stretches when its condition held.

- **Decode:**
  - both users decoding: 199 tok/s combined, about 100 each;
  - one user decoding while the other is idle: 150–160 tok/s;
  - one user decoding while the other's prompt is being read: 42–47 tok/s. The two-user scheduler is what keeps this user moving; without it, decode beside a prompt read fell to under 7 tok/s.
- **Prefill:** 1,020–1,080 tok/s of each user's own GPU time, or 800–830 tok/s on the wall clock.
- **Rounds by depth:** 47–50 ms with both users decoding from 32K to 96K, and 53 ms at 96–112K. Alone, a user's round was 32–34 ms at the same depths.
- **Throughput:**
  - Both agents resolved the task, one in 31.2 minutes and the other in 42.7.
  - Over the whole run, the pair did about 1.2× the work of one user on the same cards. That is an estimate from one run.

## Known limits

- **Context on one card is 131,072 tokens.** The weights plus a 262,144-token f16 cache do not fit in 32 GB, and a quantized cache would fall off the fast attention paths. Long agentic sessions hit this limit, so set agents to compact.
- **Two users share the cards rather than doubling them.** The pair does about 1.2× one user's work, and a user decoding while the other reads a long prompt slows to about 45 tok/s.
- **Day-to-day acceptance is lower than on benchmarks:** 0.46 accepted per drafted token and 2.56 tokens a round on mixed traffic. That compares with 0.52 on DeepSWE and 0.61–0.68 on the coding tasks.
- **The draft vocabulary is tuned for English and code.** Acceptance on Chinese is low.
- **Context shift is off,** so a full context cuts a reply short.

## Tests

Our six test programs are in `tests/`. Add `-DLLAMA_BUILD_TESTS=ON` to the configure line, then run:

```
cmake --build build --target test-mmvq-tc test-qsa-compact test-qsa-host-meta test-qsa-select test-spec-draft-grammar test-spec-rejection -j 12
ctest --test-dir build -R 'test-(mmvq-tc|qsa-compact|qsa-host-meta|qsa-select|spec-draft-grammar|spec-rejection)' --output-on-failure
```

What each test checks:
- **`test-mmvq-tc` (GPU):** the tensor-core path for few-token K-quant products. It checks accuracy against fp64 and that inf and NaN behave as on the dp4a path.
- **`test-qsa-compact` (GPU):** Flash-Next's compact selected-KV attention against dense flash attention.
- **`test-qsa-host-meta` (CPU):** the incremental block table against a full scan, on random cache histories.
- **`test-qsa-select` (CPU and GPU):** the block-selection operation against a brute-force reference, bit for bit.
- **`test-spec-draft-grammar` (CPU):** a drafted token outside the tool-call grammar does not throw when accepted.
- **`test-spec-rejection` (CPU):** rejection sampling keeps the target's distribution, checked with a chi-square test over 10^6 draws.

## Licences and credits

- **This repository:** upstream llama.cpp's MIT licence (`LICENSE`) is kept, and our changes are under the same licence. `licenses/` holds upstream's third-party notices.
- **Qwen3.8-27B GGUFs:** Unsloth's conversions (`unsloth/Qwen3.8-27B-GGUF`) of `Qwen/Qwen3.8-27B`. The GGUF metadata says `apache-2.0`; check the model card before you rely on that.
- **DFlash2 draft:** z-lab's GGUF (`z-lab/Qwen3.8-27B-DFlash2-GGUF`) of Inco AI's DFlash 2 model, `apache-2.0` per its README and metadata. The project is https://github.com/z-lab/dflash.
- **Draft vocabulary (`models/draft-vocab-qwen3.8-27b.txt`):** a ranking of token ids with no text. It was built from token counts in the 27B's and Flash-Next's replies, prose, and code (this repository and the Python standard library).
- **NInfer:** our tensor-core K-quant products follow the design of the QPN kernels in [NInfer](https://github.com/neroued/ninfer-v100) (Apache-2.0), which credits "v100-skinny". No NInfer code was copied; `ggml/src/ggml-cuda/mmvq-qpn.cu` says so.
- **Flash-Next's block-sparse prompt attention (`ggml/src/ggml-cuda/qsa-attn.cu`):** ported from PentaCoxian's V100 patch set (https://huggingface.co/pentacoxian-dev). We found no separate licence for it and treat it as MIT, like the tree it patches. It is credited in the file's header.
