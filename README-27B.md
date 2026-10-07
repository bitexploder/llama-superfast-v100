# llama-superfast-v100: Qwen3.8-27B on one V100, with speculative decoding

This fork of llama.cpp serves Qwen3.8-27B (Unsloth's `UD-Q4_K_XL` quantization) on a single Tesla V100 32 GB. The repository root's `README.md` has the short version and, below its rule, upstream llama.cpp's own README. This file is the long version for the 27B: what the fork is, how to build and run it, what we measured and under which conditions, how to reproduce it, and the licences.

## What this is

The server decodes with speculative decoding: a small draft model proposes several tokens and the 27B model checks them in one pass. There are two draft configurations:

- **MTP**: Unsloth's multi-token-prediction head as the draft. This is the one we use and the one every figure below was measured with.
- **DFlash2**: z-lab's DFlash2 draft model, supported as an alternative.

Both use rejection sampling, which accepts or rejects draft tokens in a way that keeps the model's own sampling distribution, and a draft vocabulary, which has the draft score only the 98,304 most likely token ids. Both are engine defaults on this fork rather than launch settings; `LLAMA_SPEC_REJECTION=0` and `LLAMA_SPEC_DRAFT_VOCAB=-1` restore the upstream behaviour.

The same tree also serves Qwen3.8-Flash-Next (a sparse-attention mixture-of-experts sibling) across two V100s. Its launch line is under "Run options" in `README.md` and its headline figures are at the end of "Measured speed"; its draft-vocabulary ranking file is not shipped (the engine warns and runs without the subset). Otherwise this file covers the 27B.

## What this repository is

This is a fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), based on upstream commit `6ba30d0`. It was **not cloned from upstream**: it was imported from the source tree that Unsloth's `b11030-mix-5ff778e` release asset was built from (archive sha256 `44cca07c…`), so the root commit here is that import and there is no upstream history behind it. For a reader comparing the two: diff this tree against upstream at `6ba30d0` and the difference is our work, plus whatever Unsloth's build of that commit carried that upstream's did not. We did not separate the two, so treat the import commit as the baseline and the commits after it as ours. This release is one squashed commit on top of the previous one, carrying everything that landed in between.

## Hardware and software it was built and measured on

- Tesla V100 32 GB PCIe, NVIDIA driver 580, CUDA 12.8, Ubuntu 26.04, gcc/g++ 14.
- The CUDA kernels were written and tested for that card only (`sm_70`). Other GPUs are untested and are not expected to work.

## Build

From the repository root. `ninja`, `cmake`, `gcc-14` and CUDA 12.8 must be installed.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

The binary is `build/bin/llama-server`. On 12 CPU threads the build took about 9 minutes here, with ccache on. The build fetches the server's web UI bundle from Hugging Face; if that fetch fails on the pinned checksum it falls back to the latest bundle, which is what happened here.

## Models

Three files, about 20 GB in all. They go flat in `./models`, next to the draft-vocabulary file that is already in the repository. Only the first two are needed for MTP.

| File | Repository | Size |
|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `unsloth/Qwen3.8-27B-GGUF` | 17,559,178,144 bytes (16.4 GiB) |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | `unsloth/Qwen3.8-27B-GGUF`, in `MTP/` | 1,369,590,656 bytes (1.3 GiB) |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | `z-lab/Qwen3.8-27B-DFlash2-GGUF` | 1,143,006,816 bytes (1.1 GiB) |

```
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir models
mv models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf models/ && rmdir models/MTP
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models
```

`models/draft-vocab-qwen3.8-27b.txt` is the draft vocabulary: 131,072 token ids ranked by how often the 27B model and its sibling produce them on prose and code. The default is to use the first 98,304. It ships in the repository.

## Run

```
scripts/serve-27b.sh mtp
scripts/serve-27b.sh dflash
```

Run them from the repository root. An optional second argument names the model directory (default `./models`). `LLAMA_HOST` (default `127.0.0.1`), `LLAMA_PORT` (default `8080`) and `LLAMA_BIN` set the address and the binary. On a machine with several cards, pick one with `CUDA_VISIBLE_DEVICES`. The server holds about 26 to 29 GB of the card's memory.

These are the two command lines the script runs, with `$MODELS` for the model directory. The script sets `LLAMA_SPEC_DRAFT_VOCAB_FILE`; rejection sampling and the vocabulary size are engine defaults.

```
LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

```
LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt LLAMA_DFLASH2_HEAD_FILE=$MODELS/mtp-Qwen3.8-27B-Q4_0.gguf build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

The DFlash2 configuration also needs the MTP file: its output head reads that file's draft-vocabulary rows.

## Settings

These are the settings behind every figure below, and the ones we use for everyday work: coding agents, Python and prose alike.

- **Sampling**: temperature 1.0, top-p 0.95, top-k 20.
- **Thinking**: on, at the chat template's default (`--jinja`). The client must not send a reasoning-effort field, so that the template's default applies.
- **Context**: 131,072 tokens, KV cache in f16 for both the target and the draft. The server fits in 26 to 29 GB of the card.
- **Prompt processing**: micro-batches of 2,048 tokens (`-b 4096 -ub 2048`).
- **Concurrency**: one request at a time (`--parallel 1`).
- **Drafting**: a 7-token draft window (`--spec-draft-n-max 7`), with rejection sampling and the 98,304-id draft vocabulary on by default.
- **Placement**: all layers on the GPU (`-ngl 99`), flash attention on (`-fa on`), 12 host threads (`-t 12`).

Nothing else needs tuning. `scripts/serve-27b.sh mtp` applies all of the above.

The causal f16 attention mask is built on the GPU for both decoding and prompt micro-batches. Under `--split-mode tensor`, each card has its own cell mirror and builds its own mask; the target and MTP caches use the same path. `LLAMA_KQ_MASK_DEVICE=0` restores host construction and uploads. For debugging only, `LLAMA_KQ_MASK_DEVICE_CHECK=N` compares the first N device-built masks against the host implementation on every card, bitwise, and aborts on a mismatch. This check synchronizes the devices and must be unset for timing.

## Using it from a coding agent

This is the OpenAI-compatible provider block our coding agent uses. Replace `@PORT@` with the server's port. `supportsReasoningEffort` must stay false, so that the template's thinking default applies.

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:@PORT@/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B (single V100)",
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

Everything here is from one V100 32 GB PCIe, single stream, with the settings above, on this engine: the figures were taken over the last few changes, and after each later change the 27B's output was checked bitwise identical and its speed unchanged. The card ran at 1,345 to 1,380 MHz while decoding and at about 1,290 MHz during the long prompt-processing runs, where the power cap bites. Peak memory use was 29,832 MiB as reported by nvidia-smi. Each run is one sample at temperature 1.0.

### Seven hours of agentic work: DeepSWE

The longest thing we have run. Seventeen tasks from the DeepSWE 1.1 catalogue (`datacurve-ai/deep-swe`, revision `0b9fabbb`, 113 tasks; we ran the first seventeen of a fixed shuffled order and stopped, so this is not a score on the full set), each a real feature or bug in a public Go, TypeScript, Python or Rust repository, with the catalogue's own fail-to-pass and pass-to-pass tests as the verifier. The agent was the Pi coding agent in JSON mode, in the task's container, with automatic context compaction set to trigger near 98K tokens and a three-hour cap per task. Two servers ran, one per card, with the launch above and no overrides, over two lanes.

- **Both servers ran the whole seven hours without a restart**: 2,058 requests on one card and 1,769 on the other, with no error, crash or non-finite value in either log.
- **Every task ran up to the compaction trigger**, 1 to 9 times, 52 compactions in all, and every compaction was survived: the agent carried on with a summarized context and the server re-read it.
- Volume: 12.8 task-hours, 3,686 agent turns, 6.86 million prompt tokens processed, 3.00 million tokens generated.
- Speed over the seventeen tasks: **99.4 tok/s mean decode** (86.6 to 112.3), acceptance 0.521 per drafted token (0.443 to 0.616), 4.65 tokens per round (4.10 to 5.31). Task wall time 16 to 110 minutes, mean 45.
- Outcome: 7 of 17 resolved. The unresolved tasks decoded at the same speed as the resolved ones and finished well inside the cap; several were near misses by the verifier's own counts (75 of 80 fail-to-pass tests passing, 85 of 91, 45 of 48, 46 of 49; two passed every fail-to-pass test and broke a pass-to-pass test).

| Task | Outcome | Wall | Turns | Max context | Prompt tokens | Generated | Decode tok/s | Acceptance | Tok/round | Compactions | Fail-to-pass | Pass-to-pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| go-genai streamed function args | resolved | 25 min | 131 | 99.5K | 264K | 97K | 100.9 | 0.532 | 4.72 | 2 | 6/6 | 62/62 |
| goreleaser retry publish auditing | resolved | 48 min | 269 | 99.2K | 414K | 163K | 100.8 | 0.533 | 4.73 | 3 | 29/29 | 29/29 |
| happy-dom intersection observer | not resolved | 24 min | 85 | 102.0K | 132K | 91K | 100.3 | 0.516 | 4.61 | 1 | 11/14 | 9/9 |
| superjson error stack serialization | not resolved | 16 min | 63 | 97.6K | 126K | 74K | 101.4 | 0.523 | 4.66 | 1 | 75/80 | 113/116 |
| opa template string reconstruction | resolved | 110 min | 645 | 101.8K | 993K | 423K | 97.1 | 0.510 | 4.57 | 8 | 5/5 | 4/4 |
| quill shared toolbar focus | not resolved | 62 min | 190 | 103.9K | 494K | 254K | 90.7 | 0.467 | 4.27 | 4 | 9/13 | 22/22 |
| kombu single active consumer priority | resolved | 28 min | 123 | 99.0K | 254K | 117K | 103.3 | 0.548 | 4.83 | 2 | 85/85 | 1421/1421 |
| kysely window grouping helpers | not resolved | 49 min | 188 | 99.1K | 379K | 140K | 112.3 | 0.616 | 5.31 | 3 | 184/254 | 22/22 |
| updo policy alerting | not resolved | 19 min | 85 | 98.5K | 140K | 83K | 104.6 | 0.552 | 4.87 | 1 | 14/17 | 123/123 |
| tengo destructuring bindings | not resolved | 40 min | 227 | 102.6K | 384K | 163K | 93.5 | 0.487 | 4.41 | 3 | 85/91 | 132/132 |
| httpx streaming json iteration | resolved | 19 min | 83 | 98.2K | 152K | 82K | 106.3 | 0.555 | 4.89 | 1 | 108/108 | 1404/1404 |
| scriggo method declarations | not resolved | 104 min | 683 | 103.6K | 1,190K | 392K | 95.1 | 0.498 | 4.49 | 9 | 45/48 | 1049/1049 |
| sqlfmt create table DDL formatting | not resolved | 73 min | 208 | 111.3K | 653K | 283K | 86.6 | 0.443 | 4.10 | 5 | 32/32 | 1147/1273 |
| bandit structured nosec directives | not resolved | 33 min | 159 | 98.4K | 303K | 134K | 94.0 | 0.477 | 4.34 | 2 | 69/69 | 281/282 |
| ytt jsonpath query api | resolved | 26 min | 129 | 98.3K | 176K | 127K | 105.7 | 0.557 | 4.90 | 1 | 103/103 | 1/1 |
| fd deterministic multi-key sorting | resolved | 27 min | 146 | 105.2K | 268K | 117K | 101.3 | 0.542 | 4.79 | 2 | 43/43 | 109/109 |
| meriyah explicit resource declarations | not resolved | 66 min | 272 | 109.4K | 537K | 260K | 96.5 | 0.507 | 4.55 | 4 | 46/49 | 51469/51469 |

"Max context" is the largest context the task reached; "prompt tokens" is everything the server processed as prompt over the task, including the re-reads after each compaction. Decode is the server's steady decode rate over the task. The run was on a build a few commits before this one; the 27B's output on this build is bitwise identical to that build's on our identity check, so the figures stand.

### Two V100s: one DeepSWE task, run again and again

The goreleaser task from the table above became our yardstick on two cards: the 27B under tensor parallelism (`--split-mode tensor`), 131,072 tokens of context per user, the same draft and settings, rerun as the engine changed. Every run is one sample at temperature 1.0.

- **One user on two cards: 4 of 6 runs resolved. The fastest took 22.4 minutes** (177 turns), against 48 minutes on one card.
- **Two users sharing the two cards, each agent running the task at the same time: 3 of 4 resolved.** In the latest pair both agents resolved it, in 31.2 and 42.7 minutes: two finished tasks in 43 minutes of wall time.
- **Over all eleven 27B runs of this task, 8 resolved.** The misses never broke anything: all 29 pass-to-pass tests passed every time, and the misses fell short on the 29 fail-to-pass tests (25, 7 and 2 passing).

### Qwen3.8-Flash-Next on two V100s

The same engine runs Qwen3.8-Flash-Next (ISTA-DASLab's GSQ-RCO `IQ3_XXS`) with its layers split 55/45 across two cards and a 262,144-token context. The launch is under "Run options" in `README.md`.

- **Prompt processing, cold:** 1,135 tok/s at 16K, 1,438 at 64K, 1,448 at 146,560 and 1,408 at 260,000 tokens. That is about 2 to 2.6 times what this fork managed before its tensor-core expert products, larger prompt chunks and two-card overlap.
- **The same goreleaser task: 3 of 3 runs resolved. The best took 26.8 minutes** with the full 262,144-token window, so it never needed to compact (deepest context 180,857 tokens), at 112 tok/s decode.

### Coding-agent tasks

Four feature tasks on a small Python web application, each with tests that must pass, run by a coding agent that reads files, edits them and runs the tests. Each task runs twice: from an empty context, and with the repository loaded into context first, which puts about 72K tokens in the context before the task starts. "Decode" is the server's steady decode rate over the run; "acceptance" is accepted draft tokens over drafted tokens; "context" is the task's own context at the end of the run, on top of the preload where there is one. All eight runs passed their tests.

| Task | Mode | Decode tok/s | Acceptance | Context at end | Turns | Wall |
|---|---|---|---|---|---|---|
| 1 | empty | 123.1 | 0.622 | 30K | 10 | 2.0 min |
| 1 | preloaded | 108.4 | 0.617 | 72K + 18K | 49 | 4.5 min |
| 2 | empty | 127.2 | 0.639 | 23K | 12 | 1.3 min |
| 2 | preloaded | 117.3 | 0.667 | 72K + 9K | 36 | 3.2 min |
| 3 | empty | 124.4 | 0.622 | 24K | 16 | 2.0 min |
| 3 | preloaded | 113.4 | 0.654 | 72K + 18K | 69 | 4.6 min |
| 4 | empty | 131.6 | 0.680 | 32K | 17 | 1.8 min |
| 4 | preloaded | 108.5 | 0.607 | 72K + 14K | 45 | 4.2 min |

Mean decode: 126.6 tok/s from an empty context, 111.9 tok/s preloaded. Loading the 72K-token repository into context took 103 to 107 seconds of prompt processing per run; during that phase, on the agent's own incremental prompts, the server processed 600 to 645 tok/s.

### HumanEval

Problems 0 to 31, one sample each, scored on the base tests and on the extended HumanEval+ tests.

- 32 of 32 passed on both sets. No answer hit the length cap.
- 107.7 tok/s mean steady decode. Counting prompt processing and everything else, 92.1 tok/s over the whole run.
- 3.71 tokens accepted per round, pooled over all rounds. Mean answer length 1,615 tokens.

### Prompt processing

A fixed prompt, processed cold (no cached prefix), two launches, with the card at about 1,290 MHz. The spread between the two launches is 1.2% at 16K and 0.5% at 64K.

| Prompt | Launch 1 | Launch 2 |
|---|---|---|
| 16K tokens | 935 tok/s (17.5 s) | 924 tok/s (17.7 s) |
| 64K tokens | 742 tok/s (86.3 s) | 738 tok/s (86.7 s) |

For scale: NInfer's V100 port, measured on the same card with its own settings (NVFP4 weights, 1,024-token prefill chunks, int8 KV cache), processes a cold 16K to 18K prompt at about 833 tok/s. At 16K this fork is now past that; we have no cold 64K figure for NInfer.

### Day-to-day use

On the author's own coding traffic over about 21,600 rounds, the server reported 0.457 accepted per drafted token (33,584 of 73,446) and 2.56 tokens per round. That is below the coding-agent tasks above (0.61 to 0.68) and the DeepSWE run (0.52), and well below HumanEval (3.71 per round). We have not measured why. The traffic mix differs from the benchmarks' (short exchanges, prose and tool output beside code), and the adaptive draft width, which narrows the 7-token window when a full-width round does not pay, uses a cost table measured on an older version of the verification kernel.

### Against NInfer

Measured on the previous build of this fork, which the current one beats by about 2% on the same tasks. The same four tasks, once each in both modes, both engines passing every run, both aggregated the same way: total output tokens over total decode time.

| | Decode tok/s | Acceptance | Tokens per round | Prompt processing tok/s |
|---|---|---|---|---|
| This fork (MTP, 7-token window) | 115.6 | 0.599 to 0.668 | 5.19 to 5.68 | 482 to 670 |
| NInfer (its published settings, 3-token window) | 57.8 | 0.646 to 0.714 | 3.21 to 3.57 | 557 to 734 |

NInfer's acceptance is higher. Its tokens per round are lower because it drafts 3 tokens where this fork drafts 7, so those two columns are not like for like. On the agent tasks' incremental prompts the two engines' prompt processing is close (the current build of this fork processed 600 to 645 tok/s during the preload phase of the runs above, against NInfer's 557 to 734 on the earlier comparison), and on a cold fixed prompt this fork is faster at 16K (see "Prompt processing"). NInfer's published 262,144-token context does not fit a 32 GB V100: its startup reservation needs 11.7 GB beyond the weights, against 12.6 GB free, so that run used 131,072. This fork serves 131,072 on the same card in 26 to 29 GB.

## What was changed, and what each change was worth

Rounds are at 64K context unless stated. Every change was gated against the previous build: bitwise on the output text, or by KL divergence where a numeric path changed. Several other changes were measured, did not pay for themselves, and are not in the fork.

The foundation, from the first release: tensor-core products (`mma.sync` m8n8k4) for the few-token matrix-vector step on K-quant and IQ4 weights, repacked once at load into fragment order, with the small operations around them fused; speculative decoding with the MTP head or DFlash2 as the drafter, rejection sampling, the 98,304-id draft vocabulary and a pipelined draft-and-verify loop. Since then:

- The verification step reads each cached KV head once instead of twice: 13% off the round.
- The 8-row attention kernel widened to serve any width from 2 to 8 rows: 21% off the round at a 5-token draft window.
- A streaming attention kernel at four columns per warp, with its key loads moved off the QK warps: 2.6 ms off the round at 64K, 5.1 ms at 120K.
- The attention mask built on the device rather than copied every step: 1.5 ms off the round.
- A chunked gated-delta prefill on tensor cores: prompt processing up 9.0% at 16K and 7.5% at 64K.
- The gated-delta decode recurrence at four columns per warp: 0.23 to 0.30 ms off the round.
- A fused decode kernel: 0.56 to 0.62 ms off the round.
- A checkpoint fix that stopped the draft model's KV cache being copied on every request: 241 to 511 ms off each request.
- The deployment's environment settings moved out of the launch line and into the engine as defaults.

## Known limits

- **Context is 131,072 tokens on one card.** The weights (16.4 GiB) plus a 262,144-token f16 KV cache (about 24 GB) do not fit in 32 GB, and the attention kernels' fast paths read only f16 and bf16 caches, so a quantized cache would run on the slower generic path. On the DeepSWE tasks the agent reached the compaction trigger on every task, so this is the limit that binds on long agentic work. Two cards lift it: the runs above used 131,072 tokens per user there, and Flash-Next ran at 262,144.
- Draft acceptance in day-to-day use is lower than on the benchmarks (see above).
- The draft vocabulary is tuned for English and code. Acceptance on Chinese is low.
- Context shift is off, so a full context cuts a reply short.
- Every figure is one sample per run at temperature 1.0; turn counts and paths differ between runs, and per-task figures are not like for like across engines.
- Untested: the `hf download` lines above were written from the local copies' repository ids and were not re-run from an empty directory; any card other than the V100, any driver or CUDA version other than the ones above; and more than one request at a time on one card (two users on two cards is tested, above).

## Reproducing the numbers

Every figure above was measured one request at a time, with the launch lines under "Run", on the card and software listed at the top.

### HumanEval+

`scripts/humaneval/eval.py` sends HumanEval problems 0 to 31 to a running server, one at a time, and scores them with EvalPlus. It uses EvalPlus's own chat instruction and its own answer sanitizer, temperature 1.0, top-p 0.95, top-k 20, seed 36, thinking on (the chat template's default, no `reasoning_effort` sent) and a 16,384-token cap on each reply.

```
pip install evalplus==0.3.1
scripts/serve-27b.sh mtp > server.log 2>&1 &
python3 scripts/humaneval/eval.py 8080 server.log 36
```

`server.log` must be the file the server writes to, because the script reads tokens per round from it. It writes `progress.txt`, `stage-8-summary.*` (problems 0 to 7) and `stage-32-summary.*` (problems 0 to 31) next to itself, and one JSON per problem under `run-seed36/`. On Python 3.14 the script sets the `fork` start method, because the default breaks EvalPlus's checker. The figures under "HumanEval" above are this script's output on the current build with MTP; one seed, so the pass rates carry a wide margin. DFlash2 runs the same way with `scripts/serve-27b.sh dflash`; on an earlier build it scored the same and decoded faster (124.4 tok/s mean steady), but it has not been re-measured on this one.

### The coding-agent tasks

This is a method we describe but do not ship as something you can run: the tasks are written against the author's own small web application, and the runner is tied to the author's machine layout (a sandbox, and one model server per card). If you want to repeat it on your own code base, this is what was done:

- **Agent:** the Pi coding agent (`--mode json`), pointed at the server through the provider block under "Using it from a coding agent", with `supportsReasoningEffort` false, in a sandbox that could write only its own copy of the repository.
- **Task:** a written spec ("add star ratings to the web backend", and three more of the same size) plus a test suite copied into the run's `tests/` directory. The agent was told to make `pytest tests -q -x --tb=short` pass, not to edit tests, and to stop with a one-line reply once it did. The score is the pass count of the original test files, run outside the agent's sandbox; a run also records whether the agent modified tests (none did).
- **Empty mode:** the agent starts with only the task. **Preload mode:** it first reads the repository in full (about 72K tokens of prompt before the task), and the task's own context is counted from the end of that reading.
- **Sampling and server:** as under "Settings"; MTP.

### DeepSWE

Also described rather than shipped. The catalogue is public (`datacurve-ai/deep-swe`, revision `0b9fabbb`): each task has a container image, an instruction and the verifier's tests. What was done: a fixed shuffled order of the tasks (seed 36), two lanes claiming tasks from a shared ledger, one server per card launched as under "Run", the Pi coding agent in JSON mode inside the task's container with automatic compaction near 98K tokens of context and a three-hour cap per task, and the catalogue's own verifier run on the result. Per task we recorded the outcome, wall time, turns, prompt tokens, generated tokens, acceptance, tokens per round, compactions and the verifier's counts; that is the table above.

## Tests

The six test programs we added live in `tests/` and are built with the rest of the tests. Configure with `-DLLAMA_BUILD_TESTS=ON` added to the build command above, then:

```
cmake --build build --target test-mmvq-tc test-qsa-compact test-qsa-host-meta test-qsa-select test-spec-draft-grammar test-spec-rejection -j 12
ctest --test-dir build -R 'test-(mmvq-tc|qsa-compact|qsa-host-meta|qsa-select|spec-draft-grammar|spec-rejection)' --output-on-failure
```

Each test also runs by itself as `build/bin/<name>`.

| Test | Needs a GPU | What it checks |
|---|---|---|
| `test-mmvq-tc` | yes, first CUDA device | The tensor-core path for few-token K-quant products: accuracy against an fp64 reference at 3 to 8 tokens; activations up to 1e6 stay finite and equally accurate; a token holding inf or NaN behaves as on the dp4a path (compared with a child process run with `LLAMA_MMVQ_TC=0`) without disturbing other tokens; and a tensor-core product sharing an activation quantization with dp4a products neither leads nor joins their group. |
| `test-qsa-compact` | yes | Differential test of the compact selected-KV attention for the `qwen4exp` architecture against dense flash attention with the same mask, on the same bf16 cache. Set `QSA_TEST_PER_TOKEN` for per-token output. |
| `test-qsa-host-meta` | no | The incremental block table used for block selection against a full scan, on random cache histories (appends, rollbacks, prefix reuse, copies, shifts, restores, clears), byte for byte. Run `build/bin/test-qsa-host-meta bench` to time both at 20K, 64K and 128K instead. |
| `test-qsa-select` | runs the CPU backend and every GPU backend | The block-selection op against a brute-force reference, bit for bit, including ties at the k-th place, all-zero scores and rows with fewer selectable blocks than k, and repeatability on the GPU. |
| `test-spec-draft-grammar` | no | A drafted token that does not fit the target's tool-call grammar must not throw when it is accepted into the sampler's copy (`common_sampler_accept_draft`); the control shows the plain accept does throw. It loads only a vocabulary file, `models/ggml-vocab-qwen35.gguf`, which `ctest` passes; run by hand as `build/bin/test-spec-draft-grammar models/ggml-vocab-qwen35.gguf`. |
| `test-spec-rejection` | no | The rejection step of speculative sampling keeps the target's distribution: 10^6 draws per synthetic pair of distributions, a chi-square p-value above 0.01, no token outside the target's support, and the acceptance rate printed beside its expected value. Also covers the adaptive draft length. |

This tree is the one we build and run; the published copy differs from it only in comments and documentation, and was not rebuilt separately for publication.

## Licences and credits

- **This repository:** upstream llama.cpp's MIT licence (`LICENSE`, "Copyright (c) 2023-2026 The ggml authors") is kept as it was, and our changes are released under the same licence. `licenses/` holds upstream's third-party notices (the vendored nlohmann JSON library's, for one).
- **Qwen3.8-27B, Unsloth's GGUFs:** the quantized model `Qwen3.8-27B-UD-Q4_K_XL.gguf` and the MTP head `mtp-Qwen3.8-27B-Q4_0.gguf` are Unsloth's conversions (`unsloth/Qwen3.8-27B-GGUF`) of Qwen's `Qwen/Qwen3.8-27B`. The GGUF metadata of both files says `general.license = apache-2.0` and `general.quantized_by = Unsloth`. No separate licence file for them was on the machine this was prepared on, so the metadata is the only evidence we have; check the model card before you rely on it.
- **DFlash2 draft:** `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` is z-lab's GGUF conversion (`z-lab/Qwen3.8-27B-DFlash2-GGUF`, a mirror of `incoai/Qwen3.8-27B-DFlash2-GGUF`) of Inco AI's DFlash 2 draft model for Qwen3.8-27B. The repository's README front matter and the file's GGUF metadata both say `apache-2.0`. The project link in that README is https://github.com/z-lab/dflash.
- **Draft vocabulary (`models/draft-vocab-qwen3.8-27b.txt`):** a ranking of token ids, no text. It was built by counting tokens in the 27B model's own replies (weighted four times per token), in replies from Qwen3.8-Flash-Next (a sibling model with the same tokenizer), in prose (a wiki-text evaluation corpus we did not record the licence of, this repository's documentation and our own notes) and in code (this repository's C++ and CUDA and the Python standard library). Only the counts were used, and no text is shipped. At 98,304 ids it covered at least 99.5% of tokens on the held-out job types we tried (poetry, CSV and other prompts). It does not use NInfer's data, so it needs no NInfer credit. A second, merged list that did use NInfer's token counts (Apache-2.0, https://github.com/neroued/ninfer-v100) was tried and is not shipped.
- **NInfer:** the design of our tensor-core products for K-quants follows the QPN kernels in NInfer (`ninfer-v100`, Apache-2.0), which in turn credits the "v100-skinny" project. The header comment of `ggml/src/ggml-cuda/mmvq-qpn.cu` says so, and states that no NInfer code was copied.
- **Block-sparse prompt attention for Flash-Next (`ggml/src/ggml-cuda/qsa-attn.cu`):** a port of the kernel in PentaCoxian's V100 patch set for llama.cpp, published alongside the Qwen3.8-Flash-Next GGUFs at https://huggingface.co/pentacoxian-dev. The patch modifies llama.cpp and we found no separate licence statement for it, so we treat it as MIT like the tree it patches; the port is credited in the file's header. It is used by the Flash-Next path only.
