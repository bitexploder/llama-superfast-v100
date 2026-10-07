#!/usr/bin/env bash
# Serve Qwen3.8-27B on one V100 with speculative decoding.
#   scripts/serve-27b.sh mtp    [model-dir]   MTP head (Unsloth)
#   scripts/serve-27b.sh dflash [model-dir]   DFlash2 draft (z-lab)
# model-dir defaults to ./models and must hold the files named in README-27B.md.
# Environment: LLAMA_HOST (127.0.0.1), LLAMA_PORT (8080), LLAMA_BIN (build/bin/llama-server),
# CUDA_VISIBLE_DEVICES (unset: the server uses every visible card; set it to pick one).
set -euo pipefail

MODE=${1:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${2:-./models}
BIN=${LLAMA_BIN:-$ROOT/build/bin/llama-server}
TARGET=$MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf
MTP=$MODELS/mtp-Qwen3.8-27B-Q4_0.gguf
DFLASH=$MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

# rejection sampling and the 98,304-row draft vocabulary are the engine's defaults; only the model file's path is set here
export LLAMA_SPEC_DRAFT_VOCAB_FILE=$ROOT/models/draft-vocab-qwen3.8-27b.txt

case "$MODE" in
  mtp)    DRAFT=(-md "$MTP" --spec-type draft-mtp) ;;
  dflash) export LLAMA_DFLASH2_HEAD_FILE=$MTP    # the DFlash2 head reads the MTP file's draft-vocabulary rows
          DRAFT=(-md "$DFLASH" --spec-type draft-dflash) ;;
  *) echo "usage: $0 <mtp|dflash> [model-dir]" >&2; exit 2 ;;
esac

NEED=("$BIN" "$TARGET" "$MTP" "$LLAMA_SPEC_DRAFT_VOCAB_FILE")
[ "$MODE" = dflash ] && NEED+=("$DFLASH")
for f in "${NEED[@]}"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

exec "$BIN" --host "${LLAMA_HOST:-127.0.0.1}" --port "${LLAMA_PORT:-8080}" \
  -m "$TARGET" "${DRAFT[@]}" --spec-draft-n-max 7 \
  -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 \
  -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20
