#!/usr/bin/env bash
# Train (optionally) and export one run as a verified GGUF, on the GPU server.
#
#   ./scripts/export_model.sh <model> <arm> [task] [extra bench args]
#
# This runs in the foreground. On the server prefer the tmux wrapper, which
# survives a dropped ssh connection:
#
#   ./scripts/tmux.sh export qwen words                 # paper run
#   ./scripts/tmux.sh export qwen words pizza --fresh   # retrain, then export
#
#   ./scripts/export_model.sh qwen words                       # the paper's run, Q8_0
#   ./scripts/export_model.sh qwen words pizza --fresh         # retrain first (~12 min), new tag
#   ./scripts/export_model.sh qwen words pizza --fresh --negatives 0.15 --lr 0.0001
#                                                              # also learn to answer "?" to non-orders
#   ./scripts/export_model.sh qwen words pizza --quant q8_0,q4_k_m
#   ./scripts/export_model.sh lfm2 words pizza --verify 0      # skip the llama-server check
#
# llama.cpp is cloned and built next to the repo on first use (override with
# LLAMA_CPP=/path). Output lands in models/<task>/ with a JSON manifest holding the
# sha256, the exact prompt and the verification result. models/ is gitignored, and
# PIZZA-derived files are CC-BY-NC-4.0: never commit them.
#
# Then on the Mac, copy the file the site loads:
#   scp -P <port> <host>:<remote>/models/pizza/pizza-qwen-words-...-q8_0.gguf \
#       site/public/models/pizza-words-q8_0.gguf
set -euo pipefail

MODEL="${1:?usage: export_model.sh <model> <arm> [task] [extra bench args]}"
ARM="${2:?usage: export_model.sh <model> <arm> [task] [extra bench args]}"
TASK="${3:-pizza}"
shift $(( $# < 3 ? $# : 3 ))

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOGS="runs/$MODEL/$TASK/logs"
mkdir -p "$LOGS"
LOG="$LOGS/export-$ARM.log"

echo "export  $MODEL / $TASK / $ARM  ->  models/$TASK/   (log: $LOG)"
# Same interpreter as tmux.sh: the gpu group holds torch, transformers and unsloth.
PY="${PY:-uv run --extra pizza --group gpu python}"
CUDA_VISIBLE_DEVICES="${GPU:-0}" TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -m bench export \
  --model "$MODEL" --task "$TASK" --arm "$ARM" "$@" 2>&1 | tee "$LOG"

echo
ls -lh "models/$TASK/" | grep -E "\.gguf|\.json" || true
