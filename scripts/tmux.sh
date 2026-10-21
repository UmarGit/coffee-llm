#!/usr/bin/env bash
# Run the benchmark on the GPUs in detached tmux sessions.
#
# The whole pipeline for one model and task, which is the usual way in:
#
#   ./scripts/tmux.sh all qwen pizza 3     every stage, in order, one session
#   ./scripts/tmux.sh all lfm2 pizza 3
#   ./scripts/tmux.sh all qwen coffee 3
#
# Or one stage at a time, with MODEL and TASK from the environment:
#
#   ./scripts/tmux.sh sweep 3      learning rate sweep, one shard per GPU
#   ./scripts/tmux.sh select       choose learning rates once every shard is done
#   ./scripts/tmux.sh grid 3       the main grid
#   ./scripts/tmux.sh latency 3    batch size 1 latency
#   MODEL=lfm2 ./scripts/tmux.sh grid 3
#
#   ./scripts/tmux.sh tail         follow every shard log
#   ./scripts/tmux.sh status       which sessions are alive and how far along
#   ./scripts/tmux.sh stop         kill every session this script started
#
# Shard count defaults to the number of GPUs present and is refused if it exceeds
# them, because an out of range CUDA_VISIBLE_DEVICES silently falls back to CPU.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOGS="$REPO/runs/logs"
PREFIX="bench"
PY="${PY:-uv run --extra pizza --group gpu python}"
MODEL="${MODEL:-qwen}"
TASK="${TASK:-pizza}"
GEN_BATCH="${GEN_BATCH:-8}"
LIMIT="${LIMIT:-}"

gpu_count() { nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 0; }

sessions() { tmux ls 2>/dev/null | grep -oE "^${PREFIX}-[^:]*" || true; }

require_gpus() {
  local want="$1" have
  have=$(gpu_count)
  if [ "$have" -eq 0 ]; then
    echo "no GPU visible (nvidia-smi -L); refusing to start"; exit 1
  fi
  if [ "$want" -gt "$have" ]; then
    echo "asked for $want shards but this machine has $have GPU(s)."
    echo "Shards beyond GPU $((have - 1)) would fall back to CPU. Use: $0 $STAGE $have"
    exit 1
  fi
}

launch() {
  local stage="$1" shards="$2" i name busy
  require_gpus "$shards"
  mkdir -p "$LOGS"
  for ((i = 0; i < shards; i++)); do
    name="${PREFIX}-${stage}-${i}"
    if tmux has-session -t "$name" 2>/dev/null; then
      echo "$name already running; stop it first"; exit 1
    fi
    busy=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ "${busy:-0}" -gt 2000 ]; then
      echo "warning: GPU $i already has ${busy} MiB in use by another process"
    fi
    tmux new-session -d -s "$name" \
      "cd '$REPO' && CUDA_VISIBLE_DEVICES=$i TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       $PY -m bench $stage --model $MODEL --task $TASK --shard $i/$shards \
       --gen-batch-size $GEN_BATCH \
       ${LIMIT:+--limit $LIMIT} 2>&1 | tee '$LOGS/${stage}-${i}.log'; \
       echo; echo '--- finished, press enter ---'; read"
    echo "started $name on GPU $i"
  done
  echo
  echo "follow with: $0 tail"
}

STAGE="${1:-}"
case "$STAGE" in
  all)
    # One session, every stage, in protocol order. Sharding and the joins between
    # stages are handled inside run_all.sh.
    MODEL="${2:-qwen}"
    TASK="${3:-pizza}"
    SHARDS="${4:-$(gpu_count)}"
    require_gpus "$SHARDS"
    NAME="${PREFIX}-all-${MODEL}-${TASK}"
    if tmux has-session -t "$NAME" 2>/dev/null; then
      echo "$NAME already running; $0 stop, or attach with: tmux attach -t $NAME"; exit 1
    fi
    mkdir -p "$LOGS"
    tmux new-session -d -s "$NAME" \
      "cd '$REPO' && TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       ./scripts/run_all.sh '$MODEL' '$TASK' '$SHARDS' 2>&1 \
       | tee '$LOGS/all-${MODEL}-${TASK}.log'; \
       echo; echo '--- finished, press enter ---'; read"
    echo "started $NAME across $SHARDS GPU(s)"
    echo
    echo "  follow:  tmux attach -t $NAME        (detach with ctrl-b d)"
    echo "  or:      tail -f $LOGS/all-${MODEL}-${TASK}.log"
    echo "  stages:  tail -f runs/$MODEL/$TASK/logs/*.log"
    echo "  stop:    $0 stop"
    ;;
  sweep|grid|latency)
    launch "$STAGE" "${2:-$(gpu_count)}"
    ;;
  select)
    echo "selecting learning rates from the completed sweep runs"
    CUDA_VISIBLE_DEVICES=0 $PY -m bench sweep --model "$MODEL" --task "$TASK" --shard 0/1
    LRS="runs/$MODEL/$TASK/lrs.json"
    [ -f "$LRS" ] && { echo; cat "$LRS"; }
    ;;
  single)
    # anything that is not sharded: determinism, test, grammar, fewshot, report
    shift
    require_gpus 1
    mkdir -p "$LOGS"
    for stage in "$@"; do
      echo "=== $stage $(date -Is) ==="
      CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
        $PY -m bench "$stage" 2>&1 | tee "$LOGS/${stage}.log"
    done
    ;;
  tail)
    ls "$LOGS"/*.log >/dev/null 2>&1 || { echo "no logs yet"; exit 0; }
    tail -f "$LOGS"/*.log
    ;;
  status)
    echo "sessions:"
    sessions | sed 's/^/  /' || echo "  none"
    echo "complete runs: $(grep -l '"complete": true' runs/*/metrics.json 2>/dev/null | wc -l | tr -d ' ')"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /' || true
    ;;
  stop)
    found=0
    while read -r s; do
      [ -z "$s" ] && continue
      tmux kill-session -t "$s" && echo "killed $s"
      found=1
    done < <(sessions)
    [ "$found" -eq 0 ] && echo "nothing to stop"
    ;;
  *)
    sed -n '2,17p' "$0"
    exit 1
    ;;
esac
