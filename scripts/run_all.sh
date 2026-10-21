#!/usr/bin/env bash
# Every stage for one (model, task), in order, in one process.
#
#   ./scripts/run_all.sh <model> <task> [shards]
#
# Sharded stages fan out one process per GPU and join before the next stage
# starts, so the ordering the protocol requires is enforced here rather than by
# whoever is typing. Resumable throughout: a completed run is skipped, so
# re-running after a crash costs only what did not finish.
set -euo pipefail

MODEL="${1:?usage: run_all.sh <model> <task> [shards]}"
TASK="${2:?usage: run_all.sh <model> <task> [shards]}"
SHARDS="${3:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ -z "$SHARDS" ]; then
  SHARDS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [ "$SHARDS" = "0" ] && SHARDS=1
fi

BENCH=(uv run --extra pizza python -m bench)
LOGS="runs/$MODEL/$TASK/logs"
mkdir -p "$LOGS"

say() { printf '\n== %s  [%s/%s]  %s\n' "$1" "$MODEL" "$TASK" "$(date +%H:%M:%S)"; }

# One process per GPU, then wait. A single shard still goes through here so the
# two paths cannot drift apart.
fan_out() {
  local stage="$1" pids=() i rc=0
  for ((i = 0; i < SHARDS; i++)); do
    say "$stage shard $i/$SHARDS"
    CUDA_VISIBLE_DEVICES="$i" "${BENCH[@]}" "$stage" \
      --model "$MODEL" --task "$TASK" --shard "$i/$SHARDS" \
      >"$LOGS/$stage.$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  if [ "$rc" -ne 0 ]; then
    echo "FAILED: $stage (see $LOGS/$stage.*.log)" >&2
    exit "$rc"
  fi
}

one() {
  local stage="$1"
  say "$stage"
  CUDA_VISIBLE_DEVICES=0 "${BENCH[@]}" "$stage" --model "$MODEL" --task "$TASK" \
    >"$LOGS/$stage.log" 2>&1 || {
      echo "FAILED: $stage (see $LOGS/$stage.log)" >&2; exit 1; }
}

printf '%s\n' "----------------------------------------------------------------"
printf 'model   %s\ntask    %s\nshards  %s\nlogs    %s\n' \
       "$MODEL" "$TASK" "$SHARDS" "$LOGS"
printf '%s\n' "----------------------------------------------------------------"

# Data and gates first: `check` is the one stage whose failure must stop
# everything, because it is what catches contamination and a truncating cap.
one prepare
one check

one smoke

fan_out sweep
# Choosing the learning rates is `sweep` run unsharded: every run is already
# complete so nothing retrains, and the pass writes lrs.json. It must come after
# the join above, because it needs every shard's result.
say "select learning rates"
CUDA_VISIBLE_DEVICES=0 "${BENCH[@]}" sweep --model "$MODEL" --task "$TASK" \
  --shard 0/1 >"$LOGS/select.log" 2>&1 || {
    echo "FAILED: select (see $LOGS/select.log)" >&2; exit 1; }
LRS="runs/$MODEL/$TASK/lrs.json"
[ -f "$LRS" ] && { echo "  chosen:"; sed 's/^/    /' "$LRS"; }

fan_out grid
fan_out latency

one determinism
one test
one fewshot
if [ "$TASK" = "pizza" ]; then
  one grammar
fi

one report

say "done"
echo "report: reports/$MODEL/$TASK/"
