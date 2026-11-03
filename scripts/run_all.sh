#!/usr/bin/env bash
# Every stage for one (model, task), in order, in one process.
#
#   ./scripts/run_all.sh <model> <task> [shards] [-- extra bench args]
#   ./scripts/run_all.sh qwen pizza 3 -- --negatives 0.3 --corrections 0.1
#
# Sharded stages fan out one process per GPU and join before the next stage
# starts, so the ordering the protocol requires is enforced here rather than by
# whoever is typing. Resumable throughout: a completed run is skipped, so
# re-running after a crash costs only what did not finish.
#
# Extra bench args after `--` go to every stage. BENCH_RUNS moves the run
# directory (default runs/). With STAGES_DIR set, each finished stage leaves
# <model>.<stage>.done there and is skipped next time. DRY_RUN=1 prints the
# commands instead of running them.
set -euo pipefail

MODEL="${1:?usage: run_all.sh <model> <task> [shards] [-- bench args]}"
TASK="${2:?usage: run_all.sh <model> <task> [shards] [-- bench args]}"
shift 2
SHARDS=""
if [ $# -gt 0 ] && [ "$1" != "--" ]; then SHARDS="$1"; shift; fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
EXTRA=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ -z "$SHARDS" ]; then
  SHARDS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [ "$SHARDS" = "0" ] && SHARDS=1
fi

RUNS="${BENCH_RUNS:-runs}"
BENCH=(uv run --extra pizza python -m bench)
LOGS="$RUNS/$MODEL/$TASK/logs"
DRY_RUN="${DRY_RUN:-0}"
[ "$DRY_RUN" = "1" ] || mkdir -p "$LOGS"

say() { printf '\n== %s  [%s/%s]  %s\n' "$1" "$MODEL" "$TASK" "$(date +%H:%M:%S)"; }

run() {
  local log="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    echo "  $* > $log"
    return 0
  fi
  "$@" >"$log" 2>&1
}

# One process per GPU, then wait. A single shard still goes through here so the
# two paths cannot drift apart.
fan_out() {
  local stage="$1" pids=() i rc=0
  for ((i = 0; i < SHARDS; i++)); do
    say "$stage shard $i/$SHARDS"
    CUDA_VISIBLE_DEVICES="$i" run "$LOGS/$stage.$i.log" "${BENCH[@]}" "$stage" \
      --model "$MODEL" --task "$TASK" --shard "$i/$SHARDS" ${EXTRA[@]+"${EXTRA[@]}"} &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  if [ "$rc" -ne 0 ]; then
    echo "FAILED: $stage (see $LOGS/$stage.*.log)" >&2
    exit "$rc"
  fi
}

one() {
  local stage="$1" log="${2:-$1}"; shift $(( $# < 2 ? $# : 2 ))
  say "$log"
  CUDA_VISIBLE_DEVICES=0 run "$LOGS/$log.log" "${BENCH[@]}" "$stage" \
    --model "$MODEL" --task "$TASK" "$@" ${EXTRA[@]+"${EXTRA[@]}"} || {
      echo "FAILED: $log (see $LOGS/$log.log)" >&2; exit 1; }
}

# A stage with a marker is skipped; a stage that finishes leaves one.
step() {
  local name="$1"; shift
  local mark="${STAGES_DIR:-}/$MODEL.$name.done"
  if [ -n "${STAGES_DIR:-}" ] && [ -f "$mark" ]; then
    say "$name (done, skipping)"
    return 0
  fi
  "$@"
  if [ -n "${STAGES_DIR:-}" ] && [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$STAGES_DIR"
    date -Is 2>/dev/null >"$mark" || date >"$mark"
  fi
}

printf '%s\n' "----------------------------------------------------------------"
printf 'model   %s\ntask    %s\nshards  %s\nruns    %s\nargs    %s\n' \
       "$MODEL" "$TASK" "$SHARDS" "$RUNS" "${EXTRA[*]:-(none)}"
printf '%s\n' "----------------------------------------------------------------"

# Data and gates first: `check` is the one stage whose failure must stop
# everything, because it is what catches contamination and a truncating cap.
step prepare one prepare
step check one check

step smoke one smoke

step sweep fan_out sweep
# Choosing the learning rates is `sweep` run unsharded: every run is already
# complete so nothing retrains, and the pass writes lrs.json. It must come after
# the join above, because it needs every shard's result.
step select one sweep select --shard 0/1
LRS="$RUNS/$MODEL/$TASK/lrs.json"
[ -f "$LRS" ] && { echo "  chosen:"; sed 's/^/    /' "$LRS"; }

step grid fan_out grid
step latency fan_out latency

step determinism one determinism
step test one test
step fewshot one fewshot
if [ "$TASK" = "pizza" ]; then
  step grammar one grammar
fi

step report one report

say "done"
echo "report: ${BENCH_REPORTS:-reports}/$MODEL/$TASK/"
