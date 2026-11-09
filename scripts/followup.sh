#!/usr/bin/env bash
# Follow-up runs for the paper (plan.md X1 and X3), reusing neg-v1's frozen
# learning rates and token caps. Run inside tmux, after runs-neg-v1/ exists:
#
#   ./scripts/followup.sh ablation [--dry-run]
#       X1: WORDS and EXR, full pool, 5 seeds, both models, trained with
#       negatives only (runs-ctl-neg/) and with neither (runs-ctl-none/). 40 runs.
#   ./scripts/followup.sh lr-edges [--dry-run]
#       X3: one learning rate past the grid edge for the three edge arms, at the
#       selection size and seeds (runs-lr-edges/). 9 runs.
#
# GPU_IDS=0,2 picks the GPUs (default 0,1). Re-running resumes: finished runs are
# skipped. Each directory ends with bundle/summary.txt, so the user fetches it with
#   ./scripts/fetch_campaign.sh ctl-neg     (or ctl-none, lr-edges)
set -euo pipefail

WHAT="${1:?usage: followup.sh ablation|lr-edges [--dry-run]}"
DRY=0
[ "${2:-}" = "--dry-run" ] && DRY=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SRC="runs-neg-v1"
[ -f "$SRC/campaign.json" ] || { echo "no $SRC/campaign.json on this machine" >&2; exit 1; }
IFS=, read -r -a GPUS <<<"${GPU_IDS:-0,1}"
N=${#GPUS[@]}
PY=(uv run --extra pizza --group gpu python)
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SIZE="$("${PY[@]}" -m bench.campaign size "$SRC")"

seed_dir() {
  local dir=$1 model=$2
  [ "$DRY" -eq 1 ] && return 0
  mkdir -p "$dir/$model/pizza/logs"
  cp "$SRC/$model/pizza/caps.json" "$SRC/$model/pizza/lrs.json" "$dir/$model/pizza/"
}

# fan <dir> <model> <log> <bench stage and flags...>: one shard per GPU, in parallel.
fan() {
  local dir=$1 model=$2 log=$3
  shift 3
  export BENCH_RUNS="$REPO/$dir"
  local pids=() i
  for i in "${!GPUS[@]}"; do
    local cmd=("${PY[@]}" -m bench "$@" --model "$model" --task pizza --shard "$i/$N")
    if [ "$DRY" -eq 1 ]; then
      echo "  BENCH_RUNS=$dir CUDA_VISIBLE_DEVICES=${GPUS[$i]} ${cmd[*]}"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${cmd[@]}" >"$dir/$model/pizza/logs/$log-$i.log" 2>&1 &
    pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
}

bundle() {
  local dir=$1
  [ "$DRY" -eq 1 ] && { echo "  bundle $dir"; return 0; }
  rm -rf "$dir/bundle.tmp"
  mkdir -p "$dir/bundle.tmp"
  (cd "$dir" && find . \( -name adapter -o -name tmp -o -name 'bundle*' -o -name '*.gguf' \
      -o -name '*-merged' \) -prune -o -type f -print | tar cf - -T -) | tar xf - -C "$dir/bundle.tmp"
  "${PY[@]}" - "$dir/bundle.tmp" <<'EOF' >"$dir/bundle.tmp/summary.txt"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for m in sorted(root.glob("*/pizza/*/metrics.json")):
    r = json.loads(m.read_text())
    tag = "complete" if r.get("complete") else "INCOMPLETE"
    print(f"{m.parent.parent.parent.name:5} {r['run_id']:34} dev {r['dev']['em']:.4f} "
          f"test {r['test']['em']:.4f} neg {r['negatives'].get('handled', 0):.3f} {tag}")
EOF
  rm -rf "$dir/bundle"
  mv "$dir/bundle.tmp" "$dir/bundle"
  cat "$dir/bundle/summary.txt"
}

case "$WHAT" in
  ablation)
    for mix in neg none; do
      dir="runs-ctl-$mix"
      flags=(--arms words,exr --sizes "$SIZE" --seeds 0,1,2,3,4)
      [ "$mix" = neg ] && flags+=(--negatives 0.3)
      echo "######## $dir  ${flags[*]}  $(date +%H:%M:%S)"
      for model in qwen lfm2; do
        seed_dir "$dir" "$model"
        fan "$dir" "$model" grid grid "${flags[@]}"
      done
      bundle "$dir"
    done
    ;;
  lr-edges)
    dir="runs-lr-edges"
    common=(--lr-select-size 100000 --lr-select-seeds 0,1,2 --negatives 0.3 --corrections 0.1)
    for spec in qwen:exr_short:0.001 qwen:json:0.00005 lfm2:dsl:0.00005; do
      IFS=: read -r model arm lr <<<"$spec"
      echo "######## $dir  $model $arm lr $lr  $(date +%H:%M:%S)"
      seed_dir "$dir" "$model"
      fan "$dir" "$model" "sweep-$arm" sweep --arms "$arm" --lr-candidates "$lr" "${common[@]}"
    done
    bundle "$dir"
    ;;
  *)
    echo "unknown: $WHAT (ablation or lr-edges)" >&2
    exit 1
    ;;
esac
