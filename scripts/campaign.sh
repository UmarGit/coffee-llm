#!/usr/bin/env bash
# A whole campaign in one command: every stage for both models, the GGUF export
# and a collected bundle with a checked summary.txt. Run it inside tmux:
#
#   ./scripts/tmux.sh campaign neg-v1 --negatives 0.3 --corrections 0.1
#   ./scripts/tmux.sh campaign neg-v1 --pilot          # tiny end-to-end check first
#
# or directly:
#
#   ./scripts/campaign.sh <name> [--pilot] [--dry-run] [--models qwen,lfm2]
#                         [--export qwen:words,lfm2:words] [bench flags...]
#
# Everything lands in runs-<name>/ (runs-<name>-pilot/ with --pilot), never in
# runs/, because run_id does not encode the training mix. The directory refuses
# a second campaign with different flags. Re-running the same command resumes:
# finished stages leave runs-<name>/stages/<model>.<stage>.done.
#
#   runs-<name>/campaign.json     name, flags, arms, sizes, seeds, code version, GPUs
#   runs-<name>/<model>/pizza/    the runs, as in runs/
#   runs-<name>/models/           verified Q8_0 GGUFs and their manifests
#   runs-<name>/bundle/           everything but adapters and weights, plus summary.txt
set -euo pipefail

NAME="${1:?usage: campaign.sh <name> [--pilot] [--dry-run] [bench flags]}"
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PILOT=0
DRY=0
MODELS="qwen,lfm2"
EXPORTS="qwen:words,lfm2:words"
FLAGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pilot) PILOT=1 ;;
    --dry-run) DRY=1 ;;
    --models) MODELS="$2"; shift ;;
    --export) EXPORTS="$2"; shift ;;
    *) FLAGS+=("$1") ;;
  esac
  shift
done

VERIFY=200
if [ "$PILOT" -eq 1 ]; then
  NAME="$NAME-pilot"
  FLAGS+=(--sizes 1000 --seeds 0 --lr-select-size 1000 --lr-select-seeds 0
          --lr-candidates 0.0001 --max-steps 200 --limit 50)
  export BENCH_EVAL_LIMIT=100
  VERIFY=20
fi

DIR="runs-$NAME"
export BENCH_RUNS="$REPO/$DIR" BENCH_REPORTS="$REPO/$DIR/reports" BENCH_MODELS="$REPO/$DIR/models"
export STAGES_DIR="$REPO/$DIR/stages" DRY_RUN="$DRY"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=(uv run --extra pizza --group gpu python)
SHARDS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
[ "${SHARDS:-0}" -gt 0 ] || SHARDS=1

say() { printf '\n######## %s  %s\n' "$1" "$(date +%H:%M:%S)"; }
do_() {
  if [ "$DRY" -eq 1 ]; then echo "  $*"; else "$@"; fi
}

say "campaign $NAME  ->  $DIR  (models $MODELS, $SHARDS GPU(s))"
echo "flags: ${FLAGS[*]:-(none)}"
if [ "$DRY" -eq 1 ]; then
  echo "  ${PY[*]} -m bench.campaign manifest $DIR $NAME $MODELS ${FLAGS[*]:-}"
else
  "${PY[@]}" -m bench.campaign manifest "$DIR" "$NAME" "$MODELS" ${FLAGS[@]+"${FLAGS[@]}"}
fi

IFS=, read -r -a MODEL_LIST <<<"$MODELS"
for model in "${MODEL_LIST[@]}"; do
  say "$model: every stage"
  ./scripts/run_all.sh "$model" pizza "$SHARDS" -- ${FLAGS[@]+"${FLAGS[@]}"}
done

IFS=, read -r -a EXPORT_LIST <<<"$EXPORTS"
for pair in "${EXPORT_LIST[@]}"; do
  model="${pair%%:*}"
  arm="${pair##*:}"
  mark="$STAGES_DIR/$model.export-$arm.done"
  if [ -f "$mark" ]; then
    say "export $model $arm (done, skipping)"
    continue
  fi
  say "export $model $arm"
  if [ "$DRY" -eq 1 ]; then
    size="<largest size>"
  else
    size="$("${PY[@]}" -m bench.campaign size "$DIR")"
  fi
  [ "$DRY" -eq 1 ] || mkdir -p "$DIR/$model/pizza/logs"
  if [ "$DRY" -eq 1 ]; then
    echo "  ${PY[*]} -m bench export --model $model --task pizza --arm $arm --size $size --seed 0 --verify $VERIFY ${FLAGS[*]:-}"
  else
    CUDA_VISIBLE_DEVICES=0 "${PY[@]}" -m bench export --model "$model" --task pizza \
      --arm "$arm" --size "$size" --seed 0 --verify "$VERIFY" ${FLAGS[@]+"${FLAGS[@]}"} \
      2>&1 | tee "$DIR/$model/pizza/logs/export-$arm.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: export $model $arm" >&2; exit 1; }
    date >"$mark"
  fi
done

say "collect bundle"
# Everything a report, the paper or the site needs, without adapters or weights.
do_ rm -rf "$DIR/bundle.tmp"
do_ mkdir -p "$DIR/bundle.tmp"
if [ "$DRY" -eq 1 ]; then
  echo "  (cd $DIR && find . -prune adapter tmp bundle* *.gguf -type f | tar) -> $DIR/bundle"
else
  (cd "$DIR" && find . \( -name adapter -o -name tmp -o -name 'bundle*' -o -name '*.gguf' \
      -o -name '*-merged' \) -prune -o -type f -print | tar cf - -T -) \
    | tar xf - -C "$DIR/bundle.tmp"
  rm -rf "$DIR/bundle"
  mv "$DIR/bundle.tmp" "$DIR/bundle"
fi

say "summary"
if [ "$DRY" -eq 1 ]; then
  echo "  ${PY[*]} -m bench.campaign summary $DIR"
  exit 0
fi
"${PY[@]}" -m bench.campaign summary "$DIR" || {
  echo
  echo "Some checks failed; see $DIR/bundle/summary.txt"
  exit 1
}
echo
echo "fetch on the Mac with: ./scripts/fetch_campaign.sh $NAME --models"
