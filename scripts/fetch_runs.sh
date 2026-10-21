#!/usr/bin/env bash
# Download runs/ from the GPU server to this Mac.
#
#   ./scripts/fetch_runs.sh                  # metrics, curves, predictions, logs
#   ./scripts/fetch_runs.sh --with-adapters  # also the LoRA weights (about 24 MB each)
#   ./scripts/fetch_runs.sh --list           # list remote files, download nothing
#   ./scripts/fetch_runs.sh --scp            # plain scp, pulls everything including adapters
#
# Adapters are skipped by default because every reported number is recomputed from the
# saved predictions, so the report never needs them. Pull them only to export a GGUF.
#
# Transfer uses tar over ssh, which needs only ssh, tar and gzip. Those are present on
# macOS and on the server by default. rsync is used instead when both ends have it,
# because it only moves what changed, but it is not required.
#
# Override the connection with environment variables:
#   HOST=user-8@10.32.15.90 PORT=2235 KEY=~/.ssh/custodian_ed25519 REMOTE=experiments
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Connection settings come from server.conf when present; environment variables still
# win, so `HOST=... make push` works without editing anything. See server.conf.example.
CONF_FILE="${CONF_FILE:-$REPO/server.conf}"
if [ -f "$CONF_FILE" ]; then
  _host="${HOST:-}"; _port="${PORT:-}"; _key="${KEY:-}"; _remote="${REMOTE:-}"
  # shellcheck disable=SC1090
  . "$CONF_FILE"
  [ -n "$_host" ] && HOST="$_host"
  [ -n "$_port" ] && PORT="$_port"
  [ -n "$_key" ] && KEY="$_key"
  [ -n "$_remote" ] && REMOTE="$_remote"
fi

HOST="${HOST:-user-8@10.32.15.90}"
PORT="${PORT:-2235}"
KEY="${KEY:-$HOME/.ssh/custodian_ed25519}"
REMOTE="${REMOTE:-experiments}"          # path relative to the remote home directory
LOCAL="${LOCAL:-$REPO/runs}"

WITH_ADAPTERS=0
LIST_ONLY=0
USE_SCP=0
for arg in "$@"; do
  case "$arg" in
    --with-adapters) WITH_ADAPTERS=1 ;;
    --list|--dry-run) LIST_ONLY=1 ;;
    --scp) USE_SCP=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

[ -f "$KEY" ] || { echo "ssh key not found at $KEY (override with KEY=...)"; exit 1; }
SSH=(ssh -i "$KEY" -p "$PORT")
RUNS_REMOTE="\$HOME/$REMOTE/runs"

if ! "${SSH[@]}" "$HOST" "test -d $RUNS_REMOTE"; then
  echo "cannot reach $HOST, or $REMOTE/runs does not exist there"
  exit 1
fi

EXC_NAMES=(tmp __pycache__ .ipynb_checkpoints)
if [ "$WITH_ADAPTERS" -eq 0 ]; then
  EXC_NAMES+=(adapter)
  echo "skipping adapters; pass --with-adapters to include them"
fi

# find ... \( -name a -o -name b \) -prune -o -type f -print
PRUNE_EXPR="\\( "
for i in "${!EXC_NAMES[@]}"; do
  [ "$i" -gt 0 ] && PRUNE_EXPR="$PRUNE_EXPR -o "
  PRUNE_EXPR="$PRUNE_EXPR-name ${EXC_NAMES[$i]}"
done
PRUNE_EXPR="$PRUNE_EXPR \\) -prune -o"

if [ "$LIST_ONLY" -eq 1 ]; then
  echo "remote contents of $REMOTE/runs, excluding: ${EXC_NAMES[*]}"
  "${SSH[@]}" "$HOST" "cd $RUNS_REMOTE && find . $PRUNE_EXPR -type f -print | sort | head -40"
  "${SSH[@]}" "$HOST" "cd $RUNS_REMOTE && echo \"...\" && echo \"files to transfer: \$(find . $PRUNE_EXPR -type f -print | wc -l)\""
  exit 0
fi

mkdir -p "$LOCAL"
echo "pulling $HOST:$REMOTE/runs/  ->  $LOCAL/"

HAVE_LOCAL=0
HAVE_REMOTE=0
if [ "$USE_SCP" -eq 0 ]; then
  command -v rsync >/dev/null 2>&1 && HAVE_LOCAL=1
  "${SSH[@]}" "$HOST" "command -v rsync >/dev/null 2>&1" && HAVE_REMOTE=1 || true
fi

if [ "$USE_SCP" -eq 1 ]; then
  # scp cannot exclude anything, so this pulls the adapters too. Only worth using if
  # tar or ssh piping is blocked on your network.
  echo "using scp (no excludes possible, this pulls adapters as well)"
  scp -r -i "$KEY" -P "$PORT" "$HOST:$REMOTE/runs/." "$LOCAL/"
elif [ "$HAVE_LOCAL" -eq 1 ] && [ "$HAVE_REMOTE" -eq 1 ]; then
  echo "using rsync (only moves what changed)"
  EXCLUDES=()
  for n in "${EXC_NAMES[@]}"; do EXCLUDES+=(--exclude "$n/"); done
  rsync -azh --progress --partial \
    ${EXCLUDES[@]+"${EXCLUDES[@]}"} \
    -e "ssh -i $KEY -p $PORT" \
    "$HOST:$REMOTE/runs/" "$LOCAL/"
else
  echo "using tar over ssh (no rsync on the server)"
  # The file list comes from find rather than tar --exclude, because GNU tar and bsdtar
  # anchor exclude patterns differently and this has to work whichever is on each end.
  "${SSH[@]}" "$HOST" "cd $RUNS_REMOTE && find . $PRUNE_EXPR -type f -print | tar czf - -T -" \
    | tar xzf - -C "$LOCAL"
fi

echo
echo "=== what arrived ==="
COMPLETE=$(grep -l '"complete": true' "$LOCAL"/*/metrics.json 2>/dev/null | wc -l | tr -d ' ' || true)
TOTAL=$(find "$LOCAL" -maxdepth 1 -mindepth 1 -type d -not -name logs 2>/dev/null | wc -l | tr -d ' ' || true)
echo "  ${COMPLETE:-0} complete runs of ${TOTAL:-0} run directories"
echo "  $(du -sh "$LOCAL" | cut -f1) on disk"

if [ -f "$LOCAL/lrs.json" ]; then
  echo "  chosen learning rates: $(tr -d '\n ' < "$LOCAL/lrs.json")"
else
  echo "  lrs.json not present yet (the sweep selection has not run)"
fi

CAPPED=$(grep -L '"hit_cap_rate": 0\.0[,}]' "$LOCAL"/*/metrics.json 2>/dev/null | wc -l | tr -d ' ' || true)
if [ "${CAPPED:-0}" -gt 0 ]; then
  echo "  WARNING: ${CAPPED} runs generated at the token cap; check their logs"
else
  echo "  no run hit the token cap"
fi

echo
echo "Rebuild every table and figure locally, no GPU needed:"
echo "  make paper"
