#!/usr/bin/env bash
# Download a campaign's bundle from the GPU server into runs-<name>/.
#
#   ./scripts/fetch_campaign.sh neg-v1            # bundle -> runs-neg-v1/ (resumable)
#   ./scripts/fetch_campaign.sh neg-v1 --models   # also the GGUF(s), sha256 checked
#   ./scripts/fetch_campaign.sh neg-v1-pilot
#
# The bundle mirrors runs/ without adapters or weights, so afterwards
# BENCH_RUNS=runs-<name> works for `make paper`-style tools and
# tools/export_site_data.py. Files already here with the same size are skipped,
# so a dropped connection costs only what had not arrived. Prints summary.txt.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
REMOTE="${REMOTE:-experiments}"

NAME="${1:?usage: fetch_campaign.sh <name> [--models]}"
shift
MODELS=0
for a in "$@"; do
  case "$a" in
    --models) MODELS=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

[ -f "$KEY" ] || { echo "ssh key not found at $KEY (override with KEY=...)"; exit 1; }
SSH=(ssh -i "$KEY" -p "$PORT")
SRC="\$HOME/$REMOTE/runs-$NAME/bundle"
LOCAL="$REPO/runs-$NAME"

if ! "${SSH[@]}" "$HOST" "test -f $SRC/summary.txt"; then
  echo "no finished bundle at $HOST:~/$REMOTE/runs-$NAME/bundle (is the campaign done?)"
  "${SSH[@]}" "$HOST" "ls -t \$HOME/$REMOTE/runs-$NAME/stages 2>/dev/null | head -3" || true
  exit 1
fi

mkdir -p "$LOCAL"
REMOTE_LIST=$("${SSH[@]}" "$HOST" "cd $SRC && find . -type f -printf '%s %p\n'")
NEED=$(while read -r size path; do
  [ -z "$path" ] && continue
  have=$(stat -f%z "$LOCAL/$path" 2>/dev/null || stat -c%s "$LOCAL/$path" 2>/dev/null || echo -1)
  [ "$have" = "$size" ] || printf '%s\n' "$path"
done <<<"$REMOTE_LIST")
TOTAL=$(printf '%s\n' "$REMOTE_LIST" | grep -c . || true)
COUNT=$(printf '%s\n' "$NEED" | grep -c . || true)

echo "bundle: $TOTAL files, $COUNT to fetch -> $LOCAL/"
if [ "$COUNT" -gt 0 ]; then
  printf '%s\n' "$NEED" | "${SSH[@]}" "$HOST" "cd $SRC && tar czf - -T -" | tar xzf - -C "$LOCAL"
fi

if [ "$MODELS" -eq 1 ]; then
  for m in "$LOCAL"/models/*/*.json; do
    [ -f "$m" ] || continue
    name="$(basename "$m" .json)"
    echo
    echo "model $name"
    MODELS_DIR="$REMOTE/runs-$NAME/models/pizza" OUT="$LOCAL/models" \
      "$REPO/scripts/fetch_model.sh" "$name"
  done
fi

echo
cat "$LOCAL/summary.txt"
