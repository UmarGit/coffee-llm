#!/usr/bin/env bash
# Upload the code to the GPU server. The reverse of fetch_runs.sh.
#
#   ./scripts/push_code.sh           # send code, leave runs/ and data/ alone
#   ./scripts/push_code.sh --list    # show what would be sent, send nothing
#
# Sends only source: src/, scripts/, tests/, PROTOCOL.md, pyproject.toml, README.md and
# the uv lockfile. Never sends .venv, runs/, data/, models/ or .git, so the dataset and
# the results already on the server are untouched.
#
# Transfer is tar over ssh, because the server has no rsync. Only ssh, tar and gzip are
# needed, and all three ship with macOS and the server.
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
REMOTE="${REMOTE:-experiments}"

LIST_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --list|--dry-run) LIST_ONLY=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

[ -f "$KEY" ] || { echo "ssh key not found at $KEY (override with KEY=...)"; exit 1; }
SSH=(ssh -i "$KEY" -p "$PORT")

# Explicit allow list. Safer than excluding, because a new large directory in the repo
# cannot accidentally start being uploaded.
SEND=(src scripts tests PROTOCOL.md pyproject.toml README.md)
[ -f "$REPO/uv.lock" ] && SEND+=(uv.lock)

cd "$REPO"
MISSING=()
for f in "${SEND[@]}"; do [ -e "$f" ] || MISSING+=("$f"); done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "not found in the repo: ${MISSING[*]}"; exit 1
fi

# Prune caches and anything heavy that may sit inside the sent directories.
PRUNE_EXPR="\\( -name __pycache__ -o -name .pytest_cache -o -name .ipynb_checkpoints -o -name '*.egg-info' \\) -prune -o"
FILE_FILTER="-type f ! -name '*.pyc' ! -name '.DS_Store' -print"

LIST=$(eval "find ${SEND[*]} $PRUNE_EXPR $FILE_FILTER" | sort)
COUNT=$(printf '%s\n' "$LIST" | grep -c . || true)

if [ "$LIST_ONLY" -eq 1 ]; then
  echo "would send $COUNT files to $HOST:$REMOTE/"
  printf '%s\n' "$LIST" | sed 's/^/  /'
  exit 0
fi

if ! "${SSH[@]}" "$HOST" "test -d \$HOME/$REMOTE"; then
  echo "cannot reach $HOST, or $REMOTE does not exist there"; exit 1
fi

echo "sending $COUNT files to $HOST:$REMOTE/"
printf '%s\n' "$LIST" | tar czf - -T - | "${SSH[@]}" "$HOST" "tar xzf - -C \$HOME/$REMOTE"

echo
echo "=== verifying on the server ==="
"${SSH[@]}" "$HOST" "cd \$HOME/$REMOTE && echo \"  PROTOCOL.md: \$(wc -l < PROTOCOL.md) lines\" && \
  echo \"  pizza_bench modules: \$(ls src/pizza_bench/*.py | wc -l)\" && \
  grep -E '^ +(max_steps|min_steps|patience):' src/pizza_bench/config.py | sed 's/^/  /' && \
  echo \"  runs/ untouched: \$(ls -d runs 2>/dev/null || echo 'absent')\""
echo
echo "Next on the server: make prepare && make check && make smoke"
