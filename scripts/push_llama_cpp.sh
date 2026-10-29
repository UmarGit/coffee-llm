#!/usr/bin/env bash
# Send a llama.cpp source tree to the GPU server, for servers that cannot reach GitHub.
#
#   ./scripts/push_llama_cpp.sh              # clone on this Mac, send to ~/llama.cpp there
#   LLAMA_CPP=~/src/llama.cpp ./scripts/push_llama_cpp.sh   # send an existing checkout
#
# Only sources are sent (no .git, no build/), about 60 MB. The export stage builds it on
# the server on first use, so nothing compiled on the Mac ever runs there.
#
# Connection settings come from server.conf, exactly as for push_code.sh.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONF_FILE="${CONF_FILE:-$REPO/server.conf}"
if [ -f "$CONF_FILE" ]; then
  _host="${HOST:-}"; _port="${PORT:-}"; _key="${KEY:-}"
  # shellcheck disable=SC1090
  . "$CONF_FILE"
  [ -n "$_host" ] && HOST="$_host"
  [ -n "$_port" ] && PORT="$_port"
  [ -n "$_key" ] && KEY="$_key"
fi
HOST="${HOST:-user-8@10.32.15.90}"
PORT="${PORT:-2235}"
KEY="${KEY:-$HOME/.ssh/custodian_ed25519}"
DEST="${DEST:-llama.cpp}"                 # relative to the remote home directory

[ -f "$KEY" ] || { echo "ssh key not found at $KEY (override with KEY=...)"; exit 1; }
SSH=(ssh -i "$KEY" -p "$PORT")

SRC="${LLAMA_CPP:-}"
if [ -z "$SRC" ]; then
  SRC="$(mktemp -d)/llama.cpp"
  echo "cloning llama.cpp into $SRC"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
[ -f "$SRC/convert_hf_to_gguf.py" ] || { echo "$SRC is not a llama.cpp checkout"; exit 1; }
echo "llama.cpp $(git -C "$SRC" log -1 --format='%h %cs' 2>/dev/null || echo '(no git info)')"

if "${SSH[@]}" "$HOST" "test -e \$HOME/$DEST"; then
  echo "$HOST:~/$DEST already exists; remove it there first to replace it"
  exit 1
fi

echo "sending to $HOST:~/$DEST"
# COPYFILE_DISABLE stops macOS tar adding ._ resource files.
COPYFILE_DISABLE=1 tar czf - -C "$(dirname "$SRC")" --exclude .git --exclude build \
  "$(basename "$SRC")" \
  | "${SSH[@]}" "$HOST" "mkdir -p \$HOME/$DEST && tar xzf - -C \$HOME/$DEST --strip-components=1"

echo "done. On the server, rerun the export; it builds llama.cpp on first use."
