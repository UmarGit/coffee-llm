#!/usr/bin/env bash
# Download an exported GGUF from the GPU server into the website.
#
#   ./scripts/fetch_model.sh                    # newest export, q8_0
#   ./scripts/fetch_model.sh fresh q4_k_m       # newest export whose name has "fresh", q4_k_m
#   ./scripts/fetch_model.sh --check            # print the verification result, download nothing
#
# Before downloading it prints the export's verification against the saved test
# predictions, so a broken export is caught before 800 MB cross the network.
#
# The server has no SFTP subsystem (modern scp fails with "subsystem request failed"),
# so this streams over plain ssh, like fetch_runs.sh. It resumes: rerun after a dropped
# connection and it continues from the bytes already on disk. The result is checked
# against the sha256 in the manifest.
#
# Lands in site/public/models/, which is gitignored. PIZZA-derived: CC-BY-NC-4.0.
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
TASK="${TASK:-pizza}"
OUT="${OUT:-$REPO/site/public/models}"

CHECK_ONLY=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) ARGS+=("$a") ;;
  esac
done
MATCH="${ARGS[0]:-}"
QUANT="${ARGS[1]:-q8_0}"

[ -f "$KEY" ] || { echo "ssh key not found at $KEY (override with KEY=...)"; exit 1; }
# The GGUF barely compresses, so compression only costs CPU; AES-GCM is the cheapest
# cipher on hardware with AES instructions, which both ends have.
SSH=(ssh -i "$KEY" -p "$PORT" -o Compression=no -c aes128-gcm@openssh.com,chacha20-poly1305@openssh.com)
DIR="$REMOTE/models/$TASK"

MANIFEST=$("${SSH[@]}" "$HOST" "ls -t \$HOME/$DIR/*.json 2>/dev/null | grep -F -- '${MATCH}' | head -1" || true)
[ -n "$MANIFEST" ] || { echo "no manifest matching '${MATCH}' in $HOST:~/$DIR"; exit 1; }
JSON=$("${SSH[@]}" "$HOST" "cat '$MANIFEST'")

INFO=$(MANIFEST_JSON="$JSON" QUANT="$QUANT" python3 - <<'PY'
import json, os, sys

m = json.loads(os.environ["MANIFEST_JSON"])
q = os.environ["QUANT"]
f = m["files"].get(q)
if not f:
    sys.exit("no %s file in this export; have %s" % (q, sorted(m["files"])))
say = lambda s: print(s, file=sys.stderr)
say("run       %s  (%s, %s)" % (m["run_id"], m["base"], m["arm"]))
say("file      %s  %.0f MB" % (f["file"], f["bytes"] / 1e6))
v = f.get("verify") or {}
if v:
    say("verify    agreement %.1f%% on %d test orders, em %.3f vs saved %.3f, parse %.1f%%, median %s ms"
        % (100 * v["agreement_with_saved"], v["n"], v["em_gguf"], v["em_saved_same_rows"],
           100 * v["parse_validity_gguf"], v["median_ms_server"]))
    for d in v.get("first_disagreements", [])[:5]:
        say("  differs: %r\n           saved %r\n           gguf  %r" % (d["src"], d["saved"], d["gguf"]))
    if v["agreement_with_saved"] < 0.95:
        say("WARNING   agreement is low: this export does not behave like the trained run")
else:
    say("verify    not run (--verify 0)")
print(f["file"], f["bytes"], f["sha256"])
PY
)
read -r FILE BYTES SHA <<<"$INFO"
[ "$CHECK_ONLY" -eq 1 ] && exit 0

mkdir -p "$OUT"
ARM=$(MANIFEST_JSON="$JSON" python3 -c 'import json, os; print(json.loads(os.environ["MANIFEST_JSON"])["arm"])')
DEST="$OUT/$TASK-$ARM-$QUANT.gguf"
PART="$DEST.part"
printf '%s' "$JSON" > "${DEST%.gguf}.json"

HAVE=0
[ -f "$PART" ] && HAVE=$(stat -f%z "$PART" 2>/dev/null || stat -c%s "$PART")
if [ "$HAVE" -lt "$BYTES" ]; then
  [ "$HAVE" -gt 0 ] && echo "resuming at $((HAVE / 1000000)) MB of $((BYTES / 1000000)) MB"
  PROGRESS=(cat)
  command -v pv >/dev/null && PROGRESS=(pv -s "$((BYTES - HAVE))")
  "${SSH[@]}" "$HOST" "tail -c +$((HAVE + 1)) \"\$HOME/$DIR/$FILE\"" | "${PROGRESS[@]}" >> "$PART"
fi

echo "checking sha256"
GOT=$(shasum -a 256 "$PART" | cut -d' ' -f1)
if [ "$GOT" != "$SHA" ]; then
  echo "sha256 mismatch: got $GOT, manifest says $SHA"
  echo "delete $PART and run again"
  exit 1
fi
mv "$PART" "$DEST"
echo "ok  $DEST"
