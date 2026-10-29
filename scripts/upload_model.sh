#!/usr/bin/env bash
# Publish the exported model to Hugging Face, where the website downloads it from.
#
#   ./scripts/upload_model.sh                 # to umarsheikh303/pizza-words-gguf
#   REPO=someone/else ./scripts/upload_model.sh
#
# Uploads site/public/models/pizza-words-q8_0.{gguf,json} (fetch them with
# scripts/fetch_model.sh) and the model card in hf/pizza-words-gguf/README.md.
# Needs a Hugging Face write token: `uvx --from huggingface_hub hf auth login`.
# The model is CC-BY-NC-4.0; the card says so and must travel with it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-umarsheikh303/pizza-words-gguf}"
NAME="pizza-words-q8_0"
SRC="$REPO_ROOT/site/public/models"
HF=(uvx --from huggingface_hub hf)

for f in "$SRC/$NAME.gguf" "$SRC/$NAME.json" "$REPO_ROOT/hf/pizza-words-gguf/README.md"; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done

# The manifest's sha256 must match the file being published.
WANT=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(list(m['files'].values())[0]['sha256'])" "$SRC/$NAME.json")
echo "checking sha256 of $NAME.gguf"
GOT=$(shasum -a 256 "$SRC/$NAME.gguf" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "sha256 mismatch: file $GOT, manifest $WANT"; exit 1; }

"${HF[@]}" upload "$REPO" "$REPO_ROOT/hf/pizza-words-gguf/README.md" README.md --commit-message "Model card"
"${HF[@]}" upload "$REPO" "$SRC/$NAME.json" "$NAME.json" --commit-message "Export manifest"
"${HF[@]}" upload "$REPO" "$SRC/$NAME.gguf" "$NAME.gguf" --commit-message "Q8_0 GGUF"

echo
echo "published: https://huggingface.co/$REPO"
echo "the site loads: https://huggingface.co/$REPO/resolve/main/$NAME.gguf"
