#!/bin/bash
# Fetch the published model and its charset, pinned to one revision.
#
# Three things were wrong here until 2026-08-15, and each of them made the
# script fail or lie:
#   - the org was `janakh`, not `janakhpon`, so every fetch 404'd
#   - it resolved `main`, so it could hand you a different model from the one
#     the four SDKs pin
#   - it took the charset from the repository root instead of `onnx/`
#
# The revision below must equal REVISION in python/monocr_onnx/model_manager.py,
# MODEL_REVISION in js/src/model-manager.js and rust/src/model_manager.rs, and
# ModelRevision in go/pkg/model. A model and a charset from different revisions
# decode into confident wrong text.
set -euo pipefail

DEST=${1:-"model"}
REVISION="d3d9d5e"
BASE_URL="https://huggingface.co/janakhpon/monocr/resolve/$REVISION"

echo "Downloading MonOCR model at $REVISION to $DEST..."
mkdir -p "$DEST"

# `--fail` so an HTML error page is not saved under a .onnx name. Without it
# curl exits 0 and you debug the model instead of the URL.
curl -fL "$BASE_URL/onnx/monocr.onnx"  -o "$DEST/monocr.onnx"
curl -fL "$BASE_URL/onnx/charset.txt"  -o "$DEST/charset.txt"

chars=$(python3 -c "import sys,pathlib; print(len(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').rstrip('\n')))" "$DEST/charset.txt")
if [ "$chars" -ne 276 ]; then
    echo "charset.txt has $chars characters, expected 276 at $REVISION." >&2
    echo "The revision moved or the download is truncated. Refusing to continue." >&2
    exit 1
fi

echo "Done. Model and a $chars-character charset saved to $DEST"
