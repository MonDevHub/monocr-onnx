#!/bin/bash
set -e

DEST=${1:-"model"}
BASE_URL="https://huggingface.co/janakh/monocr/resolve/main"


echo "Downloading MonOCR models to $DEST..."
mkdir -p "$DEST"

# Download ONNX
echo "Downloading monocr.onnx..."
curl -L "$BASE_URL/onnx/monocr.onnx" -o "$DEST/monocr.onnx"

# Download Charset
echo "Downloading charset.txt..."
curl -L "$BASE_URL/charset.txt" -o "$DEST/charset.txt"


echo "Done. Models saved to $DEST"
