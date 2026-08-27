# MonOCR (Python SDK)

The official Python SDK for Mon language OCR, powered by ONNX Runtime. Optimized for high-throughput batch processing and production server environments.

## Installation

> **Pin 0.3.0 or newer.** The 0.1.x releases on PyPI target a superseded model —
> 64px input height, a 225-character charset, `pixel / 255` normalisation — and a
> charset that size against a 277-class graph returns the wrong characters rather
> than merely worse ones. The version bound below is what makes the mismatch an
> install error instead of silently wrong output. See `CHANGELOG.md`.

```bash
pip install "monocr-onnx>=0.3.0"
```

## Features

- **Pinned to one network**: the v3.5 recogniser at revision `d3d9d5e`, with 160px input
  height, 276 characters, 277 CTC classes. No accuracy figure is claimed here; see
  the [model card](https://huggingface.co/janakhpon/monocr) for the held-out result
  and its caveats.
- **Parallel Processing**: Native support for multithreaded batch OCR.
- **Pinned Model**: Weights and charset are fetched from one immutable [Hugging Face](https://huggingface.co/janakhpon/monocr) revision, checksummed, and cached per revision.
- **One API for images and PDFs**: the same call shape for a line, a page and a document.
- **Line segmentation**: adaptive thresholding, with padding relative to each line's height.

## Quick Start

```python
from monocr_onnx import MonOCR

# Initialize engine (downloads model automatically on first run)
engine = MonOCR()

# Recognize single image
text = engine.predict("document.png")
print(text)

# Recognize single line (for custom layout analysis)
line_text = engine.predict_line("line_crop.png")
```

## API Reference

### `MonOCR(model_path=None, charset_path=None)`

Initialize the OCR engine. If paths are omitted, the pinned model and its charset are downloaded on first use.

Loading refuses a model whose output class count or input height disagrees with the charset — a mismatched pair still runs and still returns text, it is just the wrong text.

### `predict(image_path)` -> `str`

Recognize text from a single image file or page. Alias for `predict_page`.

### `predict_line(image)` -> `str`

Recognize text from a single cropped text line image (PIL).

### `predict_page(image_path)` -> `str`

Segment an image into lines and recognize each.

## CLI Usage

```bash
# Recognize an image
monocr image input.jpg

# Process a PDF
monocr pdf document.pdf

# Batch directory processing
monocr batch ./input

# Pre-fetch the model and charset
monocr download
```

## Model artifact

The model and its charset are pinned to `janakhpon/monocr@d3d9d5e` (v3.5: 160px
input height, 276 characters, 277 CTC classes) and verified by sha256 after
download. They are never fetched from `main` — that ref has already moved under
this package once, replacing a 64px / 225-class network with the current one.

The cache lives at `~/.monocr/models/<revision>/`, so bumping the pin misses the
cache rather than silently reusing old weights.

If you installed 0.1.0, a stale `~/.monocr/models/monocr.onnx` may still be on
disk. Nothing reads it any more; `monocr download` will point it out and it is
safe to delete.

## Requirements

- Python 3.11+ — onnxruntime 1.24.1 ships no wheel below cp311 and no sdist, so
  3.10 and below have nothing to install
- opencv-python-headless (line segmentation)
- onnxruntime 1.24.1 (CPU or GPU), pinned in `uv.lock`

## Maintenance

Maintained by [MonDevHub](https://github.com/MonDevHub).

## License

MIT
