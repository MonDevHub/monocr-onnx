# MonOCR (Python SDK)

The official Python SDK for Mon language OCR, powered by ONNX Runtime. Optimized for high-throughput batch processing and production server environments.

## Installation

```bash
pip install monocr-onnx
```

## Features

- **Production Accuracy**: Aligned with v2.0 high-precision models (128px vertical resolution).
- **Parallel Processing**: Native support for multithreaded batch OCR.
- **Pinned Model**: Weights and charset are fetched from one immutable [Hugging Face](https://huggingface.co/janakhpon/monocr) revision, checksummed, and cached per revision.
- **Comprehensive API**: Unified methods for images, PDFs, and accuracy benchmarking.
- **Robust Segmentation**: Advanced line-detection with adaptive thresholding and relative padding.

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

The model and its charset are pinned to `janakhpon/monocr@a51be11` (v2.0: 128px
input height, 315 characters, 316 CTC classes) and verified by sha256 after
download. They are never fetched from `main` — that ref has already moved under
this package once, replacing a 64px / 225-class network with the current one.

The cache lives at `~/.monocr/models/<revision>/`, so bumping the pin misses the
cache rather than silently reusing old weights.

If you installed 0.1.0, a stale `~/.monocr/models/monocr.onnx` may still be on
disk. Nothing reads it any more; `monocr download` will point it out and it is
safe to delete.

## Requirements

- Python 3.9+
- opencv-python-headless (for robust segmentation)
- onnxruntime (CPU or GPU)

## Maintenance

Maintained by [MonDevHub](https://github.com/MonDevHub).

## License

MIT
