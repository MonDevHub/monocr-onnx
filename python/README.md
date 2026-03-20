# MonOCR (Python SDK)

The official Python SDK for Mon language OCR, powered by ONNX Runtime. Optimized for high-throughput batch processing and production server environments.

## Installation

```bash
pip install monocr-onnx
```

## Features

- **Production Accuracy**: Aligned with v2.0 high-precision models (128px vertical resolution).
- **Parallel Processing**: Native support for multithreaded batch OCR.
- **Auto-Model Management**: Automated caching of model weights from [Hugging Face](https://huggingface.co/janakhpon/monocr).
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

Initialize the OCR engine. If paths are omitted, the latest production models are automatically downloaded.

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
monocr batch ./input -o results.json
```

## Requirements

- Python 3.9+
- opencv-python-headless (for robust segmentation)
- onnxruntime (CPU or GPU)

## Maintenance

Maintained by [MonDevHub](https://github.com/MonDevHub).

## License

MIT
