# MonOCR (Universal SDK)

[![MonDevHub](https://img.shields.io/badge/Maintained%20by-MonDevHub-blue.svg)](https://github.com/MonDevHub)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MonOCR** is a high-performance, production-ready Optical Character Recognition (OCR) engine specifically optimized for the Mon language (mnw). Built on **ONNX Runtime**, it provides a lightning-fast, unified API for text recognition across any platform.

## Why MonOCR?

- **Production Accuracy**: Aligned with the latest high-precision models (128px vertical resolution).
- **Universal SDK**: Native, high-performance implementations for JavaScript, Python, Go, and Rust.
- **Robust Segmentation**: Intelligent line-detection with adaptive thresholding and relative padding for varied document layouts.
- **Smart Model Management**: Zero-config setup; models are automatically fetched and cached from [Hugging Face](https://huggingface.co/janakhpon/monocr).

## Supported Platforms

| SDK                      | Directory            | Registry/Source                                            | Status        |
| :----------------------- | :------------------- | :--------------------------------------------------------- | :------------ |
| **JavaScript / Node.js** | [`js/`](js/)         | [npm: monocr](https://www.npmjs.com/package/monocr)        | Production    |
| **Python**               | [`python/`](python/) | [PyPI: monocr-onnx](https://pypi.org/project/monocr-onnx/) | Production    |
| **Go**                   | [`go/`](go/)         | `github.com/MonDevHub/monocr-onnx/go`                      | Production    |
| **Rust**                 | [`rust/`](rust/)     | [monocr-onnx](rust/)                                       | Production    |

## Quick Installation

### Python

```bash
pip install monocr-onnx
```

### Node.js

```bash
npm install monocr
```

### Go

```bash
go get github.com/MonDevHub/monocr-onnx/go
```

## Usage Example (Python)

```python
from monocr_onnx import MonOCR

# Initialize engine (downloads model automatically on first run)
engine = MonOCR()

# Simple page-level OCR
text = engine.predict("scanned_document.jpg")
print(f"Recognized Text:\n{text}")

# Or process specific lines if you have your own layout analysis
line_text = engine.predict_line("single_line_crop.png")
```

## Project Structure

- `python/`: Source code for the Python package.
- `js/`: Source code for the Node.js package (uses `sharp` for image processing).
- `go/`: Source code for the Go module.
- `rust/`: Source code for the Rust crate.
- `models/`: (Reference) Model architecture and conversion scripts.

## Model Hub

The underlying weights and multi-format exports (ONNX, MLPackage, PyTorch) are hosted on the [Hugging Face Model Hub](https://huggingface.co/janakhpon/monocr).

## License

MIT License. Developed and maintained by the **MonDevHub** team.
