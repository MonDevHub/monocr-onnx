# MonOCR (Universal SDK)

[![MonDevHub](https://img.shields.io/badge/Maintained%20by-MonDevHub-blue.svg)](https://github.com/MonDevHub)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

On-device OCR for the [Mon language](https://en.wikipedia.org/wiki/Mon_language)
(mnw), running on ONNX Runtime. Images never leave the machine.

> [!NOTE]
> Mon is classified as **vulnerable** in [UNESCO's Atlas of the World's Languages in Danger](https://en.wikipedia.org/wiki/Atlas_of_the_World%27s_Languages_in_Danger).
>
> This project digitises the Mon script, so that later work — system
> integrations, corpora, language models — has something to build on.

## Overview

This repository is the engine and its four bindings: Python, JavaScript
(Node.js), Go, and Rust. Each loads the same ONNX graph and the same 315-character
charset, pinned to one Hugging Face revision, and each decodes CTC output the same
way.

They are checked against each other rather than assumed to agree. The current
result is in [docs/CROSS_BINDING_PARITY.md](docs/CROSS_BINDING_PARITY.md):
**identical text on 5 of 7 images**, with both disagreements a single trailing
character traceable to a difference in image resampling. That is agreement, not
accuracy — no binding has been scored against ground truth here.

> [!TIP]
> The web and mobile apps cap uploads at 50 MB and 20 MB. This SDK has no such
> limit; use it directly for larger files.

## Architecture

```
Image (File/Buffer)
  LineSegmenter      → horizontal projection profile → List<LineSegment>
  ImagePreprocessor  → crop + scale + normalize [-1.0, 1.0]
  MonOcrEngine       → ONNX Runtime Session (monocr.onnx)
  CtcDecoder         → greedy CTC decode → String
```

### Model specification

| Attribute    | Specification                  |
| ------------ | ------------------------------ |
| Architecture | MobileNetV3 + BiLSTM-384 + CTC |
| Precision    | FP32 (ONNX)                    |
| Parameters   | ~6.6M                          |
| Input        | 128 × Variable (H × W)         |
| Charset      | 315 characters, 316 classes    |
| Asset Size   | ~25 MB                         |

The model is pinned to revision `a51be11` of
[`janakhpon/monocr`](https://huggingface.co/janakhpon/monocr), not to `main`. Each
binding ships the matching charset and refuses to decode if the two disagree: CTC
reserves index 0 for the blank, so a model over N characters must emit N + 1
classes. Without that check a mismatch returns well-formed Mon text that is wrong,
with no error and no lookup miss.

## Supported platforms

| SDK                      | Directory            | Registry/Source                                            | Status     |
| :----------------------- | :------------------- | :--------------------------------------------------------- | :--------- |
| **JavaScript / Node.js** | [`js/`](js/)         | [npm: monocr](https://www.npmjs.com/package/monocr)        | Published  |
| **Python**               | [`python/`](python/) | [PyPI: monocr-onnx](https://pypi.org/project/monocr-onnx/) | Published  |
| **Go**                   | [`go/`](go/)         | `github.com/MonDevHub/monocr-onnx/go`                      | Published  |
| **Rust**                 | [`rust/`](rust/)     | [`rust/`](rust/) — not on crates.io                        | Source only |

## Installation

### Python

```bash
pip install monocr-onnx
# or
uv add monocr-onnx
```

### Node.js

```bash
npm install monocr
```

### Go

```bash
go get github.com/MonDevHub/monocr-onnx/go
```

## Usage (Python)

```python
from monocr_onnx import MonOCR

# Downloads the pinned model on first run and caches it by revision.
engine = MonOCR()

# Page-level: segments into lines, then reads each one.
text = engine.predict("scanned_document.jpg")
print(text)

# Line-level: skips segmentation, for a crop you have already cut.
line_text = engine.predict_line("single_line_crop.png")
```

## Project structure

```
monocr-onnx/
├── python/           # Python package & CLI
├── js/               # Node.js package (uses sharp)
├── go/               # Go module
├── rust/             # Rust crate
├── src/              # shared engine sources
├── scripts/          # build and conversion scripts
├── examples/         # runnable per-language examples
├── data/             # test images and fixtures
└── docs/             # parity results and design notes
```

## Ecosystem

- **[MonOCR Web](https://github.com/MonDevHub/monocr-web)** — in-browser OCR
- **[MonOCR Android](https://github.com/MonDevHub/ocr-android)** — Jetpack Compose
- **[MonOCR iOS](https://github.com/MonDevHub/ocr-ios)** — SwiftUI

## Resources

- [Hugging Face model](https://huggingface.co/janakhpon/monocr) — ONNX and Core ML.
  The TFLite export was removed at revision `a51be11`.

## License

MIT
