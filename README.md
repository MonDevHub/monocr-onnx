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
(Node.js), Go, and Rust. Each loads the same ONNX graph and the same 276-character
charset, pinned to one Hugging Face revision, and each decodes CTC output the same
way.

They are checked against each other rather than assumed to agree, and the
current result is not full agreement. From
[docs/CROSS_BINDING_PARITY.md](docs/CROSS_BINDING_PARITY.md): Python, JS and Go
produce identical text on **5 of 7** images, and Rust differs on a sixth, so
across all four bindings it is **4 of 7**. Two disagreements drop a trailing Mon
character; Rust's reads a Myanmar digit six where the others read an ASCII zero.
The cause is four different image-resampling kernels, two of them the wrong
family.

That is agreement, not accuracy. No binding has been scored against ground truth
here, and four implementations reading the same wrong thing would agree
perfectly.

> [!TIP]
> The web and mobile apps cap uploads at 50 MB and 20 MB. This SDK has no such
> limit; use it directly for larger files.

## Architecture

```
Image (File/Buffer)
  ModelManager    → fetch + cache monocr.onnx at the pinned revision
  LineSegmenter   → horizontal projection profile → line boxes
  MonOCR.predict  → crop + scale + normalize to [-1.0, 1.0]
                  → ONNX Runtime session → greedy CTC decode → String
```

`ModelManager`, `LineSegmenter` and `MonOCR` are the names in this repository, in
each of the four languages. The block above previously named `ImagePreprocessor`,
`MonOcrEngine` and `CtcDecoder`, which are classes in the Android and iOS apps and
appear nowhere here.

### Model specification

| Attribute    | Specification                  |
| ------------ | ------------------------------ |
| Architecture | MobileNetV3-Large + SE + 2×BiLSTM-512 + attention + CTC |
| Precision    | FP32 (ONNX)                    |
| Parameters   | 11.55M                         |
| Input        | 160 × 1024 (H × W), both static |
| Charset      | 276 characters, 277 classes    |
| Asset Size   | 46.2 MB                        |

The model is pinned to revision `d3d9d5e` of
[`janakhpon/monocr`](https://huggingface.co/janakhpon/monocr), not to `main`. Each
binding ships the matching charset and refuses to decode if the two disagree: CTC
reserves index 0 for the blank, so a model over N characters must emit N + 1
classes. Without that check a mismatch returns well-formed Mon text that is wrong,
with no error and no lookup miss.

## Supported platforms

| SDK                      | Directory            | Registry/Source                                            | Published  | In this tree |
| :----------------------- | :------------------- | :--------------------------------------------------------- | :--------- | :----------- |
| **JavaScript / Node.js** | [`js/`](js/)         | [npm: monocr](https://www.npmjs.com/package/monocr)        | 0.1.5      | 0.2.0        |
| **Python**               | [`python/`](python/) | [PyPI: monocr-onnx](https://pypi.org/project/monocr-onnx/) | 0.1.0      | 0.2.0        |
| **Go**                   | [`go/`](go/)         | `github.com/MonDevHub/monocr-onnx/go`                      | v0.1.1     | 0.2.0        |
| **Rust**                 | [`rust/`](rust/)     | [`rust/`](rust/) — not on crates.io                        | not published | 0.2.0     |

**0.2.0 is not released.** The table used to read "Published" in every row, which
was true of the *binding* and not of the version beside it. Registry state
verified 2026-08-15 against pypi.org, registry.npmjs.org and proxy.golang.org.

This matters more than a version number usually does: the published 0.1.x
releases are the ones carrying the 225-character charset defect that the commits
in this tree fix. Installing from a registry today gets the broken charset.

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
├── src/              # monocr_full.js — an early single-file prototype,
│                     #   imported by nothing and pinned to a 64px input
├── scripts/          # download_models.sh — fetch the pinned model
├── examples/         # runnable examples: python, js, go (no rust yet)
├── data/             # test images and fixtures
└── docs/             # parity results and design notes
```

## Ecosystem

All three apps now live in one repository,
[MonDevHub/monocr](https://github.com/MonDevHub/monocr):

- `apps/web` — in-browser OCR, SvelteKit
- `apps/android` — Jetpack Compose
- `apps/ios` — SwiftUI

The former `MonDevHub/ocr-android` and `MonDevHub/ocr-ios` links were listed here
until 2026-08-15 and both return 404; `MonDevHub/monocr-web` still resolves but
has not been pushed to since the web app moved into the monorepo.

## Resources

- [Hugging Face model](https://huggingface.co/janakhpon/monocr) — ONNX and Core ML.
  The TFLite export was removed at revision `d3d9d5e`.

## License

MIT
