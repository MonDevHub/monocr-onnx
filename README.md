# MonOCR (Universal SDK)

[![MonDevHub](https://img.shields.io/badge/Maintained%20by-MonDevHub-blue.svg)](https://github.com/MonDevHub)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

On-device OCR for the [Mon language](https://en.wikipedia.org/wiki/Mon_language)
(mnw), running on ONNX Runtime. Images never leave the machine.

> [!NOTE]
> Mon is classified as **vulnerable** in [UNESCO's Atlas of the World's Languages in Danger](https://en.wikipedia.org/wiki/Atlas_of_the_World%27s_Languages_in_Danger).
>
> This project digitises the Mon script, so that later work has something to
> build on: system integrations, corpora, language models.

## Overview

This repository is the engine and its four bindings: Python, JavaScript
(Node.js), Go, and Rust. Each loads the same ONNX graph and the same 276-character
charset, pinned to one Hugging Face revision, and each decodes CTC output the same
way.

They are checked against each other rather than assumed to agree, and the current
result is not close to full agreement.

Re-measured 2026-09-04 on the **page** path — `read_image` / `predict`, which is
what the quick-start examples call — over the seven images in `data/images/`:
Python, JavaScript and Go return identical text on **1 of 7**. Pairwise, Python
matches JavaScript on 1, Python matches Go on 1, and JavaScript matches Go on 3.
The widest gap is `000028.jpg`, where Python returns 154 characters and
JavaScript and Go return 55 and 61.

[docs/CROSS_BINDING_PARITY.md](docs/CROSS_BINDING_PARITY.md) reports 5 of 7. It
measured the **line** path on pre-cropped files, which skips segmentation
entirely; that number was never a statement about reading a page. The two figures
are not in conflict, they answer different questions, and only the page one
describes what the documented API does.

The cause is two things compounding: four different image-resampling kernels, two
of them the wrong family, and four line segmenters whose density threshold has
four live values across the ports under two different formulas.

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

| SDK                      | Directory            | Registry/Source                                                                      | Registry, 2026-09-24 | This release |
| :----------------------- | :-------------------- | :------------------------------------------------------------------------------------ | :------------------- | :----------- |
| **JavaScript / Node.js** | [`js/`](js/)          | [npm: monocr](https://www.npmjs.com/package/monocr)                                   | 0.4.1                | 0.4.2        |
| **Python**               | [`python/`](python/)  | [PyPI: monocr-onnx](https://pypi.org/project/monocr-onnx/)                            | 0.4.1                | 0.4.2        |
| **Go**                   | [`go/`](go/)          | [pkg.go.dev: monocr-onnx/go](https://pkg.go.dev/github.com/MonDevHub/monocr-onnx/go)  | v0.4.1               | 0.4.2        |
| **Rust**                 | [`rust/`](rust/)      | [crates.io: monocr](https://crates.io/crates/monocr)                                  | 0.4.1                | 0.4.2        |

**0.4.2 (this release)** is one number for all four bindings and the version in
this tree. It changes no model, charset or API. It fixes the Python
`__version__`, which the 0.3.2, 0.4.0 and 0.4.1 wheels all reported as `0.3.0`,
and the install lines in the npm and crates.io READMEs, which in 0.4.1 still
pointed at 0.3.x.

> [!IMPORTANT]
> **Upgrade the JavaScript package.** Every npm release before 0.4.0 returned
> noise rather than text: the preprocessing step read a three-channel buffer as
> if it were one channel, so it sampled the wrong bytes. On a typeset page
> `monocr@0.3.2` returned 168 characters of garbage where 0.4.0 returns 1,178 of
> Mon. The Python, Go and Rust bindings were never affected.

Rust is the odd one out in naming: named `monocr` on
crates.io rather than `monocr-onnx` like the repository and the other three
registries — chosen once `monocr` was confirmed unclaimed there, before the
first publish. `[lib] name` in `rust/Cargo.toml` stays `monocr_onnx`, so
nothing importing the crate needed to change.

Registry state last queried 2026-09-24 against pypi.org, registry.npmjs.org,
crates.io, proxy.golang.org and pkg.go.dev — each package's own API, not this
repository's own claim about itself. All four answered 0.4.1, before any 0.4.2
tag was pushed; that answer is the "Registry" column. Once the 0.4.2 tags are
pushed, a registry that still answers 0.4.1 means that release has not landed.
A tag and a publish are different events, and conflating them is what let 0.2.0
and 0.2.1 sit tagged-but-unpublished for months.

The pip and npm lines below floor at 0.4.1, so they install the newest release
at or above it (npm's caret stops below 0.5.0); `cargo add` and `go get` take the
latest release. The floor is load-bearing twice over: 0.1.x carries a
225-character charset against a 277-class graph and returns wrong characters,
not merely worse ones, and every npm release before 0.4.0 returns noise, as
above. These lines previously read `>=0.3.0` and `^0.3.0`; on 0.x a caret range
stays below the next minor, so `^0.3.0` could never reach 0.4.0 and installed
0.3.2.

## Installation

### Python

```bash
pip install "monocr-onnx>=0.4.1"
# or
uv add "monocr-onnx>=0.4.1"
```

### Node.js

```bash
npm install monocr@^0.4.1
```

### Go

```bash
go get github.com/MonDevHub/monocr-onnx/go
```

### Rust

The crate is `monocr`; the library it exposes is `monocr_onnx`.

```bash
cargo add monocr
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
until 2026-08-15 and both return 404. `MonDevHub/monocr-web` still resolves but is
superseded: its `main` branch has not changed since 2026-04-08, and the web app
now lives in `apps/web` of the monorepo.

## Resources

- [Hugging Face model](https://huggingface.co/janakhpon/monocr) — ONNX and Core ML
  exports, and the model card with the held-out evaluation and its caveats.

## License

MIT
