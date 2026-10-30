# monocr-onnx

Part of the Mon OCR toolchain; see [how the pieces fit](https://github.com/MonDevHub).

On-device OCR for the [Mon language](https://en.wikipedia.org/wiki/Mon_language)
(mnw), running on ONNX Runtime, with bindings for Python, JavaScript (Node.js), Go
and Rust. Images never leave the machine. Mon is classified as **vulnerable** in
UNESCO's
[Atlas of the World's Languages in Danger](https://en.wikipedia.org/wiki/Atlas_of_the_World%27s_Languages_in_Danger);
digitising its script gives later work — system integrations, corpora, language
models — something to build on.

## Status

- **0.5.0** is the version in this tree for all four bindings. They share one
  version number, one model and one charset; see the [changelog](CHANGELOG.md).
- It reads printed Mon. Handwriting is out of scope.
- No accuracy figure is claimed here. The
  [model card](https://huggingface.co/janakhpon/monocr) has the held-out result
  and what it does not cover.
- The bindings do not yet return the same text for the same page. See
  [Limitations](#limitations).

> [!IMPORTANT]
> **Upgrade the JavaScript package.** Every npm release before 0.4.0 returned
> noise: preprocessing read a three-channel buffer as one channel. On a typeset
> page `monocr@0.3.2` returned 168 characters of garbage where 0.4.0 returns
> 1,178 of Mon. The Python, Go and Rust bindings were never affected.

## Install

```bash
pip install "monocr-onnx>=0.4.1"           # Python; or: uv add "monocr-onnx>=0.4.1"
npm install "monocr@>=0.4.1"               # Node.js
go get github.com/MonDevHub/monocr-onnx/go # Go
cargo add monocr                           # Rust
```

Keep the floors: 0.1.x pairs a 225-character charset with a 277-class graph and
returns wrong characters, not merely worse ones, and npm before 0.4.0 has the
bug above. pip and npm take the newest release above the floor; npm then saves
a caret range, which on 0.x stops below the next minor, so moving from 0.4.x to
0.5.x takes another install. The Rust crate is `monocr`, but its library is
`monocr_onnx`: `use monocr_onnx::MonOcr`.

The first run downloads the model (46.2 MB) and caches it.

### PDFs need poppler

PDFs need poppler's `pdftoppm` on the `PATH` in every binding; images need
nothing extra.

```bash
brew install poppler                 # macOS
sudo apt-get install poppler-utils   # Debian, Ubuntu
```

On Windows: `scoop install poppler`, `choco install poppler`,
`conda install -c conda-forge poppler`, or the prebuilt binaries from
[oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
with `Library\bin` added to `PATH`. Confirm with `pdftoppm -v` in a new shell.
Each binding's README notes how its PDF call finds poppler.

## Quick start

```python
from monocr_onnx import MonOCR

engine = MonOCR()                                  # fetches and caches the pinned model
text = engine.predict("scanned_document.jpg")      # page: segments into lines, reads each
line = engine.predict_line("single_line_crop.png") # one line you have already cut
```

Each binding's README has its own quick start, and [`examples/`](examples/) has
runnable Python, JavaScript and Go examples. Every binding also installs a CLI:
Python's is `monocr-onnx`; npm (`npm install -g monocr`), Go
(`go install github.com/MonDevHub/monocr-onnx/go/cmd/monocr@latest`) and Rust
(`cargo install monocr`, then `monocr page.jpg`) install one named `monocr`. That
name collides with the separate PyPI package
[`monocr`](https://pypi.org/project/monocr/): with both installed, whichever is
first on `PATH` runs.

## Supported platforms

All four run on the CPU on Linux, Windows and Apple-silicon macOS. Intel Macs are
narrower; [docs/supported-platforms.md](docs/supported-platforms.md) has the
detail per binding. CI runs on Linux only.

| Binding | Directory | Package | Requires | In this tree |
| :--- | :--- | :--- | :--- | :--- |
| Python | [`python/`](python/) | [PyPI: monocr-onnx](https://pypi.org/project/monocr-onnx/) | Python 3.11+ | 0.5.0 |
| JavaScript | [`js/`](js/) | [npm: monocr](https://www.npmjs.com/package/monocr) | Node.js 20.9+ | 0.5.0 |
| Go | [`go/`](go/) | [pkg.go.dev](https://pkg.go.dev/github.com/MonDevHub/monocr-onnx/go) | Go 1.23+, ONNX Runtime 1.18.0+ shared library | v0.5.0 |
| Rust | [`rust/`](rust/) | [crates.io: monocr](https://crates.io/crates/monocr) | Rust 1.88+ | 0.5.0 |

Go is the only binding that always needs ONNX Runtime installed separately; it
loads the shared library at run time, and [`go/README.md`](go/README.md) covers
it. Python and JavaScript bring the runtime with them, and Rust links a prebuilt one at build time where `ort` publishes one.

"In this tree" is the version the source declares, not what a registry serves. A
tag and a publish are different events: if a registry shows an older number than
a pushed tag, that release has not landed.

## How it works

```
Image (File/Buffer)
  ModelManager    → fetch + cache monocr.onnx at the pinned revision
  LineSegmenter   → horizontal projection profile → line boxes
  MonOCR.predict  → crop + scale + normalize to [-1.0, 1.0]
                  → ONNX Runtime session → greedy CTC decode → String
```

| Attribute    | Specification |
| ------------ | ------------- |
| Architecture | MobileNetV3-Large + SE + 2×BiLSTM-512 + attention + CTC |
| Precision    | FP32 (ONNX) |
| Parameters   | 11.55M |
| Input        | 160 × 1024 (H × W), both static |
| Charset      | 276 characters, 277 classes |
| Asset size   | 46.2 MB |

The model is pinned to one revision, and every binding refuses to decode when the
charset, the output shape or the output values break the contract, rather than
return wrong Mon text. [docs/model-contract.md](docs/model-contract.md) has the
revision, the cache path, the error types and each binding's class names.
[docs/image-input.md](docs/image-input.md) covers EXIF orientation, transparency
and 16-bit images.

## Limitations

**The bindings disagree on page output.** On the seven images in
[`data/images/`](data/images/), Python, JavaScript and Go return identical text on
**1 of 7** (measured 2026-10-01). The causes are different resampling kernels and
differently configured line segmenters. That is agreement, not accuracy: the
images have no ground truth. The kernels stay unaligned until a labelled set can
show that aligning them helps. [docs/binding-differences.md](docs/binding-differences.md)
has the pairwise numbers and each cause.

## Develop and release

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the test command for each binding (the
ones CI runs) and the A/B rule for any change that can alter output.
[`RELEASING.md`](RELEASING.md) describes how a release is cut. Bugs and questions
go to [Issues](https://github.com/MonDevHub/monocr-onnx/issues).

## Related

- [Model card](https://huggingface.co/janakhpon/monocr) — ONNX and Core ML
  exports, the held-out evaluation and its caveats.
- [MonDevHub/monocr](https://github.com/MonDevHub/monocr) — the apps built on this
  model: `apps/web` (SvelteKit), `apps/android` (Jetpack Compose) and `apps/ios`
  (SwiftUI). The Android and iOS apps build from source and are not in an app
  store yet. The apps cap a file you open for recognition at 50 MB; this SDK has
  no such limit.
- [MonDevHub/MonCorpusCollection](https://github.com/MonDevHub/MonCorpusCollection)
  — the Mon text corpus the training data is drawn from.
- [MonDevHub/awesome-mon-nlp](https://github.com/MonDevHub/awesome-mon-nlp) — a
  curated list of NLP resources for Mon.

## License

[MIT](LICENSE)
