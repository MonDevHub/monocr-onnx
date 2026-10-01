# MonOCR

On-device OCR for the [Mon language](https://en.wikipedia.org/wiki/Mon_language)
(mnw), running on ONNX Runtime, with bindings for Python, JavaScript (Node.js), Go
and Rust. Images never leave the machine.

Mon is classified as **vulnerable** in UNESCO's
[Atlas of the World's Languages in Danger](https://en.wikipedia.org/wiki/Atlas_of_the_World%27s_Languages_in_Danger).
This project digitises the Mon script so that later work — system integrations,
corpora, language models — has something to build on.

## Status

- **0.4.2** is the current release of all four bindings. The four share one
  version number, one model and one charset; see the [changelog](CHANGELOG.md).
- It reads printed Mon. Handwriting is out of scope.
- No accuracy figure is claimed here. The
  [model card](https://huggingface.co/janakhpon/monocr) has the held-out result
  and what it does not cover.
- The bindings do not yet return the same text for the same page. See
  [Limitations](#limitations).

> [!IMPORTANT]
> **Upgrade the JavaScript package.** Every npm release before 0.4.0 returned
> noise rather than text: the preprocessing step read a three-channel buffer as
> if it were one channel, so it sampled the wrong bytes. On a typeset page
> `monocr@0.3.2` returned 168 characters of garbage where 0.4.0 returns 1,178 of
> Mon. The Python, Go and Rust bindings were never affected.

## Install

```bash
pip install "monocr-onnx>=0.4.1"           # Python; or: uv add "monocr-onnx>=0.4.1"
npm install monocr@^0.4.1                  # Node.js
go get github.com/MonDevHub/monocr-onnx/go # Go
cargo add monocr                           # Rust
```

Keep the floors. 0.1.x pairs a 225-character charset with a 277-class graph and
returns wrong characters, not merely worse ones, and every npm release before
0.4.0 returns noise. On 0.x npm's caret stops below the next minor, so
`^0.4.1` takes 0.4.x releases only. `go get` and `cargo add` take the latest
release.

The Rust crate is `monocr`, but the library it exposes is `monocr_onnx`:
`use monocr_onnx::MonOcr`.

The first run downloads the model (46.2 MB) and caches it. PDFs need poppler's
`pdftoppm` on the `PATH` in every binding; images need nothing extra.

## Quick start

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

Each binding's README has its own quick start; the Python, JavaScript and Go
ones also cover a CLI. Runnable examples for Python, JavaScript and Go are in
[`examples/`](examples/).

## Supported platforms

All four run on the CPU, and are built for Linux, Windows and Apple-silicon
macOS. Intel Macs are narrower: `onnxruntime-node` ships no Intel-macOS binary,
the Rust crate's build has no prebuilt ONNX Runtime to link there, and neither
Microsoft's 1.24.1 release nor Homebrew has a prebuilt Intel-macOS library for
Go. Python still installs there, on onnxruntime 1.23.2 (the last release with an
Intel-macOS wheel) and Python 3.11 to 3.13. CI runs on Linux only.

| Binding | Directory | Package | Requires | Latest |
| :--- | :--- | :--- | :--- | :--- |
| Python | [`python/`](python/) | [PyPI: monocr-onnx](https://pypi.org/project/monocr-onnx/) | Python 3.11+ | 0.4.2 |
| JavaScript | [`js/`](js/) | [npm: monocr](https://www.npmjs.com/package/monocr) | Node.js 20.9+ | 0.4.2 |
| Go | [`go/`](go/) | [pkg.go.dev](https://pkg.go.dev/github.com/MonDevHub/monocr-onnx/go) | Go 1.23+, ONNX Runtime 1.18.0+ shared library | v0.4.2 |
| Rust | [`rust/`](rust/) | [crates.io: monocr](https://crates.io/crates/monocr) | nothing beyond Cargo | 0.4.2 |

Go is the only binding that always needs ONNX Runtime installed separately; it
loads the shared library at run time. [`go/README.md`](go/README.md) covers installing it.
The Python and JavaScript packages bring the runtime with them, and the Rust
crate links a prebuilt one at build time on the targets `ort` publishes one for.

"Latest" was read on 2026-10-01 from each registry's own API (registry.npmjs.org,
pypi.org, crates.io, proxy.golang.org). A tag and a publish are different
events: if a registry answers an older number than a pushed tag, that release has
not landed.

## How it works

```
Image (File/Buffer)
  ModelManager    → fetch + cache monocr.onnx at the pinned revision
  LineSegmenter   → horizontal projection profile → line boxes
  MonOCR.predict  → crop + scale + normalize to [-1.0, 1.0]
                  → ONNX Runtime session → greedy CTC decode → String
```

The names are the Python and JavaScript classes; JavaScript's page method is
`predictPage`. Rust spells the engine
`MonOcr`; Go uses `model.Manager`, `segmenter.LineSegmenter` and
`predictor.Predictor`.

| Attribute    | Specification |
| ------------ | ------------- |
| Architecture | MobileNetV3-Large + SE + 2×BiLSTM-512 + attention + CTC |
| Precision    | FP32 (ONNX) |
| Parameters   | 11.55M |
| Input        | 160 × 1024 (H × W), both static |
| Charset      | 276 characters, 277 classes |
| Asset size   | 46.2 MB |

The model is pinned to revision `d3d9d5e` of
[`janakhpon/monocr`](https://huggingface.co/janakhpon/monocr), not to `main`, and
cached under `~/.monocr/models/<revision>/`. Each binding ships the matching
charset and refuses to decode if the two disagree: CTC reserves index 0 for the
blank, so a model over N characters must emit N + 1 classes. Without that check
a mismatch returns well-formed Mon text that is wrong, with no error and no
lookup miss.

## Limitations

**The bindings disagree on page output.** Measured 2026-10-01 on the page path —
`read_image` / `predict`, the call the quick starts use — over the seven images
in [`data/images/`](data/images/): Python, JavaScript and Go return identical
text on **1 of 7**. Pairwise, Python matches JavaScript on 1, Python matches Go
on 1, and JavaScript matches Go on 3. Rust matches Python on 3. The widest gap is
`000028.jpg`, where Python and Rust return 154 characters and JavaScript and Go
return 55 and 61.

The cause is two things compounding: four different image-resampling kernels,
two of them the wrong family, and line segmenters that are not configured alike.
Python binarises adaptively where the other three use a flat threshold at 128;
Python sets its gap threshold at 0.02 of the profile's maximum with a smoothing
window of 5, where the other three use 0.05 of the non-zero mean and a window of
3; and Python and Rust cut a line too wide for the model into tiles where
JavaScript and Go squeeze it.

**That is agreement, not accuracy.** These images have no ground truth, and four
implementations reading the same wrong thing would agree perfectly.

[`docs/CROSS_BINDING_PARITY.md`](docs/CROSS_BINDING_PARITY.md) reports 5 of 7.
It measured the line path on pre-cropped files against the previous model, which
skips segmentation entirely, and its header marks the figures stale. Only the
page figure above describes what the documented API does.

## Related

- [Model card](https://huggingface.co/janakhpon/monocr) — ONNX and Core ML
  exports, the held-out evaluation and its caveats.
- [MonDevHub/monocr](https://github.com/MonDevHub/monocr) — the apps built on this
  model: `apps/web` (SvelteKit), `apps/android` (Jetpack Compose) and `apps/ios`
  (SwiftUI). The Android and iOS apps build from source and are not in an app
  store yet. The apps cap a file you open for recognition at 50 MB; this SDK has
  no such limit.
- [Issues](https://github.com/MonDevHub/monocr-onnx/issues) — bugs and
  questions. [`RELEASING.md`](RELEASING.md) describes how a release is cut.

## License

[MIT](LICENSE)
