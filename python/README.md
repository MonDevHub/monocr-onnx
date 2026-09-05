# MonOCR (Python SDK)

[![PyPI](https://img.shields.io/pypi/v/monocr-onnx.svg)](https://pypi.org/project/monocr-onnx/)

The official Python SDK for Mon language OCR, powered by ONNX Runtime. Optimized for high-throughput batch processing and production server environments.

## Installation

```bash
pip install "monocr-onnx>=0.4.0"
```

## Features

- **Pinned to one network**: the v3.5 recogniser at revision `d3d9d5e`, with 160px input
  height, 276 characters, 277 CTC classes. No accuracy figure is claimed here; see
  the [model card](https://huggingface.co/janakhpon/monocr) for the held-out result
  and its caveats.
- **Parallel Processing**: Native support for multithreaded batch OCR.
- **Pinned Model**: Weights and charset are fetched from one immutable [Hugging Face](https://huggingface.co/janakhpon/monocr) revision, checksummed, and cached per revision.
- **Images and PDFs**: images through `MonOCR`, PDFs through `read_pdf`. These are
  different calls, not one polymorphic one — see the API reference below.
- **Line segmentation**: adaptive thresholding, with padding relative to each line's height.

## What to know before you start

- **First run downloads the model.** Roughly 46 MB, fetched from the pinned
  Hugging Face revision and cached per revision. Nothing works offline until that
  has happened once.
- **PDFs need poppler on the PATH**, on every platform. `read_pdf` goes through
  `pdf2image`, which shells out to `pdftoppm`. Without it the call raises a
  `RuntimeError` naming poppler; it does not fail quietly. Images need nothing
  extra. See [Platforms](#platforms) below.
- **`MonOCR.predict` does not accept a PDF.** It opens the path as an image and
  raises `PIL.UnidentifiedImageError` on a PDF. Use `read_pdf`.
- **The CLI is `monocr-onnx`, not `monocr`.** It was `monocr` up to 0.3.2, which
  collided with the command installed by the separate `monocr` package; in an
  environment holding both, install order decided which one you got.
- **Throughput.** On an Apple M5, a typeset page is about 2 s and a 10-page
  scanned PDF about 78 s, model already cached. CPU only; no GPU path here.
- **No accuracy figure is claimed by this package.** The published numbers are
  validation figures measured on rendered lines — see the model card.

## Platforms

The wheel is pure Python (`py3-none-any`) and every native dependency —
`onnxruntime`, `opencv-python-headless`, `numpy`, `Pillow` — publishes wheels for
Linux, macOS and Windows, so `pip install` needs no compiler on any of them.

Only poppler is manual, and only for PDFs.

**macOS**

```bash
brew install poppler
```

**Linux**

```bash
sudo apt-get install poppler-utils   # Debian, Ubuntu
```

Other distributions package the same binaries, usually as `poppler-utils` or
`poppler`.

**Windows**

Not shipped with Windows and there is no single official installer. Any of these
works:

```powershell
scoop install poppler
choco install poppler
conda install -c conda-forge poppler
```

Or take the prebuilt binaries from
[oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases),
unzip, and add the `Library\bin` directory to `PATH` — that release is what
`pdf2image`'s own documentation points Windows users at. `pdf2image` also accepts
a `poppler_path` argument, but `read_pdf` does not forward one, so `PATH` is the
route here.

Check with `pdfinfo -v` in a new shell before calling `read_pdf`.

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

### `read_pdf(pdf_path, model_path=None, charset_path=None)` -> `list[str]`

Module-level, not a method. Renders every page through poppler and returns one
string per page. This is the only PDF entry point.

```python
from monocr_onnx import read_pdf

pages = read_pdf("book.pdf")
print(len(pages), "pages")
print(pages[0])
```

### `read_pdfs(paths, ...)` -> `list[list[str]]`

The same over several files.

## CLI Usage

```bash
# Recognize an image
monocr-onnx image input.jpg

# Process a PDF
monocr-onnx pdf document.pdf

# Batch directory processing
monocr-onnx batch ./input

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
