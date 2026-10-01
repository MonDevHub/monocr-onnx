# MonOCR (Python)

[![PyPI](https://img.shields.io/pypi/v/monocr-onnx.svg)](https://pypi.org/project/monocr-onnx/)

On-device OCR for the Mon language (mnw), running on ONNX Runtime. Part of
[monocr-onnx](https://github.com/MonDevHub/monocr-onnx), which also has
JavaScript, Go and Rust bindings.

## Install

```bash
pip install "monocr-onnx>=0.4.1"
```

Requires Python 3.11+: onnxruntime 1.24.1 ships no wheel below cp311 and no
sdist. Keep the floor; 0.1.x pairs a 225-character charset with a 277-class graph
and returns wrong characters.

The wheel is pure Python and every native dependency publishes wheels for Linux,
macOS and Windows, so no compiler is needed. It runs on the CPU only.

## Quick start

```python
from monocr_onnx import MonOCR, read_pdf

# Downloads the pinned model on first run and caches it by revision.
engine = MonOCR()

# A page: segmented into lines, joined with newlines.
print(engine.predict("page.png"))

# A line you have already cropped: read whole, never split.
print(engine.predict_line("line.png"))

# A PDF: one string per page. Needs poppler (below).
pages = read_pdf("book.pdf")
```

`predict` is an alias for `predict_page`. It does not accept a PDF: it opens the
path as an image and raises `PIL.UnidentifiedImageError`. Use `read_pdf`.

## API

| Call | Returns |
| :--- | :--- |
| `MonOCR(model_path=None, charset_path=None)` | The engine. Omit both paths to use the pinned model and its charset. |
| `.predict(path)` / `.predict_page(path)` | `str`, one line of text per detected line |
| `.predict_line(image)` | `str`, for a path or PIL image of one line |
| `read_image(path)` / `read_images(paths, workers=4)` | `str` / `list[str]` |
| `read_pdf(path)` / `read_pdfs(paths, workers=4)` | `list[str]` per PDF, one per page |
| `read_image_with_accuracy(path, ground_truth)` | `(str, float)`: the text, and `100 × (1 − edit distance ÷ length of the longer string)` |

The module-level functions also take `model_path` and `charset_path`.

Loading refuses a model whose output class count or input height disagrees with
the charset, raising `ModelContractError`. A mismatched pair would still run and
still return text; it would just be the wrong text.

## CLI

```bash
monocr-onnx image input.jpg
monocr-onnx pdf document.pdf
monocr-onnx batch ./input
monocr-onnx download        # pre-fetch the model and charset
```

The command is `monocr-onnx`, not `monocr`. It was `monocr` up to 0.3.2, which
collided with the command installed by the separate
[`monocr`](https://pypi.org/project/monocr/) package.

## PDFs need poppler

`read_pdf` goes through `pdf2image`, which shells out to poppler's `pdftoppm`.
Without it the call raises a `RuntimeError` naming poppler.

```bash
brew install poppler                 # macOS
sudo apt-get install poppler-utils   # Debian, Ubuntu
```

On Windows: `scoop install poppler`, `choco install poppler`,
`conda install -c conda-forge poppler`, or the prebuilt binaries from
[oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
with `Library\bin` added to `PATH`. `read_pdf` does not forward pdf2image's
`poppler_path`, so `PATH` is the route. Check with `pdfinfo -v` in a new shell.

## What to expect

- **First run downloads the model**, 46.2 MB, from revision `d3d9d5e` of
  [janakhpon/monocr](https://huggingface.co/janakhpon/monocr), never from `main`.
  Both files are checked by sha256 and cached under
  `~/.monocr/models/<revision>/`, so a new pin is a cache miss rather than a
  silent reuse of old weights. Nothing works offline until that has happened
  once.
- **Speed.** On an Apple M5 with the model cached, a typeset page takes about
  2 s.
- **No accuracy figure is claimed by this package.** The model card reports a
  held-out CER of 0.0100 on 150 unseen rendered lines in a typeface the model
  never trained on, with a 95% interval of [0.0056, 0.0147]. Those lines come
  from the same synthetic generator as the training data, so nothing on the card
  is measured on photographed pages. Read its caveats before quoting the number.
- **Bindings disagree.** The four bindings do not yet return identical text for
  the same page; see the
  [root README](https://github.com/MonDevHub/monocr-onnx#limitations).
- **Upgrading from 0.1.0:** a stale `~/.monocr/models/monocr.onnx` may still be
  on disk. Nothing reads it; `monocr-onnx download` points it out and it is safe
  to delete.

## License

MIT
