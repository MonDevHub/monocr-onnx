# MonOCR (Go)

[![Go Reference](https://pkg.go.dev/badge/github.com/MonDevHub/monocr-onnx/go.svg)](https://pkg.go.dev/github.com/MonDevHub/monocr-onnx/go)

On-device OCR for the Mon language (mnw), running on ONNX Runtime. Part of
[monocr-onnx](https://github.com/MonDevHub/monocr-onnx), which also has Python,
JavaScript and Rust bindings.

## Install

```bash
go get github.com/MonDevHub/monocr-onnx/go
```

Requires Go 1.23+, cgo (a C compiler, and `CGO_ENABLED=1`; on Windows a
MinGW-w64 `gcc`) and the ONNX Runtime shared library, 1.18.0 or newer, on the
machine. Unlike the other bindings, this one does not bundle a runtime; see
[ONNX Runtime](#onnx-runtime) below.

## Quick start

```go
package main

import (
    "fmt"

    monocr "github.com/MonDevHub/monocr-onnx/go"
)

func main() {
    // Downloads and caches the model on first use.
    text, err := monocr.ReadImage("document.jpg")
    if err != nil {
        panic(err)
    }
    fmt.Println(text)
}
```

## API

| Call | Does |
| :--- | :--- |
| `ReadImage(imagePath) (string, error)` | Segments the image into lines and reads them top to bottom, joined with newlines. |
| `ReadImages(imagePaths) ([]string, error)` | The same over several images, reusing one session. |
| `ReadPDF(pdfPath) ([]string, error)` / `ReadPDFs(pdfPaths) ([][]string, error)` | Rasterises each page at 300 dpi with poppler's `pdftoppm`, then reads it like an image. One string per page that decodes; a page that fails to decode is skipped. |
| `ReadImageWithModel(imagePath, modelPath, charset) (string, error)` | Runs a model and charset you supply. The charset is trimmed of line terminators only, then checked against the model's classifier width. |
| `ReadImageWithAccuracy(imagePath, groundTruth) (string, float64, error)` | Also returns `100 × (1 − edit distance ÷ length of the longer string)`. |
| `DefaultCharset() string` | The charset compiled into the package. |
| `RuntimeVersion() (string, error)` | The ONNX Runtime version actually loaded. |

The charset's first character is U+0020 — a space is one of the classes the
model emits — so trimming it with `strings.TrimSpace` shifts every index in the
decode by one.

## CLI

```bash
go install github.com/MonDevHub/monocr-onnx/go/cmd/monocr@latest
monocr image page.jpg
monocr pdf document.pdf
monocr batch ./input
monocr download      # pre-fetch the model
monocr runtime       # onnxruntime 1.24.1 (tested against 1.24.1, requires >= 1.18.0)
```

## The model

Weights come from revision `d3d9d5e` (`model.ModelRevision`) of
[janakhpon/monocr](https://huggingface.co/janakhpon/monocr) and are cached under
`~/.monocr/models/<revision>/`, so re-pinning is a cache miss rather than a
silent reuse. The graph takes a `[batch, 1, 160, 1024]` input and emits
`[batch, sequence, 277]` logits: 276 characters plus the CTC blank. Input
height and width are both static; batch is the only dynamic input axis. Images
are read as JPEG or PNG.

The SDK reads the real graph on load and refuses to run when it disagrees with
the charset, because a mismatched pair would still run and return the wrong text
with no error:

```
model contract violation: charset/model mismatch.
  charset: 276 characters -> expects 277 classes (276 + CTC blank)
  model (/…/monocr.onnx): 225 classes
```

## ONNX Runtime

`go.mod` pins `github.com/yalue/onnxruntime_go` v1.11.0, which is only the cgo
wrapper. The runtime itself is a shared library from the host, which no Go
manifest can pin, so the version is stated here and read back at load time.

- **Tested against 1.24.1**, the version the Python and JavaScript lockfiles
  resolve.
- **Minimum 1.18.0.** The wrapper requests C API version 18. ONNX Runtime keeps
  that call backward compatible, so anything from 1.18.0 up answers it and older
  libraries fail to load. Newer libraries work but expose no more than API 18 to
  this binding.

**Choosing the library.** `MONOCR_ONNXRUNTIME_PATH` set to an absolute path
overrides every default. Resolution order is that variable, then the Homebrew
path on Apple-silicon macOS, then the platform loader, which the wrapper asks
for the bare names `onnxruntime.so` (Linux) and `onnxruntime.dll` (Windows). If
the variable is set but no file is there, the SDK fails with that message rather
than quietly loading something else. An initialisation failure names the library
and version it loaded and what was required, or, when nothing loaded, where it
looked.

```bash
MONOCR_ONNXRUNTIME_PATH=/opt/onnxruntime-1.24.1/lib/libonnxruntime.dylib monocr image page.jpg
```

**macOS**

```bash
brew install onnxruntime
```

On Apple silicon this installs `/opt/homebrew/lib/libonnxruntime.dylib`, which
the SDK finds on its own. It is the only platform with a built-in default.
Homebrew on an Intel Mac installs under `/usr/local/lib`, which is not a
default, so set `MONOCR_ONNXRUNTIME_PATH` to the library there.

**Linux**

```bash
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-1.24.1.tgz
tar xzf onnxruntime-linux-x64-1.24.1.tgz
export MONOCR_ONNXRUNTIME_PATH="$PWD/onnxruntime-linux-x64-1.24.1/lib/libonnxruntime.so"
```

Set the variable. `LD_LIBRARY_PATH` or `ldconfig` alone does not work in 0.4.2:
the platform loader is asked for `onnxruntime.so`, and ONNX Runtime packages
ship `libonnxruntime.so`. Distribution packages work the same way; check the
version is at least 1.18.0.

**Windows**

Download `onnxruntime-win-x64-<version>.zip` from the same releases page
(`onnxruntime-win-arm64-…` on ARM), unzip it, and point at the DLL:

```powershell
$env:MONOCR_ONNXRUNTIME_PATH = "C:\onnxruntime\lib\onnxruntime.dll"
```

Copying `onnxruntime.dll` next to your `.exe` also works, because the Windows
loader searches the application's directory first. Adding its `lib` directory to
`PATH` is the weakest option: the loader searches `System32` before `PATH`, and
some Windows installs carry an older `onnxruntime.dll` there that would be
loaded instead.

## PDFs need poppler

`ReadPDF` and `ReadPDFs` need `pdftoppm` from poppler (`brew install poppler`,
`sudo apt-get install poppler-utils`). Images need nothing extra.

## Limitations

Line segmentation binarises with a flat global threshold at 128, where the
Python binding thresholds adaptively, and a line too wide for the model is
squeezed rather than cut into tiles. Python differs on both, and page output
differs between bindings; the
[root README](https://github.com/MonDevHub/monocr-onnx#limitations) has the
measurement. No accuracy figure is claimed here; see the
[model card](https://huggingface.co/janakhpon/monocr).

## License

MIT
