# MonOCR (Go)

[![Go Reference](https://pkg.go.dev/badge/github.com/MonDevHub/monocr-onnx/go.svg)](https://pkg.go.dev/github.com/MonDevHub/monocr-onnx/go)

On-device OCR for the Mon language (mnw), running on ONNX Runtime. Part of
[monocr-onnx](https://github.com/MonDevHub/monocr-onnx), which also has Python,
JavaScript and Rust bindings.

## Install

```bash
go get github.com/MonDevHub/monocr-onnx/go
```

Requires Go 1.23+ and the ONNX Runtime shared library, 1.18.0 or newer, on the
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
| `ReadPDF(pdfPath) ([]string, error)` / `ReadPDFs(pdfPaths) ([][]string, error)` | Rasterises each page at 300 dpi with poppler's `pdftoppm`, then reads it like an image. One string per page. |
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
`[batch, sequence, 277]` logits: 276 characters plus the CTC blank. Height and
width are both static; batch is the only dynamic axis.

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

- **Tested against 1.24.1**, the version the Python and JavaScript bindings pin.
- **Minimum 1.18.0.** The wrapper requests C API version 18. ONNX Runtime keeps
  that call backward compatible, so anything from 1.18.0 up answers it and older
  libraries fail to load. Newer libraries work but expose no more than API 18 to
  this binding.

**macOS**

```bash
brew install onnxruntime
```

This installs `/opt/homebrew/lib/libonnxruntime.dylib` on Apple silicon, which
the SDK finds on its own. It is the only platform with a built-in default.

**Linux**

```bash
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-1.24.1.tgz
tar xzf onnxruntime-linux-x64-1.24.1.tgz
export LD_LIBRARY_PATH="$PWD/onnxruntime-linux-x64-1.24.1/lib:$LD_LIBRARY_PATH"
```

`libonnxruntime.so` has to be somewhere the platform loader already looks:
`LD_LIBRARY_PATH`, or a directory registered with `ldconfig`. Distribution
packages work too where they exist; check the version is at least 1.18.0.

**Windows**

Download `onnxruntime-win-x64-<version>.zip` from the same releases page
(`onnxruntime-win-arm64-…` on ARM), unzip it, and either add its `lib`
directory to `PATH` — the Windows loader searches `PATH`, not
`LD_LIBRARY_PATH` — or point at the DLL directly:

```powershell
$env:MONOCR_ONNXRUNTIME_PATH = "C:\onnxruntime\lib\onnxruntime.dll"
```

**Choosing a specific library.** `MONOCR_ONNXRUNTIME_PATH` set to an absolute
path overrides every default:

```bash
MONOCR_ONNXRUNTIME_PATH=/opt/onnxruntime-1.24.1/lib/libonnxruntime.dylib monocr image page.jpg
# .so on Linux, onnxruntime.dll on Windows
```

Resolution order is `MONOCR_ONNXRUNTIME_PATH`, then the Homebrew path on macOS,
then the platform loader. If the variable is set but no file is there, the SDK
fails with that message rather than quietly loading something else. An
initialisation failure names which library was loaded, from where, and what was
required.

## PDFs need poppler

`ReadPDF` and `ReadPDFs` need `pdftoppm` from poppler (`brew install poppler`,
`sudo apt-get install poppler-utils`). Images need nothing extra.

## Limitations

Line segmentation binarises with a flat global threshold at 128, where the
Python binding thresholds adaptively, and a line too wide for the model is
squeezed rather than cut into tiles. Page output therefore differs from the
other bindings; the
[root README](https://github.com/MonDevHub/monocr-onnx#limitations) has the
measurement. No accuracy figure is claimed here; see the
[model card](https://huggingface.co/janakhpon/monocr).

## License

MIT
