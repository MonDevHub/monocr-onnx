# MonOCR (Go SDK)

[![Go Reference](https://pkg.go.dev/badge/github.com/MonDevHub/monocr-onnx/go.svg)](https://pkg.go.dev/github.com/MonDevHub/monocr-onnx/go)

The official Go SDK for Mon language OCR, powered by ONNX Runtime.

## Installation

```bash
go get github.com/MonDevHub/monocr-onnx/go
```

## The model

The SDK downloads its ONNX weights from
[janakhpon/monocr](https://huggingface.co/janakhpon/monocr), pinned to revision
`d3d9d5e` (`model.ModelRevision`). That artifact takes a
`[batch, 1, 160, 1024]` input and emits `[batch, sequence, 277]` logits: 276
characters plus the CTC blank. **Height and width are both static**; batch is
the only dynamic axis. This line previously read `[1, 1, 160, width]`, which had
both halves backwards.

The charset, the input height and the classifier width are one contract. If they
drift apart the model still runs and still returns text — it is just the wrong
text, with no error anywhere. So the SDK reads the real graph on load and
refuses to run when it disagrees with the charset it holds:

```
model contract violation: charset/model mismatch.
  charset: 276 characters -> expects 277 classes (276 + CTC blank)
  model (/…/monocr.onnx): 225 classes
```

Downloads are cached per revision under `~/.monocr/models/<revision>/`, so
re-pinning is a cache miss rather than a silent reuse of the previous artifact.

## Quick Start

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

## API Reference

### `monocr.ReadImage(imagePath string) (string, error)`

Recognize a single-line image. Downloads the model on first use.

### `monocr.ReadImages(imagePaths []string) ([]string, error)`

Same, over several images, reusing one session.

### `monocr.ReadPDF(pdfPath string) ([]string, error)` / `monocr.ReadPDFs(pdfPaths []string) ([][]string, error)`

Rasterize a PDF at 300 dpi, segment each page into lines, and recognize each
line. Requires `pdftoppm` from poppler-utils.

### `monocr.ReadImageWithModel(imagePath, modelPath, charset string) (string, error)`

Run against a model and charset you supply. The charset is trimmed of line
terminators only and then checked against the model's classifier width.

### `monocr.ReadImageWithAccuracy(imagePath, groundTruth string) (string, float64, error)`

Recognize and score against ground truth: `(1 - CER) * 100`.

### `monocr.DefaultCharset() string`

The charset compiled into the package, normalized. Its first character is
U+0020 — a space is one of the classes the model emits, so trimming it with
`strings.TrimSpace` shifts every index in the decode by one.

## ONNX Runtime

The SDK needs the ONNX Runtime **shared library** at run time. `go.mod` pins
`github.com/yalue/onnxruntime_go`, but that is the cgo wrapper. The runtime
itself comes from the host, and no Go manifest can pin it. So the version is
stated here and read back at load time instead.

| | version |
|---|---|
| Developed and tested against | **1.24.1** — same version the Python and JS bindings pin |
| Minimum | **1.18.0** |
| C API version requested | 18 (`ORT_API_VERSION` vendored by `onnxruntime_go` v1.11.0) |

The wrapper asks the library for C API version 18. ONNX Runtime keeps that call
backward compatible, so anything from 1.18.0 up answers it; older libraries
return nothing and the load fails. Newer libraries work but expose no more than
API 18 to this binding.

### Installing it

- **macOS**: `brew install onnxruntime` — installs to
  `/opt/homebrew/lib/libonnxruntime.dylib`, which the SDK finds on its own.
- **Linux**: download the release matching the table above and put it on
  `LD_LIBRARY_PATH`.

### Choosing a specific library

Set `MONOCR_ONNXRUNTIME_PATH` to an absolute path to override every default:

```bash
MONOCR_ONNXRUNTIME_PATH=/opt/onnxruntime-1.24.1/lib/libonnxruntime.dylib monocr image page.jpg
```

Resolution order is: `MONOCR_ONNXRUNTIME_PATH`, then the Homebrew path on macOS,
then the platform loader. If the variable is set but no file is there, the SDK
fails with that message rather than quietly loading something else — the point
of setting it is to choose, and a silent substitution defeats that.

### Knowing what actually ran

The loaded version is recorded at initialisation and readable, so a result can
name the runtime that produced it:

```bash
$ monocr runtime
onnxruntime 1.24.1 (tested against 1.24.1, requires >= 1.18.0)
```

```go
version, err := monocr.RuntimeVersion()
```

An initialisation failure reports the same detail instead of a bare error code:
which library was loaded, from where, and what was required.

### PDF support

`ReadPDF` also needs `pdftoppm` from poppler-utils.

## Maintenance

Maintained by [MonDevHub](https://github.com/MonDevHub).

## License

MIT
