# MonOCR (Go SDK)

The official Go SDK for Mon language OCR, powered by ONNX Runtime. Optimized for performance and native Go integration.

## Installation

```bash
go get github.com/MonDevHub/monocr-onnx/go
```

## Features

- **Production Accuracy**: Aligned with v2.0 high-precision models (128px vertical resolution).
- **Bundled Charset**: Integrated character mapping for zero-config deployments.
- **Auto-Caching**: Intelligent model download and management via [Hugging Face](https://huggingface.co/janakhpon/monocr).
- **Native Efficiency**: Direct bindings to ONNX Runtime via CGO.
- **Robust Segmentation**: Intelligent line-detection with adaptive thresholding and relative padding.

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/MonDevHub/monocr-onnx/go"
)

func main() {
    // Recognize text from a full page
    text, err := monocr.Predict("document.jpg")
    if err != nil {
        panic(err)
    }
    fmt.Println(text)
}
```

## API Reference

### `monocr.Predict(imagePath string)` -> `(string, error)`

Primary entry point for page-level OCR. Automatically handles segmentation and recognition.

### `monocr.PredictLine(img image.Image)` -> `(string, error)`

Recognize text from a single cropped text line.

## Prerequisites

The Go SDK requires the ONNX Runtime shared library (`libonnxruntime.so` or equivalent) to be present in the system's library path. 
- **macOS**: `brew install onnxruntime`
- **Linux**: Download and add to `LD_LIBRARY_PATH`.

## Maintenance

Maintained by [MonDevHub](https://github.com/MonDevHub).

## License

MIT
