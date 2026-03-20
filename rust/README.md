# MonOCR (Rust SDK)

A high-performance Rust SDK for the Mon language OCR, powered by ONNX Runtime.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
monocr-onnx = { git = "https://github.com/MonDevHub/monocr-onnx", branch = "main" }
```

## Features

- **Production Accuracy**: Fully aligned with the latest v2.0 high-precision models.
- **Auto-Model Management**: Automatically downloads and caches the required ONNX weights to `~/.monocr/models/`.
- **Memory Efficient**: Leverages `ndarray` for zero-copy tensor operations.
- **Thread Safe**: The core engine is designed for concurrent use.

## Quick Start

```rust
use monocr_onnx::MonOCR;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // 1. Initialize the engine
    let model_path = ".monocr/models/monocr.onnx";
    let charset_path = "charset.txt";
    let ocr = MonOCR::new(model_path, charset_path)?;

    // 2. Perform OCR on an image
    let img_path = Path::new("test_image.jpg");
    let result = ocr.predict(img_path)?;

    println!("Recognized Text: {}", result);
    Ok(())
}
```

## Prerequisites

This crate requires the ONNX Runtime shared library to be available on your system.
- **macOS**: `brew install onnxruntime`
- **Linux**: Download `libonnxruntime.so` and add to `LD_LIBRARY_PATH`.

## License

MIT
