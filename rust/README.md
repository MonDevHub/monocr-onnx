# MonOCR (Rust SDK)

A Rust SDK for Mon language OCR, powered by ONNX Runtime.

Mon (`mnw`) is a Mon-Khmer language of Myanmar and Thailand, written in a
Myanmar-script orthography. It is unrelated to Mongolian.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
monocr-onnx = { git = "https://github.com/MonDevHub/monocr-onnx", branch = "main" }
tokio = { version = "1", features = ["full"] }
```

## The model

Weights are downloaded from
[janakhpon/monocr](https://huggingface.co/janakhpon/monocr), pinned to revision
`d3d9d5e` (`model_manager::MODEL_REVISION`). That artifact takes a
`[batch, 1, 160, 1024]` input and emits `[batch, sequence, 277]` logits: 276
characters plus the CTC blank. **Height and width are both static**; batch is
the only dynamic axis. This line previously read `[1, 1, 160, width]`, which had
both halves backwards.

The charset, the input height and the classifier width are one contract. If they
drift apart the model still runs and still returns text — it is just the wrong
text, with no error anywhere. So the SDK reads the real graph on load and returns
a `ModelContractError` when it disagrees with the charset it holds:

```
model contract violation: charset/model mismatch.
  charset: 276 characters -> expects 277 classes (276 + CTC blank)
  model (/…/monocr.onnx): 225 classes
```

Downloads are cached per revision under `~/.monocr/models/<revision>/`, so
re-pinning is a cache miss rather than a silent reuse of the previous artifact.

## Features

- **Auto-Model Management**: Downloads and caches the pinned ONNX weights and
  their charset to `~/.monocr/models/<revision>/`.
- **Fail-closed loading**: Refuses to run a model whose input height or class
  count disagrees with the charset, instead of returning the wrong text.
- **Memory Efficient**: Uses `ndarray` for tensor construction.
- **Line segmentation**: Horizontal projection profile over a **flat global
  threshold at 128** for full-page OCR (`src/segmenter.rs:288`). Not adaptive —
  the crate's own docs at `src/segmenter.rs:265` state it correctly.

## Quick Start

```rust,no_run
use monocr_onnx::MonOcr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads and caches the model on first use.
    let mut ocr = MonOcr::builder().build().await?;

    let text = ocr.read_image("test_image.jpg").await?;
    println!("Recognized Text: {text}");
    Ok(())
}
```

Or the one-shot free functions, which build a `MonOcr` for you:

```rust,no_run
use monocr_onnx::{read_image, read_images, read_pdf};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = read_image("page.png").await?;
    let batch = read_images(&["a.png", "b.png"]).await?;
    let pages = read_pdf("document.pdf").await?; // needs poppler-utils
    println!("{} {} {}", text.len(), batch.len(), pages.len());
    Ok(())
}
```

## Using your own model

```rust,no_run
use monocr_onnx::MonOcr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let charset = std::fs::read_to_string("charset.txt")?;
    let mut ocr = MonOcr::builder()
        .model_path("./models/monocr.onnx")
        .charset(charset)
        .build()
        .await?;

    println!("{}", ocr.read_image("page.png").await?);
    Ok(())
}
```

The charset is stripped of line terminators only. Its first character is
U+0020 — a space is one of the classes the model emits, so trimming it with
`.trim()` shifts every index in the decode by one.

## Prerequisites

This crate requires the ONNX Runtime shared library to be available on your system.
- **macOS**: `brew install onnxruntime`
- **Linux**: Download `libonnxruntime.so` and add to `LD_LIBRARY_PATH`.

## License

MIT
