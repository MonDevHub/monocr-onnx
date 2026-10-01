# MonOCR (Rust)

[![crates.io](https://img.shields.io/crates/v/monocr.svg)](https://crates.io/crates/monocr)

On-device OCR for the Mon language (mnw), running on ONNX Runtime. Part of
[monocr-onnx](https://github.com/MonDevHub/monocr-onnx), which also has Python,
JavaScript and Go bindings.

## Install

```toml
[dependencies]
monocr = "0.4"
tokio = { version = "1", features = ["full"] }
```

The crate is `monocr`; the library it exposes is `monocr_onnx`, so imports read
`use monocr_onnx::MonOcr`.

Builds and runs on macOS, Linux and Windows, on the CPU. ONNX Runtime does not
need installing: `ort` fetches a prebuilt runtime for the target at build time
and links it in.

## Quick start

```rust,no_run
use monocr_onnx::MonOcr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads and caches the model on first use.
    let mut ocr = MonOcr::builder().build().await?;

    let text = ocr.read_image("page.png").await?; // lines joined with newlines
    println!("{text}");
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
    let pages = read_pdf("document.pdf").await?; // needs poppler, one string per page
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
U+0020 — a space is one of the classes the model emits — so trimming it with
`.trim()` shifts every index in the decode by one.

## The model

Weights come from revision `d3d9d5e` (`model_manager::MODEL_REVISION`) of
[janakhpon/monocr](https://huggingface.co/janakhpon/monocr) and are cached under
`~/.monocr/models/<revision>/`, so re-pinning is a cache miss rather than a
silent reuse. The graph takes a `[batch, 1, 160, 1024]` input and emits
`[batch, sequence, 277]` logits: 276 characters plus the CTC blank. Height and
width are both static; batch is the only dynamic axis.

The SDK reads the real graph on load and returns a `ModelContractError` when it
disagrees with the charset, because a mismatched pair would still run and return
the wrong text with no error:

```
model contract violation: charset/model mismatch.
  charset: 276 characters -> expects 277 classes (276 + CTC blank)
  model (/…/monocr.onnx): 225 classes
```

## PDFs need poppler

`read_pdf` shells out to poppler's `pdftoppm`. Images need nothing extra.

```bash
brew install poppler                 # macOS
sudo apt-get install poppler-utils   # Debian, Ubuntu
```

On Windows: `scoop install poppler`, `choco install poppler`,
`conda install -c conda-forge poppler`, or the prebuilt binaries from
[oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
with `Library\bin` added to `PATH`. `pdfinfo -v` in a new shell is the check.

## Limitations

Line segmentation binarises with a flat global threshold at 128, where the
Python binding thresholds adaptively. Page output therefore differs between
bindings; the
[root README](https://github.com/MonDevHub/monocr-onnx#limitations) has the
measurement. No accuracy figure is claimed here; see the
[model card](https://huggingface.co/janakhpon/monocr).

## License

MIT
