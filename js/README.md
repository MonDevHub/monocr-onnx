# MonOCR (JavaScript SDK)

The official JavaScript SDK for Mon language OCR, powered by ONNX Runtime. Designed for high-performance server-side and desktop Node.js applications.

## Installation

> **This command does not resolve yet, and that is deliberate.** 0.3.0 is built and
> tagged but not published; the bound below fails loudly rather than installing
> 0.1.5, which targets a superseded model (64px input height, a 225-character
> charset against a 277-class graph) and returns wrong characters rather than
> merely worse ones. Until the publish step runs, install from source:
> `npm install github:MonDevHub/monocr-onnx#main --prefix js`.
> See `CHANGELOG.md` and `RELEASING.md`.

```bash
npm install monocr@^0.3.0
```

## Features

- **Pinned to one network**: the v3.5 recogniser at revision `d3d9d5e`, with 160px input
  height, 276 characters, 277 CTC classes. No accuracy figure is claimed here; see
  the [model card](https://huggingface.co/janakhpon/monocr) for the held-out result
  and its caveats.
- **Pinned Model**: Weights and charset are fetched from one immutable Hugging Face revision, so two installs of the same version decode with the same network.
- **Fails Closed**: A model whose class count or input height disagrees with the charset is refused at load rather than decoded into wrong text.
- **Line segmentation**: horizontal projection profile over a **flat global
  threshold at 128**, with padding relative to each line's height. Not adaptive —
  `src/segmenter.js:58`. The reference implementation in `mon_OCR` does threshold
  adaptively; this binding does not, and `src/segmenter.js:10-15` records that the
  projection profile here is untested.

## Quick Start

```javascript
const { MonOCR } = require("monocr");

async function main() {
  const engine = new MonOCR();
  await engine.init();

  // Recognize a full page: one entry per detected line
  const lines = await engine.predictPage("scanned_text.png");
  console.log(lines.map((l) => l.text).join("\n"));
}

main();
```

Or the one-shot helpers:

```javascript
const { read_image, read_pdf } = require("monocr");

const text = await read_image("scanned_text.png");
const pages = await read_pdf("document.pdf"); // needs poppler-utils
```

## API Reference

### `new MonOCR(modelPath?, charsetPath?)`

Initialize the OCR engine. Both arguments are positional and optional.

- `modelPath`: path to a local ONNX model. Omit it to download the pinned model on first use.
- `charsetPath`: path to a local charset file. Omit it to use the charset that shipped with the model.

### `init()` -> `Promise<void>`

Load the model and charset, and verify they describe the same network. Called automatically by the predict methods.

### `predictPage(imagePath)` -> `Promise<Array<{text: string, bbox: object}>>`

Segment an image into lines and recognize each.

### `predictLine(imageSource)` -> `Promise<string>`

Recognize text from a single cropped text line image.

### `read_image` / `read_images` / `read_pdf` / `read_pdfs` / `read_image_with_accuracy`

Convenience wrappers returning plain strings.

## The model contract

The charset, the model's input height and its classifier width are one contract. If they drift apart the model still runs and still returns text — it is just the wrong text, with no error anywhere.

`init()` reads the class count and input height off the loaded session and compares them to the charset. On a mismatch it throws `ModelContractError` instead of decoding:

```javascript
const { MonOCR, ModelContractError } = require("monocr");

try {
  await new MonOCR("./my-model.onnx").init();
} catch (err) {
  if (err instanceof ModelContractError) {
    // This model does not match the bundled charset. Supply the charset it was
    // trained with, rather than decoding with the wrong vocabulary.
  }
}
```

Models are downloaded from a pinned revision of [`janakhpon/monocr`](https://huggingface.co/janakhpon/monocr), exported as `MODEL_REVISION`, and cached under `~/.monocr/models/<revision>/`. Bumping the revision is a cache miss, not a silent swap.

## CLI Interface

```bash
# Global installation for CLI usage
npm install -g monocr

# Process an image
monocr image input.jpg

# Process a PDF
monocr pdf document.pdf

# Pre-fetch the model into the cache
monocr download
```

## Development

```bash
npm install
npm test     # offline: no model download, no network
```

## Requirements

- Node.js 18.17+
- sharp (for image processing)
- onnxruntime-node
- poppler-utils, for the PDF entry points only

## Maintenance

Maintained by [MonDevHub](https://github.com/MonDevHub).

## License

MIT
