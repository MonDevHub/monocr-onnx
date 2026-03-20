# MonOCR (JavaScript SDK)

The official JavaScript SDK for Mon language OCR, powered by ONNX Runtime. Designed for high-performance server-side and desktop Node.js applications.

## Installation

```bash
npm install monocr
```

## Features

- **Production Accuracy**: Aligned with v2.0 high-precision models (128px vertical resolution).
- **Auto-Model Management**: Automated model delivery from [Hugging Face](https://huggingface.co/janakhpon/monocr).
- **Robust Segmentation**: Intelligent line-detection with adaptive thresholding and relative padding.
- **Optimized Performance**: Direct bindings to ONNX Runtime via Node.js.

## Quick Start

```javascript
const { MonOCR } = require("monocr");

async function main() {
  const engine = new MonOCR();
  await engine.init();

  // Recognize text from a full page
  const text = await engine.predict("scanned_text.png");
  console.log(text);
}

main();
```

## API Reference

### `new MonOCR(options)`

Initialize the OCR engine.
- `options.modelPath`: Optional path to a local ONNX model.
- `options.charsetPath`: Optional path to a local charset file.

### `predict(imagePath)` -> `Promise<string>`

Segment an image into lines and recognize each.

### `predictLine(imagePath)` -> `Promise<string>`

Recognize text from a single cropped text line image.

## CLI Interface

```bash
# Global installation for CLI usage
npm install -g monocr

# Process an image
monocr image input.jpg

# Process a PDF
monocr pdf document.pdf
```

## Requirements

- Node.js 16+
- sharp (for image processing)
- onnxruntime-node

## Maintenance

Maintained by [MonDevHub](https://github.com/MonDevHub).

## License

MIT
