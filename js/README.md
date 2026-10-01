# MonOCR (JavaScript)

[![npm](https://img.shields.io/npm/v/monocr.svg)](https://www.npmjs.com/package/monocr)

On-device OCR for the Mon language (mnw) in Node.js, running on ONNX Runtime.
Part of [monocr-onnx](https://github.com/MonDevHub/monocr-onnx), which also has
Python, Go and Rust bindings.

> [!IMPORTANT]
> **Upgrade from anything before 0.4.0.** Every earlier npm release returned
> noise rather than text: preprocessing read a three-channel buffer as if it were
> one channel. On a typeset page `monocr@0.3.2` returned 168 characters of
> garbage where 0.4.0 returns 1,178 of Mon.

## Install

```bash
npm install monocr@^0.4.1
```

Requires Node.js 20.9+. Runs on macOS, Linux and Windows on the CPU;
`onnxruntime-node` and `sharp` ship prebuilt binaries for all three, so no
compiler is needed. On 0.x npm's caret stops below the next minor, so `^0.4.1`
takes 0.4.x releases only.

## Quick start

```javascript
const { MonOCR, read_image, read_pdf } = require("monocr");

async function main() {
  // One-shot helpers. The first call downloads the pinned model and caches it.
  console.log(await read_image("page.png"));
  const pages = await read_pdf("document.pdf"); // one string per page; needs poppler

  // Or keep one engine for many calls.
  const engine = new MonOCR();
  await engine.init();
  const lines = await engine.predictPage("page.png"); // [{ text, bbox }, ...]
  console.log(lines.map((l) => l.text).join("\n"));
  console.log(await engine.predictLine("line.png")); // a crop of one line
}

main();
```

## API

| Call | Returns |
| :--- | :--- |
| `new MonOCR(modelPath?, charsetPath?)` | The engine. Omit both to use the pinned model and its charset. |
| `init()` | Loads the model and checks it against the charset. The predict methods call it for you. |
| `predictPage(imagePath)` | `Array<{ text, bbox }>`, one entry per detected line |
| `predictLine(imageSource)` | `string`, for a crop of one line |
| `read_image` / `read_images` | `string` / `string[]` |
| `read_pdf` / `read_pdfs` | `string[]` per PDF, one per page |
| `read_image_with_accuracy(path, groundTruth)` | `{ text, accuracy }`, accuracy as a percentage |

The `read_*` helpers also take `modelPath` and `charsetPath` after their first
argument.

`init()` throws `ModelContractError` when the model's class count or input
height disagrees with the charset. A mismatched pair would still run and still
return text; it would just be the wrong text.

```javascript
const { MonOCR, ModelContractError } = require("monocr");

async function load(modelPath, charsetPath) {
  try {
    const engine = new MonOCR(modelPath, charsetPath);
    await engine.init();
    return engine;
  } catch (err) {
    if (err instanceof ModelContractError) {
      // Supply the charset this model was trained with.
    }
    throw err;
  }
}
```

Models come from revision `MODEL_REVISION` (`d3d9d5e`) of
[janakhpon/monocr](https://huggingface.co/janakhpon/monocr) and are cached under
`~/.monocr/models/<revision>/`, so a new pin is a cache miss, not a silent swap.

## CLI

```bash
npm install -g monocr
monocr image input.jpg
monocr pdf document.pdf
monocr batch ./input
monocr download      # pre-fetch the model
```

## PDFs need poppler

`read_pdf` and `read_pdfs` shell out to poppler's `pdftoppm`. Images need
nothing extra.

```bash
brew install poppler                 # macOS
sudo apt-get install poppler-utils   # Debian, Ubuntu
```

On Windows: `scoop install poppler`, `choco install poppler`,
`conda install -c conda-forge poppler`, or the prebuilt binaries from
[oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
with `Library\bin` added to `PATH`. Confirm with `pdfinfo -v` in a new shell.

## Limitations

Line segmentation binarises with a flat global threshold at 128, where the
Python binding thresholds adaptively, and a line too wide for the model is
squeezed rather than cut into tiles. Page output therefore differs from the
other bindings; the
[root README](https://github.com/MonDevHub/monocr-onnx#limitations) has the
measurement. No accuracy figure is claimed here; see the
[model card](https://huggingface.co/janakhpon/monocr).

## Development

```bash
npm install
npm test     # offline: no model download, no network
```

## License

MIT
