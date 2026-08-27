// onnxruntime-node and sharp are native modules: loading them costs a dlopen
// and, on a fresh install, a postinstall download. They are required lazily so
// that importing this package -- and running its test suite, which uses a fake
// session and never touches either -- does not depend on them being built.
// CI installs no native deps for the js job as a result.
let ort = null;
function onnxRuntime() {
    if (ort === null) ort = require('onnxruntime-node');
    return ort;
}

let sharp = null;
function imaging() {
    if (sharp === null) sharp = require('sharp');
    return sharp;
}

const fs = require('fs');
const path = require('path');
const LineSegmenter = require('./segmenter');
const ModelManager = require('./model-manager');

/**
 * Raised when a model artifact and the charset used to decode it disagree.
 *
 * The charset, the input height and the classifier width are one contract. If
 * they drift apart the model still runs and still returns text -- it is just
 * the wrong text, with no error anywhere. Fail closed instead.
 *
 * This is not hypothetical here. monocr 0.1.5 shipped a 225-character charset
 * against a 316-class model: every index above 224 fell off the end of the
 * lookup and was swallowed by an `|| ""`, and every index below it resolved to
 * the wrong glyph, because the two charsets share only a 95-character ASCII
 * prefix before one continues into Latin-1 and the other jumps to Myanmar.
 */
class ModelContractError extends Error {
    constructor(message) {
        super(message);
        this.name = 'ModelContractError';
    }
}

/**
 * Read the shape of a session's first input or output.
 *
 * onnxruntime-node reports metadata as an array of
 * `{ name, isTensor, type, shape }`. Dimensions are numbers when static and
 * strings (the symbolic name, e.g. "width") when dynamic.
 */
function firstTensorShape(metadata, role) {
    const entry = Array.isArray(metadata) ? metadata[0] : undefined;
    if (!entry || !Array.isArray(entry.shape)) {
        throw new ModelContractError(
            `Cannot read the ${role} shape from this session, so the model ` +
            'contract cannot be checked. Refusing to decode with an unverified model.'
        );
    }
    return entry.shape;
}

/**
 * Split a charset into characters by codepoint.
 *
 * `String.prototype.length` counts UTF-16 units, so a single astral character
 * would count as two and shift every class index by one. The shipped charset is
 * entirely BMP today; counting by codepoint means it does not have to stay that
 * way for the contract check to stay honest.
 */
function charsetChars(charset) {
    return Array.from(charset);
}

/**
 * Verify that a loaded session and the charset about to decode it agree.
 *
 * Both numbers are read from the live session, never from a constant. A
 * constant is just another copy of the claim, and copies are exactly what
 * drifted: the charset, the sidecar JSON and the weights each stated a
 * different vocabulary size and nothing compared them.
 *
 * @param {object} session An ONNX Runtime InferenceSession (or anything exposing
 *   the same `inputMetadata` / `outputMetadata` shape).
 * @param {string} charset The decoding charset, blank excluded.
 * @param {number} targetHeight The input height this code preprocesses to.
 * @param {string} [modelPath] Included in the error message when known.
 */
function assertModelContract(session, charset, targetHeight, modelPath) {
    const where = modelPath ? `\n  model: ${modelPath}` : '';
    const numChars = charsetChars(charset).length;

    // Classifier width: [batch, time, classes]. CTC reserves index 0 for the
    // blank label, so an N-character charset needs exactly N + 1 classes.
    const outShape = firstTensorShape(session.outputMetadata, 'output');
    const numClasses = outShape[outShape.length - 1];
    if (typeof numClasses === 'number' && numClasses !== numChars + 1) {
        throw new ModelContractError(
            `Charset/model mismatch: the model has ${numClasses} output classes ` +
            `but the charset has ${numChars} characters, which needs ` +
            `${numChars + 1} (one CTC blank + one per character).${where}\n` +
            '  Decoding anyway would return confident, well-formed, wrong text. ' +
            'Pass a charsetPath matching this model, or delete the cached model ' +
            'so the pinned one is fetched again.'
        );
    }

    // Input height: [batch, channels, height, width], NCHW.
    const inShape = firstTensorShape(session.inputMetadata, 'input');
    if (inShape.length === 4) {
        const modelHeight = inShape[2];
        if (typeof modelHeight === 'number' && modelHeight !== targetHeight) {
            throw new ModelContractError(
                `Input height mismatch: the model expects height ${modelHeight} ` +
                `but preprocessing produces ${targetHeight}.${where}\n` +
                '  A model fed the wrong vertical resolution still runs and still ' +
                'returns text. Fail closed instead.'
            );
        }
    }

    // A dynamic class axis is not checkable here; `decode()` re-checks it against
    // the real dims of every output tensor, which are always concrete.
}

// Polarity. The model is trained on dark text on a light background and this
// binding never checked which it was given.
//
// Measured 2026-08-27 over 300 labelled crops from mon_OCR's
// data/real/digits/val, same graph, only the polarity of the input changed:
//
//     upright, with this probe    CER 0.0000   300/300 exact
//     inverted, with this probe   CER 0.0000   300/300 exact
//     upright, without it         CER 0.0036   296/300
//     inverted, without it        CER 0.0342   288/300   <- 9.5x worse
//
// Degradation rather than the total failure it might sound like, and cheap to
// close. Those crops are Myanmar digits on composited backgrounds, so the effect
// on full Mon text lines is unmeasured.
//
// A COPY of mon_OCR's `to_normalized_grayscale` steps 1-3, not a shared module:
// these bindings ship independently. Step 4, background levelling, is not ported
// and is what the 0.0036 upright row above costs.
const POLARITY_CORNER_FRACTION = 10;
const POLARITY_CORNER_FLOOR = 3;
const DARK_BACKGROUND_MEDIAN = 128;

/**
 * True when the four corner patches say this is light-text-on-dark.
 *
 * Corner-median rather than a global mean: document corners are almost always
 * background, so their median survives a dense, text-heavy page where a global
 * mean is dragged toward the ink. A page 64% covered in ink has a mean below 128
 * and must NOT be inverted.
 */
function backgroundIsDark(gray, width, height) {
    if (width <= 0 || height <= 0) return false;
    // The floor can exceed the image on a tiny crop; clamping keeps the sample
    // inside the buffer. Reading past it yields undefined, which coerces to NaN
    // in the comparison and to 0 in a sort — either way a light page could be
    // inverted. Silent and backwards.
    const ch = Math.min(height, Math.max(POLARITY_CORNER_FLOOR,
        Math.floor(height / POLARITY_CORNER_FRACTION)));
    const cw = Math.min(width, Math.max(POLARITY_CORNER_FLOOR,
        Math.floor(width / POLARITY_CORNER_FRACTION)));

    const samples = [];
    const corners = [[0, 0], [width - cw, 0], [0, height - ch], [width - cw, height - ch]];
    for (const [ox, oy] of corners) {
        for (let y = 0; y < ch; y++) {
            for (let x = 0; x < cw; x++) {
                samples.push(gray[(oy + y) * width + (ox + x)]);
            }
        }
    }
    if (samples.length === 0) return false;
    samples.sort((a, b) => a - b);
    const n = samples.length;
    const median = n % 2 ? samples[(n - 1) / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2;
    return median < DARK_BACKGROUND_MEDIAN;
}

/**
 * Invert `gray` in place when its background is dark. Returns the buffer either
 * way, unchanged when the page is already dark-on-light — which is what makes
 * this safe to run on every input.
 */
function normalizePolarity(gray, width, height) {
    if (!backgroundIsDark(gray, width, height)) return gray;
    for (let i = 0; i < gray.length; i++) gray[i] = 255 - gray[i];
    return gray;
}

/**
 * Return the page as a grayscale PNG Buffer with its polarity corrected, ready to
 * hand to the segmenter.
 *
 * Only the polarity probe runs here. Everything else the model needs — the resize,
 * the pad, the normalisation — belongs to `preprocess`, per crop.
 */
async function normalizePageForSegmentation(imagePath) {
    const { data, info } = await imaging()(imagePath)
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true });
    normalizePolarity(data, info.width, info.height);
    return imaging()(data, {
        raw: { width: info.width, height: info.height, channels: info.channels }
    })
        .png()
        .toBuffer();
}

class MonOCR {
    constructor(modelPath = null, charsetPath = null) {
        this.modelPath = modelPath;
        this.charsetPath = charsetPath;
        this.session = null;
        this.charset = "";
        this.segmenter = new LineSegmenter();
        this.modelManager = new ModelManager();

        // Metadata for the pinned v3.5 graph: MobileNetV3-Large + BiLSTM + CTC,
        // 160x1024 input, 276-character charset (277 classes with the CTC blank).
        //
        // Neither is a free parameter: `init()` refuses to run if the model
        // disagrees. Corrected 2026-08-27 — this comment described v2, naming 128
        // as the input height and a 315-character charset, three lines above the
        // constants that say 160. The code was right and the comment was two
        // generations behind.
        this.targetHeight = 160;
        this.targetWidth = 1024;
    }

    /**
     * Create the ONNX Runtime session. Split out so tests can substitute a
     * session without a model file or a network.
     */
    async _loadSession(modelPath) {
        return onnxRuntime().InferenceSession.create(modelPath);
    }

    /**
     * Read a charset file.
     *
     * Strips newlines only. A bare `.trim()` also eats the charset's leading
     * U+0020 -- the file really does start with a space -- which drops it from
     * 315 characters to 314 and shifts every index in the lookup by one. That is
     * not a hypothetical: 0.1.5 called `.trim()` here.
     */
    static readCharset(charsetPath) {
        return fs
            .readFileSync(charsetPath, 'utf-8')
            .replace(/^[\r\n]+/, '')
            .replace(/[\r\n]+$/, '');
    }

    async init() {
        if (this.session) return;

        // Ensure model exists. The download also brings the charset from the same
        // pinned revision, so weights and vocabulary cannot come from different
        // versions of the repository.
        let downloadedCharsetPath = null;
        if (!this.modelPath) {
            const artifacts = await this.modelManager.ensureModel();
            this.modelPath = artifacts.modelPath;
            downloadedCharsetPath = artifacts.charsetPath;
        }

        if (!this.charsetPath) {
            // Prefer the charset that shipped with these exact weights; fall back
            // to the bundled copy when the caller supplied their own model.
            this.charsetPath = downloadedCharsetPath || path.join(__dirname, 'charset.txt');
        }

        const session = await this._loadSession(this.modelPath);
        const charset = MonOCR.readCharset(this.charsetPath);

        assertModelContract(session, charset, this.targetHeight, this.modelPath);

        this.session = session;
        this.charset = charset;
    }

    /**
     * Replicates Python's resize_and_pad:
     * 1. Resize height to `targetHeight` (128 for the pinned model), maintain
     *    aspect ratio.
     * 2. Pad width to 1024 (white background).
     * 3. Normalize to [-1, 1].
     */
    async preprocess(imageSource) {
        let sharpImg;
        if (typeof imageSource.metadata === 'function') {
            sharpImg = imageSource;
        } else {
            sharpImg = imaging()(imageSource);
        }
        // Dimensions come from the DECODED buffer, not from `metadata()`.
        //
        // `metadata()` reads the input header and, as sharp's own docs put it,
        // "does not take into consideration any operations to be applied to the
        // output image". The segmenter hands us `image.clone().extract({...})` --
        // a crop whose `extract` has not run yet -- so `metadata()` returns the
        // PAGE's size, not the crop's.
        //
        // Measured on a 2550x3300 page with a 2400x90 line crop:
        //     crop.metadata()  ->  2550 x 3300
        //     scale = 160/3300 -> newWidth = 124   (correct: 1024)
        // Every segmented line was squeezed into 124 columns of a 1024 canvas
        // and the remaining 900 filled with white padding -- an ~8x horizontal
        // crush. `predictLine(path)` on a pre-cropped file was unaffected, which
        // is exactly the path docs/CROSS_BINDING_PARITY.md measured, so the
        // parity run could not have caught it.
        //
        // Materialising the grayscale raw buffer first costs one decode and
        // reports the true post-`extract` size in `info`.
        const { data: grayData, info } = await sharpImg
            .grayscale()
            .raw()
            .toBuffer({ resolveWithObject: true });

        normalizePolarity(grayData, info.width, info.height);

        // Scale to target height
        const scale = this.targetHeight / info.height;
        const newWidth = Math.min(this.targetWidth, Math.round(info.width * scale));

        const resizedBuffer = await imaging()(grayData, {
            raw: { width: info.width, height: info.height, channels: info.channels }
        })
            .resize({
                height: this.targetHeight,
                width: newWidth,
                fit: 'fill'
            })
            .raw()
            .toBuffer();

        // Create target canvas (1024 width, white background = 255)
        const totalSize = this.targetHeight * this.targetWidth;
        const canvas = new Float32Array(totalSize);

        // Fill canvas with resized image and normalize to [-1.0, 1.0]
        // (pix / 127.5) - 1.0
        for (let y = 0; y < this.targetHeight; y++) {
            for (let x = 0; x < this.targetWidth; x++) {
                const canvasIdx = y * this.targetWidth + x;
                if (x < newWidth) {
                    const imgIdx = y * newWidth + x;
                    const pixelValue = resizedBuffer[imgIdx];
                    canvas[canvasIdx] = (pixelValue / 127.5) - 1.0;
                } else {
                    // Padding is white (255)
                    canvas[canvasIdx] = (255 / 127.5) - 1.0; // 1.0
                }
            }
        }

        return new (onnxRuntime().Tensor)('float32', canvas, [1, 1, this.targetHeight, this.targetWidth]);
    }

    /**
     * Build the class-index -> character lookup, memoized on the charset it was
     * built from. Index 0 is the CTC blank and stays empty.
     */
    _idx2char() {
        if (this._idx2charCache && this._idx2charFor === this.charset) {
            return this._idx2charCache;
        }
        const chars = charsetChars(this.charset);
        const map = new Array(chars.length + 1);
        for (let i = 0; i < chars.length; i++) {
            map[i + 1] = chars[i];
        }
        this._idx2charCache = map;
        this._idx2charFor = this.charset;
        return map;
    }

    /**
     * CTC Greedy Decoding
     * Ignores blank (0) and contracts repeats.
     */
    decode(outputTensor) {
        const data = outputTensor.data;
        const dims = outputTensor.dims; // [Batch, Time, Classes]
        const numClasses = dims[dims.length - 1];
        const sequenceLength = dims[dims.length - 2];

        const idx2char = this._idx2char();

        // The same contract as `init()`, re-checked against the real dims of this
        // tensor. `init()` cannot check a dynamic class axis; these dims are
        // always concrete, so nothing gets through unverified.
        if (numClasses !== idx2char.length) {
            throw new ModelContractError(
                `Charset/model mismatch at decode: the model produced ${numClasses} ` +
                `classes but the charset covers ${idx2char.length - 1} characters ` +
                `(+ 1 CTC blank = ${idx2char.length}).`
            );
        }

        let decodedText = "";
        let prevIdx = -1;

        for (let t = 0; t < sequenceLength; t++) {
             let maxVal = -Infinity;
             let maxIdx = 0;
             for (let c = 0; c < numClasses; c++) {
                 const val = data[t * numClasses + c];
                 if (val > maxVal) {
                     maxVal = val;
                     maxIdx = c;
                 }
             }

             // CTC logic: 0 is blank, ignore repeats.
             // No `|| ""` fallback: the check above makes an out-of-range index
             // impossible, and that fallback is what turned a whole broken
             // vocabulary into quietly missing characters.
             if (maxIdx !== 0 && maxIdx !== prevIdx) {
                 decodedText += idx2char[maxIdx];
             }
             prevIdx = maxIdx;
        }

        return decodedText;
    }

    async predictLine(imageSource) {
        if (!this.session) await this.init();

        const inputTensor = await this.preprocess(imageSource);
        const feeds = {};
        feeds[this.session.inputNames[0]] = inputTensor;

        const results = await this.session.run(feeds);
        const outputTensor = results[this.session.outputNames[0]];

        return this.decode(outputTensor);
    }

    /**
     * Processes full page: segments into lines and predicts each.

     * NOTE (2026-08-16): this binding SQUEEZES a wide line into the model
     * canvas. The Python binding tiles instead, cutting at whitespace columns.
     *
     * This comment used to quote `v3.5 squeezed 0.1434 against tiled 0.0795` and
     * conclude "this binding is on the worse side of that". RETIRED 2026-08-22:
     * that harness was never committed and the figures do not reproduce.
     * Remeasured over 201 rendered lines, twice — Python arms and the Rust
     * binding — in mon_OCR/eval/tiling-ab-2026-08-22.md, the answer is
     * width-dependent: squeezing wins at 2 tiles, the two arms are level at 3,
     * and tiling wins from 4 up. On a real book page at 150 dpi every line
     * fitted one tile, so tiling never engaged.
     *
     * Porting `tile_line` and `cut_column` from python/monocr_onnx/segmenter.py
     * is still worth doing — squeezing's downside on very wide input is
     * unbounded where tiling's is not. Deferred because the four bindings
     * already disagree on output (docs/CROSS_BINDING_PARITY.md) and cut
     * positions would be a second axis.
     */
    async predictPage(imagePath) {
        if (!this.session) await this.init();

        // Polarity BEFORE segmentation, and this ordering is the point. The
        // segmenter treats dark as ink (`grayBuffer[idx] < 128`), so handed a
        // light-on-dark page it segments the BACKGROUND and returns the gaps
        // between lines. Inverting each crop inside `preprocess` afterwards cannot
        // recover a line that was never found. An audit caught this after the probe
        // shipped in `preprocess` alone.
        //
        // `segment` takes a path or a Buffer, so the page is normalised into a
        // Buffer first. The probe is idempotent — once the corners are light a
        // second call is a no-op — so the per-crop call still covers `predictLine`
        // without fighting this one.
        const source = await normalizePageForSegmentation(imagePath);

        const lines = await this.segmenter.segment(source);
        const results = [];

        for (const line of lines) {
            const text = await this.predictLine(line.img);
            results.push({
                text,
                bbox: line.bbox
            });
        }

        return results;
    }
}

module.exports = MonOCR;
module.exports.ModelContractError = ModelContractError;
// Exported for tests: the probe is the load-bearing half of preprocess.
module.exports.normalizePolarity = normalizePolarity;
module.exports.backgroundIsDark = backgroundIsDark;
module.exports.normalizePageForSegmentation = normalizePageForSegmentation;
module.exports.assertModelContract = assertModelContract;
