// Preprocessing must measure the CROP, not the page it came from.
//
// `LineSegmenter._extractLine` returns `image.clone().extract({...})` — a sharp
// pipeline whose crop has not executed yet. `metadata()` reads the input header
// and explicitly ignores pending operations, so reading dimensions from it gave
// every segmented line the page's size instead of the crop's.
//
// Measured before the fix, on a 2550x3300 page with a 2400x90 line:
//     scale = 160/3300  ->  newWidth = 124   (correct: 1024)
// an ~8x horizontal crush, with the other 900 columns filled as white padding.
// `predictLine(path)` on a pre-cropped file was unaffected — which is the only
// path docs/CROSS_BINDING_PARITY.md exercised, so nothing caught it.

const test = require('node:test');
const assert = require('node:assert');
const sharp = require('sharp');

const TARGET_H = 160;
const TARGET_W = 1024;

/** A white page with one horizontal black bar where the "line" is. */
async function page(width, height, band) {
    const img = sharp({
        create: { width, height, channels: 3, background: 'white' }
    });
    const buf = await img.png().toBuffer();
    return sharp(buf).composite([
        {
            input: {
                create: { width: band.width, height: band.height, channels: 3, background: 'black' }
            },
            left: band.left,
            top: band.top
        }
    ]).png().toBuffer();
}

/** The dimension derivation under test, lifted from `MonOcr.preprocess`. */
async function scaledWidthOf(pipeline) {
    const { info } = await pipeline.grayscale().raw().toBuffer({ resolveWithObject: true });
    const scale = TARGET_H / info.height;
    return { newWidth: Math.min(TARGET_W, Math.round(info.width * scale)), info };
}

test('a pending extract is measured after cropping, not before', async () => {
    const band = { left: 100, top: 1000, width: 2400, height: 90 };
    const bytes = await page(2550, 3300, band);
    const crop = sharp(bytes).clone().extract({
        left: band.left, top: band.top, width: band.width, height: band.height
    });

    const { newWidth, info } = await scaledWidthOf(crop);

    assert.strictEqual(info.width, band.width, 'raw buffer must report the crop width');
    assert.strictEqual(info.height, band.height, 'raw buffer must report the crop height');
    assert.strictEqual(newWidth, TARGET_W,
        `a 2400x90 line scaled to h=160 exceeds 1024 and must clamp there, got ${newWidth}`);
});

test('metadata() is the trap this test exists for', async () => {
    // Pinning the sharp behaviour itself. If a future sharp made metadata()
    // account for pending operations, this test fails and the workaround in
    // preprocess() could be simplified — that is worth being told about.
    const band = { left: 100, top: 1000, width: 2400, height: 90 };
    const bytes = await page(2550, 3300, band);
    const crop = sharp(bytes).clone().extract({
        left: band.left, top: band.top, width: band.width, height: band.height
    });

    const meta = await crop.metadata();
    assert.strictEqual(meta.height, 3300,
        'metadata() still reports the page height; if this changed, revisit preprocess()');

    const wrong = Math.min(TARGET_W, Math.round(meta.width * (TARGET_H / meta.height)));
    assert.strictEqual(wrong, 124, 'the historical defect reproduced exactly');
});

test('grayscale raw output is single-channel', async () => {
    // The tensor is indexed as 1 byte per pixel (`y * newWidth + x`). Three
    // channels here would read a horizontally-compressed third of the image.
    const bytes = await page(400, 200, { left: 20, top: 90, width: 360, height: 20 });
    const { info } = await sharp(bytes).grayscale().raw().toBuffer({ resolveWithObject: true });
    assert.strictEqual(info.channels, 1);
});

test('a pre-cropped file is unaffected by the fix', async () => {
    // The path that always worked must keep working: no crop pending, so the
    // decoded size and the header agree.
    const bytes = await page(2400, 90, { left: 0, top: 20, width: 2400, height: 50 });
    const { newWidth, info } = await scaledWidthOf(sharp(bytes));
    assert.strictEqual(info.height, 90);
    assert.strictEqual(newWidth, TARGET_W);
});

// The resize must hand back ONE byte per pixel.
//
// `preprocess` builds its float canvas with `resizedBuffer[y * newWidth + x]` —
// one byte per pixel. Re-wrapping a raw buffer loses the b-w colourspace that
// `.grayscale()` established, because raw input carries no colourspace, so the
// pipeline's default sRGB output applies and `.raw()` returns three interleaved
// channels. Every read then lands on the wrong byte and only the first third of
// the image is sampled.
//
// Published npm monocr 0.3.2 returned 168 characters of noise for a 925x1280
// page where the Python SDK returned 1,271 correct ones. The suite was green
// throughout: the test above lifts the dimension arithmetic out into a copy and
// never looks at how many channels come back.
test('the resized buffer is single-channel, one byte per pixel', async () => {
    const src = await page(925, 1280, { width: 800, height: 60, left: 40, top: 200 });
    const { data, info } = await sharp(src).grayscale().raw().toBuffer({ resolveWithObject: true });
    const newWidth = Math.min(TARGET_W, Math.round(info.width * (TARGET_H / info.height)));

    const fixed = await sharp(data, {
        raw: { width: info.width, height: info.height, channels: info.channels }
    })
        .resize({ height: TARGET_H, width: newWidth, fit: 'fill' })
        .toColourspace('b-w')
        .raw()
        .toBuffer();

    assert.strictEqual(
        fixed.length,
        TARGET_H * newWidth,
        `expected ${TARGET_H * newWidth} bytes for a single-channel ${newWidth}x${TARGET_H} buffer`
    );
});

// The failure this pins is silent, so pin that it is no longer silent: dropping
// `.toColourspace('b-w')` must produce a buffer the guard in `preprocess`
// rejects, rather than one it reads as if the bytes were pixels.
test('without toColourspace the buffer is 3x, which the guard must reject', async () => {
    const src = await page(925, 1280, { width: 800, height: 60, left: 40, top: 200 });
    const { data, info } = await sharp(src).grayscale().raw().toBuffer({ resolveWithObject: true });
    const newWidth = Math.min(TARGET_W, Math.round(info.width * (TARGET_H / info.height)));

    const unfixed = await sharp(data, {
        raw: { width: info.width, height: info.height, channels: info.channels }
    })
        .resize({ height: TARGET_H, width: newWidth, fit: 'fill' })
        .raw()
        .toBuffer();

    assert.strictEqual(unfixed.length, TARGET_H * newWidth * 3,
        'sharp no longer returns three channels here; the guard and its comment need re-deriving');
    assert.notStrictEqual(unfixed.length, TARGET_H * newWidth,
        'a 3x buffer must not be mistakable for a single-channel one');
});
