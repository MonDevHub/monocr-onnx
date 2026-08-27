// The model is trained on dark text on a light background; check what we feed it.
//
// Measured 2026-08-27 over 300 labelled crops from mon_OCR's
// data/real/digits/val, same graph, only the polarity of the input changed:
//
//     upright, with the probe    CER 0.0000   300/300 exact
//     inverted, with the probe   CER 0.0000   300/300 exact
//     upright, without it        CER 0.0036   296/300
//     inverted, without it       CER 0.0342   288/300   <- 9.5x worse

const test = require('node:test');
const assert = require('node:assert');
const { normalizePolarity, backgroundIsDark } = require('../src/monocr');

/** A bg-filled buffer with an ink bar across its middle. */
function page(w, h, bg, ink) {
    const g = new Uint8Array(w * h).fill(bg);
    for (let y = Math.floor(h / 3); y < Math.floor((2 * h) / 3); y++) {
        for (let x = Math.floor(w / 5); x < Math.floor((4 * w) / 5); x++) g[y * w + x] = ink;
    }
    return g;
}

test('a dark-on-light page is returned unchanged', () => {
    // THE NO-OP. Every input passes through the probe, so an ordinary page must
    // come back byte-identical or this is a regression rather than a fix.
    const g = page(200, 60, 255, 0);
    const before = Uint8Array.from(g);
    normalizePolarity(g, 200, 60);
    assert.deepStrictEqual(g, before);
});

test('a light-on-dark page is inverted', () => {
    const g = page(200, 60, 0, 255);
    normalizePolarity(g, 200, 60);
    assert.strictEqual(g[0], 255, 'dark background should have become light');
    assert.strictEqual(g[30 * 200 + 100], 0, 'light ink should have become dark');
});

test('a dense page is not mistaken for a dark one', () => {
    // Why corner-median and not a global mean: this page is ~64% ink, so its
    // mean is below 128 and a global test would invert an ordinary dense page.
    const w = 200, h = 60;
    const g = new Uint8Array(w * h).fill(255);
    for (let y = 6; y < 54; y++) for (let x = 20; x < 180; x++) g[y * w + x] = 0;
    const mean = g.reduce((a, b) => a + b, 0) / g.length;
    assert.ok(mean < 128, `fixture must actually be mean-dark, got ${mean.toFixed(1)}`);
    assert.strictEqual(backgroundIsDark(g, w, h), false);
});

test('the corner floor covers both axes', () => {
    // POLARITY_CORNER_FLOOR guards height and width separately; a test for one
    // leaves the other uncovered, which the Python port demonstrated.
    assert.strictEqual(backgroundIsDark(page(120, 8, 0, 255), 120, 8), true, 'short crop');
    assert.strictEqual(backgroundIsDark(page(8, 120, 0, 255), 8, 120), true, 'narrow crop');
});

test('a tiny light image is not inverted', () => {
    for (const [w, h] of [[1, 1], [2, 3], [3, 1], [5, 5]]) {
        const g = new Uint8Array(w * h).fill(255);
        assert.strictEqual(backgroundIsDark(g, w, h), false, `${w}x${h} light image read as dark`);
    }
});

test('a tiny DARK image is still inverted', () => {
    // What the bounds clamp is actually for, and the direction matters.
    //
    // The floor is 3, which exceeds the image below that size. Without the clamp
    // the sample runs past the buffer, and an out-of-range typed-array read in JS
    // is `undefined` — every comparison against it is false, so the median lands
    // on `undefined`, `undefined < 128` is false, and the page reads as LIGHT.
    // A tiny light image survives that by luck; a tiny dark one is silently left
    // inverted, which is the wrong answer. Testing the light direction alone
    // leaves the mutation alive, which is how this test came to exist.
    for (const [w, h] of [[1, 1], [2, 3], [3, 1]]) {
        const g = new Uint8Array(w * h).fill(0);
        assert.strictEqual(backgroundIsDark(g, w, h), true, `${w}x${h} dark image read as light`);
    }
});

test('preprocess normalises polarity before building the tensor', async () => {
    // The units above are worthless if preprocess does not call them. Asserted on
    // the tensor: an inverted page must reach the model as its upright twin.
    const sharp = require('sharp');
    const MonOCR = require('../src/monocr');

    const w = 200, h = 40;
    const light = new Uint8Array(w * h).fill(255);
    const dark = new Uint8Array(w * h).fill(0);
    for (let y = 10; y < 30; y++) {
        for (let x = 20; x < 180; x++) { light[y * w + x] = 0; dark[y * w + x] = 255; }
    }
    const raw = { raw: { width: w, height: h, channels: 1 } };
    const shim = { targetHeight: 160, targetWidth: 1024 };

    const up = await MonOCR.prototype.preprocess.call(shim, sharp(Buffer.from(light), raw));
    const inv = await MonOCR.prototype.preprocess.call(shim, sharp(Buffer.from(dark), raw));

    assert.deepStrictEqual(
        Array.from(inv.data), Array.from(up.data),
        'an inverted page must reach the model as its upright twin'
    );
});

test('a zero-sized image does not throw', () => {
    assert.strictEqual(backgroundIsDark(new Uint8Array(0), 0, 0), false);
});

test('predictPage normalises the page before segmenting it', async () => {
    // THE ORDERING. The segmenter treats dark as ink, so a light-on-dark page
    // segments the BACKGROUND and returns the gaps between lines. A probe inside
    // preprocess runs per crop, after segmentation, and cannot recover a line that
    // was never found. An audit caught that after the probe shipped there alone.
    //
    // Behavioural, not structural: an inverted page must segment into the same
    // number of lines as its upright twin.
    const sharp = require('sharp');
    const LineSegmenter = require('../src/segmenter');
    const { normalizePageForSegmentation } = require('../src/monocr');

    const W = 900, H = 260;
    async function png(bg, ink) {
        const g = new Uint8Array(W * H).fill(bg);
        for (const top of [40, 140]) {
            for (let y = top; y < top + 60; y++)
                for (let x = 60; x < W - 60; x += 20)
                    for (let i = 0; i < 12; i++) g[y * W + x + i] = ink;
        }
        return sharp(Buffer.from(g), { raw: { width: W, height: H, channels: 1 } })
            .png()
            .toBuffer();
    }

    const seg = new LineSegmenter();
    const upright = await seg.segment(await normalizePageForSegmentation(await png(255, 0)));
    const inverted = await seg.segment(await normalizePageForSegmentation(await png(0, 255)));

    assert.strictEqual(upright.length, 2, 'the control must find 2 lines');
    assert.strictEqual(
        inverted.length,
        upright.length,
        'an inverted page must segment like its upright twin — the probe is running too late'
    );
});

test('predictPage calls the page-level probe before segment', () => {
    // Paired with the behavioural test above: this one names the call site, so a
    // failure points at the wiring rather than at a band count.
    const MonOCR = require('../src/monocr');
    const src = MonOCR.prototype.predictPage.toString();
    const probeAt = src.indexOf('normalizePageForSegmentation');
    const segmentAt = src.indexOf('segmenter.segment');
    assert.ok(probeAt !== -1, 'predictPage does not call the page-level probe');
    assert.ok(segmentAt !== -1, 'predictPage no longer calls segment; update this test');
    assert.ok(probeAt < segmentAt, 'the probe must be called BEFORE segmentation');
});
