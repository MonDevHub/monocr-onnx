// The dual histogram: the gap threshold is calibrated on the SMOOTHED row profile,
// the line boundaries are detected on the RAW one.
//
// Both halves are pinned here, because either one silently reverting costs lines.
// Every number below was measured through THIS binding at ITS parameters (minLineH
// 10, smoothWindow 3, ratio 0.05 of the non-zero mean). The reference's numbers do
// not transfer: mon_OCR dilates the mask vertically before taking the profile and
// this binding does not.

const test = require('node:test');
const assert = require('node:assert');
const sharp = require('sharp');
const LineSegmenter = require('../src/segmenter');

const WIDTH = 800, BAND = 40, MARGIN = 30, GLYPH_W = 12, PITCH = 20;

/** A grayscale PNG of `bands` glyph-blob bands, `gap` pixels apart.
 *
 *  Blobs rather than solid bars for the same reason page-rules.test.js gives: a
 *  solid bar the width of a text column IS a printed rule by any definition, and
 *  suppressPageRules would delete it before the profile ever saw it. */
async function drawnPage(bands, gap, glyphs = 30) {
    const height = MARGIN * 2 + BAND * bands + gap * bands;
    const g = new Uint8Array(WIDTH * height).fill(255);
    let y = MARGIN;
    for (let b = 0; b < bands; b++) {
        for (let yy = y; yy < y + BAND; yy++)
            for (let k = 0; k < glyphs; k++) {
                const x = 100 + k * PITCH;
                if (x + GLYPH_W <= WIDTH)
                    for (let i = 0; i < GLYPH_W; i++) g[yy * WIDTH + x + i] = 0;
            }
        y += BAND + gap;
    }
    return sharp(Buffer.from(g), { raw: { width: WIDTH, height, channels: 1 } })
        .png()
        .toBuffer();
}

test('lines two pixels apart are not fused', async () => {
    // THE CASE THE DUAL HISTOGRAM EXISTS FOR.
    //
    // With the default smoothWindow of 3 the smoother averages three rows, so a gap
    // of 1px or 2px never reaches zero in the smoothed profile — the ink either side
    // bleeds into it and clears the threshold. Reading boundaries there returned 1
    // band against 29 drawn, at both gaps. 3px is the first gap the smoothed profile
    // survives, which is why it is the control and not the interesting case.
    const seg = new LineSegmenter(10, 3);
    for (const gap of [1, 2]) {
        const got = await seg.segment(await drawnPage(29, gap));
        assert.strictEqual(got.length, 29,
            `29 bands ${gap}px apart came back as ${got.length} — boundaries are `
            + 'being read off the smoothed profile again');
    }
    const control = await seg.segment(await drawnPage(29, 3));
    assert.strictEqual(control.length, 29,
        'the 3px control failed, so the regression is not the profile choice');
});

test('touching bands stay one line', async () => {
    // The opposite failure, and why it needs its own test: the raw profile is the
    // more sensitive of the two, so the risk of reading it is splitting where no gap
    // exists. Bands that touch share ink on every row, no row is clean anywhere, and
    // one band is the honest answer.
    const seg = new LineSegmenter(10, 3);
    const got = await seg.segment(await drawnPage(29, 0));
    assert.strictEqual(got.length, 1, `touching bands were split into ${got.length}`);
});

test('a wide smoother does not fuse the page', async () => {
    // smoothWindow is a constructor argument, so the exposure is caller-settable and
    // not fixed at the default's 2px.
    //
    // On the smoothed profile the break point is the smoother's full width, so
    // raising it widened the damage in step: measured here, smoothWindow 15 returned
    // 1 band for every gap from 1px to 14px. 5px and 12px are two the old form lost.
    const seg = new LineSegmenter(10, 15);
    for (const gap of [5, 12]) {
        const got = await seg.segment(await drawnPage(29, gap));
        assert.strictEqual(got.length, 29,
            `at smoothWindow 15, 29 bands ${gap}px apart came back as ${got.length}`);
    }
});

/** `bands` dense bands, then one faint band of `faintH` rows carrying exactly
 *  `faintInk` ink pixels per row. Probes the threshold LEVEL, not the profile the
 *  boundaries come from. */
async function pageWithAFaintBand(bands, gap, glyphs, faintInk, faintH) {
    const height = MARGIN * 2 + BAND * bands + gap * (bands + 1) + faintH;
    const g = new Uint8Array(WIDTH * height).fill(255);
    let y = MARGIN;
    for (let b = 0; b < bands; b++) {
        for (let yy = y; yy < y + BAND; yy++)
            for (let k = 0; k < glyphs; k++) {
                const x = 100 + k * PITCH;
                if (x + GLYPH_W <= WIDTH)
                    for (let i = 0; i < GLYPH_W; i++) g[yy * WIDTH + x + i] = 0;
            }
        y += BAND + gap;
    }
    for (let yy = y; yy < y + faintH; yy++)
        for (let i = 0; i < faintInk; i++) g[yy * WIDTH + 100 + i] = 0;
    return sharp(Buffer.from(g), { raw: { width: WIDTH, height, channels: 1 } })
        .png()
        .toBuffer();
}

test('the gap threshold is calibrated on the smoothed profile', async () => {
    // The other half of the dual histogram: the LEVEL still comes off the smoothed
    // profile.
    //
    // Calibrating on the raw profile instead RAISES the threshold, because smoothing
    // spreads ink into the rows either side of every band and those partial rows pull
    // the non-zero mean down. A band faint enough to sit between the two thresholds is
    // then dropped, and dropping a line is the failure this pipeline exists to avoid.
    //
    // THE FIXTURE IS TUNED AND THE TUNING IS THE FINDING. Measured on THIS page, with
    // the faint band's own rows counted into both means: at the default smoothWindow
    // of 3 the two thresholds are 16.13 (smoothed) and 16.98 (raw), 0.85 of an ink
    // pixel apart, and no whole number lands between them — so NO test at the default
    // window can tell the two calibrations apart. This binding hard-codes the 0.05
    // ratio, so unlike the Rust port it cannot widen the gap with a constructed ratio;
    // smoothWindow is the only knob, and it is a constructor argument. At 15 the
    // thresholds measure 12.39 and 16.99, so a faint band of 13 to 16 ink pixels per
    // row is found by the smoothed calibration and missed by the raw one. 15 is near
    // the middle of that window.
    const seg = new LineSegmenter(10, 15);
    const got = await seg.segment(await pageWithAFaintBand(8, 40, 30, 15, 20));
    assert.strictEqual(got.length, 9,
        `expected 8 dense bands plus the faint one, got ${got.length} — the threshold `
        + 'is being calibrated on the raw profile');
});
