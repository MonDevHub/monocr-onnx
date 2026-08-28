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
const { smoothProfile } = require('../src/segmenter');

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
    // On the smoothed profile the break point is the smoother's EFFECTIVE span,
    // 2*floor(smoothWindow/2)+1 and not the requested window, so raising it widened
    // the damage in step: measured here, smoothWindow 15 returned 1 band for every
    // gap from 1px to 14px. 5px and 12px are two the old form lost. 15 is odd, so
    // span and window coincide; the even-window case is pinned below against
    // smoothProfile directly.
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


// The smoother's own arithmetic, pinned separately from the segmenter that reads
// it. Three of the four bindings diverge from Python here and nothing caught it,
// because no test used an even window.

/** `lead` rows of `ink`, then `gap` zero rows, then `lead` rows of `ink`. */
function bandedProfile(lead, gap, ink) {
    const out = new Float32Array(lead * 2 + gap);
    for (let i = 0; i < lead; i++) out[i] = ink;
    for (let i = lead + gap; i < out.length; i++) out[i] = ink;
    return out;
}

test('the box spans 2*floor(window/2)+1 rows, not window', () => {
    // THE DIVERGENCE AN ODD-WINDOW-ONLY TEST CANNOT SEE, and the reason it survived
    // four ports: at an odd window this formula and Python's agree exactly.
    //
    // The loop is [i-half, i+half] with half = floor(window/2), so an even window
    // spans window+1 rows — one MORE than asked — and is byte-identical to the odd
    // window ABOVE it. A gap of exactly `window` zero rows therefore still reaches
    // zero at odd windows and does NOT at even ones. Measured end-to-end through
    // the pre-fix form, that is why the break-point table reads
    // 1,3,3,5,5,7,7,9,9,11,11,13 here against Python's 1,2,...,12.
    for (let window = 2; window <= 12; window++) {
        const span = 2 * Math.floor(window / 2) + 1;
        const atSpan = smoothProfile(bandedProfile(20, span, 9), window);
        assert.ok(Math.min(...atSpan.slice(20, 20 + span)) === 0,
            `window ${window} left no zero row across a gap of ${span} rows — its `
            + 'span is no longer 2*floor(window/2)+1');
        const under = smoothProfile(bandedProfile(20, span - 1, 9), window);
        assert.ok(Math.min(...under.slice(20, 20 + span - 1)) > 0,
            `window ${window} reached zero across a gap of only ${span - 1} rows, so `
            + 'the box is narrower than measured');
        if (window % 2 === 0) {
            const profile = bandedProfile(20, span - 1, 9);
            assert.deepStrictEqual(
                Array.from(smoothProfile(profile, window)),
                Array.from(smoothProfile(profile, window + 1)),
                `window ${window} no longer matches window ${window + 1} — the `
                + 'even-window rounding changed');
        }
    }
});

test('the divisor is the rows visited, so edge rows keep their true mean', () => {
    // The formula difference against Python and Go, asserted rather than assumed.
    //
    // numpy's mode='same' zero-pads and divides by the window, so row 0 comes back
    // at (floor(window/2)+1)/window of the true local mean — 200 not 300 at window
    // 3, 160 not 300 at window 15. This binding divides by the rows it actually
    // visited and reports 300. Recorded, not reconciled: see smoothProfile's header.
    const flat = new Float32Array(60).fill(300);
    assert.strictEqual(smoothProfile(flat, 3)[0], 300);
    assert.strictEqual(smoothProfile(flat, 15)[0], 300);
    assert.strictEqual(smoothProfile(flat, 3)[59], 300);
    assert.strictEqual(smoothProfile(flat, 15)[59], 300);
});

test('smoothing never lifts the profile above its raw peak', () => {
    // Go's even-window bug, asserted absent here. Go sums 2*(window/2)+1 terms and
    // divides by the requested window, so at an even window every interior row is
    // inflated by (window+1)/window and the smoothed peak clears the raw one — 1.5x
    // at window 2, 1.25x at window 4. Dividing by what you summed cannot do that.
    const profile = bandedProfile(30, 0, 300);
    for (let window = 2; window <= 12; window++) {
        const smoothed = smoothProfile(profile, window);
        assert.ok(Math.max(...smoothed) <= 300,
            `window ${window} smoothed to a peak of ${Math.max(...smoothed)}, above `
            + 'the raw 300 — the divisor is smaller than the row count');
    }
});
