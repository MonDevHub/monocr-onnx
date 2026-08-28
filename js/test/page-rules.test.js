// Printed-rule suppression.
//
// A printed page border adds a constant ink floor to every row it spans, and once
// that floor clears the gap threshold no in-frame row reads as a gap: the page
// returns as one band and is squeezed into the model window.
//
// Measured with THIS parameter set (global threshold 128, no smear, smoothing 3,
// ratio 0.05 of the mean) over twelve real MNEC page-ones: nine collapse to three
// bands or fewer, and the twelve go from 118 bands to 215.

const test = require('node:test');
const assert = require('node:assert');
const LineSegmenter = require('../src/segmenter');
const { suppressPageRules } = require('../src/segmenter');

const WIDTH = 800, BAND = 40, MARGIN = 30, GLYPH_W = 12, PITCH = 20, RULE_W = 4;

/** A binary mask: 1 = ink. Glyph blobs, not solid bars — a solid bar the width of
 *  a text column IS a rule by any definition. */
function mask(bands, gap, framed) {
    const height = MARGIN * 2 + BAND * bands + gap * (bands - 1);
    const m = new Uint8Array(WIDTH * height);
    let y = MARGIN;
    for (let b = 0; b < bands; b++) {
        for (let yy = y; yy < y + BAND; yy++)
            for (let x = MARGIN + 20; x < WIDTH - MARGIN - 20; x += PITCH)
                for (let i = 0; i < GLYPH_W; i++) m[yy * WIDTH + x + i] = 1;
        y += BAND + gap;
    }
    if (framed) {
        for (let yy = 0; yy < height; yy++)
            for (let i = 0; i < RULE_W; i++) {
                m[yy * WIDTH + 10 + i] = 1;
                m[yy * WIDTH + (WIDTH - 10 - RULE_W) + i] = 1;
            }
        for (let i = 0; i < RULE_W; i++)
            for (let x = 0; x < WIDTH; x++) {
                m[(10 + i) * WIDTH + x] = 1;
                m[(height - 10 - RULE_W + i) * WIDTH + x] = 1;
            }
    }
    return { m, height };
}

function clearRows(m, height) {
    let n = 0;
    for (let y = 0; y < height; y++) {
        let any = false;
        for (let x = 0; x < WIDTH && !any; x++) if (m[y * WIDTH + x]) any = true;
        if (!any) n++;
    }
    return n;
}

test('a page with no rules is untouched to the pixel', () => {
    // THE PROPERTY THAT MAKES THIS SAFE UNCONDITIONALLY. Every page gets the step
    // whether it has rules or not, so "does nothing" must be exact.
    const { m, height } = mask(4, 40, false);
    const before = Uint8Array.from(m);
    assert.strictEqual(suppressPageRules(m, WIDTH, height), false);
    assert.deepStrictEqual(m, before);
});

test('a frame is removed and the gaps come back', () => {
    // Measured against what a clean page achieves. "Some row reaches zero" is too
    // low a bar: removing one axis alone already clears a handful of rows.
    const clean = mask(4, 40, false);
    const framed = mask(4, 40, true);
    const target = clearRows(clean.m, clean.height);

    assert.strictEqual(clearRows(framed.m, framed.height), 0, 'fixture must ink every row');
    assert.strictEqual(suppressPageRules(framed.m, WIDTH, framed.height), true);
    assert.ok(clearRows(framed.m, framed.height) >= target * 0.9,
        'one rule direction is probably still being missed');
});

test('glyph-sized ink is never taken for a rule', () => {
    const { m, height } = mask(6, 10, false);
    const before = Uint8Array.from(m);
    suppressPageRules(m, WIDTH, height);
    assert.deepStrictEqual(m, before);
});

test('a horizontal rule alone is removed', () => {
    // Kills the mutation that skips the horizontal scan; the frame fixture cannot,
    // because its vertical rules alone clear enough rows. Text is present because
    // a rule with no other ink would be 100% of the page and the ink-share guard
    // correctly refuses that.
    const { m, height } = mask(4, 40, false);
    const rowY = MARGIN + BAND + 10;
    for (let i = 0; i < RULE_W; i++) for (let x = 0; x < WIDTH; x++) m[(rowY + i) * WIDTH + x] = 1;

    assert.strictEqual(suppressPageRules(m, WIDTH, height), true);
    for (let i = 0; i < RULE_W; i++)
        for (let x = 0; x < WIDTH; x++) assert.strictEqual(m[(rowY + i) * WIDTH + x], 0);
});

test('a run of exactly the minimum length counts, one less does not', () => {
    // Each axis separately: an exact-length case on one leaves the other's
    // >= / > mutation alive.
    const minH = Math.max(15, Math.floor(WIDTH * 0.5));
    const rowY = MARGIN + BAND + 10;

    const a = mask(4, 40, false);
    for (let x = 0; x < minH; x++) a.m[rowY * WIDTH + x] = 1;
    assert.strictEqual(suppressPageRules(a.m, WIDTH, a.height), true);

    const b = mask(4, 40, false);
    for (let x = 0; x < minH - 1; x++) b.m[rowY * WIDTH + x] = 1;
    assert.strictEqual(suppressPageRules(b.m, WIDTH, b.height), false);

    const c = mask(4, 40, false);
    const minV = Math.max(15, Math.floor(c.height * 0.5));
    for (let y = 0; y < minV; y++) c.m[y * WIDTH + 12] = 1;
    assert.strictEqual(suppressPageRules(c.m, WIDTH, c.height), true);

    const d = mask(4, 40, false);
    for (let y = 0; y < minV - 1; y++) d.m[y * WIDTH + 12] = 1;
    assert.strictEqual(suppressPageRules(d.m, WIDTH, d.height), false);
});

test('suppression is abandoned when it would eat the page', () => {
    const width = 900, height = 20 + 6 * 30;
    const m = new Uint8Array(width * height);
    let y = 20;
    for (let b = 0; b < 6; b++) {
        for (let yy = y; yy < y + 30; yy++)
            for (let x = 40; x < 860; x += PITCH)
                for (let i = 0; i < GLYPH_W; i++) m[yy * width + x + i] = 1;
        y += 30;
    }
    const before = Uint8Array.from(m);
    assert.strictEqual(suppressPageRules(m, width, height), false);
    assert.deepStrictEqual(m, before);
});

test('degenerate masks do not throw', () => {
    assert.strictEqual(suppressPageRules(new Uint8Array(0), 0, 0), false);
    assert.strictEqual(suppressPageRules(new Uint8Array(100 * 50), 100, 50), false);
    suppressPageRules(new Uint8Array(100 * 50).fill(1), 100, 50);
});

/** A grayscale PNG: `glyphs` blobs per line, optionally framed. Sparse text on
 *  purpose — see the test below for why the glyph count is load-bearing. */
async function pagePng(bands, gap, glyphs, framed) {
    const sharp = require('sharp');
    const height = MARGIN * 2 + BAND * bands + gap * (bands - 1);
    const g = new Uint8Array(WIDTH * height).fill(255);
    let y = MARGIN;
    for (let b = 0; b < bands; b++) {
        for (let yy = y; yy < y + BAND; yy++)
            for (let k = 0; k < glyphs; k++) {
                const x = 100 + k * PITCH;
                for (let i = 0; i < GLYPH_W; i++) g[yy * WIDTH + x + i] = 0;
            }
        y += BAND + gap;
    }
    if (framed) {
        for (let yy = 0; yy < height; yy++)
            for (let i = 0; i < RULE_W; i++) {
                g[yy * WIDTH + 10 + i] = 0;
                g[yy * WIDTH + WIDTH - 10 - RULE_W + i] = 0;
            }
        for (let i = 0; i < RULE_W; i++)
            for (let x = 0; x < WIDTH; x++) {
                g[(10 + i) * WIDTH + x] = 0;
                g[(height - 10 - RULE_W + i) * WIDTH + x] = 0;
            }
    }
    return sharp(Buffer.from(g), { raw: { width: WIDTH, height, channels: 1 } })
        .png()
        .toBuffer();
}

test('segment recovers a framed page instead of fusing it', async () => {
    // The behavioural test, and the fixture took finding.
    //
    // A DENSE framed page does not fuse at this parameter set — 30 glyphs per line
    // segments into 4 bands with or without suppression, which is why an earlier
    // version of this test was structural and could not catch a mutation that read
    // the profile from the raw buffer instead of the suppressed mask.
    //
    // Sparse text is what reproduces the real behaviour: with 8 glyphs per line the
    // profile mean drops far enough that the frame's ink floor clears the 0.05
    // threshold on every row, and the page comes back as ONE band. That is the same
    // mechanism the twelve real MNEC pages show.
    const seg = new LineSegmenter();

    const clean = await seg.segment(await pagePng(4, 40, 8, false));
    const framed = await seg.segment(await pagePng(4, 40, 8, true));

    assert.strictEqual(clean.length, 4, 'the unframed control must segment into 4 lines');
    assert.strictEqual(
        framed.length,
        clean.length,
        'a framed page must segment like the same page without a frame'
    );
});

test('segment wires in the suppression', () => {
    // Kept alongside the behavioural test: it names the call site, so a failure
    // points at the wiring rather than at the segmentation result.
    assert.ok(LineSegmenter.prototype.segment.toString().includes('suppressPageRules'));
});

// ── Crop column extents ───────────────────────────────────────────────────────
//
// Suppression buys two things, and this binding used to collect only one. The row
// profile was read from the suppressed mask, but each crop's x-range was recomputed
// from the RAW grayscale, so the frame was deleted from the profile and then
// reinstated in every crop. See the comment on _extractLine, and mon_OCR
// src/monocr/segmenter.py:392 for the reference's statement of the intent.

test('a frame does not widen the crops', async () => {
    // The framed page and the same page unframed must produce IDENTICAL x-extents.
    // Exact, not approximate: every frame pixel belongs to a full-width row run or a
    // full-height column run, so suppression leaves the framed mask byte-identical to
    // the clean one and there is nothing left for the crops to differ over.
    const seg = new LineSegmenter();

    const clean = await seg.segment(await pagePng(4, 40, 8, false));
    const framed = await seg.segment(await pagePng(4, 40, 8, true));

    assert.strictEqual(clean.length, 4, 'the unframed control must segment into 4 lines');
    assert.strictEqual(framed.length, clean.length, 'framed page must segment like the clean one');

    for (let i = 0; i < clean.length; i++) {
        assert.strictEqual(framed[i].bbox.x, clean[i].bbox.x, `line ${i} x moved`);
        assert.strictEqual(framed[i].bbox.w, clean[i].bbox.w, `line ${i} width moved`);
    }

    // And say the failure out loud rather than only as "not equal": before the fix
    // xMin landed on the left rule and xMax on the right one, so a crop 163px wide
    // came back 791px wide, spanning the whole framed area.
    for (const { bbox } of framed) {
        assert.ok(bbox.x > 10 + RULE_W,
            `crop starts at ${bbox.x}, inside the left rule at x=10..${10 + RULE_W - 1}`);
        assert.ok(bbox.x + bbox.w < WIDTH - 10 - RULE_W,
            `crop ends at ${bbox.x + bbox.w}, inside the right rule`);
    }
});

test('the crop extents of a rule-free page are unchanged by the fix', async () => {
    // The no-op claim, pinned with numbers rather than asserted in prose. On a page
    // with no rules suppressPageRules returns the mask untouched, and the mask is the
    // same `< 128` test over the same buffer the old code read — so these are the
    // values the raw-grayscale scan produced too.
    //
    // Glyphs run x=100 .. 100+7*PITCH+GLYPH_W-1 = 251. The band measures BAND+2
    // rows, not BAND: smoothing over a 3-wide window pulls the profile above the
    // 0.05 gap threshold one row either side of the ink. padX = ceil(42*0.15) = 7.
    const seg = new LineSegmenter();
    const lines = await seg.segment(await pagePng(4, 40, 8, false));

    const x0 = 100, x1 = 100 + 7 * PITCH + GLYPH_W - 1;
    const padX = Math.ceil((BAND + 2) * 0.15);
    assert.strictEqual(lines.length, 4);
    for (const { bbox } of lines) {
        assert.strictEqual(bbox.x, x0 - padX);
        assert.strictEqual(bbox.x + bbox.w, x1 + padX);
    }
});
