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
const { smoothProfile, mergeRuns, MIN_GAP_MERGE } = require('../src/segmenter');

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
    // at (floor(window/2)+1)/window of the true local mean — 200 not 300 on a flat
    // profile at window 3, 160 not 300 at window 15. This binding divides by the
    // rows it visited and reports 300. Recorded, not reconciled: see smoothProfile.
    //
    // The profile is NOT flat, deliberately. On a flat 300 the answer is 300 under
    // this formula AND under no smoothing at all, so a flat fixture cannot tell the
    // two apart. Row 0 is zeroed instead, which gives three different answers: 150
    // here (mean of rows 0 and 1), 100 under a window divisor, and 0 under a no-op.
    const dip = new Float32Array(60).fill(300);
    dip[0] = 0;
    dip[59] = 0;
    assert.strictEqual(smoothProfile(dip, 3)[0], 150,
        'row 0 is no longer the mean of the rows actually in range');
    assert.strictEqual(smoothProfile(dip, 3)[59], 150);
    assert.strictEqual(smoothProfile(dip, 5)[0], 200,
        'window 5 row 0 should be 600 over the 3 rows in range, not 900 over 5');

    const flat = new Float32Array(60).fill(300);
    assert.strictEqual(smoothProfile(flat, 3)[0], 300,
        'a flat profile must come back flat, unattenuated at the edges');
    assert.strictEqual(smoothProfile(flat, 15)[0], 300);
    assert.strictEqual(smoothProfile(flat, 15)[59], 300);
});

test('smoothing never lifts the profile above its raw peak', () => {
    // Go's even-window bug, asserted absent here. Go sums 2*(window/2)+1 terms and
    // divides by the requested window, so at an even window every interior row is
    // inflated by (window+1)/window and the smoothed peak clears the raw one — 1.5x
    // at window 2, 1.25x at window 4. Dividing by what you summed cannot do that.
    // 20 zero rows, 20 rows of ink, 20 zero rows, so the band is wider than the
    // widest span tested and its middle keeps the full 300.
    const profile = new Float32Array(60);
    profile.fill(300, 20, 40);
    for (let window = 2; window <= 12; window++) {
        const smoothed = smoothProfile(profile, window);
        // Two-sided, not `<=`. A one-sided bound also passes for a smoother that
        // attenuates everything, and for one that does not smooth at all.
        assert.strictEqual(Math.max(...smoothed), 300,
            `window ${window} peaked at ${Math.max(...smoothed)}, not the raw 300 — `
            + 'the divisor no longer equals the row count');
        // And smoothing really ran: the band's own edge rows are pulled down, which
        // a no-op smoother would leave at 300.
        assert.ok(smoothed[20] < 300,
            `window ${window} left the band's first row at 300 — nothing was smoothed`);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// The other half of the dual histogram: the gap merge.
//
// Raw-profile detection alone splits one Mon line wherever a single row dips below
// the gap threshold, between the upper diacritic zone and the consonant bodies. See
// `MIN_GAP_MERGE` and `mergeRuns` in src/segmenter.js for the measurement, taken
// through THIS binding.
//
// Every fixture below carries ORDINARY full-height lines as well as the case under
// test, and that is load-bearing rather than decoration. `mergeRuns` judges a
// fragment against the page's typical line height, so a page consisting of nothing
// but the two halves of one split line is degenerate — the median IS the
// half-height, and there is no evidence in it that they are halves rather than two
// short lines. Each test also says which clause is the sole reason its assertion
// holds, so a mutation to that clause cannot be masked by another.
//
// The first version of this paragraph was false of two of its own fixtures. The
// sub-threshold dip and the zero-ink gap each held only two runs, which is exactly
// the degenerate page the paragraph warns about, and the zero-ink one passed partly
// on the median INDEX convention: with two heights `floor(length / 2)` takes the
// upper, so `typical` was the full line's 42 rather than the fragment's 19, and
// `(length - 1) / 2` would have failed the test. Both now carry ordinary lines and
// pass under either convention.
// ─────────────────────────────────────────────────────────────────────────────

/** A PNG of one split text line: a strip of sparse upper marks, a blank gap, then a
 *  44-row body.
 *
 *  The measured geometry. Marks 2px wide against the body's 12px so the strip's row
 *  profile is faint but well clear of the gap threshold, and so the strip is a run
 *  in its own right rather than noise. */
async function splitLinePage(stripH, gapH) {
    const height = 200;
    const g = new Uint8Array(WIDTH * height).fill(255);
    const stripY = 60, bodyY = stripY + stripH + gapH;
    for (const [y0, y1, w] of [[stripY, stripY + stripH, 2], [bodyY, bodyY + 44, GLYPH_W]])
        for (let yy = y0; yy < y1; yy++)
            for (let k = 0; k < 30; k++) {
                const x = 100 + k * PITCH;
                if (x + w <= WIDTH) for (let i = 0; i < w; i++) g[yy * WIDTH + x + i] = 0;
            }
    return sharp(Buffer.from(g), { raw: { width: WIDTH, height, channels: 1 } })
        .png()
        .toBuffer();
}

/** A raw profile with ink on `bands` and nowhere else. */
function profile(length, bands) {
    const hist = new Float32Array(length);
    for (const [a, b, v] of bands) hist.fill(v === undefined ? 300 : v, a, b);
    return hist;
}

test('a diacritic strip is returned joined to its line', async () => {
    // The merge must be reached THROUGH segment(), not only unit-tested. A mutation
    // deleting the mergeRuns call from the pipeline survives every helper test below,
    // because they call the helper directly and leave the call site unguarded. That is
    // the gap se-brain rules/standards/testing.md names: a tested helper does not make
    // its call site safe.
    //
    // Geometry is the measured one: a 20-row strip of upper marks, two empty rows,
    // then a 44-row body. One line, and it must come back as one band.
    const got = await new LineSegmenter(10, 3).segment(await splitLinePage(20, 2));
    assert.strictEqual(got.length, 1,
        `the strip and its body came back as ${got.length} bands — the merge is not `
        + 'reached from segment()');
    assert.ok(got[0].bbox.h >= 66,
        `the returned band is ${got[0].bbox.h}px tall, so it does not span the strip `
        + 'and the body together');
});

test('a strip shorter than minLineH survives the merge', async () => {
    // The ORDER of the merge and the height filter, which no band count reveals.
    // Filtering first also returns one band here — the body, with its diacritics
    // deleted — so a count assertion passes either way. What separates them is where
    // the band STARTS: merged first it opens above the strip, filtered first the
    // 6-row strip is discarded and the band opens at the body.
    const got = await new LineSegmenter(10, 3).segment(await splitLinePage(6, 4));
    assert.strictEqual(got.length, 1, `expected one band, got ${got.length}`);
    assert.ok(got[0].bbox.y <= 60,
        `the band starts at row ${got[0].bbox.y}, below the strip at row 60 — the `
        + 'height filter ran before the merge and ate the diacritics');
});

test('a sub-threshold dip does not end a line', () => {
    // Both clauses on measured numbers rather than invented ones: one line, rows
    // 260-324, split by row 280 carrying 6 ink pixels against a threshold of 7.0.
    // Upstream's measurement, kept because it is the case F-69 diagnosed; this
    // binding's own instance is page 9 of the same book, where the threshold is 6.8
    // and row 377 carries 5.
    const hist = profile(600, [[260, 325], [400, 444], [500, 544]]);
    hist[280] = 6; // above zero, below the gap threshold
    assert.deepStrictEqual(
        mergeRuns([[260, 280], [281, 325], [400, 444], [500, 544]], hist, MIN_GAP_MERGE, 10),
        [[260, 325], [400, 444], [500, 544]],
        'a 1-row dip holding ink split one line in two');
});

test('a zero-ink gap still merges a fragment into its line', () => {
    // The other measured case: rows 341-360 are the upper marks and 362-404 the body
    // of one line, separated by TWO rows of genuinely zero ink. No ink test can cross
    // that; the fragment clause is what does, and it is the sole reason this merge
    // happens.
    const hist = profile(700, [[341, 360, 40], [362, 404], [450, 492], [550, 592]]);
    assert.deepStrictEqual(
        mergeRuns([[341, 360], [362, 404], [450, 492], [550, 592]], hist, MIN_GAP_MERGE, 10),
        [[341, 404], [450, 492], [550, 592]],
        'a 19-row fragment two empty rows from a 42-row line stayed separate');
});

test('two real lines two rows apart stay separate', () => {
    // The case the fragment clause must NOT swallow, and the reason it is a 2x ratio
    // and not a 1x one: same gap, same emptiness, but both runs are full height.
    //
    // The three 60-row lines make this test mean something. Without them the page
    // median would be 40, the merged band would be 82 against a ceiling of 80, and the
    // SIZE BOUND would refuse the merge whatever the ratio said — so a ratio loosened
    // to 1x would survive. With them the median is 60, the ceiling is 120, and the
    // fragment clause is the only thing holding the two lines apart.
    //
    // A vertical smear cannot tell these apart at all, which is why one was not used:
    // at reach 1 it closes 2-row gaps, and 2 rows is the tightest real line spacing.
    const runs = [[20, 60], [62, 102], [150, 210], [250, 310], [350, 410]];
    const hist = profile(500, runs);
    assert.deepStrictEqual(mergeRuns(runs.map(r => [...r]), hist, MIN_GAP_MERGE, 10), runs,
        'two 40-row lines were fused on a page whose typical line is 60 — the fragment '
        + 'ratio is no longer 2x');
});

test('a wide gap is a line boundary however much ink it holds', () => {
    // The size bound on its own. Overlapping diacritics can hold the raw profile above
    // zero right across real inter-line spacing; upstream that collapsed 3 PDF lines
    // into 1.
    //
    // The 60-row lines are again what makes the assertion about maxGap: they put the
    // ceiling at 120, and the merged band would be 95, so maxGap is the only clause
    // refusing this merge. Sized the other way the test would pass for a maxGap of any
    // value at all.
    const runs = [[20, 60], [75, 115], [150, 210], [250, 310], [350, 410]];
    const hist = profile(500, [...runs, [60, 75, 5]]);
    assert.deepStrictEqual(mergeRuns(runs.map(r => [...r]), hist, MIN_GAP_MERGE, 10), runs,
        'a 15-row gap merged, so the size bound is not being applied');
});

test('a dip between equal halves merges on ink alone', () => {
    // The ink clause on its own, which no other case here isolates: in the measured dip
    // cases the fragment clause ALSO fires, so dropping the ink test survives them.
    //
    // Here the two halves are 40 rows each on a page whose typical line is 60, so
    // neither is a fragment by the 2x ratio, and only the two rows of surviving ink in
    // the dip can merge them.
    const hist = profile(400, [[20, 60], [60, 62, 5], [62, 102], [150, 210], [260, 320]]);
    assert.deepStrictEqual(
        mergeRuns([[20, 60], [62, 102], [150, 210], [260, 320]], hist, MIN_GAP_MERGE, 10),
        [[20, 102], [150, 210], [260, 320]],
        'an ink-holding 2-row dip between two halves of a typical line did not merge');
});

test('a merge may not build a band past twice a typical line', () => {
    // The ceiling, and the cascade it is the backstop for. Four 20-row fragments
    // separated by single inked rows would chain into one 83-row band, and each merge
    // makes the accumulated run taller. Upstream, with a fragment test judged against
    // the NEIGHBOUR and no ceiling, page 47 of a 56-page book collapsed from 36 bands
    // to 10 with single bands of 534, 632 and 732 rows, losing 92% of its readable
    // characters.
    //
    // The typical line here is 40, so the chain is cut when it would pass 80: the first
    // three fragments become one 62-row band and the fourth stays its own.
    const runs = [[0, 20], [21, 41], [42, 62], [63, 83],
                  [150, 190], [210, 250], [270, 310], [330, 370], [390, 430]];
    const hist = profile(500, runs);
    for (const y of [20, 41, 62]) hist[y] = 5; // one inked row, so the gap clause allows it
    assert.deepStrictEqual(mergeRuns(runs.map(r => [...r]), hist, MIN_GAP_MERGE, 10),
        [[0, 62], [63, 83], [150, 190], [210, 250], [270, 310], [330, 370], [390, 430]],
        'the fragment chain grew past twice a typical line — the ceiling is gone');
});

test('a fragment is judged against the page median, not against its neighbour', () => {
    // Trap #1, and the only case that separates the two yardsticks while the ceiling
    // still allows the merge.
    //
    // The 21-row run is exactly half its 42-row neighbour, so a neighbour-relative
    // ratio calls it a fragment and fuses them. Against the page median of 40 it is
    // NOT a fragment — 21 is over half a typical line — and it stays its own band. The
    // merged band would be 65 against a ceiling of 80, so the ceiling is not what
    // refuses this: the yardstick is.
    //
    // Judging against the neighbour cascades, because each merge makes the accumulated
    // run taller and the next line then looks more like a fragment. Measured on this
    // binding's 56-page corpus, that form costs 165 bands and 8.5 points more sub-0.6x
    // fragments (1903 bands, 17.7%) than this one (1738, 9.2%).
    const runs = [[10, 50], [100, 142], [144, 165], [200, 240], [260, 300]];
    const hist = profile(400, runs);
    assert.deepStrictEqual(mergeRuns(runs.map(r => [...r]), hist, MIN_GAP_MERGE, 10), runs,
        'a 21-row run was fused into its 42-row neighbour on a page whose typical line '
        + 'is 40 — the fragment test is measuring against the neighbour again');
});

test('speckle does not set the typical line height', () => {
    // Two defects in one fixture, because the second was found by writing the first.
    // mergeRuns runs BEFORE the height filter, so its input holds every speck the
    // profile picked up.
    //
    // 1. Medianing over ALL runs lets noise set `typical`. Measured on the identical
    //    corpus through the Python binding, 47.6% of collected runs were under the
    //    minimum, and on 9 of 56 pages that drove `typical` below 10 — one page
    //    reached 1, so the ceiling was 2 against a real line height of 24 and every
    //    merge was refused. The pass switched itself off on the pages needing it most.
    // 2. A fragment attaching to another fragment chains. Twelve 2-row specks fuse
    //    into one 46-row band, which CLEARS the height filter and is handed to the
    //    recogniser as a line.
    //
    // ASSERTING THE SPLIT PAIR MERGES IS NOT ENOUGH. That assertion alone passes with
    // defect 2 still in place — the fragment-to-fragment mutation survived the battery
    // until the second assertion below was added. Both are needed.
    const bands = [];
    const runs = [];
    for (let i = 0; i < 12; i++) { bands.push([i * 4, i * 4 + 2, 20]); runs.push([i * 4, i * 4 + 2]); }
    // A split line whose halves really are halves: 24 rows, a 2-row inked dip, 24
    // rows, summing to the 50 an ordinary line measures here. 50 + 50 would be two
    // whole lines by this page's own standard and the ceiling would refuse the merge
    // for the right reason, which is not what this test is about.
    bands.push([100, 124], [124, 126, 5], [126, 150]);
    runs.push([100, 124], [126, 150]);
    // And three ordinary lines, so a real median exists to be found.
    for (let i = 0; i < 3; i++) { bands.push([200 + i * 60, 250 + i * 60]); runs.push([200 + i * 60, 250 + i * 60]); }

    const merged = mergeRuns(runs, profile(700, bands), MIN_GAP_MERGE, 10);

    assert.ok(merged.some(([a, b]) => a === 100 && b === 150),
        `the split pair did not merge, so speckle set the ceiling: ${JSON.stringify(merged)}`);
    const fused = merged.filter(([a, b]) => a < 100 && b - a >= 10);
    assert.strictEqual(fused.length, 0,
        `speckle fused into ${fused.length} band(s) tall enough to clear the height `
        + `filter and be read as lines: ${JSON.stringify(fused)}`);
});
