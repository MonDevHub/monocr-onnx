// Lazy for the same reason as in monocr.js: requiring sharp at import time makes
// every consumer of this package pay for a native binding they may not use.
//
// TEST COVERAGE, corrected twice now. This comment first read "the tests exercise
// the projection profile through fixtures and never open an image", which was
// false. It was then rewritten to say nothing in js/test/ references this file and
// that the profile is "entirely untested" — true on 2026-08-26 and false a day
// later. js/test/ now holds eight files, and test/page-rules.test.js requires this
// module directly.
//
// What is tested as of 2026-08-28: printed-rule suppression, in both directions and
// at the exact-length bound on each axis, including two behavioural cases driving
// segment() end to end — one on the band count, one on the crop COLUMN extents,
// which pins the horizontal padding with it. What is still NOT tested: the
// projection profile itself, the gap threshold, the histogram smoothing and the
// vertical padding — the parts that decide where a line begins.
//
// They diverge from the reference (mon_OCR src/monocr/segmenter.py) in ways that
// are recorded in that file's Canonical Algorithm Spec header — most of all the
// flat global `< 128` binarisation below, where the reference thresholds
// adaptively.
let sharp = null;
function imaging() {
    if (sharp === null) sharp = require('sharp');
    return sharp;
}

// Printed-rule suppression.
//
// A printed page border adds a constant ink floor to every row it spans, and once
// that floor clears the gap threshold no in-frame row reads as a gap: the page
// returns as one band and is squeezed into the model window.
//
// MEASURED WITH THIS PARAMETER SET (global threshold 128, no smear, smoothing 3,
// ratio 0.05 of the mean) over twelve real MNEC page-ones: nine collapse to three
// bands or fewer, and the twelve together go from 118 bands to 215. Pages carrying
// no rules are untouched.
//
// Worth stating because it is counter-intuitive: this binding gains MORE from the
// pass than the smeared implementations do, not less. A synthetic framed page does
// not fuse here, which briefly suggested the opposite; real pages settle it.
//
// A rule is an unbroken ink run spanning at least RULE_SPAN of the page in one
// direction. Implemented as a run-length scan, which is what an opening with a
// 1xL line kernel computes, in one sweep per axis.
const RULE_SPAN = 0.5;

// Suppression that would remove more than this share of the page ink has found
// text, not rules, and is abandoned. RULE_SPAN is a fraction of the page, so on a
// SHORT page a tall text block exceeds it vertically and every glyph column reads
// as a rule. Real framed pages classify 21.5%-58.8% of their ink as rules,
// rule-free pages 0.00%, and the false positive upstream found 98.7%.
const RULE_MAX_INK_SHARE = 0.8;

/**
 * Zero out printed rules in `binary` (1 = ink). Returns true when anything was
 * removed. Mutates in place.
 */
function suppressPageRules(binary, width, height) {
    const minH = Math.max(15, Math.floor(width * RULE_SPAN));
    const minV = Math.max(15, Math.floor(height * RULE_SPAN));
    const rules = new Uint8Array(width * height);

    for (let y = 0; y < height; y++) {
        const row = y * width;
        let start = 0;
        for (let x = 0; x <= width; x++) {
            if (x < width && binary[row + x]) continue;
            if (x - start >= minH) for (let i = start; i < x; i++) rules[row + i] = 1;
            start = x + 1;
        }
    }
    for (let x = 0; x < width; x++) {
        let start = 0;
        for (let y = 0; y <= height; y++) {
            if (y < height && binary[y * width + x]) continue;
            if (y - start >= minV) for (let i = start; i < y; i++) rules[i * width + x] = 1;
            start = y + 1;
        }
    }

    let ink = 0;
    for (let i = 0; i < binary.length; i++) if (binary[i]) ink++;
    if (ink === 0) return false;
    let ruleInk = 0;
    for (let i = 0; i < rules.length; i++) if (rules[i]) ruleInk++;
    if (ruleInk === 0 || ruleInk > ink * RULE_MAX_INK_SHARE) return false;

    for (let i = 0; i < rules.length; i++) if (rules[i]) binary[i] = 0;
    return true;
}

/** Box-filter the row profile. Returns `hist` itself for window <= 1.
 *
 *  TWO DELIBERATE DIVERGENCES FROM THE PYTHON BINDING, both measured, neither
 *  reconciled here — the formula is published API for anyone reading the profile.
 *
 *  1. EFFECTIVE WIDTH IS 2*floor(window/2)+1, NOT `window`. The loop is
 *     [i-half, i+half], so an even window spans window+1 rows -- one MORE than
 *     asked -- and is byte-identical to the odd window ABOVE it. Python convolves a
 *     true `window`-tap kernel and spans exactly what it was given.
 *     Measured through the pre-fix form that read boundaries off this profile, on
 *     29 drawn glyph-blob bands at minLineH 10: the first gap returning all 29
 *     bands, for windows 1 to 12, was 1,3,3,5,5,7,7,9,9,11,11,13. Python's was
 *     1,2,3,4,5,6,7,8,9,10,11,12. So the break point here is the EFFECTIVE width,
 *     not the requested one: at window 4 a gap of exactly 4px still fused. Rust
 *     and Go measure the same table; only Python differs.
 *  2. THE DIVISOR IS THE ROWS VISITED, NOT THE WINDOW. Near the top and bottom
 *     edges fewer rows are in range, and dividing by that count reports the true
 *     local mean. numpy's mode='same' zero-pads and divides by the window, which
 *     attenuates those rows to (floor(window/2)+1)/window of the true mean —
 *     two-thirds at window 3, 8/15 at window 15. Rust divides by the rows visited
 *     as this does; Go divides by the window and so matches numpy, but only at ODD
 *     windows -- at an even window Go matches neither, because it then sums
 *     window+1 rows and still divides by window.
 *
 *     Measured cost, now that the smoothed profile only sets the threshold LEVEL:
 *     the two formulas agree once every row they disagree on is zero, and the rows
 *     they disagree on are rows 0..half-1 and the mirror at the bottom, whose
 *     windows together cover rows 0..2*half-1. So the blank margin that hides the
 *     divergence is 2*floor(window/2) rows, NOT floor(window/2) -- measured on an
 *     8-band page, a margin of 1 row still left window 3 disagreeing (17.1607 here
 *     against Go's 17.1429) and 2 rows made them agree, and window 15 needed 14.
 *     The repo's own fixtures use a 30px margin, so they are all on the agreeing
 *     side. On a page cropped flush to the ink the threshold moved 0.21% at
 *     window 3 (17.2455 here against Go's 17.2096) and 1.17% at window 15, and no
 *     band count changed. Unifying the divisors would change output for users of
 *     at least one binding, so it is an owner decision, not a cleanup.
 */
function smoothProfile(hist, window) {
    if (window <= 1) return hist;
    const height = hist.length;
    const smoothed = new Float32Array(height);
    const half = Math.floor(window / 2);
    for (let i = 0; i < height; i++) {
        let sum = 0;
        let count = 0;
        for (let j = i - half; j <= i + half; j++) {
            if (j >= 0 && j < height) {
                sum += hist[j];
                count++;
            }
        }
        smoothed[i] = sum / count;
    }
    return smoothed;
}

class LineSegmenter {
    /**
     * @param {number} minLineH Minimum height of a line to be considered valid.
     * @param {number} smoothWindow Smoothing window for projection profile.
     */
    constructor(minLineH = 10, smoothWindow = 3) {
        this.minLineH = minLineH;
        this.smoothWindow = smoothWindow;
    }

    /**
     * Segment a document image into text lines.
     * @param {string|Buffer} imagePath Path to image or Buffer.
     * @returns {Promise<Array<{img: sharp.Sharp, bbox: {x: number, y: number, w: number, h: number}}>>}
     */
    async segment(imagePath) {
        const image = imaging()(imagePath);
        const { width, height } = await image.metadata();
        
        // 1. Get raw grayscale data for thresholding
        const grayBuffer = await image
            .grayscale()
            .raw()
            .toBuffer();

        // 2. Simple Adaptive-ish Thresholding
        // Since we don't have CV2's adaptiveThreshold easily, we'll do a simple threshold 
        // or just use sharp's threshold if we can get the mask.
        // Actually, to replicate Horizontal Projection, we need the sum of "text" pixels.
        // We'll treat dark pixels (< 128) as text (since background is white).
        // A mask IS materialised here, and unlike the dead one this file used to
        // carry it is read: rule suppression needs the 2-D shape of the ink, which
        // a per-row count cannot express.
        const binary = new Uint8Array(width * height);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                // Threshold: 128 is a safe bet for black text on white paper.
                // Inverted so text is "high" (1) and background is 0.
                if (grayBuffer[idx] < 128) binary[idx] = 1;
            }
        }

        suppressPageRules(binary, width, height);

        const hist = new Float32Array(height).fill(0);
        for (let y = 0; y < height; y++) {
            const row = y * width;
            for (let x = 0; x < width; x++) if (binary[row + x]) hist[y]++;
        }

        // 3. Smoothing projection profile
        //
        // `hist` stays alive past this point, because the two profiles have
        // different jobs: the gap threshold below is calibrated on the smoothed one,
        // the boundaries are detected on the raw one. No copy is needed for that
        // even though `smoothedHist` IS `hist` when smoothing is off -- smoothProfile
        // allocates a new array rather than writing into `hist`, and nothing after
        // this block mutates either. That function's header carries the two measured
        // divergences from the Python binding, width and divisor.
        const smoothedHist = smoothProfile(hist, this.smoothWindow);

        // 4. Gap Detection
        const nonZeroVals = smoothedHist.filter(v => v > 0);
        if (nonZeroVals.length === 0) return [];

        const meanDensity = nonZeroVals.reduce((a, b) => a + b, 0) / nonZeroVals.length;
        const gapThreshold = meanDensity * 0.05;

        const results = [];
        let start = null;

        for (let y = 0; y < height; y++) {
            // Boundaries come off the RAW profile, not the smoothed one.
            //
            // The threshold above stays calibrated on the smoothed profile, because
            // its non-zero mean is the steadier of the two. But the smoother
            // averages several rows together, so a gap narrower than its span never
            // reaches zero in the smoothed profile: the ink either side bleeds into
            // it, the bled rows clear the threshold, and the two lines fuse. The raw
            // profile needs one clean row.
            //
            // MEASURED HERE, at this binding's own parameters (minLineH 10, ratio
            // 0.05 of the non-zero mean) on 29 drawn bands, driving the pre-fix form
            // that read boundaries off the smoothed profile. First gap that returned
            // all 29 bands, by smoothWindow 1 to 12:
            //
            //   1 3 3 5 5 7 7 9 9 11 11 13
            //
            // So the break point is smoothProfile's EFFECTIVE span,
            // 2*floor(smoothWindow/2)+1, and NOT the requested window: at
            // smoothWindow 4 a gap of exactly 4px still fused. Python's table is
            // 1,2,...,12 because its kernel is a true window-tap box. `smoothWindow`
            // is a constructor argument, so a caller who raises it widens the failure
            // with it -- at 15 the smoothed profile lost every page whose lines sat
            // closer than 15px while the raw profile kept all 29.
            //
            // These are this binding's numbers. Do not substitute the reference's:
            // mon_OCR dilates the mask vertically before taking the profile and this
            // one does not, so its break point is 5px to 8px, not 3px.
            const isText = hist[y] > gapThreshold;
            if (isText && start === null) {
                start = y;
            } else if (!isText && start !== null) {
                const end = y;
                if (end - start >= this.minLineH) {
                    await this._extractLine(image, binary, width, height, start, end, results);
                }
                start = null;
            }
        }

        if (start !== null && (height - start) >= this.minLineH) {
            await this._extractLine(image, binary, width, height, start, height, results);
        }

        return results;
    }

    // The column extents are read from the SUPPRESSED mask, not from the raw
    // grayscale buffer.
    //
    // Reading `grayBuffer[...] < 128` here threw away half of what the suppression
    // pass buys. The frame was deleted from the row profile and then reinstated in
    // every crop's x-range: xMin landed on the left rule and xMax on the right one,
    // so each crop spanned the full framed area — the same over-wide crop that
    // squeezes a line into the model window, only now per line instead of per page.
    //
    // The reference states the intent at mon_OCR src/monocr/segmenter.py:392 —
    // suppression runs before the smear because "the crop's column extents come from
    // `dilated`, so removing rules first also keeps the border out of the crops" —
    // and its _extract_line sums `dilated` at line 648.
    // python/monocr_onnx/segmenter.py:140 already sums `binary` for the same reason;
    // this binding was the odd one out.
    //
    // On a page with NO rules this is a byte-for-byte no-op: suppressPageRules leaves
    // the mask untouched, and the mask is that identical `< 128` test over the same
    // buffer. Pinned by the tests in test/page-rules.test.js.
    async _extractLine(image, binary, width, height, rStart, rEnd, results) {
        // Find horizontal bounds within this vertical strip
        let xMin = width;
        let xMax = 0;
        let hasPixels = false;

        for (let y = rStart; y < rEnd; y++) {
            for (let x = 0; x < width; x++) {
                if (binary[y * width + x]) {
                    if (x < xMin) xMin = x;
                    if (x > xMax) xMax = x;
                    hasPixels = true;
                }
            }
        }

        if (!hasPixels) return;

        // Relative padding based on line height
        const hRaw = rEnd - rStart;
        const padY = Math.ceil(hRaw * 0.20);
        const padX = Math.ceil(hRaw * 0.15);
        const y1 = Math.max(0, rStart - padY);
        const y2 = Math.min(height, rEnd + padY);
        const x1 = Math.max(0, xMin - padX);
        const x2 = Math.min(width, xMax + padX);

        const w = x2 - x1;
        const h = y2 - y1;

        // Crop the line
        const crop = image.clone().extract({ left: x1, top: y1, width: w, height: h });
        
        results.push({
            img: crop,
            bbox: { x: x1, y: y1, w, h }
        });
    }
}

module.exports = LineSegmenter;

// Exported for tests.
module.exports.suppressPageRules = suppressPageRules;
module.exports.smoothProfile = smoothProfile;
