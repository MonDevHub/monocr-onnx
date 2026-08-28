import cv2
import numpy as np
from PIL import Image

# Printed-rule suppression. A page border adds a constant ink floor to every row
# it spans, and once that floor clears the gap threshold no in-frame row reads as
# a gap: the page comes back as one band and is squeezed into the model window.
#
# Measured 2026-08-27 on the reference implementation over twelve real MNEC
# papers: nine collapsed to a single band without this, seven of them returning
# 0-2 characters, and the twelve together went from 3,846 characters to 5,924.
# Pages carrying no rules come back byte-identical.
#
# A rule is an unbroken ink run spanning at least RULE_SPAN of the page in one
# direction. Morphological opening with a line kernel keeps exactly those runs;
# subtracting them leaves the text. No Mon, Burmese or Latin glyph holds an
# unbroken stroke half a page long, so the false-positive risk against text is
# structural rather than merely small.
#
# There is deliberately NO thickness test. "A rule is long AND thin" was written,
# measured and deleted upstream: across twelve real pages the rule pixels found
# with a thickness limit and with none were identical to the pixel, and it cannot
# work anyway -- adaptiveThreshold compares against a LOCAL mean, so the interior
# of a thick ink region is not ink and only its edges are. Every thick region
# arrives already presented as a pair of thin bands.
RULE_SPAN = 0.5

# Suppression that would remove more than this share of the page ink has found
# text, not rules, and is abandoned. RULE_SPAN is a fraction of the page, so on a
# SHORT page a tall block of text exceeds it vertically and every glyph column
# reads as a rule. Upstream this was caught by an existing test losing 98.7% of
# its ink and returning zero lines. The threshold sits in a measured gap: real
# framed pages classify 21.5%-58.8% of their ink as rules, rule-free pages 0.00%,
# and that false positive 98.7%.
RULE_MAX_INK_SHARE = 0.80


def suppress_page_rules(binary):
    """Remove printed rules from a text mask, leaving glyphs untouched.

    Returns `binary` unchanged when the page carries no rules, and also when
    suppression would remove more ink than RULE_MAX_INK_SHARE.
    """
    h, w = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * RULE_SPAN)), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * RULE_SPAN)))),
    )
    rules = cv2.bitwise_or(horizontal, vertical)
    ink = int(np.count_nonzero(binary))
    if not ink or np.count_nonzero(rules) > ink * RULE_MAX_INK_SHARE:
        return binary
    return cv2.subtract(binary, rules)


def smooth_profile(raw_hist, window):
    """Box-filter the row profile. Returns `raw_hist` itself for window <= 1.

    A TRUE `window`-tap box, and the one binding of the four that is. numpy's
    kernel has exactly `window` taps whatever the parity, and `mode='same'`
    zero-pads the ends and divides by `window`, so a run of `window` zero rows
    always drives at least one output row to zero.

    An even `window` here is a box that is not CENTRED, which is the price of
    keeping the tap count exact: `mode='same'` takes the middle `len(hist)` of
    the full convolution, so an impulse at row 5 comes back at rows {5,6} at
    window 2 and {4,5,6,7} at window 4 -- half a row forward of the raw profile.
    Measured not to move the break-point table below, so it is recorded rather
    than corrected; correcting it would shift this binding's output.

    MEASURED on 29 drawn glyph-blob bands (min_line_h 10, threshold_ratio 0.02),
    driving the pre-fix form that read boundaries off this profile: the first gap
    that returned all 29 bands was exactly `window`, at every window from 1 to 12
    -- 1,2,3,4,...,12. The three sibling bindings loop [-window//2, +window//2]
    instead, which is 2*(window//2)+1 taps, so their break points run
    1,3,3,5,5,7,7,9,9,11,11,13 -- an even window there spans window+1 rows, one
    MORE than asked, and reads the same rows as the odd window ABOVE it. In JS
    and Rust it is VALUE-identical to that odd window too; in Go it is only
    tap-identical, because Go divides by the window it was asked for rather than
    by the rows it summed. Their numbers are not this one's; see
    js/src/segmenter.js and go/pkg/segmenter/segmenter.go, which record their own.
    """
    if window <= 1:
        return raw_hist
    kernel = np.ones(window) / window
    return np.convolve(raw_hist, kernel, mode='same')


# Two runs separated by at most this many rows are one text line, provided the
# raw profile never reaches zero inside the gap OR one of them is a fragment.
#
# WHY THIS EXISTS. Detecting boundaries on the raw profile splits a single line
# wherever one row dips below the gap threshold, and in Mon that happens between
# the upper diacritic zone and the consonant bodies. The strip of glyph tops then
# decodes to digits, because a row of circle-tops IS digits, and the decapitated
# body decodes missing its asats, because the asat went with the strip. See
# mon_OCR docs/AUDIT-2026-08-B.md F-69, which measured that with a model.
#
# MEASURED HERE, at this binding's own threshold: page 20 of a 56-page Mon book
# rendered at 300 DPI, threshold 20.8 ink pixels per row (0.02 of the smoothed
# profile MAX), one text line spanning rows 474-538, and **row 493 carrying 16
# ink pixels** -- one row wide, 16 against 20.8. The line came back as a 19-row
# strip and a 44-row body, and the page returned 70 runs where the merge leaves
# 42 bands.
#
# A 1-row gap holding ink is not a line boundary at any resolution. This is the
# reference's rule (mon_OCR `_MIN_GAP_MERGE`, segmenter.py step 8), ported with
# its value, and it is the half of the dual histogram this binding left behind:
# raw detection needs a merge to be safe, and the raw-only change shipped without
# it.
#
# WHAT IS THE REFERENCE'S AND WHAT IS NOT. Only this constant and the ordering --
# merge, then filter by height -- come from mon_OCR. Its merge has exactly two
# clauses, gap at most 10 and raw minimum above zero, and its comment argues
# AGAINST anything like the fragment clause below: "If in doubt, we keep lines
# SEPARATE... A split diacritic-only sub-line decodes to empty or near-empty text,
# which is harmless." Measurement falsified that premise -- a split sub-line
# decodes to a confident run of Mon DIGITS, not to empty -- so the fragment clause
# and the ceiling are this repository's additions, carrying this repository's
# evidence. An earlier version of this comment called them the reference's, which
# borrowed authority the reference declines to give.
#
# THIS BINDING'S OWN MEASUREMENT is in `merge_runs` below. Do not substitute the
# Rust port's or the reference's: the threshold here is a fraction of the profile
# MAX at ratio 0.02, where JS, Go and Rust take a fraction of the non-zero MEAN
# at 0.05, and the default `smooth_window` here is 5 against their 3.
MIN_GAP_MERGE = 10


def merge_runs(runs, raw_hist, max_gap, min_line):
    """Fuse runs that a single sub-threshold row split apart.

    Merges ``runs[i]`` into ``runs[i-1]`` when the gap between them is at most
    ``max_gap`` rows AND (every row in the gap carries ink OR one is a fragment
    being attached to something that could BE a line), AND the merged band stays
    within twice a typical line. See ``MIN_GAP_MERGE`` for why.

    ``min_line`` is the caller's minimum line height. It is needed twice, and both
    uses exist because this function runs BEFORE the height filter and so sees
    every speck the profile picked up: it bounds which runs may set the page's
    typical line height, and it stops two runs that are each too short to be a line
    from becoming one by being adjacent.

    A module-level function taking the profile rather than a method, so the
    arithmetic is testable without a page, a mask or a model.

    MEASURED THROUGH THIS BINDING at its own parameters (min_line_h 10,
    smooth_window 5, threshold_ratio 0.02 of the profile MAX), over the 56 pages
    of a real Mon book rendered at 300 DPI:

        no merge                 1974 bands   438 sub-0.6x-median (22.2%)
        merge, all-runs median   1920 bands   323 sub-0.6x-median (16.8%)
        this merge               1796 bands   208 sub-0.6x-median (11.6%)

    The middle row is this function as first written, medianing over every run,
    and it is here because the difference is this binding's largest single effect.
    This corpus is heavily speckled: 1791 of 3765 collected runs are under
    min_line_h, 47.6%, and on 9 of the 56 pages the all-runs median put `typical`
    below 10 -- page 1 reached `typical` 1 and a ceiling of 2 against a real line
    height of 24. On 6 of those 9 the merge was a bit-for-bit no-op, returning the
    unmerged band count exactly (22, 15, 25, 25, 18 and 32 bands unchanged). The
    sibling port that found this measured 30% of runs under the minimum; here it
    is half again as bad.

    The sub-0.6x share is the fragment proxy, and not a metric invented here:
    F-69 read a model over 4,251 bands, and of the 642 landing in [0.4, 0.6) of
    the page median, 94.4% decoded to majority digits. (95.1% is that bucket's
    mean digit share -- a different column of the same table.) Each arm is
    scored against its OWN page median above, and that could have flattered the
    merge, because merging raises the median. It does not: scored against the
    unmerged arm's medians as a fixed yardstick the merged count is 157 (8.7%).

    Two things this does NOT claim. It does not remove every suspect band --
    285 of F-69's 990 sub-0.6x bands were page numbers and watermarks, read
    correctly, which is why the merge is not a thin-band filter. And the total
    band count is not monotone: 6 of the 56 pages come back with MORE bands than
    the unmerged arm, because a merge lifts a pair of fragments that were each
    below min_line_h over the filter. That is content recovered, and it is why
    step 4.5 runs before the height filter rather than after it. On page 1 that
    effect is large: 22 bands unmerged, 48 merged.
    """
    if not runs:
        return []

    # The page's own typical line height, from the runs as detected. Both tests
    # below are relative to this rather than to the neighbouring run, and that is
    # a correction rather than a preference: judging a fragment against its
    # neighbour CASCADES. The merge mutates the accumulated run, so every merge
    # makes it taller, and a taller run makes the next line look more like a
    # fragment. Measured upstream on page 47 of a 56-page book: 36 bands
    # collapsed to 10, with single bands of 534, 632 and 732 rows holding a dozen
    # text lines each, and the page lost 92% of its readable characters.
    #
    # Measured HERE: judging a fragment against the accumulated neighbour instead
    # costs 7 bands and 0.4 points of fragments over the 56-page corpus (1803
    # bands and 12.0% sub-0.6x against this form's 1796 and 11.6%). Small, and it
    # was smaller still before the height-filtered median below: with the
    # all-runs median the two forms landed within one band of each other, because
    # the ceiling was contained so tightly that neither yardstick got to matter.
    #
    # And medianed over runs that could BE a line, not over every run. The merge
    # deliberately runs before the height filter, so `runs` still holds every
    # speckle: measured on a sibling port, 30% of collected runs were under the
    # minimum, and on 8 of 55 pages that drove `typical` below 10 -- one page
    # reached a `typical` of 2 and a ceiling of 4 against a real line height of 35.
    # The ceiling then refuses every merge, so the pass switches itself OFF on
    # exactly the pages that need it most.
    #
    # Falling back to the unfiltered median when nothing clears the minimum is safe
    # rather than principled: on such a page the height filter discards everything
    # anyway, so no crop depends on the value.
    heights = sorted(h for h in (r1 - r0 for r0, r1 in runs) if h >= min_line)
    if not heights:
        heights = sorted(r1 - r0 for r0, r1 in runs)
    typical = max(1, heights[len(heights) // 2])

    # No merge may produce a band more than twice a typical line. This is the
    # backstop for the cascade above: the fragment test alone cannot bound the
    # result, and one runaway band costs a whole page. Twice rather than tighter
    # because a legitimate merge of two halves lands at about one typical line
    # and must not be refused.
    #
    # Measured here: over the 56-page corpus, dropping it takes 1796 bands down to
    # 1167 -- 629 bands, 35%, swallowed into chains of merges. The sub-0.6x share
    # moves to 14.8% while that happens, WORSE on this metric as well as on the
    # count, which is the clearest form the argument takes: under the all-runs
    # median the same mutation improved the share to 15.5% while destroying 743
    # bands, and a fragment-share metric watched alone would have called that
    # progress.
    ceiling = typical * 2

    merged = []
    for r0, r1 in runs:
        if merged:
            gap_start = merged[-1][1]
            gap_size = max(0, r0 - gap_start)
            # An empty gap cannot occur from the run collector, but a caller can
            # hand us touching or overlapping runs; treat those as already one
            # line.
            #
            # The bounds test is explicit because a numpy slice would not do it.
            # `raw_hist[gap_start:r0]` TRUNCATES silently past the end, so a gap
            # reaching beyond the profile would read as fully inked here while
            # Rust (`hist.get`), JS (`undefined > 0` is false) and Go (an explicit
            # length test) all read it as not inked. Unreachable through
            # `segment`, where runs are bounded by the profile, but `merge_runs`
            # is a public entry point with its own tests, and three bindings
            # agreeing against a fourth is a divergence whichever way it is
            # resolved.
            gap_has_ink = r0 <= len(raw_hist) and all(
                raw_hist[y] > 0 for y in range(gap_start, r0)
            )

            # A run at most half a typical line is a fragment of a line, not a
            # line. This is the clause that crosses a gap of genuinely ZERO ink,
            # which `gap_has_ink` refuses and which a floating Mon diacritic
            # produces: measured, rows 341-360 and 362-404 are the upper marks
            # and the body of one line separated by two empty rows. Two REAL
            # lines two rows apart are each a full line by this test, so they
            # stay apart -- which is what a vertical smear could not do, because
            # at reach 1 it closes 2-row gaps and 2 rows is the tightest real
            # line spacing.
            #
            # A fragment attaches to a LINE, never to another fragment. Without
            # the second half of this, a run of speckle merges with itself:
            # measured on a 12-speck fixture, twelve 2-row specks fused into one
            # 46-row band, which then CLEARS the height filter and is handed to
            # the recogniser as a line. Two pieces that are both too short to be
            # a line do not become one by being adjacent.
            ha = merged[-1][1] - merged[-1][0]
            hb = r1 - r0
            fragment = 2 * min(ha, hb) <= typical and max(ha, hb) >= min_line

            if (
                gap_size <= max_gap
                and (gap_has_ink or fragment)
                and (r1 - merged[-1][0]) <= ceiling
            ):
                merged[-1][1] = r1
                continue
        merged.append([r0, r1])

    return [(r0, r1) for r0, r1 in merged]


class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles with Smoothing.
    Handles noisy documents and touching lines by finding valleys in the projection.
    """
    def __init__(self, min_line_h=10, smooth_window=5, threshold_ratio=0.02):
        self.min_line_h = min_line_h
        self.smooth_window = smooth_window
        self.threshold_ratio = threshold_ratio

    def segment(self, image):
        """
        Segment a document image into text lines.
        Args:
            image (PIL.Image or np.ndarray): Input image.
        Returns:
            list: List of dicts with keys 'img' (PIL.Image) and 'bbox' (x, y, w, h).
        """
        # Convert to CV2 grayscale
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        h_img, width = gray.shape

        # 1. Binarize (Adaptive Thresholding)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
        
        # 1.5 Printed-rule suppression
        binary = suppress_page_rules(binary)

        # 2. Horizontal Projection Profile
        raw_hist = np.sum(binary, axis=1).astype(np.float32)

        # 3. Smoothing
        #
        # `raw_hist` stays alive past this point, because the two profiles have
        # different jobs: the threshold in step 4 is calibrated on the smoothed
        # one, the boundaries are detected on the raw one. No copy is needed for
        # that even though `smoothed_hist` IS `raw_hist` when smoothing is off --
        # neither array is written to after this block.
        #
        # A consequence worth knowing rather than papering over: this binding
        # thresholds on the profile MAX, and smoothing does not lower the peak of a
        # band taller than the window, so on an ordinary page max(smoothed) equals
        # max(raw) to the pixel and this step no longer changes any output. It bites
        # only where the tallest row is thinner than the window. The bindings that
        # calibrate on a non-zero MEAN keep a real dependence on it. Not reconciled
        # here: the basis is a tuning constant, and the reference forbids moving it.
        smoothed_hist = smooth_profile(raw_hist, self.smooth_window)

        # 4. Gap Detection
        non_zero_vals = smoothed_hist[smoothed_hist > 0]
        if len(non_zero_vals) == 0:
            return []

        # Find threshold based on ratio or density
        max_val = np.max(smoothed_hist)
        threshold = max_val * self.threshold_ratio
        
        results = []
        runs = []
        start = None

        for y in range(h_img):
            # Boundaries come off the RAW profile, not the smoothed one.
            #
            # The threshold above stays calibrated on the smoothed profile's max.
            # But the smoother averages over `smooth_window` rows, so a gap
            # narrower than the whole window never reaches zero in the smoothed
            # profile: the ink either side bleeds into it, the bled rows clear the
            # threshold, and the two lines fuse. The raw profile needs one clean
            # row.
            #
            # MEASURED HERE, at this binding's own parameters (min_line_h 10,
            # smooth_window 5, threshold_ratio 0.02) on 29 drawn bands: reading the
            # smoothed profile returned 1 band for gaps of 1px, 2px, 3px and 4px
            # against 29 drawn, and matched the raw profile from 5px up. The break
            # point is the smoother's full width at EVERY window from 1 to 12,
            # even ones included, because `smooth_profile` is a true window-tap
            # box -- the sibling bindings round an even span UP to the next odd
            # number and their tables differ; see that docstring. This binding's
            # exposure is twice
            # the Rust port's, its default window being 5 against Rust's 3.
            # `smooth_window` is a constructor argument, so a caller who raises it
            # widens the failure with it: at 15 the smoothed profile lost every
            # page whose lines sat closer than 15px while the raw profile kept all
            # 29.
            #
            # These are this binding's numbers. Do not substitute the reference's
            # or the Rust port's: their break points differ, because the reference
            # dilates the mask vertically before the profile and this one does not.
            is_text_val = raw_hist[y] > threshold
            
            if is_text_val and start is None:
                start = y
            elif not is_text_val and start is not None:
                # End of a text block. Collected, not extracted: the merge in
                # step 4.5 needs every run on the page before it can measure the
                # page's typical line height.
                runs.append((int(start), int(y)))
                start = None

        if start is not None:
            runs.append((int(start), int(h_img)))

        # 4.5 Fuse runs a single sub-threshold row split apart, BEFORE the height
        # filter. The order is the reference's and it matters: a diacritic strip
        # can be shorter than `min_line_h`, and filtering first would discard the
        # strip and leave the decapitated body behind as a whole line.
        for r0, r1 in merge_runs(runs, raw_hist, MIN_GAP_MERGE, self.min_line_h):
            if (r1 - r0) >= self.min_line_h:
                self._extract_line(binary, gray, r0, r1, image, results)

        return results

    def _extract_line(self, binary, gray, r_start, r_end, source_image, results):
        """Crop a detected line region and append to results list."""
        # Find horizontal bounds within strip
        line_slice = binary[r_start:r_end, :]
        
        col_sums = np.sum(line_slice, axis=0)
        col_indices = np.where(col_sums > 0)[0]
        
        if len(col_indices) == 0:
            return
            
        x_start, x_end = col_indices[0], col_indices[-1]
        
        # Add relative padding based on line height
        h_raw = r_end - r_start
        pad_y = int(h_raw * 0.20)
        pad_x = int(h_raw * 0.15)
        y1 = max(0, r_start - pad_y)
        y2 = min(gray.shape[0], r_end + pad_y)
        x1 = max(0, x_start - pad_x)
        x2 = min(gray.shape[1], x_end + pad_x)
        
        w = x2 - x1
        h = y2 - y1
        
        # If input was not PIL, ensure we return something consistent
        if not isinstance(source_image, Image.Image):
            source_image = Image.fromarray(source_image)
            
        crop = source_image.crop((x1, y1, x2, y2))
        
        results.append({
            'img': crop,
            'bbox': (int(x1), int(y1), int(w), int(h))
        })


# Where a tile may be cut, as a fraction of the tile width, searching backwards
# from the ideal boundary. 0.12 of a 1024px window is ~123px, roughly two Mon
# glyphs — wide enough to find a gap, narrow enough that tiles stay near full
# width.
_CUT_SEARCH_FRACTION = 0.12

# A column counts as carrying ink below this grayscale value.
_CUT_INK_THRESHOLD = 250


def cut_column(crop, x0, ideal, crop_w):
    """Where to end a tile that starts at ``x0`` and may not pass ``ideal``.

    Cutting at exactly ``ideal`` lands wherever the arithmetic falls, which is
    usually the middle of a glyph. Both halves keep their pixels, so a coverage
    check still passes, but the model reads each half as a whole character and
    one glyph becomes two. Measured upstream on 120 drawn lines this showed up
    as ``ဗော်`` read back as ``ဗေဗိာ်``.

    So search backwards from ``ideal`` for a column of white. A tile may only
    get narrower, never wider, or it stops fitting the model window. Returns
    ``ideal`` unchanged when there is no gap to cut at, which is the honest
    outcome for a continuous script: a known-bad seam beats an overflowing tile.

    Ported from mon_OCR ``segmenter._cut_column``; the constants are the same,
    so the two produce the same cuts on the same input.
    """
    if ideal >= crop_w:
        return crop_w

    window = max(1, int((ideal - x0) * _CUT_SEARCH_FRACTION))
    lo = max(x0 + 1, ideal - window)
    if lo >= ideal:
        return ideal

    band = crop.crop((lo, 0, ideal, crop.height))
    if band.mode != "L":
        # The column sum below is silently wrong on a 3-channel array: it would
        # produce a (W, 3) profile and an argmin over a flattened index.
        band = band.convert("L")
    ink = (np.asarray(band, dtype=np.uint8) < _CUT_INK_THRESHOLD).sum(axis=0)

    # Prefer a truly empty column, and the rightmost one, so tiles stay as wide
    # as the window allows. Fall back to the lightest column present.
    blank = np.flatnonzero(ink == 0)
    offset = int(blank[-1]) if blank.size else int(np.argmin(ink))
    return lo + offset


def tile_line(crop, target_h, target_w):
    """Split one line crop into pieces that each fit the model window.

    Returns ``[crop]`` unchanged when the line already fits after being scaled
    to ``target_h``. Otherwise cuts at whitespace columns and returns the pieces
    left to right, to be read separately and joined with no separator.
    """
    crop_w, crop_h = crop.size
    if crop_h <= 0 or crop_w <= 0:
        return [crop]

    scale = target_h / crop_h
    if int(crop_w * scale) <= target_w:
        return [crop]

    tile_w_src = max(1, int(target_w / scale))
    tiles, x0 = [], 0
    while x0 < crop_w:
        ideal = min(x0 + tile_w_src, crop_w)
        x1 = cut_column(crop, x0, ideal, crop_w)
        # Structural guard, not a tuning knob: cut_column can only return a
        # value in (x0, ideal], but if it ever returned x0 this loop would spin
        # forever on a page. One pixel of forced progress bounds it.
        x1 = max(x1, x0 + 1)
        tiles.append(crop.crop((x0, 0, x1, crop_h)))
        x0 = x1
    return tiles
