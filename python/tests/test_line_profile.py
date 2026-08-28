"""The dual histogram: the gap threshold is calibrated on the SMOOTHED row
profile, the line boundaries are detected on the RAW one.

Both halves are pinned here, because either one silently reverting costs lines.
Every number below was measured through THIS binding at ITS parameters
(min_line_h 10, smooth_window 5, threshold_ratio 0.02, a fraction of the profile
MAX). The Rust port's and the reference's numbers differ and do not transfer:
Rust's default window is 3, and the reference dilates the mask vertically before
taking the profile.
"""

import cv2
import numpy as np

from monocr_onnx.segmenter import (
    MIN_GAP_MERGE,
    LineSegmenter,
    merge_runs,
    smooth_profile,
    suppress_page_rules,
)

WIDTH = 800
BAND = 40
MARGIN = 30
GLYPH_W = 12
PITCH = 20


def _drawn_page(bands, gap, glyphs=30):
    """Glyph-like blobs on white, `bands` of them `gap` pixels apart.

    Blobs rather than solid bars for the same reason test_page_rules.py gives: a
    solid bar the width of a text column IS a printed rule by any definition, and
    rule suppression would delete it before the profile ever saw it.
    """
    height = MARGIN * 2 + BAND * bands + gap * bands
    page = np.full((height, WIDTH), 255, np.uint8)
    y = MARGIN
    for _ in range(bands):
        for k in range(glyphs):
            x = 100 + k * PITCH
            if x + GLYPH_W <= WIDTH:
                page[y : y + BAND, x : x + GLYPH_W] = 0
        y += BAND + gap
    return page


def test_lines_four_pixels_apart_are_not_fused():
    """THE CASE THE DUAL HISTOGRAM EXISTS FOR.

    With the default smooth_window of 5 the smoother averages five rows, so a gap
    of 1px to 4px never reaches zero in the smoothed profile -- the ink either side
    bleeds into it and clears the threshold. Reading boundaries there returned 1
    band against 29 drawn, at every one of those four gaps. 5px is the first gap
    the smoothed profile survives, which is why it is the control below and not the
    interesting case.
    """
    seg = LineSegmenter(min_line_h=10, smooth_window=5)
    for gap in (1, 2, 3, 4):
        got = seg.segment(_drawn_page(29, gap))
        assert len(got) == 29, (
            f"29 bands {gap}px apart came back as {len(got)} -- boundaries are "
            "being read off the smoothed profile again"
        )
    control = seg.segment(_drawn_page(29, 5))
    assert len(control) == 29, (
        "the 5px control failed, so the regression is not the profile choice"
    )


def test_touching_bands_stay_one_line():
    """The opposite failure, and why it needs its own test: the raw profile is the
    more sensitive of the two, so the risk of reading it is splitting where no gap
    exists. Bands that touch share ink on every row, no row is clean anywhere, and
    one band is the honest answer.
    """
    seg = LineSegmenter(min_line_h=10, smooth_window=5)
    got = seg.segment(_drawn_page(29, 0))
    assert len(got) == 1, f"touching bands were split into {len(got)}"


def test_a_wide_smoother_does_not_fuse_the_page():
    """`smooth_window` is a constructor argument, so the exposure is caller-settable
    and not fixed at the default's 4px.

    On the smoothed profile the break point is the smoother's full width, so raising
    it widened the damage in step: measured here, smooth_window 15 returned 1 band
    for every gap from 1px to 14px. 5px and 12px are two the old form lost.
    """
    seg = LineSegmenter(min_line_h=10, smooth_window=15)
    for gap in (5, 12):
        got = seg.segment(_drawn_page(29, gap))
        assert len(got) == 29, (
            f"at smooth_window 15, 29 bands {gap}px apart came back as {len(got)}"
        )


def _page_with_a_spike_and_a_faint_band(bands, gap, glyphs, faint_ink, faint_h):
    """A page whose MAX row is one pixel tall, plus a faint band.

    The thin spike is the whole point. Smoothing does not lower the peak of a band
    taller than the window -- the interior rows keep all five neighbours at full
    value -- so on an ordinary page max(smoothed) equals max(raw) exactly and NO
    threshold_ratio can tell the two calibrations apart. A 1px row of dense ink is
    a peak the smoother does flatten, which is what opens the gap between them.
    """
    height = MARGIN * 2 + BAND * bands + gap * (bands + 2) + 1 + faint_h
    page = np.full((height, WIDTH), 255, np.uint8)
    y = MARGIN
    for _ in range(bands):
        for k in range(glyphs):
            x = 100 + k * PITCH
            if x + GLYPH_W <= WIDTH:
                page[y : y + BAND, x : x + GLYPH_W] = 0
        y += BAND + gap
    for x in range(20, WIDTH - 20, 14):
        page[y : y + 1, x : x + GLYPH_W] = 0
    y += 1 + gap
    page[y : y + faint_h, 100 : 100 + faint_ink] = 0
    return page


def test_the_gap_threshold_is_calibrated_on_the_smoothed_profile():
    """The other half of the dual histogram: the LEVEL still comes off the smoothed
    profile.

    Calibrating on the raw profile instead RAISES the threshold, because the raw max
    is the unflattened spike. A band faint enough to sit between the two thresholds
    is then dropped, and dropping a line is the failure this pipeline exists to
    avoid.

    Measured on this fixture at the DEFAULT threshold_ratio of 0.02: max(smoothed)
    is 91,800 and max(raw) is 168,300, so the thresholds are 1,836 and 3,366 -- 7.2
    and 13.2 ink pixels per row. The faint band carries 10, which is inside that
    window. Unlike the Rust port, this binding needs no constructed ratio to make
    the calibration testable: its basis is the profile max rather than a non-zero
    mean, and the spike separates the two maxima by 1.83x.
    """
    seg = LineSegmenter(min_line_h=10, smooth_window=5, threshold_ratio=0.02)
    page = _page_with_a_spike_and_a_faint_band(8, 40, 30, 10, 20)
    got = seg.segment(page)
    assert len(got) == 9, (
        f"expected 8 dense bands plus the faint one, got {len(got)} -- the "
        "threshold is being calibrated on the raw profile"
    )


def test_the_two_thresholds_really_do_straddle_the_faint_band():
    """Guards the test above against its fixture rotting into a tautology.

    If a future change to binarisation or to rule suppression moved the two maxima
    together, the test above would still pass while proving nothing. This asserts
    the separation it depends on, in the same profile terms the segmenter uses.
    """
    page = _page_with_a_spike_and_a_faint_band(8, 40, 30, 10, 20)
    binary = suppress_page_rules(
        cv2.adaptiveThreshold(
            page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
    )
    raw = np.sum(binary, axis=1).astype(np.float32)
    smoothed = np.convolve(raw, np.ones(5) / 5, mode="same")

    faint = 10 * 255
    assert smoothed.max() * 0.02 < faint < raw.max() * 0.02, (
        f"the faint band's {faint} no longer sits between the smoothed threshold "
        f"{smoothed.max() * 0.02:.0f} and the raw one {raw.max() * 0.02:.0f}"
    )


# The smoother's own arithmetic, pinned separately from the segmenter that reads
# it. Three bindings of the four diverge here and nothing caught it, because no
# test used an even window.


def test_the_box_has_one_tap_per_window_row_at_every_parity():
    """The divergence an odd-window-only test cannot see.

    numpy's kernel has exactly `window` taps whatever the parity, so a run of
    `window` zero rows always drives at least one output row to zero. The three
    sibling bindings loop [-window//2, +window//2] -- 2*(window//2)+1 taps -- so an
    even window there spans window+1 rows, one MORE than asked, and reads the same
    rows as the odd window ABOVE it. Measured through the pre-fix form on 29 drawn
    bands, the first gap returning all 29 bands was `window` here and
    2*(window//2)+1 there.

    Even windows are the whole point of this test. An odd window makes the two
    formulas agree, which is why the divergence survived four ports.
    """
    for window in range(2, 13):
        hist = np.array([9.0] * 20 + [0.0] * window + [9.0] * 20, np.float32)
        smoothed = smooth_profile(hist, window)
        gap = smoothed[20 : 20 + window]
        assert gap.min() == 0.0, (
            f"window {window} left no zero row across a gap of exactly {window} "
            f"rows (min {gap.min()}) -- the box is no longer window taps wide"
        )
        narrower = smooth_profile(
            np.array([9.0] * 20 + [0.0] * (window - 1) + [9.0] * 20, np.float32),
            window,
        )
        assert narrower[20 : 20 + window - 1].min() > 0.0, (
            f"window {window} reached zero across a gap of only {window - 1} rows, "
            "so the box is narrower than it claims"
        )


def test_the_divisor_is_the_window_and_the_ends_are_zero_padded():
    """Edge handling, and the formula difference the sibling bindings carry.

    `mode='same'` zero-pads and divides by `window`, so a row near the top or
    bottom is the sum of the rows in range over the FULL window: at row 0 with
    window 3 that is 2/3 of the true mean of the two rows it could see, and with
    window 15 it is 8/15. Go matches this exactly at odd windows; JS and Rust
    divide by the number of rows they actually visited and so report the true local
    mean instead. Recorded, not reconciled -- see go/pkg/segmenter/segmenter.go.
    """
    hist = np.full(60, 300.0, np.float32)
    assert smooth_profile(hist, 3)[0] == 200.0
    assert smooth_profile(hist, 15)[0] == 160.0
    assert smooth_profile(hist, 3)[-1] == 200.0


def test_smoothing_never_lifts_the_profile_above_its_raw_peak():
    """Go's even-window bug, asserted absent here.

    Go sums 2*(window/2)+1 terms and divides by the requested `window`, so at an
    even window every interior row is inflated by (window+1)/window and the
    smoothed peak exceeds the raw one -- 1.5x at window 2. A box that divides by
    what it summed cannot do that, and this pins that it does not.
    """
    hist = np.array([0.0] * 10 + [300.0] * 30 + [0.0] * 10, np.float32)
    for window in range(2, 13):
        smoothed = smooth_profile(hist, window)
        # Two-sided, not `<=`. A one-sided bound also passes for a smoother that
        # attenuates everything, or for one that does not smooth at all.
        #
        # The tolerance is not slack: numpy multiplies each tap by 1/window and
        # sums, so it rounds `window` times where a sum-then-divide rounds once.
        # Measured drift on integer-valued profiles is around 1e-13 -- window 7
        # here peaks at 299.99999999999994. That is also why Go's identical
        # formula matches this one mathematically but not to the bit.
        assert abs(float(smoothed.max()) - float(hist.max())) < 1e-9, (
            f"window {window} smoothed to a peak of {smoothed.max()}, not the raw "
            f"{hist.max()} -- the divisor no longer equals the tap count"
        )
        # And the profile really was smoothed: the band's own edge rows are pulled
        # down, which a no-op smoother would leave at 300.
        assert smoothed[10] < 300.0, (
            f"window {window} left the band's first row at 300 -- nothing was "
            "smoothed"
        )


# ─────────────────────────────────────────────────────────────────────────────
# The other half of the dual histogram: the gap merge.
#
# Raw-profile detection alone splits one Mon line wherever a single row dips
# below the gap threshold, between the upper diacritic zone and the consonant
# bodies. See `MIN_GAP_MERGE` and `merge_runs` in monocr_onnx/segmenter.py for
# the measurement, taken through THIS binding.
#
# Every fixture below carries ORDINARY full-height lines as well as the case
# under test, and that is load-bearing rather than decoration. `merge_runs`
# judges a fragment against the page's typical line height, so a page consisting
# of nothing but the two halves of one split line is degenerate -- the median IS
# the half-height, and there is no evidence in it that they are halves rather
# than two short lines. Each test also says which clause is the sole reason its
# assertion holds, so that a mutation to that clause cannot be masked by another.
# ─────────────────────────────────────────────────────────────────────────────


def _split_line_page(strip_h, gap_h):
    """A strip of sparse upper marks, a blank gap, then a 44-row body.

    The measured geometry: one text line whose diacritics sit clear of the
    consonants. Marks 2px wide against the body's 12px so the strip's row profile
    is faint but well clear of the 0.02-of-max threshold, and so the strip is a
    run in its own right rather than noise.
    """
    height = 200
    page = np.full((height, WIDTH), 255, np.uint8)
    strip_y = 60
    body_y = strip_y + strip_h + gap_h
    for y0, y1, w in ((strip_y, strip_y + strip_h, 2), (body_y, body_y + 44, GLYPH_W)):
        for k in range(30):
            x = 100 + k * PITCH
            if x + w <= WIDTH:
                page[y0:y1, x : x + w] = 0
    return page


def test_a_diacritic_strip_is_returned_joined_to_its_line():
    """The merge must be reached THROUGH `segment`, not only unit-tested.

    A mutation deleting the `merge_runs` call from the pipeline survives every
    helper test below, because they call the helper directly and leave the call
    site unguarded. That is the gap se-brain rules/standards/testing.md names: a
    tested helper does not make its call site safe.

    Geometry is the measured one: a 20-row strip of upper marks, two empty rows,
    then a 44-row body. One line, and it must come back as one band.
    """
    got = LineSegmenter(min_line_h=10, smooth_window=5).segment(
        _split_line_page(strip_h=20, gap_h=2)
    )
    assert len(got) == 1, (
        f"the strip and its body came back as {len(got)} bands -- the merge is "
        "not reached from segment()"
    )
    assert got[0]["bbox"][3] >= 66, (
        f"the returned band is {got[0]['bbox'][3]}px tall, so it does not span "
        "the strip and the body together"
    )


def test_a_strip_shorter_than_min_line_h_survives_the_merge():
    """The ORDER of the merge and the height filter, which no band count reveals.

    Filtering first also returns one band here -- the body, with its diacritics
    deleted -- so a count assertion passes either way. What separates them is
    where the band STARTS: merged first, it opens above the strip; filtered
    first, the 6-row strip is discarded and the band opens at the body.

    A 6-row strip is below `min_line_h`, which is the case the reference's
    ordering exists for and the one the shipped raw-only change would have lost
    outright.
    """
    got = LineSegmenter(min_line_h=10, smooth_window=5).segment(
        _split_line_page(strip_h=6, gap_h=4)
    )
    assert len(got) == 1, f"expected one band, got {len(got)}"
    assert got[0]["bbox"][1] <= 60, (
        f"the band starts at row {got[0]['bbox'][1]}, below the strip at row 60 "
        "-- the height filter ran before the merge and ate the diacritics"
    )


def test_a_sub_threshold_dip_does_not_end_a_line():
    """Both clauses on the measured numbers rather than on invented ones: one
    line, rows 260-324, split by row 280 carrying 6 ink pixels against a
    threshold of 7.0. Upstream's measurement, kept because it is the case F-69
    diagnosed; this binding's own instance of it is page 20 of the same book,
    where the threshold is 20.8 and row 493 carries 16.
    """
    hist = np.zeros(400, np.float32)
    hist[260:325] = 200.0
    hist[280] = 6.0  # above zero, below the gap threshold
    assert merge_runs([(260, 280), (281, 325)], hist) == [(260, 325)], (
        "a 1-row dip holding ink split one line in two"
    )


def test_a_zero_gap_still_merges_a_fragment_into_its_line():
    """The other measured case: rows 341-360 are the upper marks and 362-404 the
    body of one line, separated by TWO rows of genuinely zero ink. No ink test
    can cross that; the fragment clause is what does, and it is the sole reason
    this merge happens.
    """
    hist = np.zeros(500, np.float32)
    hist[341:360] = 40.0
    hist[362:404] = 300.0
    assert merge_runs([(341, 360), (362, 404)], hist) == [(341, 404)], (
        "a 19-row fragment two empty rows from a 42-row line stayed separate"
    )


def test_two_real_lines_two_rows_apart_stay_separate():
    """The case the fragment clause must NOT swallow, and the reason it is a
    2x ratio and not a 1x one: same gap, same emptiness, but both runs are full
    height.

    The three 60-row lines make this test mean something. Without them the page
    median would be 40, the merged band would be 82 against a ceiling of 80, and
    the SIZE BOUND would refuse the merge whatever the ratio said -- so a ratio
    loosened to 1x would survive. With them the median is 60, the ceiling is 120,
    and the fragment clause is the only thing holding the two lines apart.

    A vertical smear cannot tell these apart at all, which is why one was not
    used: at reach 1 it closes 2-row gaps, and 2 rows is the tightest real line
    spacing.
    """
    hist = np.zeros(500, np.float32)
    for a, b in ((20, 60), (62, 102), (150, 210), (250, 310), (350, 410)):
        hist[a:b] = 300.0
    runs = [(20, 60), (62, 102), (150, 210), (250, 310), (350, 410)]
    assert merge_runs(runs, hist) == runs, (
        "two 40-row lines were fused on a page whose typical line is 60 -- the "
        "fragment ratio is no longer 2x"
    )


def test_a_wide_gap_is_a_line_boundary_however_much_ink_it_holds():
    """The size bound on its own. Overlapping diacritics can hold the raw profile
    above zero right across real inter-line spacing; upstream that collapsed 3
    PDF lines into 1.

    The 60-row lines are again what makes the assertion about `max_gap`: they put
    the ceiling at 120, and the merged band would be 95, so `max_gap` is the only
    clause refusing this merge. Sized the other way the test would pass for a
    `max_gap` of any value at all.
    """
    hist = np.zeros(500, np.float32)
    hist[20:60] = 300.0
    hist[60:75] = 5.0  # 15 rows of ink between two lines
    for a, b in ((75, 115), (150, 210), (250, 310), (350, 410)):
        hist[a:b] = 300.0
    runs = [(20, 60), (75, 115), (150, 210), (250, 310), (350, 410)]
    # `max_gap` passed explicitly, so the bound under test is the shipped
    # constant and not whatever the default argument happens to be.
    assert merge_runs(runs, hist, MIN_GAP_MERGE) == runs, (
        "a 15-row gap merged, so the size bound is not being applied"
    )


def test_a_dip_between_equal_halves_merges_on_ink_alone():
    """The ink clause on its own, which no other case here isolates: in the
    measured dip cases the fragment clause ALSO fires, so dropping the ink test
    survives them.

    Here the two halves are 40 rows each on a page whose typical line is 60, so
    neither is a fragment by the 2x ratio, and only the two rows of surviving ink
    in the dip can merge them.
    """
    hist = np.zeros(400, np.float32)
    hist[20:60] = 300.0
    hist[60:62] = 5.0  # two rows of ink: below any threshold, above zero
    for a, b in ((62, 102), (150, 210), (260, 320)):
        hist[a:b] = 300.0
    assert merge_runs([(20, 60), (62, 102), (150, 210), (260, 320)], hist) == [
        (20, 102),
        (150, 210),
        (260, 320),
    ], "an ink-holding 2-row dip between two halves of a typical line did not merge"


def test_a_merge_may_not_build_a_band_past_twice_a_typical_line():
    """The ceiling, and the cascade it is the backstop for.

    Four 20-row fragments separated by single inked rows would chain into one
    83-row band, and each merge makes the accumulated run taller. Upstream, with
    a fragment test judged against the NEIGHBOUR and no ceiling, page 47 of a
    56-page book collapsed from 36 bands to 10 with single bands of 534, 632 and
    732 rows, losing 92% of its readable characters.

    The typical line here is 40, so the chain is cut when it would pass 80: the
    first three fragments become one 62-row band and the fourth stays its own.
    """
    hist = np.zeros(500, np.float32)
    for a, b in ((0, 20), (21, 41), (42, 62), (63, 83)):
        hist[a:b] = 300.0
    for y in (20, 41, 62):
        hist[y] = 5.0  # a single inked row, so the gap clause allows the merge
    for a, b in ((150, 190), (210, 250), (270, 310), (330, 370), (390, 430)):
        hist[a:b] = 300.0
    runs = [
        (0, 20),
        (21, 41),
        (42, 62),
        (63, 83),
        (150, 190),
        (210, 250),
        (270, 310),
        (330, 370),
        (390, 430),
    ]
    assert merge_runs(runs, hist) == [
        (0, 62),
        (63, 83),
        (150, 190),
        (210, 250),
        (270, 310),
        (330, 370),
        (390, 430),
    ], "the fragment chain grew past twice a typical line -- the ceiling is gone"
