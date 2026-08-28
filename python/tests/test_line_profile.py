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

from monocr_onnx.segmenter import LineSegmenter, smooth_profile, suppress_page_rules

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
