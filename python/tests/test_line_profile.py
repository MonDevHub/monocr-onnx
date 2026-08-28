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

from monocr_onnx.segmenter import LineSegmenter, suppress_page_rules

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
