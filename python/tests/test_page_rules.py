"""Printed-rule suppression: a page border must not swallow the page.

A page border adds a constant ink floor to every row it spans, and once that floor
clears the gap threshold no in-frame row reads as a gap -- the page returns as one
band and is squeezed into the model window.

Measured 2026-08-27 on the reference implementation over twelve real MNEC papers:
nine collapsed to a single band without this, seven of them returning 0-2
characters, and the twelve together went from 3,846 characters to 5,924. Pages
carrying no rules come back byte-identical.
"""

import cv2
import numpy as np
import pytest

from monocr_onnx.segmenter import LineSegmenter as SEGMENTER_CLASS
from monocr_onnx.segmenter import RULE_MAX_INK_SHARE, suppress_page_rules

WIDTH = 800
BAND = 40
MARGIN = 30
GLYPH_W = 12
PITCH = 20
RULE_W = 4


def _binary(page):
    return cv2.adaptiveThreshold(
        page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )


def _page(bands=4, gap=40, framed=False, width=WIDTH):
    """Glyph-like blobs, not solid bars: a solid bar the width of a text column IS
    a rule by any definition, so a fixture drawn that way tests nothing."""
    h = MARGIN * 2 + BAND * bands + gap * (bands - 1)
    page = np.full((h, width), 255, np.uint8)
    y = MARGIN
    for _ in range(bands):
        for x in range(MARGIN + 20, width - MARGIN - 20, PITCH):
            page[y : y + BAND, x : x + GLYPH_W] = 0
        y += BAND + gap
    if framed:
        page[:, 10 : 10 + RULE_W] = 0
        page[:, width - 10 - RULE_W : width - 10] = 0
        page[10 : 10 + RULE_W, :] = 0
        page[h - 10 - RULE_W : h - 10, :] = 0
    return page


def test_a_page_with_no_rules_is_untouched_to_the_pixel():
    """THE PROPERTY THAT MAKES THIS SAFE TO RUN UNCONDITIONALLY.

    Every caller gets this step whether their pages have rules or not, so "does
    nothing on a clean page" has to be exact. Asserted on the mask rather than a
    band count: an equal band count would also hold if the pass had quietly
    altered the ink.
    """
    b = _binary(_page())
    assert np.array_equal(suppress_page_rules(b), b)


def test_a_frame_is_removed():
    b = _binary(_page(framed=True))
    assert not np.array_equal(suppress_page_rules(b), b)




def test_glyph_sized_ink_is_never_taken_for_a_rule():
    """The false-positive direction, and what pins RULE_SPAN. Lowering it toward a
    glyph width fails here."""
    b = _binary(_page(bands=6, gap=10))
    assert np.array_equal(suppress_page_rules(b), b)


def test_suppression_is_abandoned_when_it_would_eat_the_page():
    """What RULE_MAX_INK_SHARE is for.

    RULE_SPAN is a fraction of the page, so on a SHORT page a tall block of text
    exceeds it vertically and every glyph column reads as a rule. Leaving the page
    alone is strictly better than emptying it.
    """
    page = np.full((20 + 6 * 30, 900), 255, np.uint8)
    y = 20
    for _ in range(6):
        for x in range(40, 860, PITCH):
            page[y : y + 30, x : x + GLYPH_W] = 0
        y += 30
    b = _binary(page)
    assert np.array_equal(suppress_page_rules(b), b), (
        "suppression removed most of the page ink; the guard should have stopped it"
    )


def test_the_guard_threshold_is_the_measured_one():
    assert RULE_MAX_INK_SHARE == 0.80


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (50, 50)])
def test_degenerate_masks_do_not_trap(shape):
    blank = np.zeros(shape, np.uint8)
    assert np.array_equal(suppress_page_rules(blank), blank)
    suppress_page_rules(np.full(shape, 255, np.uint8))



def test_the_frame_no_longer_inks_every_row_STRONG():
    """The mechanism, measured against what a clean page achieves.

    An earlier version asserted only that SOME row reaches zero, and that was too
    weak to matter: removing the horizontal rules alone already clears 8 rows, so
    a mutation dropping vertical-rule detection passed. Both directions together
    clear 180. The right bar is the clean page.
    """
    clean_clear = int((_binary(_page()).sum(axis=1) == 0).sum())
    framed = _binary(_page(framed=True))
    assert (framed.sum(axis=1) > 0).all(), "fixture must actually ink every row"

    cleaned_clear = int((suppress_page_rules(framed).sum(axis=1) == 0).sum())
    assert cleaned_clear >= clean_clear * 0.9, (
        f"after suppression only {cleaned_clear} rows reach zero, against "
        f"{clean_clear} on the same page without a frame -- one rule direction "
        "is probably still being missed"
    )


def test_segment_wires_in_the_suppression():
    """Structural, because the synthetic fixture cannot reproduce the collapse.

    A synthetic framed page does NOT fuse in this binding, with or without
    suppression, and the reason is specific: this segmenter has no morphological
    smear and thresholds at 2% of the profile MAX. The frame floor on the fixture
    is ~2,040 against a threshold of ~4,080, so the gaps stay visible.
    mon_OCR fuses on the same fixture because its `_SMEAR_X = 11` widens each rule
    column by 10px and triples that floor.

    REAL pages do fuse here. Measured 2026-08-27 over the twelve MNEC page-ones
    through this segmenter, bands returned without suppression -> with:

        vs MON College Reform No 3          3 -> 11
        Early Childhood No 5 (2020in)       1 -> 16
        Early Childhood No 5 (in, x2)       2 -> 13
        ...  total across all twelve      124 -> 169

    The other eight are unchanged. So the pass earns its place on real input and
    cannot be pinned by a fixture, which is why this test reads the call site
    instead of the behaviour. Parsed rather than grepped: a text search would
    match the comment naming the function.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(SEGMENTER_CLASS.segment)))
    called = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert "suppress_page_rules" in called, (
        "segment() does not call suppress_page_rules; the unit tests above pass "
        "regardless, so this is the only thing pinning the wiring"
    )
