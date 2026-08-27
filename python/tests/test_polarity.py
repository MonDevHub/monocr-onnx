"""The model is trained on dark text on a light background; check what we feed it.

Measured 2026-08-27 over 300 labelled crops from mon_OCR's `data/real/digits/val`,
same graph, only the polarity of the input changed:

    upright, with the probe      CER 0.0000   300/300 exact
    inverted, with the probe     CER 0.0000   300/300 exact
    upright, without it          CER 0.0036   296/300
    inverted, without it         CER 0.0342   288/300   <- 9.5x worse

Degradation rather than total failure, and worth closing for the cost of reading
four corner patches.
"""

import numpy as np
import pytest
from PIL import Image

from monocr_onnx.predictor import normalize_polarity


def _page(bg: int, ink: int, w: int = 200, h: int = 60) -> Image.Image:
    a = np.full((h, w), bg, dtype=np.uint8)
    a[h // 3 : 2 * h // 3, w // 5 : 4 * w // 5] = ink
    return Image.fromarray(a, mode="L")


def test_a_dark_on_light_page_is_returned_unchanged():
    """THE NO-OP. Every input goes through this, so an ordinary page must be
    byte-identical or the probe is a regression rather than a fix."""
    page = _page(bg=255, ink=0)
    assert np.array_equal(np.asarray(normalize_polarity(page)), np.asarray(page))


def test_a_light_on_dark_page_is_inverted():
    page = _page(bg=0, ink=255)
    out = np.asarray(normalize_polarity(page))
    assert out[0, 0] == 255, "the dark background should have become light"
    assert out[30, 100] == 0, "the light ink should have become dark"


def test_inverting_twice_returns_the_original():
    """The probe is idempotent in the sense that matters: applying it to its own
    output must not flip the page back."""
    page = _page(bg=0, ink=255)
    once = normalize_polarity(page)
    twice = normalize_polarity(once)
    assert np.array_equal(np.asarray(once), np.asarray(twice))


def test_a_dense_page_is_not_mistaken_for_a_dark_one():
    """Why corner-median and not a global mean.

    This page is 70% ink, so its mean luminance is below 128 and a global test
    would invert it — destroying a perfectly ordinary dense page. The corners are
    background, so the median sees through it.
    """
    a = np.full((60, 200), 255, dtype=np.uint8)
    a[6:54, 20:180] = 0  # ~64% of the page, but the corners stay white
    page = Image.fromarray(a, mode="L")
    assert np.asarray(page).mean() < 128, "fixture must actually be mean-dark"
    assert np.array_equal(np.asarray(normalize_polarity(page)), np.asarray(page))


@pytest.mark.parametrize("size", [(1, 1), (2, 3), (3, 1)])
def test_a_tiny_image_does_not_trap(size):
    """The corner patch has a floor of 3px, which can exceed the image."""
    h, w = size
    page = Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode="L")
    normalize_polarity(page)


def test_the_probe_runs_inside_preprocess():
    """The unit above is worthless if nothing calls it.

    Asserted on the tensor rather than by mocking: an inverted page and its
    upright twin must reach the model as the same pixels.
    """
    from monocr_onnx import predictor as P

    class _FakeSession:
        def get_inputs(self):
            raise AssertionError("not needed")

    upright = _page(bg=255, ink=0)
    inverted = _page(bg=0, ink=255)

    pre = P.MonOcrOnnx.preprocess if hasattr(P, "MonOcrOnnx") else None
    if pre is None:  # class renamed; find whatever owns preprocess
        cls = next(
            v
            for v in vars(P).values()
            if isinstance(v, type) and callable(getattr(v, "preprocess", None))
        )
        pre = cls.preprocess

    class _Shim:
        input_height = 160
        input_width = 1024

    a = pre(_Shim(), upright)
    b = pre(_Shim(), inverted)
    assert a is not None and b is not None
    assert np.allclose(a, b), "an inverted page must reach the model as its upright twin"


def test_a_short_crop_still_gets_a_usable_corner_patch():
    """What `_POLARITY_CORNER_FLOOR` is for.

    The patch is `height // 10`, which is 0 for anything under 10px tall. Without
    the floor the corner sample is empty, `np.median([])` is nan, `nan < 128` is
    False, and a dark-background crop is silently left inverted — a wrong answer
    rather than a crash, which is why this needs its own test.
    """
    a = np.full((8, 120), 0, dtype=np.uint8)
    a[3:5, 10:110] = 255
    page = Image.fromarray(a, mode="L")

    out = np.asarray(normalize_polarity(page))
    assert out[0, 0] == 255, "an 8px-tall dark-background crop must still invert"


def test_a_narrow_crop_still_gets_a_usable_corner_patch():
    """The same floor on the other axis, pinned separately.

    `_POLARITY_CORNER_FLOOR` guards width as well as height, and a test for one
    does not cover the other — removing the width floor alone left the height
    test passing.
    """
    a = np.full((60, 8), 0, dtype=np.uint8)
    a[10:50, 3:5] = 255
    page = Image.fromarray(a, mode="L")

    out = np.asarray(normalize_polarity(page))
    assert out[0, 0] == 255, "an 8px-wide dark-background crop must still invert"


def test_polarity_runs_before_segmentation_not_only_per_crop():
    """THE ORDERING, which is what makes the probe useful on a page.

    The segmenter binarises with THRESH_BINARY_INV, so it treats dark as ink. On a
    light-on-dark page it segments the BACKGROUND and returns the gaps between
    lines; inverting each crop afterwards cannot recover a line never found.

    A probe in `preprocess` alone does not fix a page, and that is what shipped
    first. Asserted on the band count: an inverted page must segment into the same
    number of lines as its upright twin.
    """
    from monocr_onnx import predictor as P

    def page(bg, ink, w=900, h=260):
        a = np.full((h, w), bg, dtype=np.uint8)
        for top in (40, 140):
            for x in range(60, w - 60, 20):
                a[top : top + 60, x : x + 12] = ink
        return Image.fromarray(a, mode="L")

    seg = P.LineSegmenter()
    upright = len(seg.segment(P.normalize_polarity(page(255, 0))))
    inverted = len(seg.segment(P.normalize_polarity(page(0, 255))))

    assert upright == 2, f"the control must find 2 lines, found {upright}"
    assert inverted == upright, (
        f"an inverted page found {inverted} bands against the upright page's "
        f"{upright} — polarity is not being applied before segmentation"
    )


def test_predict_page_normalises_before_it_segments():
    """The ordering inside `predict_page`, pinned where it can actually fail.

    The test above exercises the probe and the segmenter together, which passes
    whether or not `predict_page` itself calls the probe first — verified by
    mutation: deleting the call from `predict_page` left it green. This reads the
    two call sites out of the function body and asserts their order.

    Source order rather than a runtime assertion because `predict_page` needs the
    46MB graph and a real inference to reach the segmenter, none of which bears on
    the question.
    """
    import ast
    import inspect
    import textwrap

    from monocr_onnx.predictor import MonOCR

    tree = ast.parse(textwrap.dedent(inspect.getsource(MonOCR.predict_page)))
    polarity_at = segment_at = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name == "normalize_polarity" and polarity_at is None:
            polarity_at = node.lineno
        if name == "segment" and segment_at is None:
            segment_at = node.lineno

    assert polarity_at is not None, (
        "predict_page does not call normalize_polarity. The segmenter binarises "
        "with THRESH_BINARY_INV, so a light-on-dark page segments the background "
        "and returns the gaps between lines; a probe in preprocess alone runs too "
        "late to help."
    )
    assert segment_at is not None, "predict_page no longer calls segment; update this test"
    assert polarity_at < segment_at, (
        f"normalize_polarity is called at line {polarity_at} and segment at "
        f"{segment_at}: the probe must run BEFORE segmentation, not after"
    )
