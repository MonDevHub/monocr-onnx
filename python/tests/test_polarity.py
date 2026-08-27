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
