"""
Preprocessing has to reproduce the training tensor exactly.

0.1.0 divided by 255, giving [0, 1], and fed a variable-width tensor with no
padding. The network was trained on ``canvas / 127.5 - 1.0`` over a
height-scaled, white-padded 1024px canvas.
"""

import numpy as np
import pytest
from PIL import Image


def test_white_maps_to_plus_one_and_black_to_minus_one(make_ocr):
    """
    The regression test for the /255 bug. Under [0, 1] normalisation white is
    1.0 and black is 0.0, so the black assertion is what fails.
    """
    ocr = make_ocr()
    white = ocr.preprocess(Image.new("L", (200, 40), color=255))
    black = ocr.preprocess(Image.new("L", (200, 40), color=0))

    assert white.min() == pytest.approx(1.0)
    assert white.max() == pytest.approx(1.0)
    assert black[:, :, :, :200].min() == pytest.approx(-1.0)


def test_mid_grey_maps_to_zero(make_ocr):
    """127.5 is the midpoint of the range; /255 would put it at 0.5."""
    ocr = make_ocr()
    arr = ocr.preprocess(Image.new("L", (200, 40), color=128))
    assert arr[0, 0, 0, 0] == pytest.approx(128 / 127.5 - 1.0, abs=1e-6)
    assert abs(arr[0, 0, 0, 0]) < 0.01


def test_output_range_never_leaves_minus_one_to_one(make_ocr):
    ocr = make_ocr()
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 256, (37, 300), dtype=np.uint8), mode="L")
    arr = ocr.preprocess(noise)
    assert arr.min() >= -1.0
    assert arr.max() <= 1.0


def test_tensor_is_padded_to_the_full_canvas(make_ocr):
    """
    Shape is [1, 1, H, W] with W the training canvas width, not the scaled
    image width. 0.1.0 returned a tensor whose width varied per image.
    """
    ocr = make_ocr()
    arr = ocr.preprocess(Image.new("L", (100, 50), color=0))
    assert arr.shape == (1, 1, 160, 1024)
    assert arr.dtype == np.float32


def test_padding_is_white_not_black(make_ocr):
    """
    White padding is +1.0 after normalisation. Zero-padding would look like
    mid-grey ink to the network and hallucinate characters past the line end.
    """
    ocr = make_ocr()
    arr = ocr.preprocess(Image.new("L", (100, 50), color=0))
    scaled_width = int(100 * (160 / 50))
    assert np.allclose(arr[0, 0, :, scaled_width:], 1.0)


def test_aspect_ratio_is_preserved_below_the_canvas_width(make_ocr):
    ocr = make_ocr()
    arr = ocr.preprocess(Image.new("L", (200, 40), color=0))
    ink_columns = int(np.count_nonzero(arr[0, 0, 0, :] < 0))
    assert ink_columns == pytest.approx(200 * (160 / 40), abs=1)


def test_wide_lines_are_squeezed_not_truncated(make_ocr):
    """
    A line whose height-scaled width exceeds the canvas is compressed into it.
    Truncating instead would drop the tail of the line with no error.
    """
    ocr = make_ocr()
    arr = ocr.preprocess(Image.new("L", (4000, 40), color=0))
    assert arr.shape[3] == 1024
    assert np.all(arr[0, 0, :, -1] < 0)  # ink reaches the right edge


def test_dimensions_come_from_the_graph_not_a_constant(make_ocr):
    """A model traced at a static width must be fed that width."""
    ocr = make_ocr(width=512)
    assert ocr.input_width == 512
    assert ocr.preprocess(Image.new("L", (100, 50))).shape == (1, 1, 160, 512)


def test_rgb_input_is_converted_to_single_channel(make_ocr):
    ocr = make_ocr()
    arr = ocr.preprocess(Image.new("RGB", (100, 50), color=(255, 255, 255)))
    assert arr.shape == (1, 1, 160, 1024)


def test_degenerate_image_returns_none(make_ocr):
    ocr = make_ocr()
    assert ocr.preprocess(Image.new("L", (10, 0))) is None
    assert ocr.preprocess(Image.new("L", (0, 10))) is None


def test_predict_line_feeds_the_normalized_tensor(make_ocr):
    """End-to-end through predict_line: the session sees [-1, 1], not [0, 1]."""
    ocr = make_ocr()
    ocr.predict_line(Image.new("L", (200, 40), color=255))
    assert ocr.fake_session.last_input.min() == pytest.approx(1.0)
