"""Cutting a wide line into model-window tiles, instead of squeezing it.

`preprocess` clamps a line's width to the canvas, which compresses the whole
line horizontally and breaks the aspect ratio the model was trained on.

This docstring used to price that at `CER 0.1434 squeezed against 0.0795 tiled`.
RETIRED 2026-08-22: that harness was never committed and the figures do not
reproduce. Remeasured over 201 rendered lines in
`mon_OCR/eval/tiling-ab-2026-08-22.md`, the cost is width-dependent — squeezing
wins at 2 tiles, the two arms are level at 3, and tiling wins from 4 up.

These tests pin the mechanism rather than the preference, which is why the
retirement does not change any assertion below.

The cut position is the whole point. Cutting at an arbitrary pixel lands inside
a glyph, and the model then reads each half as a whole character — upstream this
turned `ဗော်` into `ဗေဗိာ်`.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from monocr_onnx.segmenter import _CUT_INK_THRESHOLD, cut_column, tile_line


def line_with_gap(width=3000, height=100, gap_at=None, gap_width=60):
    """A line of ink with one deliberate white column band."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 20, width, height - 20], fill=0)
    if gap_at is not None:
        draw.rectangle([gap_at, 0, gap_at + gap_width, height], fill=255)
    return img


# ---------------------------------------------------------------------------
# tile_line
# ---------------------------------------------------------------------------


def test_a_line_that_already_fits_is_returned_untouched():
    """No tiling, and the same object — not a re-crop that could shift a pixel."""
    img = Image.new("L", (200, 100), 255)
    tiles = tile_line(img, target_h=160, target_w=1024)
    assert len(tiles) == 1
    assert tiles[0] is img


def test_a_line_exactly_at_the_canvas_width_is_not_tiled():
    """The boundary case. 640x100 scales to 1024x160 exactly."""
    img = Image.new("L", (640, 100), 255)
    assert len(tile_line(img, target_h=160, target_w=1024)) == 1


def test_a_line_one_pixel_over_is_tiled():
    """The other side of the same boundary, so the comparison cannot be >= by accident."""
    img = Image.new("L", (641, 100), 255)
    assert len(tile_line(img, target_h=160, target_w=1024)) > 1


def test_tiles_cover_the_line_with_no_gap_and_no_overlap():
    """Every source pixel column appears in exactly one tile.

    Coverage is what stops tiling from silently dropping text; it is the
    property a naive fixed-stride implementation gets right and a
    cut-at-whitespace one can get wrong.
    """
    img = line_with_gap(width=3000, height=100, gap_at=1400)
    tiles = tile_line(img, target_h=160, target_w=1024)
    assert sum(t.width for t in tiles) == img.width


def test_no_tile_exceeds_the_model_window_after_scaling():
    """A tile wider than the canvas would be squeezed again, defeating the point."""
    img = line_with_gap(width=5000, height=100, gap_at=2000)
    tiles = tile_line(img, target_h=160, target_w=1024)
    scale = 160 / img.height
    assert tiles, "a wide line must produce tiles"
    for tile in tiles:
        assert int(tile.width * scale) <= 1024, f"tile of {tile.width}px overflows the window"


def test_a_solid_line_with_no_gap_still_terminates():
    """The anti-hang guard. With no whitespace anywhere, cut_column returns the
    ideal boundary and the loop must still advance."""
    img = Image.new("L", (4000, 100), 0)  # ink edge to edge
    tiles = tile_line(img, target_h=160, target_w=1024)
    assert sum(t.width for t in tiles) == 4000
    assert all(t.width > 0 for t in tiles)


def test_a_degenerate_crop_is_returned_rather_than_divided_by_zero():
    for size in ((0, 100), (100, 0)):
        img = Image.new("L", size, 255)
        assert tile_line(img, target_h=160, target_w=1024) == [img]


# ---------------------------------------------------------------------------
# cut_column
# ---------------------------------------------------------------------------


def test_the_cut_lands_on_the_whitespace_not_the_ideal_boundary():
    """The behaviour the whole port exists for."""
    gap_at, gap_width = 900, 60
    img = line_with_gap(width=3000, height=100, gap_at=gap_at, gap_width=gap_width)

    cut = cut_column(img, x0=0, ideal=1000, crop_w=3000)

    assert gap_at <= cut <= gap_at + gap_width, (
        f"cut at {cut} is outside the white band [{gap_at}, {gap_at + gap_width}]"
    )
    assert cut != 1000, "cutting at the ideal boundary is the defect being fixed"


def test_the_cut_prefers_the_rightmost_blank_column():
    """Tiles should stay as wide as the search window allows."""
    img = line_with_gap(width=3000, height=100, gap_at=900, gap_width=60)
    cut = cut_column(img, x0=0, ideal=1000, crop_w=3000)
    column = np.asarray(img.crop((cut - 1, 0, cut, img.height)), dtype=np.uint8)
    assert (column >= _CUT_INK_THRESHOLD).all(), "the column left of the cut carries ink"


def test_the_cut_never_moves_forward_past_the_ideal():
    """A wider tile would not fit the model window."""
    img = line_with_gap(width=3000, height=100, gap_at=1500)
    assert cut_column(img, x0=0, ideal=1000, crop_w=3000) <= 1000


def test_the_cut_never_returns_the_start():
    """Returning x0 would make tile_line spin forever."""
    img = Image.new("L", (3000, 100), 0)
    assert cut_column(img, x0=500, ideal=1500, crop_w=3000) > 500


def test_the_last_tile_ends_at_the_line_end():
    """When the ideal boundary is the end of the line there is nothing to search."""
    img = line_with_gap(width=1200, height=100, gap_at=600)
    assert cut_column(img, x0=0, ideal=1200, crop_w=1200) == 1200
    assert cut_column(img, x0=0, ideal=1500, crop_w=1200) == 1200


def test_the_lightest_column_is_used_when_nothing_is_fully_blank():
    """A continuous script may offer no gap. A known-bad seam beats an overflow."""
    img = Image.new("L", (3000, 100), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([950, 0, 952, 40], fill=255)  # lighter, not blank

    cut = cut_column(img, x0=0, ideal=1000, crop_w=3000)

    assert 0 < cut <= 1000
    assert 940 <= cut <= 960, f"expected the cut near the lightest column, got {cut}"


def test_a_three_channel_crop_is_converted_before_the_column_sum():
    """On an RGB array the profile would be (W, 3) and the argmin a flat index."""
    img = line_with_gap(width=3000, height=100, gap_at=900).convert("RGB")
    cut = cut_column(img, x0=0, ideal=1000, crop_w=3000)
    assert 900 <= cut <= 960


# ---------------------------------------------------------------------------
# predict_page wiring
# ---------------------------------------------------------------------------


def test_a_wide_line_reaches_the_model_more_than_once(make_ocr, monkeypatch):
    """The tiles must actually be predicted, and joined with no separator."""
    ocr = make_ocr()
    seen = []

    def fake_predict_line(crop):
        seen.append(crop.width)
        return "x"

    monkeypatch.setattr(ocr, "predict_line", fake_predict_line)
    text = ocr._read_line_tiled(line_with_gap(width=3000, height=100, gap_at=900))

    assert len(seen) > 1, "a 3000px line at height 160 needs more than one window"
    assert text == "x" * len(seen), "tiles must join with no separator"


def test_a_narrow_line_is_predicted_once(make_ocr, monkeypatch):
    ocr = make_ocr()
    calls = []
    monkeypatch.setattr(ocr, "predict_line", lambda crop: calls.append(crop) or "y")

    assert ocr._read_line_tiled(Image.new("L", (200, 100), 255)) == "y"
    assert len(calls) == 1


def test_tiling_terminates_even_if_the_cut_never_advances():
    """The anti-hang guard, forced onto a path that cannot occur naturally.

    `cut_column` cannot return `x0` — `lo` is `max(x0 + 1, ...)` — so the
    `max(x1, x0 + 1)` in `tile_line` is unreachable through the public API and
    no ordinary test can kill it. That is exactly why it is worth pinning: a
    later change to `cut_column` could make it reachable, and the symptom would
    be a page that never finishes rather than a wrong answer.

    Stubbing the cut to stand still and asserting the loop still terminates is
    the only way to exercise it. The work runs on a thread so a regression fails
    the test in two seconds instead of hanging the suite.
    """
    import threading

    from monocr_onnx import segmenter as seg

    original = seg.cut_column
    seg.cut_column = lambda crop, x0, ideal, crop_w: x0  # never advances
    result = {}

    def run():
        # Small enough that 1px-at-a-time still finishes quickly, wide enough
        # that 700 * (160/100) = 1120 exceeds the 1024 canvas and tiling starts.
        result["tiles"] = seg.tile_line(Image.new("L", (700, 100), 255), 160, 1024)

    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        worker.join(timeout=2.0)
    finally:
        seg.cut_column = original

    assert not worker.is_alive(), (
        "tile_line did not terminate when the cut stood still — the "
        "max(x1, x0 + 1) guard is gone and a page would hang"
    )
    tiles = result["tiles"]
    assert sum(t.width for t in tiles) == 700
    assert all(t.width >= 1 for t in tiles)
