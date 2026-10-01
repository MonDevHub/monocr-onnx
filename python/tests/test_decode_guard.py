"""
A NaN or an infinity in the logits must fail the read, not decode.

Greedy CTC takes an argmax per timestep, and ``np.argmax`` over a row holding
NaN returns the NaN's index without complaint. A NaN at class 0 -- the blank --
deletes the timestep; anywhere else it emits that class's character. Either
way a numeric failure comes back as a plausible line of text. On real input
the pinned model's scores are finite, so the guard only fires on such a failure.
"""

import numpy as np
import pytest
from PIL import Image

from monocr_onnx.predictor import ModelContractError, ModelOutputError, check_logits

NUM_CLASSES = 277


def _line():
    a = np.full((40, 120), 255, dtype=np.uint8)
    a[10:30, 10:110] = 0
    return Image.fromarray(a, mode="L")


def _logits_for(indices, num_classes=NUM_CLASSES):
    """Logits that argmax to `indices`, one class per timestep."""
    out = np.full((1, len(indices), num_classes), -10.0, dtype=np.float32)
    for t, idx in enumerate(indices):
        out[0, t, idx] = 10.0
    return out


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "+inf", "-inf"])
def test_a_non_finite_logit_fails_the_read(make_ocr, bad):
    logits = _logits_for([2, 0, 3])
    logits[0, 1, 5] = bad
    ocr = make_ocr(logits=logits)
    with pytest.raises(ModelOutputError, match="non-finite"):
        ocr.predict_line(_line())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "+inf", "-inf"])
def test_check_logits_rejects_each_non_finite_value(bad):
    logits = _logits_for([1, 2])
    logits[0, 0, 0] = bad
    with pytest.raises(ModelOutputError):
        check_logits(logits, NUM_CLASSES)


def test_nan_would_otherwise_decode_silently():
    """Why the guard exists: argmax over NaN picks the NaN, no error raised."""
    row = np.full(NUM_CLASSES, -10.0, dtype=np.float32)
    row[7] = 10.0
    row[0] = np.nan
    assert int(np.argmax(row)) == 0  # the blank: this timestep vanishes


def test_finite_logits_still_decode(make_ocr):
    ocr = make_ocr(logits=_logits_for([2, 2, 0, 3]))
    expected = ocr.idx2char[2] + ocr.idx2char[3]
    assert ocr.predict_line(_line()) == expected


def test_a_class_axis_that_disagrees_with_the_charset_is_refused(make_ocr):
    """The graph's class axis can be dynamic, so the tensor is checked too."""
    ocr = make_ocr(logits=_logits_for([1, 2], num_classes=NUM_CLASSES - 1))
    with pytest.raises(ModelContractError, match="shape"):
        ocr.predict_line(_line())
