"""
Fail closed when the artifact and the charset describe different output spaces.

A mismatched pair still loads, still runs and still returns text. 0.1.0 shipped
a 225-character charset against a 316-class model and reported nothing at all —
``idx2char.get(idx, "")`` turned roughly half the predicted indices into empty
strings.
"""

import numpy as np
import pytest

from monocr_onnx.predictor import MonOCR, ModelContractError


def test_matching_artifact_loads(make_ocr):
    ocr = make_ocr(num_classes=316, height=128)
    assert len(ocr.charset) == 315
    assert ocr.num_classes == 316
    assert ocr.input_height == 128


def test_too_few_classes_is_refused(patched_session):
    """The exact shape of the shipped bug, from the other side."""
    patched_session(num_classes=226)
    with pytest.raises(ModelContractError) as exc:
        MonOCR(model_path="fake.onnx")
    assert "316" in str(exc.value) or "226" in str(exc.value)


def test_too_many_classes_is_refused(patched_session):
    patched_session(num_classes=400)
    with pytest.raises(ModelContractError):
        MonOCR(model_path="fake.onnx")


def test_off_by_one_class_count_is_refused(patched_session):
    """
    315 classes against a 315-character charset is the CTC blank going missing,
    or the charset's leading space being stripped. One character of drift
    corrupts every glyph past it, so it has to be an error and not a warning.
    """
    patched_session(num_classes=315)
    with pytest.raises(ModelContractError):
        MonOCR(model_path="fake.onnx")


def test_wrong_input_height_is_refused(patched_session):
    """
    A 64px model is the artifact that used to live at this URL. It is a
    different network, not something to resize into.
    """
    patched_session(num_classes=316, height=64)
    with pytest.raises(ModelContractError) as exc:
        MonOCR(model_path="fake.onnx")
    assert "height" in str(exc.value).lower()


def test_contract_error_names_both_sides_and_their_sources(patched_session):
    patched_session(num_classes=226)
    with pytest.raises(ModelContractError) as exc:
        MonOCR(model_path="fake.onnx")
    message = str(exc.value)
    assert "charset" in message
    assert "fake.onnx" in message
    assert "225 characters" in message  # what 226 classes implies


def test_empty_charset_is_refused(patched_session, tmp_path):
    patched_session()
    empty = tmp_path / "charset.txt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ModelContractError):
        MonOCR(model_path="fake.onnx", charset_path=str(empty))


def test_dynamic_class_axis_skips_the_load_check(patched_session):
    """Nothing to compare against — but decoding still refuses bad indices."""
    patched_session(num_classes="classes")
    ocr = MonOCR(model_path="fake.onnx")
    assert ocr.num_classes is None


def test_decode_refuses_an_index_outside_the_charset(patched_session):
    """
    The last line of defence, reachable only when the class axis is dynamic.
    0.1.0 dropped these characters silently instead.
    """
    patched_session(num_classes="classes")
    ocr = MonOCR(model_path="fake.onnx")
    with pytest.raises(ModelContractError) as exc:
        ocr.decode(np.array([1, 999, 2]))
    assert "999" in str(exc.value)


def test_decode_collapses_repeats_and_drops_blanks(make_ocr):
    ocr = make_ocr()
    ocr.idx2char = {1: "a", 2: "b", 3: "c"}
    ocr.charset = "abc"
    assert ocr.decode(np.array([1, 1, 0, 1, 2, 2, 0, 3])) == "aabc"


def test_decode_keeps_the_space_character(make_ocr):
    """Index 1 is U+0020; a decode that trims it loses real content."""
    ocr = make_ocr()
    assert ocr.decode(np.array([1])) == " "
