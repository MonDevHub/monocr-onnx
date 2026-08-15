"""
The charset is an index-ordered list, so its identity is its exact bytes.

Two ways it went wrong in 0.1.0, both silent:
  * the bundled file was a 225-character subset of the model's then-315-character
    output space, aligned only for the first 95 ASCII entries;
  * it was read with a bare ``.strip()``, which eats the leading U+0020 and
    shifts every index by one.
"""

import hashlib
import importlib.resources
from pathlib import Path

import pytest

from monocr_onnx.model_manager import ModelManager
from monocr_onnx.predictor import _read_charset

CHARSET_PATH = Path(importlib.resources.files("monocr_onnx") / "charset.txt")


def bundled_text():
    return CHARSET_PATH.read_text(encoding="utf-8")


def test_bundled_charset_is_byte_identical_to_the_pinned_remote_copy():
    """
    The strongest form of the check: the shipped charset must be the same file
    the pinned revision publishes next to the weights. This catches both a
    wrong charset and any drift between the bundled copy and the download.
    """
    digest = hashlib.sha256(CHARSET_PATH.read_bytes()).hexdigest()
    assert digest == ModelManager._SHA256[ModelManager.CHARSET_FILENAME]


def test_bundled_charset_has_276_characters():
    """277 model classes = 276 characters + the CTC blank at index 0.

    315 until 2026-08-15, when the pinned revision moved from the v2 network
    to v3.5. The number here and MODEL_REVISION move together or not at all.
    """
    assert len(_read_charset(bundled_text())) == 276


def test_charset_begins_with_a_space():
    """U+0020 is a character the model predicts, not leading whitespace."""
    assert bundled_text().startswith(" ")
    assert _read_charset(bundled_text())[0] == " "


def test_bare_strip_would_lose_the_leading_space():
    """
    Pins the trap itself. `.strip()` silently returns 314 characters, which
    still loads, still decodes, and produces the wrong glyph for every index.
    """
    text = bundled_text()
    assert len(text.strip()) == len(_read_charset(text)) - 1
    assert text.strip()[0] != " "


def test_read_charset_keeps_spaces_and_drops_only_line_endings():
    assert _read_charset(" abc \n") == " abc "
    assert _read_charset(" abc \r\n") == " abc "
    assert _read_charset(" abc ") == " abc "


def test_charset_is_not_the_old_225_character_subset():
    """
    The old bundle diverged at index 95 (0-based): it jumped straight from
    ASCII '~' into Myanmar, where the real charset continues into Latin-1.
    """
    charset = _read_charset(bundled_text())
    assert charset[95] == "£"  # £
    assert charset[95] != "က"  # က


@pytest.mark.parametrize("index", [0, 94, 95, 275])
def test_charset_indices_are_stable(index):
    """Guards against a re-ordered charset, which would shift every decode."""
    expected = {0: " ", 94: "~", 95: "£", 275: "−"}
    assert _read_charset(bundled_text())[index] == expected[index]
