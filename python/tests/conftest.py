"""
Shared fixtures.

Nothing here touches the network or loads a real ONNX file. The contract this
package has to defend is expressible as two numbers off the graph — the class
count and the input height — so a fake session that reports those numbers
exercises it exactly as a 26MB download would.
"""

import numpy as np
import pytest

from monocr_onnx import predictor


class FakeIO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    """Stands in for ``onnxruntime.InferenceSession``.

    ``shape`` entries that are strings are dynamic axes, matching how
    onnxruntime reports them (the real pinned model reports
    ``[1, 1, 160, 1024]`` and ``[1, 'sequence', 277]``).
    """

    def __init__(self, num_classes=277, height=160, width=1024, logits=None):
        self._inputs = [FakeIO("input", [1, 1, height, width])]
        self._outputs = [FakeIO("logits", [1, "sequence", num_classes])]
        self._logits = logits
        self.last_input = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        self.last_input = next(iter(feed.values()))
        if self._logits is None:
            return [np.zeros((1, 4, self._outputs[0].shape[-1]), dtype=np.float32)]
        return [self._logits]


@pytest.fixture
def make_ocr(monkeypatch):
    """Build a MonOCR wired to a FakeSession, with the bundled charset."""

    def _make(**session_kwargs):
        session = FakeSession(**session_kwargs)
        monkeypatch.setattr(
            predictor.ort, "InferenceSession", lambda path, providers=None: session
        )
        ocr = predictor.MonOCR(model_path="does-not-exist.onnx")
        ocr.fake_session = session
        return ocr

    return _make


@pytest.fixture
def patched_session(monkeypatch):
    """Install a FakeSession without constructing MonOCR (for failure cases)."""

    def _patch(**session_kwargs):
        session = FakeSession(**session_kwargs)
        monkeypatch.setattr(
            predictor.ort, "InferenceSession", lambda path, providers=None: session
        )
        return session

    return _patch
