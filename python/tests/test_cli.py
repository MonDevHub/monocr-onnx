"""
``monocr batch`` exits non-zero when a file fails.

It used to report each failure on stderr and still exit 0, so a script or CI
step checking the status saw success even when every file had failed.
"""

from click.testing import CliRunner

from monocr_onnx import cli, predictor


class _OCR:
    def __init__(self, *a, **kw):
        pass

    def predict(self, path):
        if "bad" in str(path):
            raise ValueError("cannot decode")
        return "text"


def _run(tmp_path, monkeypatch, names):
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(predictor, "MonOCR", _OCR)
    return CliRunner().invoke(cli.main, ["batch", str(tmp_path)])


def test_batch_exits_nonzero_when_a_file_fails(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, ["a.png", "bad.png"])
    assert result.exit_code == 1
    assert "--- a.png ---" in result.output


def test_batch_exits_nonzero_when_every_file_fails(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, ["bad1.png", "bad2.jpg"])
    assert result.exit_code == 1


def test_batch_exits_zero_when_every_file_succeeds(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, ["a.png", "b.JPG"])
    assert result.exit_code == 0
