"""
read_pdf rasterises one page at a time.

It used to call ``convert_from_path`` once for the whole document, which at
300 dpi holds every page in memory together (about 25 MB of RGB per A4 page)
before the first is recognised. Nothing here needs poppler: the two pdf2image
calls are replaced.
"""

from PIL import Image

from monocr_onnx import ocr


class _Recorder:
    def __init__(self):
        self.calls = []
        self.predicted = []


def test_read_pdf_converts_one_page_per_call(monkeypatch):
    rec = _Recorder()

    def fake_info(path):
        return {"Pages": 3}

    def fake_convert(path, dpi=200, first_page=None, last_page=None, **kw):
        rec.calls.append((first_page, last_page))
        assert first_page is not None and first_page == last_page, (
            "convert_from_path asked for more than one page at once"
        )
        return [Image.new("L", (4, 4), color=first_page)]

    class FakeOCR:
        def __init__(self, *a, **kw):
            pass

        def predict(self, img):
            rec.predicted.append(img.getpixel((0, 0)))
            return f"page {img.getpixel((0, 0))}"

    monkeypatch.setattr(ocr, "pdfinfo_from_path", fake_info)
    monkeypatch.setattr(ocr, "convert_from_path", fake_convert)
    monkeypatch.setattr(ocr, "MonOCR", FakeOCR)

    assert ocr.read_pdf("doc.pdf") == ["page 1", "page 2", "page 3"]
    assert rec.calls == [(1, 1), (2, 2), (3, 3)]
