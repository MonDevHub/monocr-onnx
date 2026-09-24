from .ocr import read_image, read_images, read_pdf, read_pdfs, read_image_with_accuracy
from .predictor import MonOCR, ModelContractError
from .model_manager import ModelManager, ModelDownloadError

# Read from the installed metadata, so pyproject.toml is the only place a
# version is written.
#
# This was a hand-maintained literal that nothing checked, and it drifted: the
# 0.3.2, 0.4.0 and 0.4.1 wheels all reported `__version__ == "0.3.0"`. A
# version bump touches pyproject.toml and nothing else, so a second copy goes
# stale on every release. Deriving it removes that class of mistake instead of
# detecting it.
try:
    from importlib.metadata import PackageNotFoundError, version as _installed_version

    __version__ = _installed_version("monocr-onnx")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "read_image",
    "read_images",
    "read_pdf",
    "read_pdfs",
    "read_image_with_accuracy",
    "MonOCR",
    "ModelContractError",
    "ModelManager",
    "ModelDownloadError",
]
