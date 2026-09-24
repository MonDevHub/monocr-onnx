"""The version number, and the one place it is written.

`__version__` was a hand-maintained literal. The 0.3.2, 0.4.0 and 0.4.1 wheels
all reported "0.3.0", because a version bump edits pyproject.toml and nothing
else, and no test compared the two. It is now read from the installed metadata;
these tests keep it that way.
"""

import re
import tomllib
from importlib.metadata import version as installed_version
from pathlib import Path

import monocr_onnx

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "monocr_onnx" / "__init__.py"


def declared_version():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_version_matches_pyproject():
    """pyproject.toml, the installed metadata and `__version__` agree.

    `__version__` reads the metadata, so the first two are the only independent
    sources: this catches an install that has drifted from the manifest it was
    built from, and a regression to a literal that has drifted from both.
    """
    assert declared_version() == installed_version("monocr-onnx")
    assert monocr_onnx.__version__ == declared_version()


def test_version_is_not_the_uninstalled_fallback():
    """`0.0.0+unknown` means the package is imported from a source tree that
    was never installed, and every version it reports is fiction."""
    assert monocr_onnx.__version__ != "0.0.0+unknown"


def test_no_release_version_is_hardcoded_in_the_package():
    """The literal this replaced. The uninstalled sentinel is allowed; a
    semver literal is not."""
    init = INIT.read_text()
    assert "importlib.metadata" in init, "__version__ must come from the installed metadata"
    hardcoded = re.search(r'__version__\s*=\s*["\']\d+\.\d+\.\d+["\']', init)
    assert hardcoded is None, (
        f"__version__ is assigned {hardcoded.group(0) if hardcoded else ''} again; "
        "it drifts from pyproject.toml on the next release"
    )
