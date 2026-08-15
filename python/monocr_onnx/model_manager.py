"""
Download and cache the MonOCR ONNX artifact and the charset that belongs to it.

The two files are one unit. A model emits a fixed number of classes and the
charset names them; take the weights from one revision and the charset from
another and every index above the first divergence decodes to the wrong
character, with no error anywhere. So both are fetched from the same pinned
revision and both are checksummed.

The pin is not cosmetic. This package used to fetch
``.../resolve/main/onnx/monocr.onnx`` and ``main`` has already moved under it:
revision 881d167 published a 57,759,911-byte network with a 64px input height
and 225 output classes, and revision a51be11 publishes a 26,355,440-byte
network with a 128px input height and 316 output classes. Anyone who installed
before the reupload has the old file sitting in the cache under the same name.
"""

import hashlib
import shutil
from pathlib import Path

import requests
from tqdm import tqdm


class ModelDownloadError(RuntimeError):
    """Raised when an artifact could not be fetched, or arrived corrupted."""


class ModelManager:
    """Resolve the pinned model and charset, downloading them once per revision."""

    REPO_ID = "janakhpon/monocr"

    # Pinned revision. Bump REVISION and the two digests together — never one
    # alone. The cache is keyed on this string, so changing it is also what
    # invalidates the cache.
    REVISION = "a51be11"

    MODEL_FILENAME = "monocr.onnx"
    CHARSET_FILENAME = "charset.txt"

    # Path inside the repo -> local filename.
    _REMOTE_PATHS = {
        MODEL_FILENAME: "onnx/monocr.onnx",
        CHARSET_FILENAME: "onnx/charset.txt",
    }

    # sha256 of each file at REVISION. Verified after download, so a truncated
    # transfer or an HTML error page saved under a .onnx name fails loudly
    # instead of becoming a permanent poisoned cache entry.
    _SHA256 = {
        MODEL_FILENAME: "f212ab7e76c4dc7f120e600fe60ce3bd99227efa7e5ba402f446daa04b6271db",
        CHARSET_FILENAME: "0b61abeb13e1e5058c8792582d79cf91fa3b6f37295c1fd502bffdf04ba50cb9",
    }

    def __init__(self, cache_root=None, revision=None):
        self.revision = revision or self.REVISION
        self.cache_root = Path(cache_root) if cache_root else Path.home() / ".monocr" / "models"
        # Revision-scoped, which is the cache invalidation. A flat
        # `~/.monocr/models/monocr.onnx` guarded by `path.exists()` cannot tell
        # a stale artifact from a fresh one — the filename is identical either
        # way — so bumping the pin would silently keep serving the old weights.
        self.cache_dir = self.cache_root / self.revision

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def get_model_path(self):
        """Absolute path to the pinned ONNX model, downloading it if absent."""
        return self.ensure(self.MODEL_FILENAME)

    def get_charset_path(self):
        """Absolute path to the charset published alongside the pinned model."""
        return self.ensure(self.CHARSET_FILENAME)

    def url_for(self, filename):
        remote = self._REMOTE_PATHS[filename]
        return f"https://huggingface.co/{self.REPO_ID}/resolve/{self.revision}/{remote}"

    def clear_cache(self):
        """Delete this revision's cached files. Returns True if anything went."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            return True
        return False

    def legacy_cache_files(self):
        """
        Pre-pin cache entries: files written flat into ``~/.monocr/models``.

        Those predate the revision pin and may hold a different network
        entirely. Nothing reads them any more; this exists so the CLI can tell
        the user they are dead weight.
        """
        if not self.cache_root.is_dir():
            return []
        return sorted(p for p in self.cache_root.iterdir() if p.is_file())

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def ensure(self, filename):
        """Return the cached path for one pinned file, downloading it if needed."""
        dest = self.cache_dir / filename
        if dest.exists() and self._digest(dest) == self._SHA256.get(filename):
            return dest
        if dest.exists():
            # Present but wrong: a partial write, or a file from before the
            # digests were pinned. Re-fetch rather than trust it.
            dest.unlink()
        self._download(filename, dest)
        return dest

    def _download(self, filename, dest):
        url = self.url_for(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        print(f"Downloading {filename} ({self.REPO_ID}@{self.revision})...")
        try:
            response = requests.get(url, stream=True, allow_redirects=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    bar.update(f.write(chunk))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise ModelDownloadError(f"failed to download {url}: {e}") from e

        expected = self._SHA256.get(filename)
        actual = self._digest(tmp)
        if expected and actual != expected:
            tmp.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"checksum mismatch for {filename} from {url}\n"
                f"  expected sha256 {expected}\n"
                f"  got      sha256 {actual}\n"
                "The pinned revision is immutable, so this is a corrupted or "
                "intercepted download, not a new release."
            )

        # Atomic-ish: the file only appears at its final name once it is whole
        # and verified, so an interrupted run cannot leave a half-model behind.
        tmp.replace(dest)
        print(f"Saved to {dest}")

    @staticmethod
    def _digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
