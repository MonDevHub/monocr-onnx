"""
The download must be pinned, verified, and cached in a way that can see a
revision change.

0.1.0 fetched ``.../resolve/main/onnx/monocr.onnx`` and guarded the cache with
``path.exists()``. ``main`` has already moved: the artifact went from a 57.7MB /
H=64 / 225-class network to a 26.4MB / H=128 / 316-class one, then to a
46.2MB / H=160 / 277-class one, all under the same
filename, so every existing install kept the old weights forever.
"""

import hashlib

import pytest

from monocr_onnx.model_manager import ModelDownloadError, ModelManager


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.payload), chunk_size):
            yield self.payload[i : i + chunk_size]


@pytest.fixture
def serve(monkeypatch):
    """Serve fixed bytes for any URL, and record what was requested."""
    requested = []

    def _serve(payload):
        def fake_get(url, **kwargs):
            requested.append(url)
            return FakeResponse(payload)

        monkeypatch.setattr("monocr_onnx.model_manager.requests.get", fake_get)
        return requested

    return _serve


@pytest.fixture
def manager(tmp_path):
    return ModelManager(cache_root=tmp_path)


# ----------------------------------------------------------------------
# The pin
# ----------------------------------------------------------------------


def test_urls_carry_the_pinned_revision_and_never_main(manager):
    for filename in (manager.MODEL_FILENAME, manager.CHARSET_FILENAME):
        url = manager.url_for(filename)
        assert f"/resolve/{ModelManager.REVISION}/" in url
        assert "/resolve/main/" not in url


def test_repo_org_is_janakhpon(manager):
    """``janakh`` is a different account and returns 401."""
    assert manager.REPO_ID == "janakhpon/monocr"
    for filename in (manager.MODEL_FILENAME, manager.CHARSET_FILENAME):
        assert "huggingface.co/janakhpon/monocr/" in manager.url_for(filename)


def test_model_and_charset_come_from_the_same_revision(manager):
    model_url = manager.url_for(manager.MODEL_FILENAME)
    charset_url = manager.url_for(manager.CHARSET_FILENAME)
    prefix = f"/resolve/{ModelManager.REVISION}/"
    assert model_url.split(prefix)[0] == charset_url.split(prefix)[0]


def test_every_pinned_file_has_a_pinned_digest():
    assert set(ModelManager._SHA256) == set(ModelManager._REMOTE_PATHS)


# ----------------------------------------------------------------------
# Cache invalidation
# ----------------------------------------------------------------------


def test_cache_dir_is_scoped_by_revision(manager, tmp_path):
    assert manager.cache_dir == tmp_path / ModelManager.REVISION


def test_bumping_the_revision_misses_the_cache(tmp_path, serve):
    payload = b"weights-v1"
    digest = hashlib.sha256(payload).hexdigest()

    old = ModelManager(cache_root=tmp_path, revision="aaaaaaa")
    new = ModelManager(cache_root=tmp_path, revision="bbbbbbb")
    for m in (old, new):
        m._SHA256 = dict(m._SHA256, **{m.MODEL_FILENAME: digest})

    requested = serve(payload)
    old.get_model_path()
    assert len(requested) == 1

    # Same filename, same bytes on disk under the old revision — and it must
    # still re-fetch, because `path.exists()` cannot see a revision change.
    new.get_model_path()
    assert len(requested) == 2
    assert old.get_model_path() != new.get_model_path()


def test_a_cached_file_is_reused_without_a_second_request(tmp_path, serve):
    payload = b"weights"
    m = ModelManager(cache_root=tmp_path)
    m._SHA256 = dict(m._SHA256, **{m.MODEL_FILENAME: hashlib.sha256(payload).hexdigest()})
    requested = serve(payload)

    first = m.get_model_path()
    second = m.get_model_path()
    assert first == second
    assert len(requested) == 1


def test_clear_cache_removes_only_this_revision(tmp_path, serve):
    payload = b"weights"
    m = ModelManager(cache_root=tmp_path)
    m._SHA256 = dict(m._SHA256, **{m.MODEL_FILENAME: hashlib.sha256(payload).hexdigest()})
    serve(payload)
    path = m.get_model_path()

    other = tmp_path / "other-rev" / "monocr.onnx"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")

    assert m.clear_cache() is True
    assert not path.exists()
    assert other.exists()
    assert m.clear_cache() is False


def test_legacy_flat_cache_files_are_reported_not_read(tmp_path, serve):
    """
    The pre-pin layout wrote ``~/.monocr/models/monocr.onnx``. Those files may
    be an entirely different network; nothing reads them any more.
    """
    stale = tmp_path / "monocr.onnx"
    stale.write_bytes(b"an-old-network")

    payload = b"weights"
    m = ModelManager(cache_root=tmp_path)
    m._SHA256 = dict(m._SHA256, **{m.MODEL_FILENAME: hashlib.sha256(payload).hexdigest()})
    serve(payload)

    assert m.get_model_path().read_bytes() == payload
    assert stale in m.legacy_cache_files()


# ----------------------------------------------------------------------
# Integrity
# ----------------------------------------------------------------------


def test_a_corrupt_download_raises_and_leaves_nothing_behind(manager, serve):
    """
    The pinned revision is immutable, so wrong bytes are a bad transfer — an
    HTML error page, a truncated stream — and must never become a cache entry.
    """
    serve(b"<html>not a model</html>")
    with pytest.raises(ModelDownloadError) as exc:
        manager.get_model_path()
    assert "checksum" in str(exc.value)
    assert not (manager.cache_dir / manager.MODEL_FILENAME).exists()
    assert list(manager.cache_dir.glob("*.part")) == []


def test_a_stale_cache_entry_with_the_wrong_digest_is_refetched(tmp_path, serve):
    payload = b"weights"
    m = ModelManager(cache_root=tmp_path)
    m._SHA256 = dict(m._SHA256, **{m.MODEL_FILENAME: hashlib.sha256(payload).hexdigest()})

    poisoned = m.cache_dir / m.MODEL_FILENAME
    poisoned.parent.mkdir(parents=True)
    poisoned.write_bytes(b"wrong-bytes")

    requested = serve(payload)
    assert m.get_model_path().read_bytes() == payload
    assert len(requested) == 1


def test_a_network_failure_leaves_no_partial_file(manager, monkeypatch):
    def boom(url, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr("monocr_onnx.model_manager.requests.get", boom)
    with pytest.raises(ModelDownloadError):
        manager.get_model_path()
    assert not (manager.cache_dir / manager.MODEL_FILENAME).exists()


def test_the_bundled_charset_digest_matches_the_pinned_one():
    """
    Ties the offline fallback to the download. If the bundled file and the
    pinned remote file ever diverge, the charset used depends on whether the
    download succeeded — which is exactly the class of bug this pin exists to
    remove.
    """
    import importlib.resources

    path = importlib.resources.files("monocr_onnx") / "charset.txt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == ModelManager._SHA256[ModelManager.CHARSET_FILENAME]
