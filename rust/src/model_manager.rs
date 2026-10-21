//! Model Manager
//!
//! This module handles downloading and caching the ONNX model used for OCR.
//! Models are downloaded from HuggingFace and stored in the user's cache directory.

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The Hugging Face repository holding the ONNX artifact.
pub const MODEL_REPO: &str = "janakhpon/monocr";

/// The pinned revision.
///
/// `main` is a moving ref and the artifact has already changed under it: the
/// model served at one point had a 64-pixel input and 225 output classes, v2
/// had 128 and 316, and the one served now has 160 and 277. A cache that gates on "the file exists"
/// cannot tell those apart, so the revision is part of the cache path.
pub const MODEL_REVISION: &str = "d3d9d5e";

/// Filename of the ONNX model within the repository's `onnx/` directory.
pub const MODEL_FILENAME: &str = "monocr.onnx";

/// Filename of the charset that belongs to the same revision.
pub const CHARSET_FILENAME: &str = "charset.txt";

/// sha256 of [`MODEL_FILENAME`] at [`MODEL_REVISION`].
///
/// Copied from `python/monocr_onnx/model_manager.py` (`_SHA256`), which is the
/// source of truth; a test fails if the two drift. Bump the revision and both
/// digests together, never one alone.
///
/// All four bindings share `~/.monocr/models/<revision>/`, so a corrupt file
/// written by any of them would otherwise be reused by every one, forever.
pub const MODEL_SHA256: &str = "b95de1ea0e3dc99a5c31bea32e220835da801f683ed17091eaec9954c80a4c04";

/// sha256 of [`CHARSET_FILENAME`] at [`MODEL_REVISION`]. See [`MODEL_SHA256`].
pub const CHARSET_SHA256: &str = "edfd75f688e4155c64aeee0dbac755da0e7ba45a388a2d178a84190fb3d7e953";

/// Manages downloading and caching of OCR models
///
/// This struct handles the lifecycle of the ONNX model file, including:
/// - Determining the cache location (`~/.monocr/models/<revision>/`)
/// - Downloading the model and its charset from HuggingFace if not present
/// - Providing the path to the model file for loading
pub struct ModelManager {
    /// Directory where this revision's files are cached
    cache_dir: PathBuf,
    /// Base URL for downloading, already pinned to a revision
    base_url: String,
    /// Filename of the model file
    model_filename: String,
    /// Digests each file must have. Fields rather than the constants read
    /// directly, so a test can inject a tiny fixture's digest.
    model_sha256: String,
    charset_sha256: String,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    /// Create a new ModelManager with default settings
    ///
    /// # Default Values
    ///
    /// - Cache directory: `~/.monocr/models/<revision>/`
    /// - Base URL: `https://huggingface.co/janakhpon/monocr/resolve/<revision>`
    /// - Model filename: `monocr.onnx`
    ///
    /// # Panics
    ///
    /// Panics if the user's home directory cannot be determined
    pub fn new() -> Self {
        let home = dirs::home_dir().expect("Failed to get home directory");
        // Revision-scoped: re-pinning MODEL_REVISION is a cache miss rather
        // than a silent reuse of whatever was downloaded last time.
        let cache_dir = home.join(".monocr").join("models").join(MODEL_REVISION);

        Self {
            cache_dir,
            base_url: format!("https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"),
            model_filename: MODEL_FILENAME.to_string(),
            model_sha256: MODEL_SHA256.to_string(),
            charset_sha256: CHARSET_SHA256.to_string(),
        }
    }

    /// The directory this manager downloads into.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// The pinned download URL for the ONNX model.
    pub fn model_url(&self) -> String {
        format!("{}/onnx/{}", self.base_url, self.model_filename)
    }

    /// The pinned download URL for the charset.
    ///
    /// Same revision as the weights — that is the only way to be sure the two
    /// agree.
    pub fn charset_url(&self) -> String {
        format!("{}/onnx/{}", self.base_url, CHARSET_FILENAME)
    }

    /// Get the path to the ONNX model file, downloading it if the cache for
    /// this revision is empty or holds a file with the wrong sha256.
    pub fn get_model_path(&self) -> io::Result<PathBuf> {
        let model_path = self.cache_dir.join(&self.model_filename);

        if !cached_and_verified(&model_path, &self.model_sha256)? {
            println!(
                "Model {MODEL_REVISION} missing or corrupt at {:?}. Downloading...",
                model_path
            );
            self.download(&self.model_url(), &model_path, &self.model_sha256)?;
            println!("Download complete");
        }

        Ok(model_path)
    }

    /// Get the charset published alongside the pinned model.
    ///
    /// Preferred over the embedded copy because it comes from the same revision
    /// as the weights.
    pub fn get_charset(&self) -> io::Result<String> {
        let charset_path = self.cache_dir.join(CHARSET_FILENAME);

        if !cached_and_verified(&charset_path, &self.charset_sha256)? {
            self.download(&self.charset_url(), &charset_path, &self.charset_sha256)?;
        }

        fs::read_to_string(&charset_path)
    }

    /// Download `url` to `dest` via a temporary file.
    ///
    /// The rename is the last step, so an interrupted transfer never leaves a
    /// truncated artifact behind. The sha256 is checked before it, so a corrupt
    /// transfer never reaches the cached path either; a mismatch is an
    /// `InvalidData` error, as the Python binding raises.
    fn download(&self, url: &str, dest: &Path, expected_sha256: &str) -> io::Result<()> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let client = Client::new();
        let mut response = client
            .get(url)
            .send()
            .map_err(|e| io::Error::other(format!("Failed to download from {}: {}", url, e)))?;

        if !response.status().is_success() {
            return Err(io::Error::other(format!(
                "Failed to download {}: {}",
                url,
                response.status()
            )));
        }

        let total_size = response.content_length().unwrap_or(0);
        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars(">-"));

        let tmp_path = dest.with_extension("part");
        // Every failure below removes the temp file, not only the mismatch.
        let result = (|| -> io::Result<()> {
            let mut file = fs::File::create(&tmp_path)?;
            io::copy(&mut response, &mut file)?;
            drop(file);

            let actual = file_sha256(&tmp_path)?;
            if actual != expected_sha256 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "checksum mismatch for {} from {url}\n  expected sha256 {expected_sha256}\n  \
                         got      sha256 {actual}\nThe pinned revision is immutable, so this is a \
                         corrupted or intercepted download, not a new release.\nRetry; if it \
                         persists, a proxy may be rewriting the download: fetch {url} manually \
                         into {}.",
                        dest.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                        dest.parent().unwrap_or(Path::new(".")).display(),
                    ),
                ));
            }

            fs::rename(&tmp_path, dest)
        })();

        if result.is_err() {
            let _ = fs::remove_file(&tmp_path);
        }
        pb.finish_and_clear();
        result
    }
}

/// Whether `path` exists with the `expected` sha256, as the Python binding
/// checks.
///
/// A file that exists with any other digest -- a partial write, or one cached
/// before digests were checked -- is removed, so the caller fetches it once; a
/// second mismatch is an error from `download`.
///
/// The digest is recomputed on every call rather than remembered in a sidecar
/// marker. That is one read of the 46 MB model per load, small next to creating
/// the ONNX session, and a marker could outlive a change to the file it vouches
/// for.
fn cached_and_verified(path: &Path, expected: &str) -> io::Result<bool> {
    let actual = match file_sha256(path) {
        Ok(digest) => digest,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    if actual == expected {
        return Ok(true);
    }
    match fs::remove_file(path) {
        Ok(()) => Ok(false),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Hex sha256 of a file, streamed so the model is not held in memory twice.
fn file_sha256(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    fn hex_sha256(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    const FAKE_MODEL: &[u8] = b"fake onnx weights";
    const FAKE_CHARSET: &[u8] = b"abc\n";

    /// A one-file HTTP server on localhost standing in for Hugging Face, so no
    /// test here touches the network. Every request gets `body`; the request
    /// paths are recorded.
    fn serve(body: &'static [u8]) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request.split_whitespace().nth(1).unwrap_or("").to_string();
                log.lock().unwrap().push(path);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            }
        });
        (base, seen)
    }

    /// A manager on a temp cache and a local server, with the tiny fixtures'
    /// digests injected in place of the pinned ones.
    fn test_manager(cache_dir: &Path, base_url: String) -> ModelManager {
        ModelManager {
            cache_dir: cache_dir.to_path_buf(),
            base_url,
            model_filename: MODEL_FILENAME.to_string(),
            model_sha256: hex_sha256(FAKE_MODEL),
            charset_sha256: hex_sha256(FAKE_CHARSET),
        }
    }

    /// `python/monocr_onnx/model_manager.py` is the source of truth for the
    /// digests. All bindings share one cache directory, so if their digests
    /// disagree, one of them rejects -- and deletes -- a file another one just
    /// verified.
    #[test]
    fn digests_match_the_python_binding() {
        let py = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python")
            .join("monocr_onnx")
            .join("model_manager.py");
        let Ok(src) = fs::read_to_string(&py) else {
            eprintln!("skipping: {py:?} is not alongside this crate");
            return;
        };
        for (key, ours) in [
            ("MODEL_FILENAME", MODEL_SHA256),
            ("CHARSET_FILENAME", CHARSET_SHA256),
        ] {
            // `_REMOTE_PATHS` uses the same keys, so match on the value's shape
            // too: the digest is the entry whose value is 64 hex characters.
            let theirs = src
                .lines()
                .filter(|l| l.trim_start().starts_with(&format!("{key}: \"")))
                .filter_map(|l| l.split('"').nth(1))
                .find(|v| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()))
                .unwrap_or_else(|| panic!("no {key} digest in the Python model manager"));
            assert_eq!(ours, theirs, "{key} digest drifted from Python");
        }
    }

    #[test]
    fn embedded_charset_has_the_pinned_digest() {
        assert_eq!(hex_sha256(include_bytes!("charset.txt")), CHARSET_SHA256);
    }

    /// The cache used to accept any file that existed. A corrupt one must be
    /// deleted and fetched again, and a verified one must not be.
    #[test]
    fn a_corrupt_cached_model_is_refetched() {
        let dir = tempfile::tempdir().unwrap();
        let (base, seen) = serve(FAKE_MODEL);
        let m = test_manager(dir.path(), base);
        fs::write(dir.path().join(MODEL_FILENAME), b"truncated or poisoned").unwrap();

        let path = m.get_model_path().unwrap();
        assert_eq!(fs::read(&path).unwrap(), FAKE_MODEL);
        assert_eq!(
            seen.lock().unwrap().len(),
            1,
            "the corrupt model was not refetched"
        );

        m.get_model_path().unwrap();
        assert_eq!(
            seen.lock().unwrap().len(),
            1,
            "a verified cache hit downloaded again"
        );
    }

    #[test]
    fn a_corrupt_cached_charset_is_refetched() {
        let dir = tempfile::tempdir().unwrap();
        let (base, _) = serve(FAKE_CHARSET);
        let m = test_manager(dir.path(), base);
        fs::write(dir.path().join(CHARSET_FILENAME), b"xyz\n").unwrap();

        assert_eq!(m.get_charset().unwrap().as_bytes(), FAKE_CHARSET);
    }

    /// A transfer with the wrong digest must fail loudly and never reach the
    /// cached path, where every binding would trust it from then on.
    #[test]
    fn a_download_with_the_wrong_digest_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let wrong = b"fake onnx WEIGHTS";
        assert_eq!(wrong.len(), FAKE_MODEL.len(), "only the digest may differ");
        let (base, _) = serve(wrong);
        let m = test_manager(dir.path(), base);

        let err = m.get_model_path().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
        let next_step = format!("manually into {}.", dir.path().display());
        assert!(err.to_string().contains(&next_step), "{err}");
        let left: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect();
        assert!(
            left.is_empty(),
            "a refused download left {left:?} in the cache"
        );
    }

    /// `main` is a moving ref and the artifact has already changed under it.
    #[test]
    fn download_urls_are_pinned() {
        let m = ModelManager::new();
        for url in [m.model_url(), m.charset_url()] {
            assert!(
                !url.contains("/resolve/main/"),
                "still tracking the moving ref `main`: {url}"
            );
            assert!(
                url.contains(&format!("/resolve/{MODEL_REVISION}/")),
                "not pinned to {MODEL_REVISION}: {url}"
            );
        }
    }

    /// The charset has to come from the same revision as the weights.
    #[test]
    fn charset_is_fetched_from_the_model_revision() {
        let m = ModelManager::new();
        let model_dir = m.model_url().trim_end_matches(MODEL_FILENAME).to_string();
        let charset_dir = m
            .charset_url()
            .trim_end_matches(CHARSET_FILENAME)
            .to_string();
        assert_eq!(model_dir, charset_dir);
    }

    /// The cache used to gate on file existence alone, so an artifact from an
    /// older revision was reused forever.
    #[test]
    fn cache_dir_is_scoped_by_revision() {
        let m = ModelManager::new();
        assert_eq!(
            m.cache_dir().file_name().and_then(|s| s.to_str()),
            Some(MODEL_REVISION),
            "cache directory {:?} is not revision-scoped",
            m.cache_dir()
        );
    }
}
