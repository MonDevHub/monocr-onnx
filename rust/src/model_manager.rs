//! Model Manager
//!
//! This module handles downloading and caching the ONNX model used for OCR.
//! Models are downloaded from HuggingFace and stored in the user's cache directory.

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The Hugging Face repository holding the ONNX artifact.
pub const MODEL_REPO: &str = "janakhpon/monocr";

/// The pinned revision.
///
/// `main` is a moving ref and the artifact has already changed under it: the
/// model served at one point had a 64-pixel input and 225 output classes, the
/// one served now has 128 and 316. A cache that gates on "the file exists"
/// cannot tell those apart, so the revision is part of the cache path.
pub const MODEL_REVISION: &str = "d3d9d5e";

/// Filename of the ONNX model within the repository's `onnx/` directory.
pub const MODEL_FILENAME: &str = "monocr.onnx";

/// Filename of the charset that belongs to the same revision.
pub const CHARSET_FILENAME: &str = "charset.txt";

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
    /// this revision is empty.
    pub fn get_model_path(&self) -> io::Result<PathBuf> {
        let model_path = self.cache_dir.join(&self.model_filename);

        if !model_path.exists() {
            println!(
                "Model {MODEL_REVISION} not found at {:?}. Downloading...",
                model_path
            );
            self.download(&self.model_url(), &model_path)?;
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

        if !charset_path.exists() {
            self.download(&self.charset_url(), &charset_path)?;
        }

        fs::read_to_string(&charset_path)
    }

    /// Download `url` to `dest` via a temporary file.
    ///
    /// The rename is the last step, so an interrupted transfer never leaves a
    /// truncated artifact behind for the existence check to accept.
    fn download(&self, url: &str, dest: &Path) -> io::Result<()> {
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
        let mut file = fs::File::create(&tmp_path)?;
        let copied = io::copy(&mut response, &mut file);
        drop(file);

        if let Err(e) = copied {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        fs::rename(&tmp_path, dest)?;
        pb.finish_and_clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
