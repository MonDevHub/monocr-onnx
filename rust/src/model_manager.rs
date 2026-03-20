//! Model Manager
//!
//! This module handles downloading and caching the ONNX model used for OCR.
//! Models are downloaded from HuggingFace and stored in the user's cache directory.

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Manages downloading and caching of OCR models
///
/// This struct handles the lifecycle of the ONNX model file, including:
/// - Determining the cache location (~/.monocr/models/)
/// - Downloading the model from HuggingFace if not present
/// - Providing the path to the model file for loading
///
/// # Default Behavior
///
/// By default, models are downloaded from:
/// `https://huggingface.co/janakhpon/monocr/resolve/main/onnx/monocr.onnx`
///
/// The model is cached in `~/.monocr/models/` on Unix-like systems.
pub struct ModelManager {
    /// Directory where models are cached
    cache_dir: PathBuf,
    /// Base URL for downloading models
    base_url: String,
    /// Filename of the model file
    model_filename: String,
}

impl ModelManager {
    /// Create a new ModelManager with default settings
    ///
    /// # Default Values
    ///
    /// - Cache directory: `~/.monocr/models/` (where `~` is the user's home directory)
    /// - Base URL: `https://huggingface.co/janakhpon/monocr/resolve/main`
    /// - Model filename: `monocr.onnx`
    ///
    /// # Returns
    ///
    /// A new `ModelManager` instance configured with default settings
    ///
    /// # Panics
    ///
    /// Panics if the user's home directory cannot be determined
    ///
    /// # Example
    ///
    /// ```ignore
    /// use monocr_onnx::model_manager::ModelManager;
    ///
    /// let manager = ModelManager::new();
    /// let model_path = manager.get_model_path().expect("Failed to get model");
    /// ```
    pub fn new() -> Self {
        let home = dirs::home_dir().expect("Failed to get home directory");
        let cache_dir = home.join(".monocr").join("models");

        Self {
            cache_dir,
            base_url: "https://huggingface.co/janakhpon/monocr/resolve/main".to_string(),
            model_filename: "monocr.onnx".to_string(),
        }
    }

    /// Get the path to the ONNX model file
    ///
    /// This method checks if the model file exists in the cache directory.
    /// If the model does not exist, it automatically downloads it from HuggingFace.
    ///
    /// # Returns
    ///
    /// * `Ok(PathBuf)` - Path to the model file if successful
    /// * `Err(io::Error)` - If the model cannot be downloaded or the path cannot be determined
    ///
    /// # Behavior
    ///
    /// 1. Constructs the expected model path from cache_dir and model_filename
    /// 2. Checks if the file exists
    /// 3. If not found, calls [`download_model`](Self::download_model) to download it
    /// 4. Returns the path to the existing/downloaded model
    ///
    /// # Download Process
    ///
    /// When downloading:
    /// - Creates the cache directory if it doesn't exist
    /// - Shows a progress bar during download
    /// - Uses HTTP GET request to download from HuggingFace
    ///
    /// # Example
    ///
    /// ```ignore
    /// use monocr_onnx::model_manager::ModelManager;
    ///
    /// async fn main() {
    ///     let manager = ModelManager::new();
    ///     match manager.get_model_path() {
    ///         Ok(path) => println!("Model loaded from: {:?}", path),
    ///         Err(e) => eprintln!("Error: {}", e),
    ///     }
    /// }
    /// ```
    pub fn get_model_path(&self) -> io::Result<PathBuf> {
        let model_path = self.cache_dir.join(&self.model_filename);

        if !model_path.exists() {
            println!("Model not found at {:?}. Downloading...", model_path);
            self.download_model(&model_path)?;
        }

        Ok(model_path)
    }

    /// Download the ONNX model from HuggingFace
    ///
    /// This is an internal method called by [`get_model_path`](Self::get_model_path)
    /// when the model is not found in the cache.
    ///
    /// # Arguments
    ///
    /// * `dest` - The destination path where the model should be saved
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the download completes successfully
    /// * `Err(io::Error)` - If the download fails or the HTTP response indicates an error
    ///
    /// # Process
    ///
    /// 1. Creates parent directories if they don't exist
    /// 2. Constructs the download URL: `{base_url}/onnx/{model_filename}`
    /// 3. Makes an HTTP GET request
    /// 4. Validates the response status (checks for 2xx success)
    /// 5. Displays a progress bar during download
    /// 6. Writes the response body to the destination file
    ///
    /// # Requirements
    ///
    /// - Requires network access to HuggingFace
    /// - Requires `reqwest` HTTP client
    /// - Shows progress using `indicatif` crate
    fn download_model(&self, dest: &Path) -> io::Result<()> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let url = format!("{}/onnx/{}", self.base_url, self.model_filename);
        let client = Client::new();
        let mut response = client.get(&url).send().map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to download from {}: {}", url, e),
            )
        })?;

        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to download model: {}", response.status()),
            ));
        }

        let total_size = response.content_length().unwrap_or(0);
        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars(">-"));

        let mut file = fs::File::create(dest)?;

        io::copy(&mut response, &mut file)?;

        pb.finish_with_message("Download complete");
        Ok(())
    }
}
