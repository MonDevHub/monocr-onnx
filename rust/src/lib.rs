//! MonOcr - Mongolian OCR library using ONNX models
//!
//! This library provides OCR (Optical Character Recognition) functionality for Mongolian text
//! using deep learning models. It supports reading text from images and PDFs, with optional
//! accuracy measurement against ground truth text.
//!
//! # Quick Start
//!
//! ```ignore
//! use monocr_onnx::read_image;
//!
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let text = read_image("path/to/image.png").await?;
//!     println!("Recognized text: {}", text);
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - Read text from single images (PNG, JPG, etc.)
//! - Read text from multiple images in batch
//! - Read text from PDF files (requires poppler-utils)
//! - Measure OCR accuracy against ground truth
//! - Customizable model paths and character sets
//! - Line segmentation for full page OCR

use anyhow::Result;
use std::path::Path;

mod model_manager;
mod monocr;
mod segmenter;
mod utils;

pub use monocr::{LineResult, MonOcr, MonOcrBuilder};
pub use utils::calculate_accuracy;

/// Read text from a single image file
///
/// This function initializes a new MonOcr instance with default settings and performs
/// OCR on the given image. The image is automatically segmented into lines, and each
/// line is recognized using the ONNX model.
///
/// # Arguments
///
/// * `image_path` - Path to the image file (PNG, JPG, BMP, etc.)
///
/// # Returns
///
/// Returns a `Result<String>` containing the recognized text, with lines separated by newlines.
///
/// # Example
///
/// ```ignore
/// use monocr_onnx::read_image;
///
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let text = read_image("document.png").await?;
///     println!("Recognized: {}", text);
///     Ok(())
/// }
/// ```
pub async fn read_image(image_path: impl AsRef<Path>) -> Result<String> {
    let mut ocr = MonOcr::builder().build().await?;
    ocr.read_image(image_path).await
}

/// Read text from multiple image files
///
/// This function processes multiple images in sequence, returning a vector of recognized texts.
/// Each image is segmented into lines and processed individually.
///
/// # Arguments
///
/// * `image_paths` - A slice of paths to image files
///
/// # Returns
///
/// Returns a `Result<Vec<String>>` where each element contains the recognized text
/// from the corresponding image. Lines within each text are separated by newlines.
///
/// # Example
///
/// ```ignore
/// use monocr_onnx::read_images;
///
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let paths = vec!["page1.png", "page2.png", "page3.png"];
///     let results = read_images(&paths).await?;
///     for (i, text) in results.iter().enumerate() {
///         println!("Page {}: {}", i + 1, text);
///     }
///     Ok(())
/// }
/// ```
pub async fn read_images(image_paths: &[impl AsRef<Path>]) -> Result<Vec<String>> {
    let mut ocr = MonOcr::builder().build().await?;
    ocr.read_images(image_paths).await
}

/// Read text from a PDF file
///
/// This function converts a PDF document to images (using pdftoppm from poppler-utils)
/// and performs OCR on each page. Each page is treated as a separate image.
///
/// # Arguments
///
/// * `pdf_path` - Path to the PDF file
///
/// # Returns
///
/// Returns a `Result<Vec<String>>` where each element contains the recognized text
/// from the corresponding page.
///
/// # Requirements
///
/// This function requires `pdftoppm` from the poppler-utils package to be installed:
/// - Ubuntu/Debian: `sudo apt-get install poppler-utils`
/// - macOS: `brew install poppler`
/// - Fedora/RHEL: `sudo dnf install poppler-utils`
///
/// # Example
///
/// ```ignore
/// use monocr_onnx::read_pdf;
///
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let pages = read_pdf("document.pdf").await?;
///     for (i, text) in pages.iter().enumerate() {
///         println!("=== Page {} ===\n{}", i + 1, text);
///     }
///     Ok(())
/// }
/// ```
pub async fn read_pdf(pdf_path: impl AsRef<Path>) -> Result<Vec<String>> {
    let mut ocr = MonOcr::builder().build().await?;
    ocr.read_pdf(pdf_path).await
}

/// Read text from an image with accuracy measurement
///
/// This function performs OCR on an image and calculates the accuracy by comparing
/// the recognized text against the ground truth using Levenshtein distance.
///
/// # Arguments
///
/// * `image_path` - Path to the image file
/// * `ground_truth` - The expected/ground truth text to compare against
///
/// # Returns
///
/// Returns a `Result<OcrResult>` containing:
/// - `text`: The recognized text from the image
/// - `accuracy`: A percentage (0-100) representing how close the recognized text is
///   to the ground truth
///
/// # Accuracy Calculation
///
/// Accuracy is calculated as: `(1 - CER) * 100` where CER is the Character Error Rate
/// (Levenshtein distance divided by the maximum length of the two strings).
/// This gives a percentage score where 100% means perfect recognition.
///
/// # Example
///
/// ```ignore
/// use monocr_onnx::read_image_with_accuracy;
///
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let result = read_image_with_accuracy("image.png", "Монгол бичг").await?;
///     println!("Recognized: {}", result.text);
///     println!("Accuracy: {:.2}%", result.accuracy);
///     Ok(())
/// }
/// ```
pub async fn read_image_with_accuracy(
    image_path: impl AsRef<Path>,
    ground_truth: &str,
) -> Result<OcrResult> {
    let mut ocr = MonOcr::builder().build().await?;
    ocr.read_image_with_accuracy(image_path, ground_truth).await
}

/// OCR result containing recognized text and accuracy measurement
///
/// This struct is returned by [`read_image_with_accuracy`] and contains both
/// the recognized text and the accuracy score when compared against ground truth.
///
/// # Fields
///
/// * `text` - The recognized text from the OCR process
/// * `accuracy` - A percentage value (0-100) indicating how closely the recognized
///   text matches the ground truth
///
/// # Example
///
/// ```ignore
/// use monocr_onnx::read_image_with_accuracy;
///
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let result = read_image_with_accuracy("test.png", "Hello World").await?;
///     if result.accuracy >= 90.0 {
///         println!("Good recognition: {}", result.text);
///     } else {
///         println!("Poor recognition: {} ({}% accuracy)", result.text, result.accuracy);
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// The recognized text from the image
    pub text: String,
    /// Accuracy percentage (0-100) based on Levenshtein distance
    pub accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires network access to download model from HuggingFace"]
    async fn test_builder() {
        let builder = MonOcr::builder();
        assert!(builder.build().await.is_ok());
    }
}
