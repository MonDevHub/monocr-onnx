//! MonOcr - Mon language OCR library using ONNX models
//!
//! This library provides OCR (Optical Character Recognition) functionality for Mon text
//! using deep learning models. It supports reading text from images and PDFs, with optional
//! accuracy measurement against ground truth text.
//!
//! # The model
//!
//! Weights are downloaded from [janakhpon/monocr](https://huggingface.co/janakhpon/monocr),
//! pinned to revision [`model_manager::MODEL_REVISION`]. That artifact takes a
//! `[1, 1, 160, 1024]` input and emits `[1, sequence, 277]` logits: 276
//! characters plus the CTC blank. The width is static: v3.5 accepts 1024 and
//! nothing else, where v2 accepted any width.
//!
//! The charset, the input height and the classifier width are one contract. If
//! they drift apart the model still runs and still returns text — it is just the
//! wrong text, with no error anywhere. So the graph is read on load and a
//! disagreement yields [`ModelContractError`] instead of a result.
//!
//! # Quick Start
//!
//! ```no_run
//! use monocr_onnx::read_image;
//!
//! #[tokio::main]
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
//! - Lines too wide for the 1024px model window are tiled at whitespace columns
//!   rather than squeezed into it; see [`MonOcr::predict_page`] for the measured
//!   reason

use anyhow::Result;
use std::path::Path;

pub mod model_manager;
mod monocr;
pub mod segmenter;
mod utils;

pub use model_manager::ModelManager;
pub use monocr::{
    normalize_charset, normalize_polarity, page_text, BBox, LineResult, ModelContractError, MonOcr,
    MonOcrBuilder, DEFAULT_INPUT_WIDTH, EXPECTED_INPUT_HEIGHT,
};
pub use segmenter::{
    cut_column, tile_line, CUT_INK_THRESHOLD, CUT_SEARCH_FRACTION, DEFAULT_DENSITY_THRESHOLD_RATIO,
};
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
/// ```no_run
/// use monocr_onnx::read_image;
///
/// #[tokio::main]
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
/// ```no_run
/// use monocr_onnx::read_images;
///
/// #[tokio::main]
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
/// ```no_run
/// use monocr_onnx::read_pdf;
///
/// #[tokio::main]
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
/// ```no_run
/// use monocr_onnx::read_image_with_accuracy;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let result = read_image_with_accuracy("image.png", "ဘာသာမန်").await?;
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
/// ```no_run
/// use monocr_onnx::read_image_with_accuracy;
///
/// #[tokio::main]
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

    /// The pinned model (`model_manager::MODEL_REVISION`) emits 277 classes:
    /// 276 characters plus the CTC blank at index 0.
    const PINNED_CHARSET_LEN: usize = 276;

    const EMBEDDED_CHARSET: &str = include_str!("charset.txt");

    #[test]
    fn embedded_charset_matches_the_pinned_model() {
        let n = normalize_charset(EMBEDDED_CHARSET).chars().count();
        assert_eq!(
            n,
            PINNED_CHARSET_LEN,
            "bundled charset has {n} characters, the pinned model expects {PINNED_CHARSET_LEN} \
             ({} classes minus the CTC blank)",
            PINNED_CHARSET_LEN + 1
        );
    }

    /// The charset's first character is U+0020. A bare `.trim()` eats it,
    /// dropping 276 to 275 and shifting every index in the decode by one — the
    /// model still runs and still returns text, just the wrong text.
    #[test]
    fn embedded_charset_keeps_its_leading_space() {
        let charset = normalize_charset(EMBEDDED_CHARSET);
        assert_eq!(
            charset.chars().next(),
            Some(' '),
            "charset must start with U+0020"
        );
        assert_eq!(
            charset.trim().chars().count(),
            PINNED_CHARSET_LEN - 1,
            "expected .trim() to drop exactly the leading space"
        );
    }

    #[test]
    fn normalize_charset_trims_only_line_terminators() {
        assert_eq!(normalize_charset(" abc"), " abc");
        assert_eq!(normalize_charset(" abc\n"), " abc");
        assert_eq!(normalize_charset(" abc\r\n"), " abc");
        assert_eq!(normalize_charset("\n abc\n"), " abc");
        // A trailing space is a class too.
        assert_eq!(normalize_charset(" abc "), " abc ");
    }

    #[tokio::test]
    #[ignore = "requires network access to download model from HuggingFace"]
    async fn test_builder() {
        let builder = MonOcr::builder();
        assert!(builder.build().await.is_ok());
    }
}
