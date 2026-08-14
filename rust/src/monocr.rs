//! Main OCR implementation
//!
//! This module contains the core OCR functionality including the MonOcr struct,
//! the builder pattern for configuration, and the prediction/inference logic.

use anyhow::Result;
use image::{imageops::FilterType, GrayImage};
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::model_manager::ModelManager;
use crate::segmenter::LineSegmenter;
use crate::utils::calculate_accuracy;
use crate::OcrResult;

/// Default embedded charset
///
/// This constant includes the default character set for Mon OCR, embedded from
/// the charset.txt file at compile time. It contains all supported characters
/// that the model can recognize, in the order the classifier emits them.
const DEFAULT_CHARSET: &str = include_str!("charset.txt");

/// Input height this binding preprocesses for.
///
/// The charset, the input height and the classifier width are one contract. If
/// they drift apart the model still runs and still returns text — it is just
/// the wrong text, with no error anywhere. So this is declared here, checked
/// against the graph in [`MonOcr::new`], and a disagreement refuses to load.
pub const EXPECTED_INPUT_HEIGHT: u32 = 128;

/// Padded canvas width fed to the model. The model's width axis is dynamic;
/// this is the binding's choice, not a model constraint.
pub const DEFAULT_INPUT_WIDTH: u32 = 1024;

/// A model artifact that disagrees with the charset or the input geometry this
/// binding was built for.
///
/// Returned instead of running, because running would produce confident
/// nonsense rather than an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelContractError(pub String);

impl fmt::Display for ModelContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model contract violation: {}", self.0)
    }
}

impl std::error::Error for ModelContractError {}

/// Strip line terminators, and nothing else.
///
/// The charset's first character really is U+0020 — a space is one of the
/// classes the model emits. A bare `.trim()` eats it, which drops the charset
/// from 315 characters to 314 and shifts every index in the decode by one, so
/// every character comes back as its neighbour.
pub fn normalize_charset(charset: &str) -> &str {
    charset
        .trim_start_matches(['\n', '\r'])
        .trim_end_matches(['\n', '\r'])
}

/// Read `shape[axis]` when it is a fixed positive size.
///
/// ONNX reports dynamic axes as -1; those return `None` because there is
/// nothing to compare them against.
fn static_dim(shape: &[i64], axis: usize) -> Option<usize> {
    match shape.get(axis) {
        Some(&d) if d > 0 => Some(d as usize),
        _ => None,
    }
}

/// Compare the charset and geometry this binding holds against what the ONNX
/// graph actually declares.
///
/// `model_classes` and `model_height` are `None` when the graph leaves that axis
/// dynamic, in which case there is nothing to compare and the check passes —
/// decoding re-derives the class count from the real output tensor and fails
/// there instead.
fn check_contract(
    charset_len: usize,
    model_classes: Option<usize>,
    model_height: Option<usize>,
    source: &str,
) -> Result<(), ModelContractError> {
    if charset_len == 0 {
        return Err(ModelContractError(
            "no charset available; cannot decode model output".to_string(),
        ));
    }
    if let Some(classes) = model_classes {
        let expected = charset_len + 1;
        if classes != expected {
            return Err(ModelContractError(format!(
                "charset/model mismatch.\n  \
                 charset: {charset_len} characters -> expects {expected} classes \
                 ({charset_len} + CTC blank)\n  \
                 model ({source}): {classes} classes\n\
                 Every index above the first divergence would decode to the wrong character."
            )));
        }
    }
    if let Some(height) = model_height {
        if height != EXPECTED_INPUT_HEIGHT as usize {
            return Err(ModelContractError(format!(
                "input height mismatch: this binding preprocesses to height {EXPECTED_INPUT_HEIGHT} \
                 but {source} expects {height}"
            )));
        }
    }
    Ok(())
}

/// Builder for configuring and creating MonOcr instances
///
/// The builder pattern allows flexible configuration of OCR settings before
/// creating an instance. All settings have sensible defaults.
///
/// # Configuration Options
///
/// - `model_path`: Custom path to the ONNX model file (default: download from HuggingFace)
/// - `charset`: Custom character set for OCR (default: built-in Mon charset)
/// - `min_line_height`: Minimum height for line segmentation (default: 10 pixels)
/// - `smooth_window`: Window size for smoothing projection profile (default: 3)
///
/// # Example
///
/// ```no_run
/// use monocr_onnx::MonOcr;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut ocr = MonOcr::builder()
///         .min_line_height(15)
///         .smooth_window(5)
///         .build()
///         .await?;
///
///     let text = ocr.read_image("document.png").await?;
///     println!("{text}");
///     Ok(())
/// }
/// ```
pub struct MonOcrBuilder {
    /// Optional custom path to ONNX model file
    model_path: Option<PathBuf>,
    /// Optional custom charset string
    charset: Option<String>,
    /// Minimum line height for segmentation (in pixels)
    min_line_height: u32,
    /// Smoothing window size for projection profile
    smooth_window: u32,
}

impl Default for MonOcrBuilder {
    /// Create a MonOcrBuilder with default settings
    ///
    /// Default values:
    /// - model_path: None (will download from HuggingFace)
    /// - charset: None (uses the charset published with the pinned model,
    ///   falling back to the built-in Mon charset)
    /// - min_line_height: 10 pixels
    /// - smooth_window: 3
    fn default() -> Self {
        Self {
            model_path: None,
            charset: None,
            min_line_height: 10,
            smooth_window: 3,
        }
    }
}

impl MonOcrBuilder {
    /// Create a new builder with default settings
    ///
    /// This is equivalent to calling `MonOcrBuilder::default()`.
    ///
    /// # Returns
    ///
    /// A new `MonOcrBuilder` instance with default configuration
    ///
    /// # Example
    ///
    /// ```
    /// use monocr_onnx::MonOcrBuilder;
    ///
    /// let builder = MonOcrBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the path to the ONNX model file
    ///
    /// By default, the model is downloaded from HuggingFace if not found in cache.
    /// Use this method to specify a custom model file location.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// The builder with the model path set
    ///
    /// # Example
    ///
    /// ```no_run
    /// use monocr_onnx::MonOcr;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let ocr = MonOcr::builder()
    ///         .model_path("./models/monocr.onnx")
    ///         .build()
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the charset string directly
    ///
    /// The charset defines all characters that the OCR model can recognize.
    /// It should be a string containing all valid characters in order.
    ///
    /// # Arguments
    ///
    /// * `charset` - A string containing the character set
    ///
    /// # Returns
    ///
    /// The builder with the charset set
    ///
    /// # Note
    ///
    /// The charset must match the one used during model training.
    /// The default charset is built-in and suitable for Mon text.
    pub fn charset(mut self, charset: impl Into<String>) -> Self {
        self.charset = Some(charset.into());
        self
    }

    /// Set the minimum line height for segmentation
    ///
    /// During line segmentation, any detected region shorter than this value
    /// will be ignored. This helps filter out noise and small artifacts.
    ///
    /// # Arguments
    ///
    /// * `height` - Minimum line height in pixels (default: 10)
    ///
    /// # Returns
    ///
    /// The builder with the minimum line height set
    ///
    /// # Recommendation
    ///
    /// Increase this value for noisy documents or decrease for documents
    /// with small font sizes.
    pub fn min_line_height(mut self, height: u32) -> Self {
        self.min_line_height = height;
        self
    }

    /// Set the smoothing window for projection profile
    ///
    /// The smoothing window is used when computing the horizontal projection
    /// profile for line detection. A larger window produces smoother results
    /// but may merge close lines.
    ///
    /// # Arguments
    ///
    /// * `window` - Window size for smoothing (default: 3, use 1 for no smoothing)
    ///
    /// # Returns
    ///
    /// The builder with the smooth window set
    pub fn smooth_window(mut self, window: u32) -> Self {
        self.smooth_window = window;
        self
    }

    /// Build the MonOcr instance
    ///
    /// This method initializes the ONNX runtime session and prepares the OCR
    /// engine for use. It may download the model if not cached.
    ///
    /// # Returns
    ///
    /// * `Ok(MonOcr)` - Ready-to-use OCR instance
    /// * `Err(anyhow::Error)` - If model loading fails
    ///
    /// # Async
    ///
    /// This function is async because model initialization may involve
    /// downloading the model file from the network.
    pub async fn build(self) -> Result<MonOcr> {
        MonOcr::new(
            self.model_path,
            self.charset,
            self.min_line_height,
            self.smooth_window,
        )
        .await
    }
}

/// Main OCR engine for text recognition
///
/// This struct encapsulates the OCR pipeline including:
/// - ONNX runtime session for model inference
/// - Character set for decoding predictions
/// - Line segmenter for page layout analysis
/// - Image preprocessing utilities
///
/// # Usage
///
/// Typically, you would create a `MonOcr` instance using the builder:
///
/// ```no_run
/// use monocr_onnx::MonOcr;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut ocr = MonOcr::builder().build().await?;
///     let text = ocr.read_image("document.png").await?;
///     println!("Recognized: {}", text);
///     Ok(())
/// }
/// ```
///
/// The instance must be mutable because internal state is modified during
/// inference (e.g., the ONNX session).
pub struct MonOcr {
    /// ONNX runtime session for model inference
    session: Session,
    /// Character set for decoding model output
    charset: Vec<char>,
    /// Line segmenter for page layout analysis
    segmenter: LineSegmenter,
    /// Target height for model input, taken from the model graph once the
    /// contract check has confirmed it matches [`EXPECTED_INPUT_HEIGHT`]
    target_height: u32,
    /// Target width for model input (the model's width axis is dynamic)
    target_width: u32,
}

/// Result from line prediction
///
/// This struct contains the recognized text and its bounding box location
/// for a single line in the image.
#[derive(Debug, Clone)]
pub struct LineResult {
    /// The recognized text for this line
    pub text: String,
    /// The bounding box of this text line in the original image
    pub bbox: BBox,
}

/// Bounding box for a line or text region
///
/// Represents a rectangular region in the image with pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    /// X coordinate of the top-left corner
    pub x: u32,
    /// Y coordinate of the top-left corner
    pub y: u32,
    /// Width of the bounding box
    pub w: u32,
    /// Height of the bounding box
    pub h: u32,
}

impl MonOcr {
    /// Create a builder for configuring MonOcr
    ///
    /// This is the entry point for creating a customized OCR instance.
    /// Use the builder methods to configure options, then call `build()`.
    ///
    /// # Returns
    ///
    /// A new `MonOcrBuilder` instance
    ///
    /// # Example
    ///
    /// ```no_run
    /// use monocr_onnx::MonOcr;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut ocr = MonOcr::builder()
    ///         .min_line_height(15)
    ///         .build()
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn builder() -> MonOcrBuilder {
        MonOcrBuilder::new()
    }

    /// Internal constructor (not part of public API)
    ///
    /// This method is called by the builder's `build()` method.
    /// It initializes the ONNX session, loads the charset, and creates
    /// the line segmenter.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Optional custom path to ONNX model
    /// * `charset` - Optional custom charset string
    /// * `min_line_height` - Minimum line height for segmentation
    /// * `smooth_window` - Smoothing window size
    async fn new(
        model_path: Option<PathBuf>,
        charset: Option<String>,
        min_line_height: u32,
        smooth_window: u32,
    ) -> Result<Self> {
        // Get or download model. When the model comes from the manager, its
        // charset comes from the same pinned revision, so the two agree by
        // construction; the embedded copy is the offline fallback.
        let (model_path, published_charset) = match model_path {
            Some(path) => (path, None),
            None => {
                let manager = ModelManager::new();
                let path = manager.get_model_path()?;
                let published = manager.get_charset().ok();
                (path, published)
            }
        };

        // Get charset
        let charset_str = charset
            .or(published_charset)
            .unwrap_or_else(|| DEFAULT_CHARSET.to_string());
        let charset: Vec<char> = normalize_charset(&charset_str).chars().collect();

        // Create ONNX session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;

        // Read the real graph rather than assuming the input height or the
        // class count. Both have changed under this SDK before.
        let source = model_path.display().to_string();
        let in_shape = session
            .inputs()
            .first()
            .and_then(|i| i.dtype().tensor_shape())
            .ok_or_else(|| ModelContractError(format!("{source} has no tensor input")))?
            .to_vec();
        let out_shape = session
            .outputs()
            .first()
            .and_then(|o| o.dtype().tensor_shape())
            .ok_or_else(|| ModelContractError(format!("{source} has no tensor output")))?
            .to_vec();

        if in_shape.len() != 4 {
            return Err(ModelContractError(format!(
                "expected a 4-D [batch, channel, height, width] input, {source} declares {in_shape:?}"
            ))
            .into());
        }
        if out_shape.len() != 3 {
            return Err(ModelContractError(format!(
                "expected a 3-D [batch, sequence, classes] output, {source} declares {out_shape:?}"
            ))
            .into());
        }

        let model_height = static_dim(&in_shape, 2);
        let model_classes = static_dim(&out_shape, 2);
        check_contract(charset.len(), model_classes, model_height, &source)?;

        let segmenter = LineSegmenter::new(min_line_height, smooth_window);

        Ok(Self {
            session,
            charset,
            segmenter,
            target_height: model_height
                .map(|h| h as u32)
                .unwrap_or(EXPECTED_INPUT_HEIGHT),
            target_width: DEFAULT_INPUT_WIDTH,
        })
    }

    /// Read text from a single image
    ///
    /// This method performs OCR on a single image file. The image is automatically
    /// segmented into lines, and each line is recognized using the ONNX model.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file (PNG, JPG, BMP, etc.)
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Recognized text with lines separated by newlines
    /// * `Err(anyhow::Error)` - If the image cannot be read or OCR fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use monocr_onnx::MonOcr;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut ocr = MonOcr::builder().build().await?;
    ///     let text = ocr.read_image("document.png").await?;
    ///     println!("Recognized text:\n{}", text);
    ///     Ok(())
    /// }
    /// ```
    pub async fn read_image(&mut self, image_path: impl AsRef<Path>) -> Result<String> {
        let results = self.predict_page(image_path).await?;
        let texts: Vec<String> = results.into_iter().map(|r| r.text).collect();
        Ok(texts.join("\n"))
    }

    /// Read text from multiple images
    ///
    /// This method processes multiple images in sequence, returning a vector of
    /// recognized texts. Each image is segmented into lines and processed individually.
    ///
    /// # Arguments
    ///
    /// * `image_paths` - A slice of paths to image files
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - Vector of recognized texts, one per image
    /// * `Err(anyhow::Error)` - If any image cannot be processed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use monocr_onnx::MonOcr;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut ocr = MonOcr::builder().build().await?;
    ///     let paths = vec!["page1.png", "page2.png", "page3.png"];
    ///     let results = ocr.read_images(&paths).await?;
    ///     for (i, text) in results.iter().enumerate() {
    ///         println!("Page {}: {}", i + 1, text);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn read_images(&mut self, image_paths: &[impl AsRef<Path>]) -> Result<Vec<String>> {
        let mut results = Vec::new();
        for path in image_paths {
            let text = self.read_image(path).await?;
            results.push(text);
        }
        Ok(results)
    }

    /// Read text from a PDF file
    ///
    /// This method converts a PDF document to images using pdftoppm and performs
    /// OCR on each page. Each page is processed as a separate image.
    ///
    /// # Arguments
    ///
    /// * `pdf_path` - Path to the PDF file
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - Vector of recognized texts, one per page
    /// * `Err(anyhow::Error)` - If PDF conversion fails or OCR fails
    ///
    /// # Requirements
    ///
    /// Requires `pdftoppm` from poppler-utils to be installed:
    /// - Ubuntu/Debian: `sudo apt-get install poppler-utils`
    /// - macOS: `brew install poppler`
    pub async fn read_pdf(&mut self, pdf_path: impl AsRef<Path>) -> Result<Vec<String>> {
        use std::process::Stdio;
        use tokio::process::Command;

        let pdf_path = pdf_path.as_ref();

        // Check for pdftoppm
        let check = Command::new("which").arg("pdftoppm").output().await;

        if check.is_err() || !check.as_ref().map(|o| o.status.success()).unwrap_or(false) {
            anyhow::bail!("pdftoppm not found: please install poppler-utils");
        }
        if check.as_ref().map(|o| o.stdout.is_empty()).unwrap_or(true) {
            anyhow::bail!("pdftoppm not found: please install poppler-utils");
        }

        // Create temp directory
        let temp_dir = tempfile::tempdir()?;
        let output_prefix = temp_dir.path().join("page");

        // Convert PDF to images
        let output = Command::new("pdftoppm")
            .args(&["-png", "-r", "300"])
            .arg(pdf_path)
            .arg(&output_prefix)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await?;

        if !output.success() {
            anyhow::bail!("Failed to convert PDF to images");
        }

        // Read generated images
        let mut entries: Vec<_> = std::fs::read_dir(temp_dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "png")
                    .unwrap_or(false)
            })
            .collect();

        // Sort by page number
        entries.sort_by(|a, b| {
            let name_a = a.file_name();
            let name_b = b.file_name();
            let num_a: u32 = name_a
                .to_string_lossy()
                .split('-')
                .last()
                .and_then(|s| s.trim_end_matches(".png").parse().ok())
                .unwrap_or(0);
            let num_b: u32 = name_b
                .to_string_lossy()
                .split('-')
                .last()
                .and_then(|s| s.trim_end_matches(".png").parse().ok())
                .unwrap_or(0);
            num_a.cmp(&num_b)
        });

        if entries.is_empty() {
            anyhow::bail!("No images generated from PDF");
        }

        // Process each page
        let mut pages = Vec::new();
        for entry in entries {
            let text = self.read_image(entry.path()).await?;
            pages.push(text);
        }

        Ok(pages)
    }

    /// Read image with accuracy measurement
    ///
    /// This method performs OCR on an image and calculates accuracy by comparing
    /// the recognized text against ground truth using Levenshtein distance.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file
    /// * `ground_truth` - The expected/ground truth text to compare against
    ///
    /// # Returns
    ///
    /// * `Ok(OcrResult)` - Contains recognized text and accuracy percentage
    /// * `Err(anyhow::Error)` - If OCR fails
    ///
    /// # Accuracy Calculation
    ///
    /// Accuracy = (1 - CER) * 100, where CER is Character Error Rate
    /// calculated as Levenshtein distance / max(len(predicted), len(ground_truth))
    pub async fn read_image_with_accuracy(
        &mut self,
        image_path: impl AsRef<Path>,
        ground_truth: &str,
    ) -> Result<OcrResult> {
        let text = self.read_image(image_path).await?;
        let accuracy = calculate_accuracy(&text, ground_truth);
        Ok(OcrResult { text, accuracy })
    }

    /// Predict text from a single line image
    ///
    /// This is an internal method that runs the ONNX model on a single
    /// pre-segmented line image. It performs preprocessing, inference,
    /// and CTC decoding.
    ///
    /// # Arguments
    ///
    /// * `image` - Pre-processed grayscale image of a single text line
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Recognized text for this line
    /// * `Err(anyhow::Error)` - If inference fails
    async fn predict_line(&mut self, image: &GrayImage) -> Result<String> {
        let input_tensor = self.preprocess(image)?;

        // Run inference
        let input = ort::value::Tensor::from_array(input_tensor)?;
        let outputs = self.session.run(ort::inputs![input])?;

        // Get output tensor
        let output = outputs[0].downcast_ref::<ort::value::DynTensorValueType>()?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let output_shape: Vec<usize> = shape.iter().cloned().map(|x| x as usize).collect();
        let output_data: Vec<f32> = data.to_vec();
        drop(outputs);

        // Decode
        self.decode_owned(&output_data, &output_shape)
    }

    /// Predict text from a full page image
    ///
    /// This method segments the image into lines and recognizes each line
    /// using the ONNX model. Returns results with text and bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the full page image
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<LineResult>)` - Vector of line results with text and bounding boxes
    /// * `Err(anyhow::Error)` - If segmentation or OCR fails
    ///
    /// # Process
    ///
    /// 1. Segment the page into individual text lines using horizontal projection
    /// 2. For each line:
    ///    - Preprocess the image for the model
    ///    - Run inference with the ONNX model
    ///    - Decode CTC output to text
    /// 3. Return results with text and bounding boxes
    pub async fn predict_page(&mut self, image_path: impl AsRef<Path>) -> Result<Vec<LineResult>> {
        let image_path = image_path.as_ref();
        let lines = self.segmenter.segment(image_path)?;

        let mut results = Vec::new();
        for line in lines {
            let text = self.predict_line(&line.img).await?;
            results.push(LineResult {
                text,
                bbox: BBox {
                    x: line.bbox.x,
                    y: line.bbox.y,
                    w: line.bbox.w,
                    h: line.bbox.h,
                },
            });
        }

        Ok(results)
    }

    /// Preprocess image for model input
    ///
    /// This method transforms a grayscale image into the tensor format
    /// expected by the ONNX model.
    ///
    /// # Processing Steps
    ///
    /// 1. **Scaling**: Scale the image to fit the model's input height (128) by
    ///    [`DEFAULT_INPUT_WIDTH`] (1024)
    ///    while maintaining aspect ratio
    /// 2. **Resizing**: Resize using Triangle filter for quality
    /// 3. **Normalization**: Convert pixel values from [0, 255] to [-1, 1]
    /// 4. **Padding**: Pad with white (1.0) if width is less than target
    ///
    /// # Arguments
    ///
    /// * `image` - Source grayscale image
    ///
    /// # Returns
    ///
    /// * `Ok(Array4<f32>)` - 4D tensor with shape [1, 1, target_height, target_width]
    /// * `Err(anyhow::Error)` - If preprocessing fails
    fn preprocess(&self, image: &GrayImage) -> Result<Array4<f32>> {
        let (width, height) = image.dimensions();

        // Calculate new width maintaining aspect ratio
        let scale = self.target_height as f32 / height as f32;
        let new_width = (width as f32 * scale).round() as u32;
        let new_width = new_width.min(self.target_width);

        // Resize image
        let resized =
            image::imageops::resize(image, new_width, self.target_height, FilterType::Triangle);

        // Create tensor and normalize
        let mut tensor = Array4::<f32>::zeros((
            1,
            1,
            self.target_height as usize,
            self.target_width as usize,
        ));

        for y in 0..self.target_height {
            for x in 0..self.target_width {
                let value = if x < new_width {
                    let pixel = resized.get_pixel(x, y);
                    (pixel[0] as f32 / 127.5) - 1.0 // Normalize to [-1, 1]
                } else {
                    1.0 // White padding
                };
                tensor[[0, 0, y as usize, x as usize]] = value;
            }
        }

        Ok(tensor)
    }

    /// CTC Greedy Decoding
    ///
    /// Converts the model output tensor to text using CTC (Connectionist
    /// Temporal Classification) greedy decoding.
    ///
    /// # CTC Decoding Process
    ///
    /// 1. For each timestep, find the class with the highest score
    /// 2. Skip the blank class (index 0)
    /// 3. Skip repeated characters - only keep the first of consecutive same chars
    /// 4. Map class index `n` to `charset[n - 1]`
    ///
    /// # Arguments
    ///
    /// * `data` - Flattened output data in row-major order
    /// * `shape` - Tensor shape [batch, sequence_length, num_classes]
    ///
    /// # Contract
    ///
    /// The stride comes from the output tensor's own shape, never from the
    /// charset. A charset that disagrees with the tensor is refused here rather
    /// than silently decoding every index to its neighbour.
    fn decode_owned(&self, data: &[f32], shape: &[usize]) -> Result<String> {
        decode_ctc(&self.charset, data, shape)
    }
}

/// CTC greedy decode of a flat logits buffer.
///
/// Free-standing so it can be exercised without an ONNX session.
fn decode_ctc(charset: &[char], data: &[f32], shape: &[usize]) -> Result<String> {
    if shape.len() != 3 {
        return Err(ModelContractError(format!(
            "expected a 3-D [batch, sequence, classes] output tensor, got shape {shape:?}"
        ))
        .into());
    }
    let sequence_length = shape[1];
    let num_classes = shape[2];
    if sequence_length == 0 || num_classes == 0 {
        return Err(ModelContractError(format!(
            "output tensor has an empty axis: shape {shape:?}"
        ))
        .into());
    }

    let expected = charset.len() + 1;
    if num_classes != expected {
        return Err(ModelContractError(format!(
            "charset/model mismatch at decode time: charset has {} characters -> \
             expects {expected} classes, tensor has {num_classes}",
            charset.len()
        ))
        .into());
    }
    if data.len() < sequence_length * num_classes {
        return Err(ModelContractError(format!(
            "output tensor holds {} values, shape {shape:?} needs {}",
            data.len(),
            sequence_length * num_classes
        ))
        .into());
    }

    let mut decoded = String::new();
    let mut prev_idx: i32 = -1;

    for t in 0..sequence_length {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;

        let base = t * num_classes;
        for c in 0..num_classes {
            let val = data[base + c];
            if val > max_val {
                max_val = val;
                max_idx = c;
            }
        }

        // Index 0 is the CTC blank; 1..=N map onto charset[0..N-1].
        if max_idx != 0 && max_idx as i32 != prev_idx {
            decoded.push(charset[max_idx - 1]);
        }
        prev_idx = max_idx as i32;
    }

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned model: input [1, 1, 128, width], output [1, sequence, 316].
    const PINNED_CLASSES: usize = 316;
    const PINNED_CHAR_LEN: usize = 315;

    fn charset_of_len(n: usize) -> Vec<char> {
        // Leading U+0020, as the real charset has.
        std::iter::once(' ')
            .chain(std::iter::repeat('x').take(n - 1))
            .collect()
    }

    #[test]
    fn contract_accepts_the_pinned_model() {
        check_contract(
            PINNED_CHAR_LEN,
            Some(PINNED_CLASSES),
            Some(EXPECTED_INPUT_HEIGHT as usize),
            "model.onnx",
        )
        .expect("the pinned pair should pass");
    }

    /// The bundled charset used to be 225 characters against a 316-class model.
    #[test]
    fn contract_rejects_charset_mismatch() {
        let err = check_contract(
            225,
            Some(PINNED_CLASSES),
            Some(EXPECTED_INPUT_HEIGHT as usize),
            "model.onnx",
        )
        .expect_err("225 characters vs 316 classes must be refused");
        assert!(err.0.contains("226") && err.0.contains("316"), "{err}");
    }

    /// `.trim()` eating the leading space is a one-character mismatch, and one
    /// character is enough to shift the whole decode.
    #[test]
    fn contract_rejects_off_by_one_charset() {
        check_contract(
            PINNED_CHAR_LEN - 1,
            Some(PINNED_CLASSES),
            Some(EXPECTED_INPUT_HEIGHT as usize),
            "model.onnx",
        )
        .expect_err("314 characters vs 316 classes must be refused");
    }

    /// This binding hard-coded `target_height: 64` while the pinned model's
    /// input is a static 128.
    #[test]
    fn contract_rejects_height_mismatch() {
        check_contract(
            PINNED_CHAR_LEN,
            Some(PINNED_CLASSES),
            Some(64),
            "stale.onnx",
        )
        .expect_err("a 64-pixel input must be refused");
    }

    #[test]
    fn contract_rejects_empty_charset() {
        check_contract(0, Some(PINNED_CLASSES), Some(128), "model.onnx")
            .expect_err("an empty charset must be refused");
    }

    /// A dynamic axis reports as -1; there is nothing to compare at load time,
    /// so the load passes and decoding re-checks the real output tensor.
    #[test]
    fn contract_skips_dynamic_axes() {
        check_contract(PINNED_CHAR_LEN, None, None, "dynamic.onnx")
            .expect("dynamic axes should defer the check");
    }

    #[test]
    fn static_dim_reads_only_fixed_axes() {
        let shape = [1i64, 1, 128, -1];
        assert_eq!(static_dim(&shape, 2), Some(128));
        assert_eq!(static_dim(&shape, 3), None, "dynamic axis");
        assert_eq!(static_dim(&shape, 9), None, "out of range");
    }

    fn synthetic_logits(seq_len: usize, num_classes: usize) -> Vec<f32> {
        (0..seq_len * num_classes)
            .map(|i| (i as f32 * 0.37).sin())
            .collect()
    }

    /// The decode stride must come from the output tensor, never from the
    /// charset. A charset that disagrees is refused outright rather than
    /// reinterpreting the whole buffer.
    #[test]
    fn decode_stride_comes_from_the_tensor() {
        let seq_len = 128;
        let data = synthetic_logits(seq_len, PINNED_CLASSES);
        let shape = [1, seq_len, PINNED_CLASSES];

        let text = decode_ctc(&charset_of_len(PINNED_CHAR_LEN), &data, &shape)
            .expect("the matching charset should decode");
        assert!(!text.is_empty());

        // The `.trim()` victim: one character short.
        decode_ctc(&charset_of_len(PINNED_CHAR_LEN - 1), &data, &shape)
            .expect_err("a 314-character charset against a 316-class tensor must be refused");

        // The old bundled charset.
        decode_ctc(&charset_of_len(225), &data, &shape)
            .expect_err("a 225-character charset against a 316-class tensor must be refused");
    }

    #[test]
    fn decode_rejects_unexpected_shapes() {
        let charset = charset_of_len(PINNED_CHAR_LEN);
        let data = synthetic_logits(8, PINNED_CLASSES);

        decode_ctc(&charset, &data, &[8, PINNED_CLASSES]).expect_err("2-D output");
        decode_ctc(&charset, &data, &[1, 0, PINNED_CLASSES]).expect_err("empty sequence axis");
        decode_ctc(&charset, &data, &[1, 16, PINNED_CLASSES])
            .expect_err("shape larger than the buffer");
    }

    /// CTC: index 0 is blank, repeats collapse, index n maps to charset[n - 1].
    #[test]
    fn decode_ctc_semantics() {
        let charset: Vec<char> = "abc".chars().collect();
        let num_classes = charset.len() + 1;

        // Timesteps: a, a, blank, a, b, c
        let argmax = [1usize, 1, 0, 1, 2, 3];
        let mut data = vec![0.0f32; argmax.len() * num_classes];
        for (t, &want) in argmax.iter().enumerate() {
            data[t * num_classes + want] = 1.0;
        }

        let got = decode_ctc(&charset, &data, &[1, argmax.len(), num_classes]).unwrap();
        assert_eq!(got, "aabc");
    }
}
