//! Line Segmentation
//!
//! This module handles the segmentation of document images into individual text lines
//! using horizontal projection profile analysis.

use anyhow::Result;
use image::{GrayImage, ImageBuffer};
use std::path::Path;

/// Bounding box for a line segment
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

/// Result of line segmentation
///
/// Contains the cropped image of a single text line and its bounding box
/// in the original image.
#[derive(Debug, Clone)]
pub struct LineSegment {
    /// Cropped grayscale image containing only this text line
    pub img: GrayImage,
    /// Bounding box of this line in the original image
    pub bbox: BBox,
}

/// Line segmenter using horizontal projection profile
///
/// This segmenter detects text lines in a document image by analyzing the
/// horizontal projection profile - the sum of dark pixels in each row.
///
/// # Algorithm
///
/// 1. Convert image to grayscale and binarize (threshold at 128)
/// 2. Compute horizontal projection profile (sum of dark pixels per row)
/// 3. Apply smoothing to reduce noise
/// 4. Find gaps between text regions (where projection is near zero)
/// 5. Extract each text region as a separate line
///
/// # Parameters
///
/// - `min_line_height`: Minimum height to consider as a valid text line
/// - `smooth_window`: Window size for smoothing the projection profile
pub struct LineSegmenter {
    /// Minimum height for a valid text line (in pixels)
    min_line_height: u32,
    /// Window size for histogram smoothing
    smooth_window: u32,
}

impl LineSegmenter {
    /// Create a new line segmenter with specified parameters
    ///
    /// # Arguments
    ///
    /// * `min_line_height` - Minimum height in pixels to consider as a valid text line
    /// * `smooth_window` - Window size for smoothing the projection profile (1 = no smoothing)
    ///
    /// # Returns
    ///
    /// A new `LineSegmenter` instance
    ///
    /// # Example
    ///
    /// ```ignore
    /// use monocr_onnx::segmenter::LineSegmenter;
    ///
    /// // Create segmenter with default parameters
    /// let segmenter = LineSegmenter::new(10, 3);
    /// ```
    pub fn new(min_line_height: u32, smooth_window: u32) -> Self {
        Self {
            min_line_height,
            smooth_window,
        }
    }

    /// Segment an image into text lines
    ///
    /// This is the main method that performs line segmentation on a document image.
    /// It uses horizontal projection profile analysis to detect text lines.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<LineSegment>)` - Vector of segmented lines with images and bounding boxes
    /// * `Err(anyhow::Error)` - If the image cannot be opened or processed
    ///
    /// # Algorithm Details
    ///
    /// 1. **Binarization**: Convert to grayscale and threshold at 128 (pixels < 128 are text)
    /// 2. **Projection**: Compute horizontal projection profile (sum of text pixels per row)
    /// 3. **Smoothing**: Apply moving average filter if smooth_window > 1
    /// 4. **Gap Detection**: Find gaps where projection is below 5% of mean density
    /// 5. **Line Extraction**: Extract each region between gaps as a separate line
    /// 6. **Padding**: Add 4-pixel padding around each line for edge character capture
    pub fn segment(&self, image_path: impl AsRef<Path>) -> Result<Vec<LineSegment>> {
        let image_path = image_path.as_ref();
        let img = image::open(image_path)?;
        let gray_img = img.to_luma8();
        let (width, height) = gray_img.dimensions();

        // 1. Get grayscale data and apply threshold
        let mut binary = vec![0u8; (width * height) as usize];
        let mut hist = vec![0f32; height as usize];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let pixel = gray_img.get_pixel(x, y);
                // Threshold: 128, inverted so text is high (1)
                if pixel[0] < 128 {
                    binary[idx] = 1;
                    hist[y as usize] += 1.0;
                }
            }
        }

        // 2. Smooth projection profile
        let smoothed_hist = if self.smooth_window > 1 {
            self.smooth_histogram(&hist)
        } else {
            hist
        };

        // 3. Gap detection
        let non_zero_vals: Vec<f32> = smoothed_hist
            .iter()
            .filter(|&&v| v > 0.0)
            .copied()
            .collect();

        if non_zero_vals.is_empty() {
            return Ok(Vec::new());
        }

        let mean_density: f32 = non_zero_vals.iter().sum::<f32>() / non_zero_vals.len() as f32;
        let gap_threshold = mean_density * 0.05;

        // 4. Find line regions
        let mut results = Vec::new();
        let mut start: Option<u32> = None;

        for y in 0..height {
            let is_text = smoothed_hist[y as usize] > gap_threshold;

            if is_text && start.is_none() {
                start = Some(y);
            } else if !is_text && start.is_some() {
                let end = y;
                let line_height = end - start.unwrap();

                if line_height >= self.min_line_height {
                    self.extract_line(
                        &gray_img,
                        &binary,
                        width,
                        height,
                        start.unwrap(),
                        end,
                        &mut results,
                    )?;
                }
                start = None;
            }
        }

        // Handle last line if image ends with text
        if let Some(s) = start {
            let line_height = height - s;
            if line_height >= self.min_line_height {
                self.extract_line(&gray_img, &binary, width, height, s, height, &mut results)?;
            }
        }

        Ok(results)
    }

    /// Smooth histogram using moving average
    ///
    /// Applies a moving average filter to the projection histogram to reduce
    /// noise and smooth out variations. This helps identify text regions more accurately.
    ///
    /// # Arguments
    ///
    /// * `hist` - Input projection histogram (one value per row)
    ///
    /// # Returns
    ///
    /// Smoothed histogram with the same length as input
    ///
    /// # Algorithm
    ///
    /// For each position, computes the average of values within the window:
    /// `[i - half_window, i + half_window]`
    fn smooth_histogram(&self, hist: &[f32]) -> Vec<f32> {
        let height = hist.len();
        let mut smoothed = vec![0f32; height];
        let half = (self.smooth_window / 2) as i32;

        for i in 0..height {
            let mut sum = 0f32;
            let mut count = 0u32;

            for j in (i as i32 - half)..=(i as i32 + half) {
                if j >= 0 && j < height as i32 {
                    sum += hist[j as usize];
                    count += 1;
                }
            }

            smoothed[i] = if count > 0 { sum / count as f32 } else { 0.0 };
        }

        smoothed
    }

    /// Extract a single line from the image and add to results
    ///
    /// This method extracts a rectangular region from the grayscale image
    /// corresponding to a detected text line.
    ///
    /// # Process
    ///
    /// 1. Find horizontal bounds (x_min, x_max) of text pixels in the region
    /// 2. Add 4-pixel padding around the detected text
    /// 3. Crop the region from the original image
    /// 4. Create a LineSegment with the cropped image and bounding box
    ///
    /// # Arguments
    ///
    /// * `gray_img` - Source grayscale image
    /// * `binary` - Binary image (1 = text, 0 = background)
    /// * `width` - Width of source image
    /// * `height` - Height of source image
    /// * `r_start` - Starting row (y coordinate) of the line region
    /// * `r_end` - Ending row (y coordinate) of the line region
    /// * `results` - Vector to append the extracted line to
    fn extract_line(
        &self,
        gray_img: &GrayImage,
        binary: &[u8],
        width: u32,
        height: u32,
        r_start: u32,
        r_end: u32,
        results: &mut Vec<LineSegment>,
    ) -> Result<()> {
        // Find horizontal bounds
        let mut x_min = width;
        let mut x_max = 0u32;
        let mut has_pixels = false;

        for y in r_start..r_end {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if binary[idx] == 1 {
                    if x < x_min {
                        x_min = x;
                    }
                    if x > x_max {
                        x_max = x;
                    }
                    has_pixels = true;
                }
            }
        }

        if !has_pixels {
            return Ok(());
        }

        // Add padding around detected text regions to capture edge characters
        // 4 pixels provides enough margin without including excessive background
        let pad = 4;
        let y1 = r_start.saturating_sub(pad);
        let y2 = (r_end + pad).min(height);
        let x1 = x_min.saturating_sub(pad);
        let x2 = (x_max + pad).min(width);

        let w = x2 - x1;
        let h = y2 - y1;

        // Extract the region
        let mut line_img = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let src_x = x1 + x;
                let src_y = y1 + y;
                let pixel = gray_img.get_pixel(src_x, src_y);
                line_img.put_pixel(x, y, *pixel);
            }
        }

        results.push(LineSegment {
            img: line_img,
            bbox: BBox { x: x1, y: y1, w, h },
        });

        Ok(())
    }
}
