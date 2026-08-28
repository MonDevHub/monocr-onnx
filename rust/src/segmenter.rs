//! Line Segmentation
//!
//! This module handles the segmentation of document images into individual text lines
//! using horizontal projection profile analysis.

use anyhow::Result;
use image::{imageops::crop_imm, GrayImage, ImageBuffer};
use std::ops::Range;
use std::path::Path;

/// Where a tile may be cut, as a fraction of the tile width, searching backwards
/// from the ideal boundary. 0.12 of a 1024px window is ~123px, roughly two Mon
/// glyphs — wide enough to find a gap, narrow enough that tiles stay near full
/// width.
pub const CUT_SEARCH_FRACTION: f64 = 0.12;

/// A column counts as carrying ink below this grayscale value.
pub const CUT_INK_THRESHOLD: u8 = 250;

/// Where to end a tile that starts at `x0` and may not pass `ideal`.
///
/// Cutting at exactly `ideal` lands wherever the arithmetic falls, which is
/// usually the middle of a glyph. Both halves keep their pixels, so a coverage
/// check still passes, but the model reads each half as a whole character and one
/// glyph becomes two. Measured upstream on 120 drawn lines this showed up as
/// `ဗော်` read back as `ဗေဗိာ်`.
///
/// So search backwards from `ideal` for a column of white. A tile may only get
/// narrower, never wider, or it stops fitting the model window. Returns `ideal`
/// unchanged when there is no gap to cut at, which is the honest outcome for a
/// continuous script: a known-bad seam beats an overflowing tile.
///
/// Ported from `monocr_onnx.segmenter.cut_column`; the constants and the
/// tie-breaking are the same, so the two produce the same cuts on the same
/// input. The shared fixture in `monocr-monorepo/shared/segmentation-fixtures`
/// is what holds them together.
pub fn cut_column(crop: &GrayImage, x0: u32, ideal: u32, crop_w: u32) -> u32 {
    if ideal >= crop_w {
        return crop_w;
    }

    // `as u32` truncates toward zero, which is what Python's `int()` does, so
    // the two ports pick the same window on the same input.
    let window = (((ideal - x0) as f64 * CUT_SEARCH_FRACTION) as u32).max(1);
    // Python computes `ideal - window` in unbounded integers and can go
    // negative; `max(x0 + 1, ...)` then discards it. Saturating at 0 reaches the
    // same answer because x0 + 1 is always the larger value there.
    let lo = (x0 + 1).max(ideal.saturating_sub(window));
    if lo >= ideal {
        return ideal;
    }

    let height = crop.height();
    let mut rightmost_blank: Option<u32> = None;
    let mut lightest_offset = 0u32;
    let mut lightest_ink = u32::MAX;

    for x in lo..ideal {
        let mut ink = 0u32;
        for y in 0..height {
            if crop.get_pixel(x, y)[0] < CUT_INK_THRESHOLD {
                ink += 1;
            }
        }

        let offset = x - lo;
        if ink == 0 {
            rightmost_blank = Some(offset);
        }
        // Strict `<` keeps the leftmost of equally light columns, which is what
        // numpy's argmin returns. The fixture pins this: on solid ink every
        // column ties and the cut must land on `lo`.
        if ink < lightest_ink {
            lightest_ink = ink;
            lightest_offset = offset;
        }
    }

    // Prefer a truly empty column, and the rightmost one, so tiles stay as wide
    // as the window allows. Fall back to the lightest column present.
    lo + rightmost_blank.unwrap_or(lightest_offset)
}

/// Split one line crop into pieces that each fit the model window.
///
/// Returns the crop unchanged when the line already fits after being scaled to
/// `target_h`. Otherwise cuts at whitespace columns and returns the pieces left
/// to right, to be read separately and joined with no separator.
///
/// `target_h` and `target_w` come from the model contract and must both be
/// positive. A zero `target_w` would ask for one-pixel tiles, which is garbage
/// in, garbage out rather than an error worth a result type.
pub fn tile_line(crop: &GrayImage, target_h: u32, target_w: u32) -> Vec<GrayImage> {
    let (crop_w, crop_h) = crop.dimensions();
    if crop_h == 0 || crop_w == 0 {
        return vec![crop.clone()];
    }

    let scale = target_h as f64 / crop_h as f64;
    // Truncation again matches Python's `int()`. It matters at the boundary: a
    // line that scales to exactly target_w is left alone, one pixel over is
    // tiled.
    if (crop_w as f64 * scale) as u32 <= target_w {
        return vec![crop.clone()];
    }

    // Must stay an f64 division in this order. Integer arithmetic on
    // target_w * crop_h / target_h would round differently, and the fixture's
    // 1.6 scale is a case where the difference is a whole pixel per tile.
    let tile_w_src = ((target_w as f64 / scale) as u32).max(1);
    let mut tiles = Vec::new();
    let mut x0 = 0u32;
    while x0 < crop_w {
        let ideal = x0.saturating_add(tile_w_src).min(crop_w);
        // Structural guard, not a tuning knob: cut_column can only return a
        // value in (x0, ideal], but if it ever returned x0 this loop would spin
        // forever on a page. One pixel of forced progress bounds it.
        let x1 = cut_column(crop, x0, ideal, crop_w).max(x0 + 1);
        tiles.push(crop_imm(crop, x0, 0, x1 - x0, crop_h).to_image());
        x0 = x1;
    }
    tiles
}

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

/// A page and the text mask derived from it, kept together.
///
/// The two are only meaningful as a pair: `binary` is a flat row-major slice
/// that can only be indexed with the image's own width as the stride. Passing
/// them as separate arguments made it possible to hand one function a mask and a
/// width that did not agree; here the stride comes from `gray` so it cannot
/// disagree.
struct BinarizedPage<'a> {
    /// Grayscale source that line crops are taken from
    gray: &'a GrayImage,
    /// Text mask, 1 = text and 0 = background, row-major over `gray`
    binary: &'a [u8],
}

/// A printed rule -- a page border, a table rule, an underline -- spans at least
/// this fraction of the page in one direction.
///
/// Deliberately coarse: no Mon, Burmese or Latin glyph holds an unbroken stroke
/// half a page long, so the false-positive risk against text is structural
/// rather than merely small. Lowering it toward a glyph's width is what would
/// make rule suppression dangerous.
const RULE_SPAN: f64 = 0.5;

/// A rule must span at least this many pixels whatever [`RULE_SPAN`] works out
/// to. On a 20px-wide crop `width * RULE_SPAN` is 10px, which a single character
/// can reach; the floor is what keeps the span out of glyph range on small
/// crops.
const RULE_MIN_SPAN_PX: usize = 15;

/// Suppression that would remove more than this share of the page's ink has
/// found text, not rules, and is abandoned.
///
/// [`RULE_SPAN`] is a fraction of the page, so on a SHORT page a tall block of
/// text can exceed it vertically and be deleted wholesale. Upstream that was not
/// hypothetical: without this guard an existing test -- six 30px bands touching
/// on a 200px page, so each glyph column is 180px of unbroken ink -- lost 98.7%
/// of its ink and returned zero lines.
///
/// The threshold sits in a measured gap rather than being a round number: real
/// framed pages classify 21.5%-58.8% of their ink as rules, every page carrying
/// no rules 0.00%, and that false positive 98.7%. 1.36x above the worst
/// legitimate case and 1.23x below the true positive.
const RULE_MAX_INK_SHARE: f64 = 0.80;

/// Zero out printed rules in `mask` (1 = ink), in place, and report whether
/// anything was removed.
///
/// A printed page border adds a constant ink floor to every row it spans, and
/// once that floor clears the gap threshold no in-frame row reads as a gap: the
/// page comes back as one band and is squeezed into the model window. Nothing
/// downstream can recover from that, because the line was never found.
///
/// MEASURED WITH THIS PARAMETER SET (global threshold 128, no smear, smoothing
/// 3, ratio 0.05 of the mean) over twelve real MNEC page-ones: nine collapse to
/// three bands or fewer, and the twelve together go from 118 bands to 215. Pages
/// carrying no rules come back byte-identical.
///
/// A run-length scan rather than a generic erode-then-dilate: opening with a 1xL
/// line kernel keeps exactly those ink runs at least L long, which one sweep per
/// axis computes directly. That is the form `js/src/segmenter.js` and
/// `go/pkg/segmenter/segmenter.go` use; the reference
/// (`mon_OCR/src/monocr/segmenter.py` `_suppress_page_rules`) reaches the same
/// answer with `cv2.morphologyEx`, and the shared fixture
/// `monocr-monorepo/shared/segmentation-fixtures/rule-cases.json` is what holds
/// the four together.
///
/// There is deliberately NO thickness test. "A rule is long AND thin" was
/// written, measured and deleted upstream: across twelve real pages the rule
/// pixels found with a thickness limit and with none were identical to the
/// pixel.
fn suppress_page_rules(mask: &mut [u8], width: u32, height: u32) -> bool {
    let (w, h) = (width as usize, height as usize);
    // A mask shorter than its own stated dimensions cannot be indexed safely,
    // and guessing at the real shape would corrupt a page rather than skip it.
    if w == 0 || h == 0 || mask.len() < w * h {
        return false;
    }

    // `as usize` truncates toward zero, matching Python's `int()` and Go's
    // `int()`, so all four ports pick the same span on an odd page width. The
    // fixture's "odd width, run at the truncated span" case is what pins it.
    let min_h = ((width as f64 * RULE_SPAN) as usize).max(RULE_MIN_SPAN_PX);
    let min_v = ((height as f64 * RULE_SPAN) as usize).max(RULE_MIN_SPAN_PX);

    // Rules are collected into a separate plane and only subtracted at the end.
    // Clearing them as they are found would let the horizontal sweep break a
    // vertical rule before the vertical sweep ever sees it, which is
    // order-dependent and silently loses one axis.
    let mut rules = vec![0u8; w * h];

    for y in 0..h {
        let row = y * w;
        let mut start = 0usize;
        // The extra step past the end closes a run that reaches the edge; a
        // border is exactly that run, so stopping at `w` would miss every
        // full-width rule.
        for x in 0..=w {
            if x < w && mask[row + x] != 0 {
                continue;
            }
            if x - start >= min_h {
                rules[row + start..row + x].fill(1);
            }
            start = x + 1;
        }
    }
    for x in 0..w {
        let mut start = 0usize;
        for y in 0..=h {
            if y < h && mask[y * w + x] != 0 {
                continue;
            }
            if y - start >= min_v {
                for i in start..y {
                    rules[i * w + x] = 1;
                }
            }
            start = y + 1;
        }
    }

    let ink = mask[..w * h].iter().filter(|&&v| v != 0).count();
    let rule_ink = rules.iter().filter(|&&v| v != 0).count();
    if ink == 0 || rule_ink == 0 || rule_ink as f64 > ink as f64 * RULE_MAX_INK_SHARE {
        // Found the text. Leaving the page alone is strictly better than
        // emptying it, and the caller is no worse off than before this step
        // existed.
        return false;
    }

    for (cell, &rule) in mask.iter_mut().zip(rules.iter()) {
        if rule != 0 {
            *cell = 0;
        }
    }
    true
}

/// Line segmenter using horizontal projection profile
///
/// This segmenter detects text lines in a document image by analyzing the
/// horizontal projection profile - the sum of dark pixels in each row.
///
/// # Algorithm
///
/// 1. Convert image to grayscale and binarize (threshold at 128)
/// 2. Suppress printed rules — page borders, table rules, underlines — so their
///    ink floor cannot hide every gap; see `suppress_page_rules`
/// 3. Compute horizontal projection profile (sum of dark pixels per row)
/// 4. Apply smoothing to reduce noise
/// 5. Find gaps between text regions (where projection is near zero)
/// 6. Extract each text region as a separate line
///
/// # Parameters
///
/// - `min_line_height`: Minimum height to consider as a valid text line
/// - `smooth_window`: Window size for smoothing the projection profile
/// - `density_threshold_ratio`: Fraction of mean row density that still counts
///   as a gap
pub struct LineSegmenter {
    /// Minimum height for a valid text line (in pixels)
    min_line_height: u32,
    /// Window size for histogram smoothing
    smooth_window: u32,
    /// Fraction of the mean non-empty row density below which a row counts as a
    /// gap between lines
    density_threshold_ratio: f32,
}

/// The gap threshold this segmenter has always used, kept as the default so
/// existing callers segment identically.
///
/// Every port of this pipeline picked a different number (canonical mon_OCR
/// 0.12, the Python binding 0.02 of max, web and Android 0.03, iOS 0.03), which
/// is the sign that it belongs to the input class rather than to the algorithm.
pub const DEFAULT_DENSITY_THRESHOLD_RATIO: f32 = 0.05;

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
        Self::with_density_ratio(
            min_line_height,
            smooth_window,
            DEFAULT_DENSITY_THRESHOLD_RATIO,
        )
    }

    /// Create a segmenter with an explicit gap threshold ratio.
    ///
    /// See [`crate::MonOcrBuilder::density_threshold_ratio`] for what the ratio
    /// does and why it is worth setting per input class. The caller is
    /// responsible for passing a finite, positive ratio; the builder validates
    /// it.
    pub fn with_density_ratio(
        min_line_height: u32,
        smooth_window: u32,
        density_threshold_ratio: f32,
    ) -> Self {
        Self {
            min_line_height,
            smooth_window,
            density_threshold_ratio,
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
    /// 2. **Rule suppression**: Remove printed rules, so a page border cannot
    ///    fuse the whole page into one band (`suppress_page_rules`)
    /// 3. **Projection**: Compute horizontal projection profile (sum of text pixels per row)
    /// 4. **Smoothing**: Apply moving average filter if smooth_window > 1
    /// 5. **Gap Detection**: Find gaps where the RAW projection is below
    ///    `density_threshold_ratio` of the SMOOTHED profile's mean non-empty row
    ///    density (default 5%). The two profiles are deliberately different: the
    ///    smoothed mean is the steadier calibration, and the raw profile is the
    ///    only one that still reaches zero between tightly set lines
    /// 6. **Line Extraction**: Extract each region between gaps as a separate line
    /// 7. **Padding**: Add 4-pixel padding around each line for edge character capture
    ///
    /// # Polarity
    ///
    /// The threshold treats dark as ink, so a light-on-dark page must be
    /// inverted before it reaches here or the BACKGROUND is what gets segmented.
    /// [`crate::normalize_polarity`] is that step and `MonOcr::predict_page`
    /// runs it. This method does not, because it is also the entry point for a
    /// caller who has already corrected polarity.
    pub fn segment(&self, image_path: impl AsRef<Path>) -> Result<Vec<LineSegment>> {
        let img = image::open(image_path.as_ref())?;
        self.segment_image(&img.to_luma8())
    }

    /// Segment an image that is already decoded and grayscale.
    ///
    /// The path-taking [`Self::segment`] is a thin wrapper over this. The split
    /// exists because polarity has to be corrected BEFORE segmentation — the
    /// threshold below treats dark as ink, so a light-on-dark page segments the
    /// BACKGROUND and returns the gaps between lines — and the caller doing that
    /// correction is holding an image, not a path. `go/monocr.go`'s
    /// `predictImage` and `js/src/monocr.js`'s `normalizePageForSegmentation`
    /// are the same arrangement.
    pub fn segment_image(&self, gray_img: &GrayImage) -> Result<Vec<LineSegment>> {
        let (width, height) = gray_img.dimensions();

        // 1. Get grayscale data and apply threshold
        //
        // The mask is materialised before the profile, and the profile is
        // counted from the mask afterwards rather than in this loop, because
        // rule suppression needs the 2-D shape of the ink: a per-row count
        // cannot express "is there an unbroken run this long". Folding the two
        // back together is what would silently compute the profile from the
        // unsuppressed page.
        let mut binary = vec![0u8; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let pixel = gray_img.get_pixel(x, y);
                // Threshold: 128, inverted so text is high (1)
                if pixel[0] < 128 {
                    binary[idx] = 1;
                }
            }
        }

        // 1.5 Printed-rule suppression, before the profile.
        //
        // See `suppress_page_rules` for what a page border costs: its ink floor
        // clears the gap threshold on every row it spans, and at THIS parameter
        // set the twelve measured MNEC pages went from 118 bands to 215. It also
        // runs before `extract_line` reads the mask for column extents, so
        // removing rules here keeps the border out of the crops too.
        //
        // The character-count figure quoted in the Python binding (3,846 to
        // 5,924) belongs to the reference's adaptive threshold and smear, not to
        // this one, so it is not repeated here.
        suppress_page_rules(&mut binary, width, height);

        let mut hist = vec![0f32; height as usize];
        for y in 0..height {
            let row = (y * width) as usize;
            for x in 0..width as usize {
                if binary[row + x] != 0 {
                    hist[y as usize] += 1.0;
                }
            }
        }

        // 2. Smooth projection profile
        //
        // `hist` is kept alive because the two profiles have different jobs: the
        // threshold below is calibrated on the smoothed one, the boundaries are
        // detected on the raw one. See step 4 for why.
        let smoothed_hist = if self.smooth_window > 1 {
            self.smooth_histogram(&hist)
        } else {
            hist.clone()
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
        let gap_threshold = mean_density * self.density_threshold_ratio;

        // 4. Find line regions
        let page = BinarizedPage {
            gray: gray_img,
            binary: &binary,
        };
        let mut results = Vec::new();
        let mut start: Option<u32> = None;

        for y in 0..height {
            // Boundaries come off the RAW profile, not the smoothed one.
            //
            // The threshold above stays calibrated on the smoothed profile,
            // because its non-zero mean is steadier. But the smoother averages
            // over `smooth_window` rows, so a gap narrower than the whole
            // window never reaches zero in the smoothed profile: the ink either
            // side bleeds into it, the bled rows clear the threshold, and the
            // two lines fuse. The raw profile needs one clean row.
            //
            // Measured HERE, at this port's own parameters (min_line_height 10,
            // smooth_window 3, density_threshold_ratio 0.05) on 29 drawn bands:
            // reading the smoothed profile returned 1 band at gaps of 1px and
            // 2px, against 29 lines drawn, and matched the raw profile from 3px
            // up. The break point is the smoother's full width, so a caller who
            // raises `smooth_window` widens the failure with it — at 15 the
            // smoothed profile lost every page whose lines sat closer than 15px
            // while the raw profile kept all 29.
            //
            // Rust's break point is far tighter than the monorepo apps', which
            // fused at 5px to 8px, because those ports dilate the mask
            // vertically before the profile and this one does not. Their
            // measurements do not transfer; these are this port's.
            let is_text = hist[y as usize] > gap_threshold;

            if is_text && start.is_none() {
                start = Some(y);
            } else if !is_text && start.is_some() {
                let end = y;
                let line_height = end - start.unwrap();

                if line_height >= self.min_line_height {
                    self.extract_line(&page, start.unwrap()..end, &mut results)?;
                }
                start = None;
            }
        }

        // Handle last line if image ends with text
        if let Some(s) = start {
            let line_height = height - s;
            if line_height >= self.min_line_height {
                self.extract_line(&page, s..height, &mut results)?;
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

        // The window reads `hist` while the position writes `smoothed`, two
        // separate allocations, so iterating the write target does not disturb
        // the values being averaged.
        for (i, out) in smoothed.iter_mut().enumerate() {
            let mut sum = 0f32;
            let mut count = 0u32;

            for j in (i as i32 - half)..=(i as i32 + half) {
                if j >= 0 && j < height as i32 {
                    sum += hist[j as usize];
                    count += 1;
                }
            }

            *out = if count > 0 { sum / count as f32 } else { 0.0 };
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
    /// * `page` - Source grayscale image and its matching text mask
    /// * `rows` - Half-open row range (y coordinates) of the line region
    /// * `results` - Vector to append the extracted line to
    fn extract_line(
        &self,
        page: &BinarizedPage,
        rows: Range<u32>,
        results: &mut Vec<LineSegment>,
    ) -> Result<()> {
        let gray_img = page.gray;
        let binary = page.binary;
        let (width, height) = gray_img.dimensions();
        let (r_start, r_end) = (rows.start, rows.end);

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use serde_json::Value;
    use std::path::PathBuf;

    /// Override for the shared fixture, for checkouts that do not sit next to
    /// the monorepo.
    const FIXTURE_ENV: &str = "MONOCR_TILING_FIXTURE";

    /// The fixture is shared with the web, Android and iOS ports on purpose: one
    /// file, generated from the Python implementation, so a port that drifts
    /// fails here instead of in production. Transcribing the numbers into Rust
    /// would defeat that, so the tests read the JSON.
    fn fixture_path() -> PathBuf {
        if let Some(path) = std::env::var_os(FIXTURE_ENV) {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../monocr-monorepo/shared/segmentation-fixtures/tiling-cases.json")
    }

    /// A missing fixture fails loudly. Skipping would report a green run for a
    /// port nothing checked, which is the exact failure this fixture exists to
    /// prevent.
    fn load_fixture() -> Value {
        let path = fixture_path();
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read the shared tiling fixture at {}: {e}\n\
                 set {FIXTURE_ENV} to point at \
                 monocr-monorepo/shared/segmentation-fixtures/tiling-cases.json",
                path.display()
            )
        });
        serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", path.display()))
    }

    fn u32_field(value: &Value, key: &str) -> u32 {
        value
            .get(key)
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("fixture entry is missing an integer '{key}': {value}"))
            as u32
    }

    /// Ink is grey 0, background grey 255, per the fixture contract.
    fn build_image(width: u32, height: u32, ink: &Value) -> GrayImage {
        let kind = ink
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("ink rule has no 'kind': {ink}"));
        let modulus = u32_field(ink, "modulus");

        let mut img = GrayImage::from_pixel(width, height, Luma([255u8]));
        for x in 0..width {
            let is_ink = match kind {
                "mod_eq" => x % modulus == 0,
                "mod_ne" => x % modulus != 0,
                "solid" => true,
                "blank" => false,
                other => panic!("unknown ink rule '{other}' in the fixture"),
            };
            if is_ink {
                for y in 0..height {
                    img.put_pixel(x, y, Luma([0u8]));
                }
            }
        }
        img
    }

    struct Fixture {
        target_height: u32,
        target_width: u32,
        root: Value,
    }

    fn fixture() -> Fixture {
        let root = load_fixture();
        let target_height = u32_field(&root, "target_height");
        let target_width = u32_field(&root, "target_width");

        // The constants live in two places, so pin them to each other here.
        assert_eq!(
            root.get("cut_search_fraction").and_then(Value::as_f64),
            Some(CUT_SEARCH_FRACTION),
            "fixture and port disagree on the cut search fraction"
        );
        assert_eq!(
            root.get("cut_ink_threshold").and_then(Value::as_u64),
            Some(CUT_INK_THRESHOLD as u64),
            "fixture and port disagree on the ink threshold"
        );

        Fixture {
            target_height,
            target_width,
            root,
        }
    }

    /// An empty array would make every assertion below vacuous, so an empty one
    /// is a fixture failure rather than a pass.
    fn cases(root: &Value, key: &str) -> Vec<Value> {
        let cases = root
            .get(key)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("fixture has no '{key}' array"))
            .clone();
        assert!(!cases.is_empty(), "fixture '{key}' is empty");
        cases
    }

    fn case_image(case: &Value) -> (GrayImage, String) {
        let name = case
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unnamed>")
            .to_string();
        let ink = case
            .get("ink")
            .unwrap_or_else(|| panic!("case '{name}' has no ink rule"));
        let img = build_image(u32_field(case, "width"), u32_field(case, "height"), ink);
        (img, name)
    }

    #[test]
    fn tile_widths_match_the_shared_fixture() {
        let f = fixture();
        for case in cases(&f.root, "cases") {
            let (img, name) = case_image(&case);
            let expected: Vec<u32> = case
                .get("expected_tile_widths")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("case '{name}' has no expected_tile_widths"))
                .iter()
                .map(|v| {
                    v.as_u64()
                        .unwrap_or_else(|| panic!("case '{name}' has a non-integer tile width"))
                        as u32
                })
                .collect();

            let widths: Vec<u32> = tile_line(&img, f.target_height, f.target_width)
                .iter()
                .map(|t| t.width())
                .collect();

            assert_eq!(widths, expected, "case '{name}'");
        }
    }

    /// Tiles must cover the line exactly once. Concatenating them back is the
    /// direct proof: a gap or an overlap changes the pixels, not just the count.
    #[test]
    fn tiles_partition_the_line() {
        let f = fixture();
        for case in cases(&f.root, "cases") {
            let (img, name) = case_image(&case);
            let tiles = tile_line(&img, f.target_height, f.target_width);

            let total: u32 = tiles.iter().map(|t| t.width()).sum();
            assert_eq!(
                total,
                img.width(),
                "case '{name}': tile widths must sum to the line width"
            );

            let mut rebuilt = GrayImage::new(img.width(), img.height());
            let mut x_off = 0u32;
            for tile in &tiles {
                assert_eq!(
                    tile.height(),
                    img.height(),
                    "case '{name}': a tile must keep the full line height"
                );
                assert!(tile.width() > 0, "case '{name}': empty tile");
                for x in 0..tile.width() {
                    for y in 0..tile.height() {
                        rebuilt.put_pixel(x_off + x, y, *tile.get_pixel(x, y));
                    }
                }
                x_off += tile.width();
            }
            assert!(
                rebuilt.as_raw() == img.as_raw(),
                "case '{name}': tiles do not reassemble into the source line"
            );
        }
    }

    /// The single-line path (`MonOcr::predict_single_line`) runs a caller's
    /// already-cropped line through the same `tile_line`, so a wide crop must
    /// come back as several tiles covering the full width. If it ever returned
    /// one tile the crop would be squeezed into the model window. Measured cost of
    /// that on this binding: nothing at 3 tiles, 4.1x the error at 4, and 23x at 8
    /// (`examples/tiling_ab.rs`, `mon_OCR/eval/tiling-ab-2026-08-22.md`).
    #[test]
    fn a_wide_crop_is_tiled_not_squeezed() {
        let f = fixture();
        let mut checked = 0;

        for case in cases(&f.root, "cases") {
            let expected_count = case
                .get("expected_tile_widths")
                .and_then(Value::as_array)
                .map(|a| a.len())
                .unwrap_or(0);
            if expected_count < 2 {
                continue;
            }

            let (img, name) = case_image(&case);
            let tiles = tile_line(&img, f.target_height, f.target_width);
            assert!(
                tiles.len() > 1,
                "case '{name}': a crop this wide must be tiled, got {} tile(s)",
                tiles.len()
            );
            assert_eq!(
                tiles.iter().map(|t| t.width()).sum::<u32>(),
                img.width(),
                "case '{name}': tiles must cover the whole crop"
            );
            for tile in &tiles {
                assert!(
                    tile.width() <= img.width(),
                    "case '{name}': a tile cannot be wider than the crop"
                );
            }
            checked += 1;
        }

        assert!(checked > 0, "the fixture has no multi-tile case to check");
    }

    /// Override for the shared printed-rule fixture, for checkouts that do not
    /// sit next to the monorepo.
    const RULE_FIXTURE_ENV: &str = "MONOCR_RULE_FIXTURE";

    /// The printed-rule fixture is the oracle four ports share, generated from
    /// the reference implementation by
    /// `monocr-monorepo/shared/segmentation-fixtures/generate-rule-cases.py`. It
    /// describes each mask as a PRNG seed plus a list of rules, so a port builds
    /// the same 23 masks without shipping any pixels, and checks the result
    /// against an ink count and a position-weighted checksum.
    ///
    /// The checksum is the part that matters: a bare ink count would not notice
    /// suppression that removed the right NUMBER of pixels in the wrong places,
    /// which is exactly what an off-by-one in a run-length scan produces.
    fn rule_fixture_path() -> PathBuf {
        if let Some(path) = std::env::var_os(RULE_FIXTURE_ENV) {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../monocr-monorepo/shared/segmentation-fixtures/rule-cases.json")
    }

    /// A missing fixture fails loudly, for the same reason `load_fixture` does:
    /// skipping would report a green run for a port nothing checked.
    fn load_rule_fixture() -> Value {
        let path = rule_fixture_path();
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read the shared printed-rule fixture at {}: {e}\n\
                 set {RULE_FIXTURE_ENV} to point at \
                 monocr-monorepo/shared/segmentation-fixtures/rule-cases.json",
                path.display()
            )
        });
        serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", path.display()))
    }

    fn i64_field(value: &Value, key: &str) -> i64 {
        value
            .get(key)
            .and_then(Value::as_i64)
            .unwrap_or_else(|| panic!("fixture entry is missing an integer '{key}': {value}"))
    }

    /// Rebuild one fixture mask: xorshift32 noise, then the rules drawn over it.
    ///
    /// The PRNG is transcribed from the fixture's own `prng` field rather than
    /// invented here: `x ^= x<<13; x ^= x>>17; x ^= x<<5`, seeded 2463534242,
    /// pixel `i` ink where `x % 100 < density` with `x` taken after the i-th
    /// step. Rust's `<<` on `u32` discards the high bits, which is the `&
    /// 0xFFFFFFFF` the generator writes explicitly.
    fn rule_mask(case: &Value) -> (Vec<u8>, u32, u32) {
        let width = u32_field(case, "width");
        let height = u32_field(case, "height");
        let (w, h) = (width as usize, height as usize);
        let density = u32_field(case, "density");

        let mut x: u32 = 2_463_534_242;
        let mut mask = vec![0u8; w * h];
        for cell in mask.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if x % 100 < density {
                *cell = 1;
            }
        }

        let run_length = i64_field(case, "run_length");
        let run_start = i64_field(case, "run_start") as usize;
        for row in cases_array(case, "rule_rows") {
            let row = row as usize;
            let (len, start) = if run_length < 0 {
                (w, 0)
            } else {
                (run_length as usize, run_start)
            };
            for cell in mask[row * w + start..row * w + w.min(start + len)].iter_mut() {
                *cell = 1;
            }
        }

        let col_length = i64_field(case, "col_length");
        let col_start = i64_field(case, "col_start") as usize;
        for col in cases_array(case, "rule_cols") {
            let col = col as usize;
            let (len, start) = if col_length < 0 {
                (h, 0)
            } else {
                (col_length as usize, col_start)
            };
            for y in start..h.min(start + len) {
                mask[y * w + col] = 1;
            }
        }

        (mask, width, height)
    }

    fn cases_array(case: &Value, key: &str) -> Vec<u64> {
        case.get(key)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("fixture case has no '{key}' array: {case}"))
            .iter()
            .map(|v| {
                v.as_u64()
                    .unwrap_or_else(|| panic!("non-integer entry in '{key}'"))
            })
            .collect()
    }

    /// Ink count and position-weighted checksum of a mask, flattened row-major,
    /// exactly as the fixture generator's `signature` computes them.
    fn rule_signature(mask: &[u8], modulus: u64) -> (u64, u64) {
        let mut ink = 0u64;
        let mut sum = 0u64;
        for (i, &v) in mask.iter().enumerate() {
            if v != 0 {
                ink += 1;
                sum += i as u64 + 1;
            }
        }
        (ink, sum % modulus)
    }

    /// The whole printed-rule contract, against the oracle the other ports use.
    ///
    /// 23 cases, including the pair that pins `>=` on each axis (a run of exactly
    /// the span and one pixel short), the 15px floor on a narrow crop, the
    /// truncated span on an odd width, and the ink-share ceiling both firing and
    /// exactly at the boundary.
    #[test]
    fn page_rules_match_the_shared_fixture() {
        let root = load_rule_fixture();

        // The constants live in two places, so pin them to each other here, the
        // same way `fixture()` does for the tiling constants.
        assert_eq!(
            root.get("rule_span").and_then(Value::as_f64),
            Some(RULE_SPAN),
            "fixture and port disagree on the rule span"
        );
        assert_eq!(
            root.get("rule_max_ink_share").and_then(Value::as_f64),
            Some(RULE_MAX_INK_SHARE),
            "fixture and port disagree on the ink-share ceiling"
        );
        let modulus = root
            .get("checksum_modulus")
            .and_then(Value::as_u64)
            .expect("fixture has no checksum_modulus");

        for case in cases(&root, "cases") {
            let name = case
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("<unnamed>")
                .to_string();
            let (mut mask, width, height) = rule_mask(&case);

            let changed = suppress_page_rules(&mut mask, width, height);
            assert_eq!(
                changed,
                case.get("expected_changed")
                    .and_then(Value::as_bool)
                    .unwrap_or_else(|| panic!("case '{name}' has no expected_changed")),
                "case '{name}': wrong answer on whether anything was suppressed"
            );

            let (ink, checksum) = rule_signature(&mask, modulus);
            assert_eq!(
                ink,
                case.get("expected_ink")
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| panic!("case '{name}' has no expected_ink")),
                "case '{name}': wrong ink count after suppression"
            );
            assert_eq!(
                checksum,
                case.get("expected_checksum")
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| panic!("case '{name}' has no expected_checksum")),
                "case '{name}': right ink count, wrong pixels — an off-by-one in \
                 one of the run-length scans"
            );
        }
    }

    // A realistic page, in the shape `go/pkg/segmenter/page_rules_test.go` uses:
    // glyph blobs rather than solid bars, because a solid bar the width of a text
    // column IS a rule by any definition and would prove nothing.
    const T_WIDTH: u32 = 800;
    const T_BAND: u32 = 40;
    const T_MARGIN: u32 = 30;
    const T_GLYPH_W: u32 = 12;
    const T_PITCH: u32 = 20;
    const T_RULE_W: u32 = 4;

    /// Build a page as a grayscale image: ink 0, background 255.
    fn drawn_page(bands: u32, gap: u32, glyphs: u32, framed: bool) -> GrayImage {
        let height = T_MARGIN * 2 + T_BAND * bands + gap * (bands - 1);
        let mut img = GrayImage::from_pixel(T_WIDTH, height, Luma([255u8]));
        let mut y = T_MARGIN;
        for _ in 0..bands {
            for yy in y..y + T_BAND {
                for k in 0..glyphs {
                    let x0 = 100 + k * T_PITCH;
                    for i in 0..T_GLYPH_W {
                        if x0 + i < T_WIDTH {
                            img.put_pixel(x0 + i, yy, Luma([0u8]));
                        }
                    }
                }
            }
            y += T_BAND + gap;
        }
        if framed {
            for yy in 0..height {
                for i in 0..T_RULE_W {
                    img.put_pixel(10 + i, yy, Luma([0u8]));
                    img.put_pixel(T_WIDTH - 10 - T_RULE_W + i, yy, Luma([0u8]));
                }
            }
            for i in 0..T_RULE_W {
                for x in 0..T_WIDTH {
                    img.put_pixel(x, 10 + i, Luma([0u8]));
                    img.put_pixel(x, height - 10 - T_RULE_W + i, Luma([0u8]));
                }
            }
        }
        img
    }

    /// THE PROPERTY THAT MAKES THIS SAFE UNCONDITIONALLY. Every page gets the
    /// step whether it carries rules or not, so "does nothing" has to be exact
    /// rather than approximate.
    #[test]
    fn a_page_with_no_rules_is_untouched_to_the_pixel() {
        let img = drawn_page(4, 40, 30, false);
        let (w, h) = img.dimensions();
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                if img.get_pixel(x, y)[0] < 128 {
                    mask[(y * w + x) as usize] = 1;
                }
            }
        }
        let before = mask.clone();

        assert!(
            !suppress_page_rules(&mut mask, w, h),
            "suppression reported a change on a page with no rules"
        );
        assert_eq!(
            mask, before,
            "glyph-sized ink was classified as a rule and removed"
        );
    }

    /// The behavioural test, and the fixture took finding.
    ///
    /// A DENSE framed page does not fuse at this parameter set — 30 glyphs per
    /// line segments the same with or without suppression, which is why a
    /// structural check on the mask alone cannot catch a profile computed from
    /// the wrong buffer. SPARSE text reproduces the real mechanism: with 8 glyphs
    /// per line the profile mean drops far enough that the frame's ink floor
    /// clears the 0.05 threshold on every row, and the page comes back as one
    /// band. Ported from `go/pkg/segmenter/page_rules_test.go`
    /// `TestSegmentRecoversAFramedPage`.
    #[test]
    fn segmenting_recovers_a_framed_page() {
        let seg = LineSegmenter::new(10, 3);
        let clean = seg.segment_image(&drawn_page(4, 40, 8, false)).unwrap();
        let framed = seg.segment_image(&drawn_page(4, 40, 8, true)).unwrap();

        assert_eq!(
            clean.len(),
            4,
            "the unframed control must segment into 4 lines, or the comparison \
             below proves nothing"
        );
        assert_eq!(
            framed.len(),
            clean.len(),
            "a framed page came back as {} line(s) where the same page unframed \
             gave {} — the page border is fusing the profile",
            framed.len(),
            clean.len()
        );
    }

    /// Degenerate shapes reach this from real callers: a 1px crop, and a mask
    /// whose length disagrees with its stated dimensions. Indexing is what would
    /// panic, so the guards are worth a test even though they assert nothing but
    /// survival.
    #[test]
    fn degenerate_masks_do_not_panic() {
        assert!(!suppress_page_rules(&mut [], 0, 0));
        assert!(!suppress_page_rules(&mut [], 10, 10));
        assert!(!suppress_page_rules(&mut vec![0u8; 50 * 50], 50, 50));
        assert!(!suppress_page_rules(&mut vec![1u8; 50 * 50], 50, 50));
        assert!(!suppress_page_rules(&mut [1u8; 1], 1, 1));
    }

    #[test]
    fn cut_column_matches_the_shared_fixture() {
        let f = fixture();
        for probe in cases(&f.root, "cut_column_probes") {
            let (img, name) = case_image(&probe);
            let got = cut_column(
                &img,
                u32_field(&probe, "x0"),
                u32_field(&probe, "ideal"),
                img.width(),
            );
            assert_eq!(got, u32_field(&probe, "expected_cut"), "probe '{name}'");
        }
    }

    /// A page of `bands` dense bands plus one faint band carrying exactly
    /// `faint_ink` ink pixels per row, used to probe the threshold LEVEL rather
    /// than the profile the boundaries come from.
    fn page_with_a_faint_band(
        bands: u32,
        gap: u32,
        glyphs: u32,
        faint_ink: u32,
        faint_h: u32,
    ) -> GrayImage {
        let height = T_MARGIN * 2 + T_BAND * bands + gap * bands + faint_h;
        let mut img = GrayImage::from_pixel(T_WIDTH, height, Luma([255u8]));
        let mut y = T_MARGIN;
        for _ in 0..bands {
            for yy in y..y + T_BAND {
                for k in 0..glyphs {
                    let x0 = 100 + k * T_PITCH;
                    for i in 0..T_GLYPH_W {
                        if x0 + i < T_WIDTH {
                            img.put_pixel(x0 + i, yy, Luma([0u8]));
                        }
                    }
                }
            }
            y += T_BAND + gap;
        }
        for yy in y..y + faint_h {
            for i in 0..faint_ink {
                img.put_pixel(100 + i, yy, Luma([0u8]));
            }
        }
        img
    }

    /// THE CASE THE DUAL HISTOGRAM EXISTS FOR, measured at this port's own
    /// parameters rather than borrowed from another port.
    ///
    /// With the default `smooth_window` of 3 the smoother averages three rows,
    /// so a gap of 1px or 2px never reaches zero in the smoothed profile — the
    /// ink either side bleeds into it and clears the threshold. Reading
    /// boundaries there returned 1 band against 29 drawn. 3px is the first gap
    /// the smoothed profile survives, which is why it is the control here and
    /// not the interesting case.
    #[test]
    fn lines_two_pixels_apart_are_not_fused() {
        let seg = LineSegmenter::new(10, 3);
        for gap in [1u32, 2] {
            let got = seg.segment_image(&drawn_page(29, gap, 30, false)).unwrap();
            assert_eq!(
                got.len(),
                29,
                "29 bands {gap}px apart came back as {} — boundaries are being \
                 read off the smoothed profile again",
                got.len()
            );
        }
        let control = seg.segment_image(&drawn_page(29, 3, 30, false)).unwrap();
        assert_eq!(
            control.len(),
            29,
            "the 3px control failed, so the regression is not the profile choice"
        );
    }

    /// The opposite failure, and the reason this needs its own test: the raw
    /// profile is the more sensitive of the two, so the risk of reading it is
    /// splitting where no gap exists. Bands that touch share ink on every row,
    /// there is no clean row anywhere, and one band is the honest answer.
    #[test]
    fn touching_bands_stay_one_line() {
        let seg = LineSegmenter::new(10, 3);
        let got = seg.segment_image(&drawn_page(29, 0, 30, false)).unwrap();
        assert_eq!(got.len(), 1, "touching bands were split into {}", got.len());
    }

    /// `smooth_window` is a constructor argument, and on the smoothed profile
    /// raising it widened the damage: the break point is the smoother's full
    /// width, so at 15 every page whose lines sat closer than 15px collapsed to
    /// one band. Measured at 5px and 12px, both of which the old form lost.
    #[test]
    fn a_wide_smoother_does_not_fuse_the_page() {
        let seg = LineSegmenter::new(10, 15);
        for gap in [5u32, 12] {
            let got = seg.segment_image(&drawn_page(29, gap, 30, false)).unwrap();
            assert_eq!(
                got.len(),
                29,
                "at smooth_window 15, 29 bands {gap}px apart came back as {}",
                got.len()
            );
        }
    }

    /// The other half of the dual histogram: the LEVEL still comes off the
    /// smoothed profile.
    ///
    /// Calibrating on the raw profile instead raises the threshold, because
    /// smoothing spreads ink into the rows either side of every band and those
    /// partial rows pull the non-zero mean down. A band faint enough to sit
    /// between the two thresholds is then dropped, and dropping a line is the
    /// failure this pipeline is built to avoid.
    ///
    /// The fixture is tuned, and the tuning is the finding: at the default ratio
    /// of 0.05 the two thresholds sit 0.86px apart on this page, and no whole
    /// number of ink pixels lands between them, so no test at the default can
    /// tell the two calibrations apart. At 0.5 — the ratio the reference
    /// recommends for wide-spaced layouts, and a constructor argument rather
    /// than a default — they are 165.6 (smoothed) and 174.4 (raw). Measured: a
    /// faint band of 166 to 174 ink pixels per row is found by the smoothed
    /// calibration and missed by the raw one. 170 is the middle of that window.
    #[test]
    fn the_gap_threshold_is_calibrated_on_the_smoothed_profile() {
        let seg = LineSegmenter::with_density_ratio(10, 3, 0.5);
        let img = page_with_a_faint_band(8, 12, 30, 170, 20);
        let got = seg.segment_image(&img).unwrap();
        assert_eq!(
            got.len(),
            9,
            "expected 8 dense bands plus the faint one, got {} — the threshold \
             is being calibrated on the raw profile",
            got.len()
        );
    }
}
