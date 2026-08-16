import cv2
import numpy as np
from PIL import Image

class LineSegmenter:
    """
    Robust line segmenter using Horizontal Projection Profiles with Smoothing.
    Handles noisy documents and touching lines by finding valleys in the projection.
    """
    def __init__(self, min_line_h=10, smooth_window=5, threshold_ratio=0.02):
        self.min_line_h = min_line_h
        self.smooth_window = smooth_window
        self.threshold_ratio = threshold_ratio

    def segment(self, image):
        """
        Segment a document image into text lines.
        Args:
            image (PIL.Image or np.ndarray): Input image.
        Returns:
            list: List of dicts with keys 'img' (PIL.Image) and 'bbox' (x, y, w, h).
        """
        # Convert to CV2 grayscale
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        h_img, width = gray.shape

        # 1. Binarize (Adaptive Thresholding)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
        
        # 2. Horizontal Projection Profile
        raw_hist = np.sum(binary, axis=1).astype(np.float32)

        # 3. Smoothing
        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            smoothed_hist = np.convolve(raw_hist, kernel, mode='same')
        else:
            smoothed_hist = raw_hist

        # 4. Gap Detection
        non_zero_vals = smoothed_hist[smoothed_hist > 0]
        if len(non_zero_vals) == 0:
            return []

        # Find threshold based on ratio or density
        max_val = np.max(smoothed_hist)
        threshold = max_val * self.threshold_ratio
        
        results = []
        start = None
        
        for y in range(h_img):
            is_text_val = smoothed_hist[y] > threshold
            
            if is_text_val and start is None:
                start = y
            elif not is_text_val and start is not None:
                # End of a text block
                end = y
                if (end - start) >= self.min_line_h:
                    self._extract_line(binary, gray, start, end, image, results)
                start = None
                
        if start is not None:
            if (h_img - int(start)) >= self.min_line_h:
                self._extract_line(binary, gray, start, h_img, image, results)
            
        return results

    def _extract_line(self, binary, gray, r_start, r_end, source_image, results):
        """Crop a detected line region and append to results list."""
        # Find horizontal bounds within strip
        line_slice = binary[r_start:r_end, :]
        
        col_sums = np.sum(line_slice, axis=0)
        col_indices = np.where(col_sums > 0)[0]
        
        if len(col_indices) == 0:
            return
            
        x_start, x_end = col_indices[0], col_indices[-1]
        
        # Add relative padding based on line height
        h_raw = r_end - r_start
        pad_y = int(h_raw * 0.20)
        pad_x = int(h_raw * 0.15)
        y1 = max(0, r_start - pad_y)
        y2 = min(gray.shape[0], r_end + pad_y)
        x1 = max(0, x_start - pad_x)
        x2 = min(gray.shape[1], x_end + pad_x)
        
        w = x2 - x1
        h = y2 - y1
        
        # If input was not PIL, ensure we return something consistent
        if not isinstance(source_image, Image.Image):
            source_image = Image.fromarray(source_image)
            
        crop = source_image.crop((x1, y1, x2, y2))
        
        results.append({
            'img': crop,
            'bbox': (int(x1), int(y1), int(w), int(h))
        })


# Where a tile may be cut, as a fraction of the tile width, searching backwards
# from the ideal boundary. 0.12 of a 1024px window is ~123px, roughly two Mon
# glyphs — wide enough to find a gap, narrow enough that tiles stay near full
# width.
_CUT_SEARCH_FRACTION = 0.12

# A column counts as carrying ink below this grayscale value.
_CUT_INK_THRESHOLD = 250


def cut_column(crop, x0, ideal, crop_w):
    """Where to end a tile that starts at ``x0`` and may not pass ``ideal``.

    Cutting at exactly ``ideal`` lands wherever the arithmetic falls, which is
    usually the middle of a glyph. Both halves keep their pixels, so a coverage
    check still passes, but the model reads each half as a whole character and
    one glyph becomes two. Measured upstream on 120 drawn lines this showed up
    as ``ဗော်`` read back as ``ဗေဗိာ်``.

    So search backwards from ``ideal`` for a column of white. A tile may only
    get narrower, never wider, or it stops fitting the model window. Returns
    ``ideal`` unchanged when there is no gap to cut at, which is the honest
    outcome for a continuous script: a known-bad seam beats an overflowing tile.

    Ported from mon_OCR ``segmenter._cut_column``; the constants are the same,
    so the two produce the same cuts on the same input.
    """
    if ideal >= crop_w:
        return crop_w

    window = max(1, int((ideal - x0) * _CUT_SEARCH_FRACTION))
    lo = max(x0 + 1, ideal - window)
    if lo >= ideal:
        return ideal

    band = crop.crop((lo, 0, ideal, crop.height))
    if band.mode != "L":
        # The column sum below is silently wrong on a 3-channel array: it would
        # produce a (W, 3) profile and an argmin over a flattened index.
        band = band.convert("L")
    ink = (np.asarray(band, dtype=np.uint8) < _CUT_INK_THRESHOLD).sum(axis=0)

    # Prefer a truly empty column, and the rightmost one, so tiles stay as wide
    # as the window allows. Fall back to the lightest column present.
    blank = np.flatnonzero(ink == 0)
    offset = int(blank[-1]) if blank.size else int(np.argmin(ink))
    return lo + offset


def tile_line(crop, target_h, target_w):
    """Split one line crop into pieces that each fit the model window.

    Returns ``[crop]`` unchanged when the line already fits after being scaled
    to ``target_h``. Otherwise cuts at whitespace columns and returns the pieces
    left to right, to be read separately and joined with no separator.
    """
    crop_w, crop_h = crop.size
    if crop_h <= 0 or crop_w <= 0:
        return [crop]

    scale = target_h / crop_h
    if int(crop_w * scale) <= target_w:
        return [crop]

    tile_w_src = max(1, int(target_w / scale))
    tiles, x0 = [], 0
    while x0 < crop_w:
        ideal = min(x0 + tile_w_src, crop_w)
        x1 = cut_column(crop, x0, ideal, crop_w)
        # Structural guard, not a tuning knob: cut_column can only return a
        # value in (x0, ideal], but if it ever returned x0 this loop would spin
        # forever on a page. One pixel of forced progress bounds it.
        x1 = max(x1, x0 + 1)
        tiles.append(crop.crop((x0, 0, x1, crop_h)))
        x0 = x1
    return tiles
