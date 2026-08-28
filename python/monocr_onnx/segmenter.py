import cv2
import numpy as np
from PIL import Image

# Printed-rule suppression. A page border adds a constant ink floor to every row
# it spans, and once that floor clears the gap threshold no in-frame row reads as
# a gap: the page comes back as one band and is squeezed into the model window.
#
# Measured 2026-08-27 on the reference implementation over twelve real MNEC
# papers: nine collapsed to a single band without this, seven of them returning
# 0-2 characters, and the twelve together went from 3,846 characters to 5,924.
# Pages carrying no rules come back byte-identical.
#
# A rule is an unbroken ink run spanning at least RULE_SPAN of the page in one
# direction. Morphological opening with a line kernel keeps exactly those runs;
# subtracting them leaves the text. No Mon, Burmese or Latin glyph holds an
# unbroken stroke half a page long, so the false-positive risk against text is
# structural rather than merely small.
#
# There is deliberately NO thickness test. "A rule is long AND thin" was written,
# measured and deleted upstream: across twelve real pages the rule pixels found
# with a thickness limit and with none were identical to the pixel, and it cannot
# work anyway -- adaptiveThreshold compares against a LOCAL mean, so the interior
# of a thick ink region is not ink and only its edges are. Every thick region
# arrives already presented as a pair of thin bands.
RULE_SPAN = 0.5

# Suppression that would remove more than this share of the page ink has found
# text, not rules, and is abandoned. RULE_SPAN is a fraction of the page, so on a
# SHORT page a tall block of text exceeds it vertically and every glyph column
# reads as a rule. Upstream this was caught by an existing test losing 98.7% of
# its ink and returning zero lines. The threshold sits in a measured gap: real
# framed pages classify 21.5%-58.8% of their ink as rules, rule-free pages 0.00%,
# and that false positive 98.7%.
RULE_MAX_INK_SHARE = 0.80


def suppress_page_rules(binary):
    """Remove printed rules from a text mask, leaving glyphs untouched.

    Returns `binary` unchanged when the page carries no rules, and also when
    suppression would remove more ink than RULE_MAX_INK_SHARE.
    """
    h, w = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * RULE_SPAN)), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * RULE_SPAN)))),
    )
    rules = cv2.bitwise_or(horizontal, vertical)
    ink = int(np.count_nonzero(binary))
    if not ink or np.count_nonzero(rules) > ink * RULE_MAX_INK_SHARE:
        return binary
    return cv2.subtract(binary, rules)


def smooth_profile(raw_hist, window):
    """Box-filter the row profile. Returns `raw_hist` itself for window <= 1.

    A TRUE `window`-tap box, and the one binding of the four that is. numpy's
    kernel has exactly `window` taps whatever the parity, and `mode='same'`
    zero-pads the ends and divides by `window`, so a run of `window` zero rows
    always drives at least one output row to zero.

    MEASURED on 29 drawn glyph-blob bands (min_line_h 10, threshold_ratio 0.02),
    driving the pre-fix form that read boundaries off this profile: the first gap
    that returned all 29 bands was exactly `window`, at every window from 1 to 12
    -- 1,2,3,4,...,12. The three sibling bindings loop [-window//2, +window//2]
    instead, which is 2*(window//2)+1 taps, so their break points run
    1,3,3,5,5,7,7,9,9,11,11,13 -- an even window there spans window+1 rows, one
    MORE than asked, and behaves as the odd window ABOVE it. Their numbers are
    not this one's; see js/src/segmenter.js and
    go/pkg/segmenter/segmenter.go, which record their own.
    """
    if window <= 1:
        return raw_hist
    kernel = np.ones(window) / window
    return np.convolve(raw_hist, kernel, mode='same')


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
        
        # 1.5 Printed-rule suppression
        binary = suppress_page_rules(binary)

        # 2. Horizontal Projection Profile
        raw_hist = np.sum(binary, axis=1).astype(np.float32)

        # 3. Smoothing
        #
        # `raw_hist` stays alive past this point, because the two profiles have
        # different jobs: the threshold in step 4 is calibrated on the smoothed
        # one, the boundaries are detected on the raw one. No copy is needed for
        # that even though `smoothed_hist` IS `raw_hist` when smoothing is off --
        # neither array is written to after this block.
        #
        # A consequence worth knowing rather than papering over: this binding
        # thresholds on the profile MAX, and smoothing does not lower the peak of a
        # band taller than the window, so on an ordinary page max(smoothed) equals
        # max(raw) to the pixel and this step no longer changes any output. It bites
        # only where the tallest row is thinner than the window. The bindings that
        # calibrate on a non-zero MEAN keep a real dependence on it. Not reconciled
        # here: the basis is a tuning constant, and the reference forbids moving it.
        smoothed_hist = smooth_profile(raw_hist, self.smooth_window)

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
            # Boundaries come off the RAW profile, not the smoothed one.
            #
            # The threshold above stays calibrated on the smoothed profile's max.
            # But the smoother averages over `smooth_window` rows, so a gap
            # narrower than the whole window never reaches zero in the smoothed
            # profile: the ink either side bleeds into it, the bled rows clear the
            # threshold, and the two lines fuse. The raw profile needs one clean
            # row.
            #
            # MEASURED HERE, at this binding's own parameters (min_line_h 10,
            # smooth_window 5, threshold_ratio 0.02) on 29 drawn bands: reading the
            # smoothed profile returned 1 band for gaps of 1px, 2px, 3px and 4px
            # against 29 drawn, and matched the raw profile from 5px up. The break
            # point is the smoother's full width at EVERY window from 1 to 12,
            # even ones included, because `smooth_profile` is a true window-tap
            # box -- the sibling bindings round the width down to odd and their
            # tables differ; see that docstring. This binding's exposure is twice
            # the Rust port's, its default window being 5 against Rust's 3.
            # `smooth_window` is a constructor argument, so a caller who raises it
            # widens the failure with it: at 15 the smoothed profile lost every
            # page whose lines sat closer than 15px while the raw profile kept all
            # 29.
            #
            # These are this binding's numbers. Do not substitute the reference's
            # or the Rust port's: their break points differ, because the reference
            # dilates the mask vertically before the profile and this one does not.
            is_text_val = raw_hist[y] > threshold
            
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
