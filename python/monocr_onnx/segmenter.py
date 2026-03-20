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
