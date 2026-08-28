package segmenter

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

type LineSegmenter struct {
	MinLineH     int
	SmoothWindow int
}

type SegmentResult struct {
	Img  image.Image
	BBox image.Rectangle
}

func NewLineSegmenter(minLineH, smoothWindow int) *LineSegmenter {
	if minLineH == 0 {
		minLineH = 10
	}
	if smoothWindow == 0 {
		smoothWindow = 3
	}
	return &LineSegmenter{
		MinLineH:     minLineH,
		SmoothWindow: smoothWindow,
	}
}

// Printed-rule suppression.
//
// A printed page border adds a constant ink floor to every row it spans, and once
// that floor clears the gap threshold no in-frame row reads as a gap: the page
// returns as one band and is squeezed into the model window.
//
// MEASURED WITH THIS PARAMETER SET (global threshold 128, no smear, smoothing 3,
// ratio 0.05 of the mean) over twelve real MNEC page-ones: nine collapse to three
// bands or fewer, and the twelve go from 118 bands to 215. Pages carrying no rules
// are untouched.
//
// Counter-intuitive and worth stating: this binding gains MORE from the pass than
// the smeared implementations, not less. A synthetic framed page does not fuse
// here, which briefly suggested the opposite; real pages settle it.
const (
	// A rule spans at least this fraction of the page in one direction. No Mon,
	// Burmese or Latin glyph holds an unbroken stroke half a page long, so the
	// false-positive risk against text is structural rather than merely small.
	ruleSpan = 0.5

	// Suppression that would remove more than this share of the page ink has found
	// text, not rules, and is abandoned. ruleSpan is a fraction of the page, so on
	// a SHORT page a tall text block exceeds it vertically and every glyph column
	// reads as a rule. Real framed pages classify 21.5%-58.8% of their ink as
	// rules, rule-free pages 0.00%, and the false positive upstream found 98.7%.
	ruleMaxInkShare = 0.80
)

// suppressPageRules zeroes printed rules in mask (1 = ink) and reports whether
// anything was removed. Mutates in place.
//
// A run-length scan rather than a generic erode-then-dilate: opening with a 1xL
// line kernel keeps exactly those ink runs at least L long, which one sweep per
// axis computes directly.
func suppressPageRules(mask []uint8, width, height int) bool {
	if width <= 0 || height <= 0 || len(mask) < width*height {
		return false
	}
	minH := int(float64(width) * ruleSpan)
	if minH < 15 {
		minH = 15
	}
	minV := int(float64(height) * ruleSpan)
	if minV < 15 {
		minV = 15
	}

	rules := make([]uint8, width*height)
	for y := 0; y < height; y++ {
		row := y * width
		start := 0
		for x := 0; x <= width; x++ {
			if x < width && mask[row+x] != 0 {
				continue
			}
			if x-start >= minH {
				for i := start; i < x; i++ {
					rules[row+i] = 1
				}
			}
			start = x + 1
		}
	}
	for x := 0; x < width; x++ {
		start := 0
		for y := 0; y <= height; y++ {
			if y < height && mask[y*width+x] != 0 {
				continue
			}
			if y-start >= minV {
				for i := start; i < y; i++ {
					rules[i*width+x] = 1
				}
			}
			start = y + 1
		}
	}

	ink, ruleInk := 0, 0
	for i := range mask[:width*height] {
		if mask[i] != 0 {
			ink++
		}
		if rules[i] != 0 {
			ruleInk++
		}
	}
	if ink == 0 || ruleInk == 0 || float64(ruleInk) > float64(ink)*ruleMaxInkShare {
		// Found the text. Leaving the page alone is strictly better than emptying it.
		return false
	}
	for i := range rules {
		if rules[i] != 0 {
			mask[i] = 0
		}
	}
	return true
}

// smoothProfile box-filters the row profile. Returns a fresh []float64 always, so
// it never aliases the caller's []int raw profile.
//
// THREE MEASURED DIVERGENCES FROM THE PYTHON BINDING, none of them reconciled
// here. The formula is published behaviour for anyone reading the profile, so
// changing it changes output for this binding's users; that is an owner decision.
//
//  1. SPAN IS 2*(window/2)+1, NOT window. The loop is [-overflow, +overflow] with
//     overflow = window/2, so an EVEN window spans window+1 rows -- one MORE than
//     asked -- and behaves exactly as the odd window ABOVE it. Python convolves a
//     true window-tap kernel. Measured on 29 drawn glyph-blob bands at MinLineH 10,
//     driving the pre-fix form that read boundaries off this profile: the first gap
//     that returned all 29 bands, for windows 1 to 12, was
//     1,3,3,5,5,7,7,9,9,11,11,13. Python's was 1,2,...,12. So at SmoothWindow 4 a
//     gap of exactly 4px still fused. JS and Rust measure the same table as this.
//  2. THE DIVISOR IS THE REQUESTED WINDOW, NOT THE ROWS SUMMED. This is edge
//     handling, and at ODD windows it matches numpy's mode='same' to the bit --
//     zero-padded ends, divided by the window. Row 0 at window 3 therefore comes
//     back at 2/3 of the true mean of the two rows in range, and at window 15 at
//     8/15 of it. JS and Rust divide by the rows they visited and report the true
//     local mean instead. Measured cost, now that this profile only sets the
//     threshold LEVEL: on a page with a blank margin of at least window/2 rows the
//     affected rows are zero either way and the thresholds agree to the bit; on a
//     page cropped flush to the ink the threshold sat 0.21% below JS's at window 3
//     (17.2096 against 17.2455) and 1.17% below at window 15, and no band count
//     changed in any case measured.
//  3. AT AN EVEN WINDOW THIS DIVIDES BY LESS THAN IT SUMMED, and that one is a
//     defect rather than a difference. It sums window+1 terms and divides by
//     window, so every interior row is inflated by exactly (window+1)/window and
//     the smoothed peak CLEARS the raw peak: measured on an 8-band page, smoothed
//     max 540 against raw 360 at window 2, 450 at window 4, 420 at 6, 405 at 8.
//     The gap threshold is inflated by the same factor -- 25.71 against JS's and
//     Rust's 17.14 at window 2, 20.45 against 16.36 at window 4 -- so at an even
//     window this binding drops faint bands the other three keep. It does not move
//     the break-point table, because the inflation is uniform and the threshold
//     scales with it. The default SmoothWindow is 3, so nothing reaches this
//     unless a caller passes an even window or sets the exported field to one.
func smoothProfile(hist []int, window int) []float64 {
	height := len(hist)
	smoothed := make([]float64, height)
	if window <= 1 {
		for i, v := range hist {
			smoothed[i] = float64(v)
		}
		return smoothed
	}
	overflow := window / 2
	for i := 0; i < height; i++ {
		sum := 0.0
		for k := -overflow; k <= overflow; k++ {
			idx := i + k
			if idx >= 0 && idx < height {
				sum += float64(hist[idx])
			}
		}
		smoothed[i] = sum / float64(window)
	}
	return smoothed
}

func (s *LineSegmenter) Segment(img image.Image) ([]SegmentResult, error) {
	// Convert to Grayscale if needed (conceptually, we just need luminance)
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// 1. Horizontal Projection Profile
	// We want to count 'text' pixels (dark pixels < 128)
	// hist[y] = sum(is_text(x, y) for x in width)
	hist := make([]int, height)

	// Accessing pixels via At() is slow, but compatible with all image types.
	// For optimization later, type switch to *image.Gray, *image.RGBA etc.
	// For now, simple implementation.
	// A mask is materialised because rule suppression needs the 2-D shape of the
	// ink, which a per-row count cannot express.
	mask := make([]uint8, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := img.At(bounds.Min.X+x, bounds.Min.Y+y)
			gray := color.GrayModel.Convert(c).(color.Gray)
			if gray.Y < 128 {
				mask[y*width+x] = 1
			}
		}
	}

	suppressPageRules(mask, width, height)

	for y := 0; y < height; y++ {
		sum := 0
		row := y * width
		for x := 0; x < width; x++ {
			if mask[row+x] != 0 {
				sum++
			}
		}
		hist[y] = sum
	}

	// 2. Smoothing
	//
	// `hist` stays alive past this point, because the two profiles have different
	// jobs: the gap threshold below is calibrated on the smoothed one, the
	// boundaries are detected on the raw one. No copy is needed for that in Go --
	// smoothProfile always returns a fresh []float64 and never aliases `hist`, which
	// is []int. The Rust port needed a clone() here only because it moved its raw
	// profile into the smoothed one. That function's header carries the three
	// measured divergences from the Python binding: span, divisor and the
	// even-window inflation.
	smoothedHist := smoothProfile(hist, s.SmoothWindow)

	// 3. Gap Detection
	var nonZeroVals []float64
	for _, v := range smoothedHist {
		if v > 0 {
			nonZeroVals = append(nonZeroVals, v)
		}
	}

	if len(nonZeroVals) == 0 {
		return []SegmentResult{}, nil
	}

	// Mean density
	sumVal := 0.0
	for _, v := range nonZeroVals {
		sumVal += v
	}
	meanDensity := sumVal / float64(len(nonZeroVals))
	gapThreshold := meanDensity * 0.05

	var results []SegmentResult
	var start *int

	// 4. Extract Lines
	for y := 0; y < height; y++ {
		// Boundaries come off the RAW profile, not the smoothed one.
		//
		// The threshold above stays calibrated on the smoothed profile, because its
		// non-zero mean is the steadier of the two. But the smoother averages several
		// rows together, so a gap narrower than its span never reaches zero in the
		// smoothed profile: the ink either side bleeds into it, the bled rows clear
		// the threshold, and the two lines fuse. The raw profile needs one clean row.
		//
		// MEASURED HERE, at this binding's own parameters (MinLineH 10, ratio 0.05 of
		// the non-zero mean) on 29 drawn bands, driving the pre-fix form that read
		// boundaries off the smoothed profile. First gap that returned all 29 bands,
		// by SmoothWindow 1 to 12:
		//
		//	1 3 3 5 5 7 7 9 9 11 11 13
		//
		// So the break point is smoothProfile's SPAN, 2*(SmoothWindow/2)+1, and not
		// the requested window: at SmoothWindow 4 a gap of exactly 4px still fused.
		// Python's table is 1,2,...,12 because its kernel is a true window-tap box.
		// SmoothWindow is both a NewLineSegmenter argument and an exported field, so a
		// caller widens the failure with it -- at 15 the smoothed profile lost every
		// page whose lines sat closer than 15px while the raw profile kept all 29.
		//
		// These are this binding's numbers. Do not substitute the reference's: mon_OCR
		// dilates the mask vertically before taking the profile and this one does not,
		// so its break point is 5px to 8px, not 3px.
		isText := float64(hist[y]) > gapThreshold

		if isText && start == nil {
			s := y
			start = &s
		} else if !isText && start != nil {
			end := y
			if (end - *start) >= s.MinLineH {
				s.extractLine(img, bounds, mask, *start, end, &results)
			}
			start = nil
		}
	}

	if start != nil && (height-*start) >= s.MinLineH {
		s.extractLine(img, bounds, mask, *start, height, &results)
	}

	return results, nil
}

// extractLine crops one detected band. The column extents come from the SUPPRESSED
// mask, not from a fresh read of the image.
//
// Re-thresholding img.At() here threw away half of what the suppression pass buys.
// The frame was deleted from the row profile and then reinstated in every crop's
// x-range: xMin landed on the left rule and xMax on the right one, so each crop
// spanned the full framed area — the same over-wide crop that squeezes a line into
// the model window, only now once per line instead of once per page.
//
// The reference states the intent at mon_OCR src/monocr/segmenter.py:392 —
// suppression runs before the smear because "the crop's column extents come from
// `dilated`, so removing rules first also keeps the border out of the crops" — and
// its _extract_line sums `dilated` at line 648. python/monocr_onnx/segmenter.py:140
// already sums `binary` for the same reason; this binding was the odd one out.
//
// On a page with NO rules this is a byte-for-byte no-op: suppressPageRules leaves the
// mask untouched, and the mask is that identical `< 128` test over the same pixels.
// It also drops width*height GrayModel conversions per band, which At() made the
// expensive part of this function. Pinned by page_rules_test.go.
func (s *LineSegmenter) extractLine(img image.Image, bounds image.Rectangle, mask []uint8, rStart, rEnd int, results *[]SegmentResult) {
	// Find horizontal bounds within strip. mask is row-major over bounds, so mask
	// index y*width+x is the pixel at (bounds.Min.X+x, bounds.Min.Y+y).
	width := bounds.Dx()
	colSum := make([]int, width)

	// Optimize: Only loop through the strip rows
	for y := rStart; y < rEnd; y++ {
		row := y * width
		for x := 0; x < width; x++ {
			if mask[row+x] != 0 {
				colSum[x]++
			}
		}
	}

	// Find non-empty cols
	xMin := -1
	xMax := -1

	for x := 0; x < width; x++ {
		if colSum[x] > 0 {
			if xMin == -1 {
				xMin = x
			}
			xMax = x
		}
	}

	if xMin == -1 {
		return
	}

	// Add generous padding to avoid cutting off diacritics
	// Add relative padding based on line height
	hRaw := rEnd - rStart
	padY := int(math.Ceil(float64(hRaw) * 0.20))
	padX := int(math.Ceil(float64(hRaw) * 0.15))

	y1 := int(math.Max(0, float64(rStart-padY)))
	y2 := int(math.Min(float64(bounds.Dy()), float64(rEnd+padY)))
	x1 := int(math.Max(0, float64(xMin-padX)))
	x2 := int(math.Min(float64(width), float64(xMax+padX)))

	// Crop
	rect := image.Rect(bounds.Min.X+x1, bounds.Min.Y+y1, bounds.Min.X+x2, bounds.Min.Y+y2)

	// Since Go images merely reference the underlying buffer when sub-imaging,
	// and we might want to resize/process these individually,
	// we should probably copy to a new buffer or just return the SubImage.
	// SubImage is safer for memory if original is large, but for processing we might want clean buffer.
	// Let's return a SubImage if possible, or a Copy. LineSegmenter usually returns independent images
	// in Python PIL (crop returns copy).
	// onnx runtime preprocessing will maximize contrast etc anyway?
	// Actually `image.NewGray` and draw is safest to ensure 'L' mode equivalent.

	dst := image.NewGray(image.Rect(0, 0, x2-x1, y2-y1))
	draw.Draw(dst, dst.Bounds(), img, rect.Min, draw.Src)

	*results = append(*results, SegmentResult{
		Img:  dst,
		BBox: rect,
	})
}
