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
	smoothedHist := make([]float64, height)
	if s.SmoothWindow > 1 {
		window := float64(s.SmoothWindow)
		overflow := s.SmoothWindow / 2
		for i := 0; i < height; i++ {
			sum := 0.0
			count := 0.0
			for k := -overflow; k <= overflow; k++ {
				idx := i + k
				if idx >= 0 && idx < height {
					sum += float64(hist[idx])
					count++
				}
			}
			smoothedHist[i] = sum / window // Python code divides by window size (kernel is 1/window)
			// Actually numpy convolution handling at edges: 'same' mode zero-pads?
			// Python: np.convolve(hist, np.ones(window)/window, mode='same')
			// Let's stick to a simple moving average.
		}
	} else {
		for i, v := range hist {
			smoothedHist[i] = float64(v)
		}
	}

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
		isText := smoothedHist[y] > gapThreshold

		if isText && start == nil {
			s := y
			start = &s
		} else if !isText && start != nil {
			end := y
			if (end - *start) >= s.MinLineH {
				s.extractLine(img, bounds, *start, end, &results)
			}
			start = nil
		}
	}

	if start != nil && (height-*start) >= s.MinLineH {
		s.extractLine(img, bounds, *start, height, &results)
	}

	return results, nil
}

func (s *LineSegmenter) extractLine(img image.Image, bounds image.Rectangle, rStart, rEnd int, results *[]SegmentResult) {
	// Find horizontal bounds within strip
	// strip corresponds to y inside [bounds.Min.Y + rStart, bounds.Min.Y + rEnd)
	// We need to sum columns to find x range.

	width := bounds.Dx()
	colSum := make([]int, width)

	// Optimize: Only loop through the strip rows
	for y := rStart; y < rEnd; y++ {
		actualY := bounds.Min.Y + y
		for x := 0; x < width; x++ {
			actualX := bounds.Min.X + x
			c := img.At(actualX, actualY)
			gray := color.GrayModel.Convert(c).(color.Gray)
			if gray.Y < 128 {
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
