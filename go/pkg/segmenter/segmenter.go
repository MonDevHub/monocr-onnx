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
	for y := 0; y < height; y++ {
		sum := 0
		for x := 0; x < width; x++ {
			c := img.At(bounds.Min.X+x, bounds.Min.Y+y)
			gray := color.GrayModel.Convert(c).(color.Gray)
			if gray.Y < 128 {
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
