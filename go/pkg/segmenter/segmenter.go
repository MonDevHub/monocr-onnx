package segmenter

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"sort"
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
//     asked -- and reads exactly the same rows as the odd window ABOVE it. Only the
//     rows, not the values: see divergence 3, which divides that same sum by a
//     different number. In JS and Rust the two windows ARE value-identical, because
//     they divide by what they summed. Here, on an 8-band page carrying a faint band
//     of 18 ink pixels per row, SmoothWindow 2 returns 8 bands and SmoothWindow 3
//     returns 9.
//
//     Python convolves a true window-tap kernel and spans exactly what it was
//     given. Measured on 29 drawn glyph-blob bands at MinLineH 10, driving the
//     pre-fix form that read boundaries off this profile: the first gap that
//     returned all 29 bands, for windows 1 to 12, was 1,3,3,5,5,7,7,9,9,11,11,13.
//     Python's was 1,2,...,12. So at SmoothWindow 4 a gap of exactly 4px still
//     fused. JS and Rust measure the same table as this.
//
//  2. THE DIVISOR IS THE REQUESTED WINDOW, NOT THE ROWS SUMMED. This is edge
//     handling, and at ODD windows it is numpy's mode='same' formula exactly --
//     zero-padded ends, divided by the window. Mathematically exactly, not to the
//     bit: numpy multiplies each tap by 1/window and sums, rounding window times
//     where this rounds once, which measured about 1e-13 of drift on an
//     integer-valued profile. At EVEN windows it matches neither numpy nor the
//     siblings, for the reason in divergence 3.
//
//     Row 0 at window 3 comes back at 2/3 of the true mean of the two rows in
//     range, and at window 15 at 8/15 of it. JS and Rust divide by the rows they
//     visited and report the true local mean instead. Measured cost, now that this
//     profile only sets the threshold LEVEL: the two formulas disagree only on rows
//     0..overflow-1 and their mirror at the bottom, and the windows of those rows
//     together cover rows 0..2*overflow-1, so the blank margin that hides the
//     divergence is 2*(window/2) rows and NOT window/2. Measured on an 8-band page:
//     a 1-row margin still left window 3 disagreeing (17.1429 here against JS's
//     17.1607) and a 2-row margin made them agree, and window 15 needed 14. Every
//     fixture in this repo uses a 30px margin, so all of them sit on the agreeing
//     side. On a page cropped flush to the ink the threshold sat 0.21% below JS's at
//     window 3 (17.2096 against 17.2455) and 1.17% below at window 15, and no band
//     count changed in any case measured.
//
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

// Two runs separated by at most this many rows are one text line, provided the raw
// profile never reaches zero inside the gap OR one of them is a fragment.
//
// WHY THIS EXISTS. Detecting boundaries on the raw profile splits a single line
// wherever one row dips below the gap threshold, and in Mon that happens between the
// upper diacritic zone and the consonant bodies. The strip of glyph tops then decodes
// to digits, because a row of circle-tops IS digits, and the decapitated body decodes
// missing its asats, because the asat went with the strip. See mon_OCR
// docs/AUDIT-2026-08-B.md F-69, which measured that with a model.
//
// MEASURED HERE, at this binding's own threshold: page 9 of a 56-page Mon book
// rendered at 300 DPI, gapThreshold 6.8 ink pixels per row (0.05 of the smoothed
// profile's non-zero mean), one text line spanning rows 357-422, and ROW 377 CARRYING
// 5 INK PIXELS -- one row wide, 5 against 6.8. The line came back as a 20-row strip
// and a 44-row body, and that page returned 38 runs where the merge leaves 23.
//
// A 1-row gap holding ink is not a line boundary at any resolution. This is the
// reference's rule (mon_OCR _MIN_GAP_MERGE, segmenter.py step 8), ported with its
// value, and it is the half of the dual histogram this binding left behind: raw
// detection needs a merge to be safe, and the raw-only change shipped without it.
//
// THIS BINDING'S OWN MEASUREMENT is in mergeRuns below. Do not substitute the Python
// binding's or the reference's: Python calibrates on the profile MAX at ratio 0.02
// where this takes 0.05 of the non-zero MEAN, and its default smooth window is 5
// against this one's 3, so its threshold on this same page is 13.7 rather than 6.8.
//
// The figures below were measured through THIS binding and then found to equal the JS
// binding's to the band. That is a result, not a transplant: JS shares the parameter
// set but divides the smoother by the rows it visited rather than by the window, and
// per smoothProfile's own header the two formulas disagree only on rows within
// window/2 of a page edge. These renders carry blank margins, so every disagreeing row
// is zero and both bindings compute a threshold of 6.8163 on page 9. A page cropped
// flush to its ink would separate them.
const minGapMerge = 10

// mergeRuns fuses runs that a single sub-threshold row split apart.
//
// Merges runs[i] into runs[i-1] when the gap between them is at most maxGap rows AND
// (every row in the gap carries ink OR one of the two is a fragment of a line), AND
// the merged band stays within twice a typical line. See minGapMerge for why.
//
// A free function taking the profile rather than a method, so the arithmetic is
// testable without a page, a mask or a model. It writes into a fresh slice and leaves
// both arguments alone.
//
// MEASURED THROUGH THIS BINDING at its own parameters (MinLineH 10, SmoothWindow 3,
// ratio 0.05 of the non-zero mean), over the 56 pages of a real Mon book rendered at
// 300 DPI:
//
//	no merge    2132 bands   576 under 0.6x the page median (27.0%)
//	this merge  1893 bands   288 under 0.6x the page median (15.2%)
//
// The sub-0.6x share is the fragment proxy, and not a metric invented here: F-69 read
// a model over 4,251 bands, and of the 642 landing in [0.4, 0.6) of the page median,
// 94.4% decoded to majority digits. (95.1% is that bucket's mean digit share -- a
// different column of the same table.) Each arm is scored against its OWN page
// median above, and that could have flattered the merge, because merging raises the
// median. It does not: scored against the unmerged arm's medians as a fixed yardstick
// the merged count is 272 (14.4%).
//
// Two things this does NOT claim. It does not remove every suspect band -- 285 of
// F-69's 990 sub-0.6x bands were page numbers and watermarks, read correctly, which is
// why the merge is not a thin-band filter. And the band count is not monotone: 6 of
// the 56 pages come back with MORE bands, because a merge lifts a pair of fragments
// that were each below MinLineH over the filter. That is content recovered, and it is
// why the merge runs before the height filter.
func mergeRuns(runs [][2]int, hist []int, maxGap int) [][2]int {
	if len(runs) == 0 {
		return nil
	}

	// The page's own typical line height, from the runs as detected. Both tests below
	// are relative to this rather than to the neighbouring run, and that is a
	// correction rather than a preference: judging a fragment against its neighbour
	// CASCADES. The merge mutates the accumulated run, so every merge makes it taller,
	// and a taller run makes the next line look more like a fragment. Measured upstream
	// on page 47 of a 56-page book: 36 bands collapsed to 10, with single bands of 534,
	// 632 and 732 rows holding a dozen text lines each, and the page lost 92% of its
	// readable characters.
	//
	// Measured HERE, and it holds up in this binding rather than only upstream: judging
	// a fragment against the accumulated neighbour instead, ceiling and all, costs both
	// metrics over the 56-page corpus -- 1921 bands and 17.4% sub-0.6x against this
	// form's 1893 and 15.2%. The Python binding measures the two forms as near-equal,
	// so this is not a shared result; the numbers here are this binding's own.
	heights := make([]int, len(runs))
	for i, r := range runs {
		heights[i] = r[1] - r[0]
	}
	sort.Ints(heights)
	typical := heights[len(heights)/2]
	if typical < 1 {
		typical = 1
	}

	// No merge may produce a band more than twice a typical line. This is the backstop
	// for the cascade above: the fragment test alone cannot bound the result, and one
	// runaway band costs a whole page. Twice rather than tighter because a legitimate
	// merge of two halves lands at about one typical line and must not be refused.
	//
	// Measured here: over the 56-page corpus, dropping it takes 1893 bands down to
	// 1670 -- 223 bands, 12%, swallowed into chains of merges. The sub-0.6x share even
	// IMPROVES to 12.8% while that happens, which is the reason a fragment-share metric
	// cannot be the only one watched.
	ceiling := typical * 2

	merged := make([][2]int, 0, len(runs))
	for _, r := range runs {
		if n := len(merged); n > 0 {
			last := &merged[n-1]
			gapStart := last[1]
			gapSize := r[0] - gapStart
			if gapSize < 0 {
				// An empty gap cannot occur from the run collector, but a caller can
				// hand us touching or overlapping runs; treat those as already one line.
				gapSize = 0
			}

			// A row outside the profile is not ink, so an out-of-range gap never merges
			// on this clause.
			gapHasInk := true
			for y := gapStart; y < r[0]; y++ {
				if y < 0 || y >= len(hist) || hist[y] <= 0 {
					gapHasInk = false
					break
				}
			}

			// A run at most half a typical line is a fragment of a line, not a line.
			// This is the clause that crosses a gap of genuinely ZERO ink, which
			// gapHasInk refuses and which a floating Mon diacritic produces: measured,
			// rows 341-360 and 362-404 are the upper marks and the body of one line
			// separated by two empty rows. Two REAL lines two rows apart are each a full
			// line by this test, so they stay apart -- which is what a vertical smear
			// could not do, because at reach 1 it closes 2-row gaps and 2 rows is the
			// tightest real line spacing.
			fragment := 2*(r[1]-r[0]) <= typical || 2*(last[1]-last[0]) <= typical

			if gapSize <= maxGap && (gapHasInk || fragment) && r[1]-last[0] <= ceiling {
				last[1] = r[1]
				continue
			}
		}
		merged = append(merged, r)
	}
	return merged
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
	runs := make([][2]int, 0)
	// -1 rather than a *int: the old form wrote `s := y` inside the loop, which
	// shadowed the receiver `s` in one branch while the next branch read `s.MinLineH`
	// off the receiver.
	start := -1

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

		if isText && start < 0 {
			start = y
		} else if !isText && start >= 0 {
			// Collected, not extracted: the merge below needs every run on the page
			// before it can measure the page's typical line height.
			runs = append(runs, [2]int{start, y})
			start = -1
		}
	}

	if start >= 0 {
		runs = append(runs, [2]int{start, height})
	}

	// 5. Fuse runs a single sub-threshold row split apart, BEFORE the height filter.
	// The order is the reference's and it matters: a diacritic strip can be shorter than
	// MinLineH, and filtering first would discard the strip and leave the decapitated
	// body behind as a whole line.
	for _, r := range mergeRuns(runs, hist, minGapMerge) {
		if r[1]-r[0] >= s.MinLineH {
			s.extractLine(img, bounds, mask, r[0], r[1], &results)
		}
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
