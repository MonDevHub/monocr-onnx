package segmenter

// The dual histogram: the gap threshold is calibrated on the SMOOTHED row profile,
// the line boundaries are detected on the RAW one.
//
// Both halves are pinned here, because either one silently reverting costs lines.
// Every number below was measured through THIS binding at ITS parameters (MinLineH
// 10, SmoothWindow 3, ratio 0.05 of the non-zero mean). The reference's numbers do
// not transfer: mon_OCR dilates the mask vertically before taking the profile and
// this binding does not.

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// drawnPage renders bands of glyph-like blobs on white, gap pixels apart.
//
// Blobs rather than solid bars for the same reason page_rules_test.go gives: a solid
// bar the width of a text column IS a printed rule by any definition, and
// suppressPageRules would delete it before the profile ever saw it.
func drawnPage(bands, gap, glyphs int) image.Image {
	height := tmargin*2 + tband*bands + gap*bands
	img := image.NewGray(image.Rect(0, 0, tw, height))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	y := tmargin
	for b := 0; b < bands; b++ {
		for yy := y; yy < y+tband; yy++ {
			for k := 0; k < glyphs; k++ {
				x := 100 + k*tpitch
				if x+tglyphW <= tw {
					for i := 0; i < tglyphW; i++ {
						img.SetGray(x+i, yy, color.Gray{Y: 0})
					}
				}
			}
		}
		y += tband + gap
	}
	return img
}

// TestLinesTwoPixelsApartAreNotFused is THE CASE THE DUAL HISTOGRAM EXISTS FOR.
//
// With the default SmoothWindow of 3 the smoother averages three rows, so a gap of
// 1px or 2px never reaches zero in the smoothed profile — the ink either side bleeds
// into it and clears the threshold. Reading boundaries there returned 1 band against
// 29 drawn, at both gaps. 3px is the first gap the smoothed profile survives, which
// is why it is the control below and not the interesting case.
func TestLinesTwoPixelsApartAreNotFused(t *testing.T) {
	seg := NewLineSegmenter(10, 3)
	for _, gap := range []int{1, 2} {
		got, err := seg.Segment(drawnPage(29, gap, 30))
		if err != nil {
			t.Fatalf("Segment: %v", err)
		}
		if len(got) != 29 {
			t.Fatalf("29 bands %dpx apart came back as %d -- boundaries are being "+
				"read off the smoothed profile again", gap, len(got))
		}
	}
	control, err := seg.Segment(drawnPage(29, 3, 30))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(control) != 29 {
		t.Fatalf("the 3px control returned %d bands, so the regression is not the "+
			"profile choice", len(control))
	}
}

// TestTouchingBandsStayOneLine is the opposite failure, and why it needs its own
// test: the raw profile is the more sensitive of the two, so the risk of reading it
// is splitting where no gap exists. Bands that touch share ink on every row, no row
// is clean anywhere, and one band is the honest answer.
func TestTouchingBandsStayOneLine(t *testing.T) {
	got, err := NewLineSegmenter(10, 3).Segment(drawnPage(29, 0, 30))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("touching bands were split into %d", len(got))
	}
}

// TestAWideSmootherDoesNotFuseThePage: SmoothWindow is both a NewLineSegmenter
// argument and an exported field, so the exposure is caller-settable and not fixed
// at the default's 2px.
//
// On the smoothed profile the break point is the smoother's full width, so raising it
// widened the damage in step: measured here, SmoothWindow 15 returned 1 band for
// every gap from 1px to 14px. 5px and 12px are two the old form lost.
func TestAWideSmootherDoesNotFuseThePage(t *testing.T) {
	seg := NewLineSegmenter(10, 15)
	for _, gap := range []int{5, 12} {
		got, err := seg.Segment(drawnPage(29, gap, 30))
		if err != nil {
			t.Fatalf("Segment: %v", err)
		}
		if len(got) != 29 {
			t.Fatalf("at SmoothWindow 15, 29 bands %dpx apart came back as %d", gap, len(got))
		}
	}
}

// pageWithAFaintBand draws bands dense bands, then one faint band of faintH rows
// carrying exactly faintInk ink pixels per row. It probes the threshold LEVEL, not
// the profile the boundaries come from.
func pageWithAFaintBand(bands, gap, glyphs, faintInk, faintH int) image.Image {
	height := tmargin*2 + tband*bands + gap*(bands+1) + faintH
	img := image.NewGray(image.Rect(0, 0, tw, height))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	y := tmargin
	for b := 0; b < bands; b++ {
		for yy := y; yy < y+tband; yy++ {
			for k := 0; k < glyphs; k++ {
				x := 100 + k*tpitch
				if x+tglyphW <= tw {
					for i := 0; i < tglyphW; i++ {
						img.SetGray(x+i, yy, color.Gray{Y: 0})
					}
				}
			}
		}
		y += tband + gap
	}
	for yy := y; yy < y+faintH; yy++ {
		for i := 0; i < faintInk; i++ {
			img.SetGray(100+i, yy, color.Gray{Y: 0})
		}
	}
	return img
}

// TestTheGapThresholdIsCalibratedOnTheSmoothedProfile pins the other half of the dual
// histogram: the LEVEL still comes off the smoothed profile.
//
// Calibrating on the raw profile instead RAISES the threshold, because smoothing
// spreads ink into the rows either side of every band and those partial rows pull the
// non-zero mean down. A band faint enough to sit between the two thresholds is then
// dropped, and dropping a line is the failure this pipeline exists to avoid.
//
// THE FIXTURE IS TUNED AND THE TUNING IS THE FINDING. Measured on THIS page, with the
// faint band's own rows counted into both means: at the default SmoothWindow of 3 the
// two thresholds are 16.13 (smoothed) and 16.99 (raw), 0.85 of an ink pixel apart,
// and no whole number lands between them — so NO test at the default window can tell
// the two calibrations apart. This binding hard-codes the 0.05 ratio, so unlike the
// Rust port it cannot widen the gap with a constructed ratio; SmoothWindow is the only
// knob, and it is a NewLineSegmenter argument. At 15 the thresholds measure 12.39 and
// 16.99, so a faint band of 13 to 16 ink pixels per row is found by the smoothed
// calibration and missed by the raw one. 15 is near the middle of that window.
func TestTheGapThresholdIsCalibratedOnTheSmoothedProfile(t *testing.T) {
	got, err := NewLineSegmenter(10, 15).Segment(pageWithAFaintBand(8, 40, 30, 15, 20))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(got) != 9 {
		t.Fatalf("expected 8 dense bands plus the faint one, got %d -- the threshold "+
			"is being calibrated on the raw profile", len(got))
	}
}

// The smoother's own arithmetic, pinned separately from the Segment that reads it.
// Three of the four bindings diverge from Python here and nothing caught it, because
// no test used an even window. Every number below is measured at this binding's own
// parameters; smoothProfile's header records where it came from.

// bandedProfile is lead rows of ink, gap zero rows, then lead rows of ink again.
func bandedProfile(lead, gap, ink int) []int {
	out := make([]int, lead*2+gap)
	for i := 0; i < lead; i++ {
		out[i] = ink
	}
	for i := lead + gap; i < len(out); i++ {
		out[i] = ink
	}
	return out
}

func minOf(vals []float64) float64 {
	m := vals[0]
	for _, v := range vals {
		if v < m {
			m = v
		}
	}
	return m
}

func maxOf(vals []float64) float64 {
	m := vals[0]
	for _, v := range vals {
		if v > m {
			m = v
		}
	}
	return m
}

// TestTheBoxSpansOneMoreRowThanAnEvenWindowAsks is the divergence an odd-window-only
// test cannot see, and the reason it survived four ports: at an odd window this
// formula and Python's agree exactly.
//
// The loop is [-window/2, +window/2], so an even window spans window+1 rows -- one
// MORE than asked -- and reads exactly the same rows as the odd window ABOVE it. A
// gap of exactly `window` zero rows therefore still reaches zero at odd windows and
// does NOT at even ones, which is why the measured break-point table reads
// 1,3,3,5,5,7,7,9,9,11,11,13 here against Python's 1,2,...,12.
//
// Unlike JS and Rust, an even window here is not VALUE-identical to the odd one
// above, only tap-identical: the divisor is the requested window, so the two divide
// the same sum by window and window+1. That is divergence 3 in smoothProfile's
// header, and it is asserted below as equal sums.
func TestTheBoxSpansOneMoreRowThanAnEvenWindowAsks(t *testing.T) {
	for window := 2; window <= 12; window++ {
		span := 2*(window/2) + 1
		atSpan := smoothProfile(bandedProfile(20, span, 9), window)
		if got := minOf(atSpan[20 : 20+span]); got != 0 {
			t.Errorf("window %d left no zero row across a gap of %d rows (min %v) -- its span is no longer 2*(window/2)+1", window, span, got)
		}
		under := smoothProfile(bandedProfile(20, span-1, 9), window)
		if got := minOf(under[20 : 20+span-1]); got <= 0 {
			t.Errorf("window %d reached zero across a gap of only %d rows -- the box is narrower than measured", window, span-1)
		}
		if window%2 == 0 {
			profile := bandedProfile(20, span-1, 9)
			even := smoothProfile(profile, window)
			odd := smoothProfile(profile, window+1)
			for i := range even {
				// Same tap set, so the reconstructed sums must match exactly even
				// though the quotients do not.
				lhs := even[i] * float64(window)
				rhs := odd[i] * float64(window+1)
				if math.Abs(lhs-rhs) > 1e-9 {
					t.Fatalf("window %d and window %d summed different rows at row %d (%v vs %v) -- the even-window rounding changed", window, window+1, i, lhs, rhs)
				}
			}
		}
	}
}

// TestTheDivisorIsTheRequestedWindow pins the edge handling, which at ODD windows is
// numpy's mode='same' to the bit: zero-padded ends divided by the window. Row 0 at
// window 3 is 2/3 of the true mean of the two rows in range, and at window 15 it is
// 8/15 of it. JS and Rust divide by the rows they visited and report the true mean.
// Recorded, not reconciled -- unifying the divisors changes output for at least one
// binding's users, so it is an owner decision. See smoothProfile's header.
func TestTheDivisorIsTheRequestedWindow(t *testing.T) {
	flat := make([]int, 60)
	for i := range flat {
		flat[i] = 300
	}
	cases := []struct {
		window int
		want   float64
	}{{3, 200}, {5, 180}, {15, 160}}
	for _, c := range cases {
		got := smoothProfile(flat, c.window)
		if got[0] != c.want {
			t.Errorf("window %d row 0 = %v, want %v (attenuated to (window/2+1)/window of the true 300)", c.window, got[0], c.want)
		}
		if got[59] != c.want {
			t.Errorf("window %d last row = %v, want %v", c.window, got[59], c.want)
		}
	}
}

// TestAnEvenWindowInflatesTheProfileAboveItsRawPeak pins the one divergence of the
// three that is a defect rather than a difference: this smoother sums window+1 terms
// at an even window and divides by window, so every interior row comes back
// (window+1)/window too high and the smoothed peak clears the raw one. JS, Rust and
// numpy all keep the peak.
//
// The consequence is a threshold inflated by the same factor, so at an even window
// this binding drops faint bands the other three keep -- measured on an 8-band page,
// gap threshold 25.71 against JS's and Rust's 17.14 at window 2, and 20.45 against
// 16.36 at window 4. The default SmoothWindow is 3, so only a caller who passes an
// even window or sets the exported field to one reaches it.
//
// This test pins the behaviour as it ships. Changing the divisor is an owner
// decision, and this test is where the change would announce itself.
func TestAnEvenWindowInflatesTheProfileAboveItsRawPeak(t *testing.T) {
	profile := bandedProfile(30, 0, 300)
	for window := 2; window <= 12; window++ {
		peak := maxOf(smoothProfile(profile, window))
		want := 300.0
		if window%2 == 0 {
			want = 300.0 * float64(window+1) / float64(window)
		}
		if peak != want {
			t.Errorf("window %d smoothed to a peak of %v, want %v (raw peak 300)", window, peak, want)
		}
	}
}
