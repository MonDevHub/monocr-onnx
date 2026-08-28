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
