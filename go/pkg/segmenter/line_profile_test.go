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
// On the smoothed profile the break point is the smoother's SPAN,
// 2*(SmoothWindow/2)+1 and not the requested window, so raising it widened the
// damage in step: measured here, SmoothWindow 15 returned 1 band for every gap from
// 1px to 14px. 5px and 12px are two the old form lost. 15 is odd, so span and window
// coincide; the even-window case is pinned against smoothProfile directly, below.
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
// numpy's mode='same' formula exactly -- zero-padded ends divided by the window --
// though not bit-identical to numpy, which rounds once per tap. Row 0 at window 3 is
// 2/3 of the true mean of the two rows in range, and at window 15 it is 8/15 of it.
// JS and Rust divide by the rows they visited and report the true mean.
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

// ─────────────────────────────────────────────────────────────────────────────
// The other half of the dual histogram: the gap merge.
//
// Raw-profile detection alone splits one Mon line wherever a single row dips below the
// gap threshold, between the upper diacritic zone and the consonant bodies. See
// minGapMerge and mergeRuns in segmenter.go for the measurement, taken through THIS
// binding.
//
// Every fixture below carries ORDINARY full-height lines as well as the case under
// test, and that is load-bearing rather than decoration. mergeRuns judges a fragment
// against the page's typical line height, so a page consisting of nothing but the two
// halves of one split line is degenerate -- the median IS the half-height, and there is
// no evidence in it that they are halves rather than two short lines. Each test also
// says which clause is the sole reason its assertion holds, so a mutation to that
// clause cannot be masked by another.
// ─────────────────────────────────────────────────────────────────────────────

// splitLinePage renders one split text line: a strip of sparse upper marks, a blank
// gap, then a 44-row body.
//
// The measured geometry. Marks 2px wide against the body's tglyphW so the strip's row
// profile is faint but well clear of the gap threshold, and so the strip is a run in
// its own right rather than noise.
func splitLinePage(stripH, gapH int) image.Image {
	const height = 200
	img := image.NewGray(image.Rect(0, 0, tw, height))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	stripY := 60
	bodyY := stripY + stripH + gapH
	for _, band := range [][3]int{{stripY, stripY + stripH, 2}, {bodyY, bodyY + 44, tglyphW}} {
		for yy := band[0]; yy < band[1]; yy++ {
			for k := 0; k < 30; k++ {
				x := 100 + k*tpitch
				if x+band[2] <= tw {
					for i := 0; i < band[2]; i++ {
						img.SetGray(x+i, yy, color.Gray{Y: 0})
					}
				}
			}
		}
	}
	return img
}

// profileOf builds a raw row profile with ink on the given bands and nowhere else.
// Each band is {from, to, value}.
func profileOf(length int, bands [][3]int) []int {
	hist := make([]int, length)
	for _, b := range bands {
		for y := b[0]; y < b[1]; y++ {
			hist[y] = b[2]
		}
	}
	return hist
}

func sameRuns(got, want [][2]int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestADiacriticStripIsReturnedJoinedToItsLine drives the merge THROUGH Segment, not
// only through the helper.
//
// A mutation deleting the mergeRuns call from the pipeline survives every helper test
// below, because they call the helper directly and leave the call site unguarded. That
// is the gap se-brain rules/standards/testing.md names: a tested helper does not make
// its call site safe.
//
// Geometry is the measured one: a 20-row strip of upper marks, two empty rows, then a
// 44-row body. One line, and it must come back as one band.
func TestADiacriticStripIsReturnedJoinedToItsLine(t *testing.T) {
	got, err := NewLineSegmenter(10, 3).Segment(splitLinePage(20, 2))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("the strip and its body came back as %d bands -- the merge is not "+
			"reached from Segment", len(got))
	}
	if got[0].BBox.Dy() < 66 {
		t.Errorf("the returned band is %dpx tall, so it does not span the strip and "+
			"the body together", got[0].BBox.Dy())
	}
}

// TestAStripShorterThanMinLineHSurvivesTheMerge pins the ORDER of the merge and the
// height filter, which no band count reveals.
//
// Filtering first also returns one band here -- the body, with its diacritics deleted
// -- so a count assertion passes either way. What separates them is where the band
// STARTS: merged first it opens above the strip, filtered first the 6-row strip is
// discarded and the band opens at the body.
func TestAStripShorterThanMinLineHSurvivesTheMerge(t *testing.T) {
	got, err := NewLineSegmenter(10, 3).Segment(splitLinePage(6, 4))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected one band, got %d", len(got))
	}
	if got[0].BBox.Min.Y > 60 {
		t.Errorf("the band starts at row %d, below the strip at row 60 -- the height "+
			"filter ran before the merge and ate the diacritics", got[0].BBox.Min.Y)
	}
}

// TestASubThresholdDipDoesNotEndALine uses measured numbers rather than invented ones:
// one line, rows 260-324, split by row 280 carrying 6 ink pixels against a threshold of
// 7.0. Upstream's measurement, kept because it is the case F-69 diagnosed; this
// binding's own instance is page 9 of the same book, where the threshold is 6.8163 and
// row 377 carries 5.
func TestASubThresholdDipDoesNotEndALine(t *testing.T) {
	hist := profileOf(400, [][3]int{{260, 325, 200}, {280, 281, 6}})
	want := [][2]int{{260, 325}}
	if got := mergeRuns([][2]int{{260, 280}, {281, 325}}, hist, minGapMerge); !sameRuns(got, want) {
		t.Errorf("a 1-row dip holding ink split one line in two: got %v, want %v", got, want)
	}
}

// TestAZeroInkGapStillMergesAFragmentIntoItsLine is the other measured case: rows
// 341-360 are the upper marks and 362-404 the body of one line, separated by TWO rows
// of genuinely zero ink. No ink test can cross that; the fragment clause is what does,
// and it is the sole reason this merge happens.
func TestAZeroInkGapStillMergesAFragmentIntoItsLine(t *testing.T) {
	hist := profileOf(500, [][3]int{{341, 360, 40}, {362, 404, 300}})
	want := [][2]int{{341, 404}}
	if got := mergeRuns([][2]int{{341, 360}, {362, 404}}, hist, minGapMerge); !sameRuns(got, want) {
		t.Errorf("a 19-row fragment two empty rows from a 42-row line stayed separate: "+
			"got %v, want %v", got, want)
	}
}

// TestTwoRealLinesTwoRowsApartStaySeparate is the case the fragment clause must NOT
// swallow, and the reason it is a 2x ratio and not a 1x one: same gap, same emptiness,
// but both runs are full height.
//
// The three 60-row lines make this test mean something. Without them the page median
// would be 40, the merged band would be 82 against a ceiling of 80, and the SIZE BOUND
// would refuse the merge whatever the ratio said -- so a ratio loosened to 1x would
// survive. With them the median is 60, the ceiling is 120, and the fragment clause is
// the only thing holding the two lines apart.
//
// A vertical smear cannot tell these apart at all, which is why one was not used: at
// reach 1 it closes 2-row gaps, and 2 rows is the tightest real line spacing.
func TestTwoRealLinesTwoRowsApartStaySeparate(t *testing.T) {
	runs := [][2]int{{20, 60}, {62, 102}, {150, 210}, {250, 310}, {350, 410}}
	hist := profileOf(500, [][3]int{
		{20, 60, 300}, {62, 102, 300}, {150, 210, 300}, {250, 310, 300}, {350, 410, 300},
	})
	if got := mergeRuns(runs, hist, minGapMerge); !sameRuns(got, runs) {
		t.Errorf("two 40-row lines were fused on a page whose typical line is 60 -- "+
			"the fragment ratio is no longer 2x: got %v", got)
	}
}

// TestAWideGapIsALineBoundaryHoweverMuchInkItHolds is the size bound on its own.
// Overlapping diacritics can hold the raw profile above zero right across real
// inter-line spacing; upstream that collapsed 3 PDF lines into 1.
//
// The 60-row lines are again what makes the assertion about maxGap: they put the ceiling
// at 120, and the merged band would be 95, so maxGap is the only clause refusing this
// merge. Sized the other way the test would pass for a maxGap of any value at all.
func TestAWideGapIsALineBoundaryHoweverMuchInkItHolds(t *testing.T) {
	runs := [][2]int{{20, 60}, {75, 115}, {150, 210}, {250, 310}, {350, 410}}
	hist := profileOf(500, [][3]int{
		{20, 60, 300}, {60, 75, 5}, {75, 115, 300},
		{150, 210, 300}, {250, 310, 300}, {350, 410, 300},
	})
	if got := mergeRuns(runs, hist, minGapMerge); !sameRuns(got, runs) {
		t.Errorf("a 15-row gap merged, so the size bound is not being applied: got %v", got)
	}
}

// TestADipBetweenEqualHalvesMergesOnInkAlone isolates the ink clause, which no other
// case here does: in the measured dip cases the fragment clause ALSO fires, so dropping
// the ink test survives them.
//
// Here the two halves are 40 rows each on a page whose typical line is 60, so neither is
// a fragment by the 2x ratio, and only the two rows of surviving ink in the dip can
// merge them.
func TestADipBetweenEqualHalvesMergesOnInkAlone(t *testing.T) {
	hist := profileOf(400, [][3]int{
		{20, 60, 300}, {60, 62, 5}, {62, 102, 300}, {150, 210, 300}, {260, 320, 300},
	})
	want := [][2]int{{20, 102}, {150, 210}, {260, 320}}
	got := mergeRuns([][2]int{{20, 60}, {62, 102}, {150, 210}, {260, 320}}, hist, minGapMerge)
	if !sameRuns(got, want) {
		t.Errorf("an ink-holding 2-row dip between two halves of a typical line did "+
			"not merge: got %v, want %v", got, want)
	}
}

// TestAMergeMayNotBuildABandPastTwiceATypicalLine pins the ceiling, and the cascade it
// is the backstop for.
//
// Four 20-row fragments separated by single inked rows would chain into one 83-row band,
// and each merge makes the accumulated run taller. Upstream, with a fragment test judged
// against the NEIGHBOUR and no ceiling, page 47 of a 56-page book collapsed from 36
// bands to 10 with single bands of 534, 632 and 732 rows, losing 92% of its readable
// characters.
//
// The typical line here is 40, so the chain is cut when it would pass 80: the first
// three fragments become one 62-row band and the fourth stays its own.
func TestAMergeMayNotBuildABandPastTwiceATypicalLine(t *testing.T) {
	runs := [][2]int{
		{0, 20}, {21, 41}, {42, 62}, {63, 83},
		{150, 190}, {210, 250}, {270, 310}, {330, 370}, {390, 430},
	}
	bands := make([][3]int, 0, len(runs)+3)
	for _, r := range runs {
		bands = append(bands, [3]int{r[0], r[1], 300})
	}
	// One inked row in each gap of the chain, so the gap clause allows the merge and the
	// ceiling is the only thing that can refuse it.
	for _, y := range []int{20, 41, 62} {
		bands = append(bands, [3]int{y, y + 1, 5})
	}
	want := [][2]int{
		{0, 62}, {63, 83},
		{150, 190}, {210, 250}, {270, 310}, {330, 370}, {390, 430},
	}
	if got := mergeRuns(runs, profileOf(500, bands), minGapMerge); !sameRuns(got, want) {
		t.Errorf("the fragment chain grew past twice a typical line -- the ceiling is "+
			"gone: got %v, want %v", got, want)
	}
}

// TestAFragmentIsJudgedAgainstThePageMedian is the only case that separates the two
// yardsticks while the ceiling still allows the merge.
//
// The 21-row run is exactly half its 42-row neighbour, so a neighbour-relative ratio
// calls it a fragment and fuses them. Against the page median of 40 it is NOT a fragment
// -- 21 is over half a typical line -- and it stays its own band. The merged band would
// be 65 against a ceiling of 80, so the ceiling is not what refuses this: the yardstick
// is.
//
// Judging against the neighbour cascades, because each merge makes the accumulated run
// taller and the next line then looks more like a fragment. Measured on this binding's
// 56-page corpus, that form costs 28 bands and 2.2 points more sub-0.6x fragments (1921
// bands, 17.4%) than this one (1893, 15.2%).
func TestAFragmentIsJudgedAgainstThePageMedian(t *testing.T) {
	runs := [][2]int{{10, 50}, {100, 142}, {144, 165}, {200, 240}, {260, 300}}
	bands := make([][3]int, 0, len(runs))
	for _, r := range runs {
		bands = append(bands, [3]int{r[0], r[1], 300})
	}
	if got := mergeRuns(runs, profileOf(400, bands), minGapMerge); !sameRuns(got, runs) {
		t.Errorf("a 21-row run was fused into its 42-row neighbour on a page whose "+
			"typical line is 40 -- the fragment test is measuring against the "+
			"neighbour again: got %v", got)
	}
}
