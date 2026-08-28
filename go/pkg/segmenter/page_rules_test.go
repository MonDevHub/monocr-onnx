package segmenter

// Printed-rule suppression.
//
// A printed page border adds a constant ink floor to every row it spans, and once
// that floor clears the gap threshold no in-frame row reads as a gap: the page
// returns as one band and is squeezed into the model window.
//
// Measured with THIS parameter set (global threshold 128, no smear, smoothing 3,
// ratio 0.05 of the mean) over twelve real MNEC page-ones: nine collapse to three
// bands or fewer, and the twelve go from 118 bands to 215.

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

const (
	tw      = 800
	tband   = 40
	tmargin = 30
	tglyphW = 12
	tpitch  = 20
	truleW  = 4
)

// maskOf builds a binary mask: 1 = ink. Glyph blobs, not solid bars — a solid bar
// the width of a text column IS a rule by any definition.
func maskOf(bands, gap, glyphs int, framed bool) ([]uint8, int) {
	height := tmargin*2 + tband*bands + gap*(bands-1)
	m := make([]uint8, tw*height)
	y := tmargin
	for b := 0; b < bands; b++ {
		for yy := y; yy < y+tband; yy++ {
			for k := 0; k < glyphs; k++ {
				x := 100 + k*tpitch
				for i := 0; i < tglyphW && x+i < tw; i++ {
					m[yy*tw+x+i] = 1
				}
			}
		}
		y += tband + gap
	}
	if framed {
		for yy := 0; yy < height; yy++ {
			for i := 0; i < truleW; i++ {
				m[yy*tw+10+i] = 1
				m[yy*tw+tw-10-truleW+i] = 1
			}
		}
		for i := 0; i < truleW; i++ {
			for x := 0; x < tw; x++ {
				m[(10+i)*tw+x] = 1
				m[(height-10-truleW+i)*tw+x] = 1
			}
		}
	}
	return m, height
}

func clearRows(m []uint8, height int) int {
	n := 0
	for y := 0; y < height; y++ {
		any := false
		for x := 0; x < tw && !any; x++ {
			if m[y*tw+x] != 0 {
				any = true
			}
		}
		if !any {
			n++
		}
	}
	return n
}

// TestCleanPageIsUntouched is THE PROPERTY THAT MAKES THIS SAFE UNCONDITIONALLY.
// Every page gets the step whether it has rules or not, so "does nothing" must be
// exact rather than approximate.
func TestCleanPageIsUntouched(t *testing.T) {
	m, h := maskOf(4, 40, 30, false)
	before := append([]uint8(nil), m...)
	if suppressPageRules(m, tw, h) {
		t.Fatal("suppression reported a change on a page with no rules")
	}
	if !bytes.Equal(m, before) {
		t.Fatal("mask was modified on a page with no rules")
	}
}

// TestFrameIsRemoved measures against what a clean page achieves. "Some row
// reaches zero" is too low a bar: removing one axis alone already clears rows.
func TestFrameIsRemoved(t *testing.T) {
	clean, ch := maskOf(4, 40, 30, false)
	framed, fh := maskOf(4, 40, 30, true)
	target := clearRows(clean, ch)

	if got := clearRows(framed, fh); got != 0 {
		t.Fatalf("fixture must ink every row, %d rows are clear", got)
	}
	if !suppressPageRules(framed, tw, fh) {
		t.Fatal("suppression reported no change on a framed page")
	}
	if got := clearRows(framed, fh); got < int(float64(target)*0.9) {
		t.Fatalf("only %d rows clear after suppression, against %d on the clean page "+
			"-- one rule direction is probably still being missed", got, target)
	}
}

func TestGlyphInkIsNeverARule(t *testing.T) {
	m, h := maskOf(6, 10, 30, false)
	before := append([]uint8(nil), m...)
	suppressPageRules(m, tw, h)
	if !bytes.Equal(m, before) {
		t.Fatal("glyph-sized ink was classified as a rule")
	}
}

// TestHorizontalRuleAlone kills the mutation that skips the horizontal scan; the
// frame fixture cannot, because its vertical rules alone clear enough rows.
//
// Text is present deliberately: a rule with no other ink is 100% of the page, and
// the ink-share guard correctly refuses that.
func TestHorizontalRuleAlone(t *testing.T) {
	m, h := maskOf(4, 40, 30, false)
	rowY := tmargin + tband + 10
	for i := 0; i < truleW; i++ {
		for x := 0; x < tw; x++ {
			m[(rowY+i)*tw+x] = 1
		}
	}
	if !suppressPageRules(m, tw, h) {
		t.Fatal("a horizontal rule was not removed")
	}
	for i := 0; i < truleW; i++ {
		for x := 0; x < tw; x++ {
			if m[(rowY+i)*tw+x] != 0 {
				t.Fatalf("rule pixel survived at (%d,%d)", x, rowY+i)
			}
		}
	}
}

// TestExactMinimumLength pins the >= bound on EACH axis. A case on one axis alone
// leaves the other's off-by-one alive, and a full-page frame is far past the
// threshold so it cannot catch either.
func TestExactMinimumLength(t *testing.T) {
	rowY := tmargin + tband + 10
	minH := int(float64(tw) * ruleSpan)

	m, h := maskOf(4, 40, 30, false)
	for x := 0; x < minH; x++ {
		m[rowY*tw+x] = 1
	}
	if !suppressPageRules(m, tw, h) {
		t.Fatal("a horizontal run of exactly the minimum length was not removed")
	}

	m2, h2 := maskOf(4, 40, 30, false)
	for x := 0; x < minH-1; x++ {
		m2[rowY*tw+x] = 1
	}
	if suppressPageRules(m2, tw, h2) {
		t.Fatal("a horizontal run one pixel short was removed")
	}

	m3, h3 := maskOf(4, 40, 30, false)
	minV := int(float64(h3) * ruleSpan)
	for y := 0; y < minV; y++ {
		m3[y*tw+12] = 1
	}
	if !suppressPageRules(m3, tw, h3) {
		t.Fatal("a vertical run of exactly the minimum length was not removed")
	}

	m4, h4 := maskOf(4, 40, 30, false)
	for y := 0; y < minV-1; y++ {
		m4[y*tw+12] = 1
	}
	if suppressPageRules(m4, tw, h4) {
		t.Fatal("a vertical run one pixel short was removed")
	}
}

// TestSuppressionAbandonedWhenItWouldEatThePage: ruleSpan is a fraction of the
// page, so on a SHORT page a tall text block exceeds it vertically and every glyph
// column reads as a rule.
func TestSuppressionAbandonedWhenItWouldEatThePage(t *testing.T) {
	width, height := 900, 20+6*30
	m := make([]uint8, width*height)
	y := 20
	for b := 0; b < 6; b++ {
		for yy := y; yy < y+30; yy++ {
			for x := 40; x < 860; x += tpitch {
				for i := 0; i < tglyphW; i++ {
					m[yy*width+x+i] = 1
				}
			}
		}
		y += 30
	}
	before := append([]uint8(nil), m...)
	if suppressPageRules(m, width, height) {
		t.Fatal("suppression removed most of the page ink; the guard should have stopped it")
	}
	if !bytes.Equal(m, before) {
		t.Fatal("mask was modified despite the guard")
	}
}

func TestDegenerateMasksDoNotPanic(t *testing.T) {
	suppressPageRules(nil, 0, 0)
	suppressPageRules(make([]uint8, 0), 10, 10) // short buffer
	suppressPageRules(make([]uint8, 50*50), 50, 50)
	solid := make([]uint8, 50*50)
	for i := range solid {
		solid[i] = 1
	}
	suppressPageRules(solid, 50, 50)
}

// pngOf renders a mask as a grayscale PNG so Segment can be driven end to end.
func pngOf(m []uint8, height int) []byte {
	img := image.NewGray(image.Rect(0, 0, tw, height))
	for y := 0; y < height; y++ {
		for x := 0; x < tw; x++ {
			v := uint8(255)
			if m[y*tw+x] != 0 {
				v = 0
			}
			img.SetGray(x, y, color.Gray{Y: v})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}

// TestSegmentRecoversAFramedPage is the behavioural test, and the fixture took
// finding.
//
// A DENSE framed page does not fuse at this parameter set — 30 glyphs per line
// segments the same with or without suppression, which is why a structural check
// alone cannot catch a profile computed from the wrong buffer.
//
// SPARSE text reproduces the real mechanism: with 8 glyphs per line the profile
// mean drops far enough that the frame's ink floor clears the 0.05 threshold on
// every row, and the page comes back as one band.
func TestSegmentRecoversAFramedPage(t *testing.T) {
	seg := NewLineSegmenter(10, 3)

	cleanMask, ch := maskOf(4, 40, 8, false)
	framedMask, fh := maskOf(4, 40, 8, true)

	clean, err := decodeAndSegment(seg, pngOf(cleanMask, ch))
	if err != nil {
		t.Fatal(err)
	}
	framed, err := decodeAndSegment(seg, pngOf(framedMask, fh))
	if err != nil {
		t.Fatal(err)
	}
	if len(clean) != 4 {
		t.Fatalf("the unframed control segmented into %d lines, expected 4", len(clean))
	}
	if len(framed) != len(clean) {
		t.Fatalf("framed page gave %d lines, the same page unframed gave %d", len(framed), len(clean))
	}
}

func decodeAndSegment(s *LineSegmenter, data []byte) ([]SegmentResult, error) {
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return s.Segment(img)
}

// ── Crop column extents ───────────────────────────────────────────────────────
//
// Suppression buys two things, and this binding used to collect only one. The row
// profile was read from the suppressed mask, but each crop's x-range was recomputed
// by re-thresholding img.At(), so the frame was deleted from the profile and then
// reinstated in every crop. See the comment on extractLine, and mon_OCR
// src/monocr/segmenter.py:392 for the reference's statement of the intent.

// TestFrameDoesNotWidenTheCrops requires the framed page and the same page unframed
// to produce IDENTICAL x-extents. Exact, not approximate: every frame pixel belongs
// to a full-width row run or a full-height column run, so suppression leaves the
// framed mask byte-identical to the clean one and the crops have nothing to differ
// over.
func TestFrameDoesNotWidenTheCrops(t *testing.T) {
	seg := NewLineSegmenter(10, 3)

	cleanMask, ch := maskOf(4, 40, 8, false)
	framedMask, fh := maskOf(4, 40, 8, true)

	clean, err := decodeAndSegment(seg, pngOf(cleanMask, ch))
	if err != nil {
		t.Fatal(err)
	}
	framed, err := decodeAndSegment(seg, pngOf(framedMask, fh))
	if err != nil {
		t.Fatal(err)
	}
	if len(clean) != 4 {
		t.Fatalf("the unframed control segmented into %d lines, expected 4", len(clean))
	}
	if len(framed) != len(clean) {
		t.Fatalf("framed page gave %d lines, the same page unframed gave %d", len(framed), len(clean))
	}

	for i := range clean {
		if framed[i].BBox.Min.X != clean[i].BBox.Min.X || framed[i].BBox.Max.X != clean[i].BBox.Max.X {
			t.Errorf("line %d x-extent moved: framed [%d,%d), clean [%d,%d)",
				i, framed[i].BBox.Min.X, framed[i].BBox.Max.X,
				clean[i].BBox.Min.X, clean[i].BBox.Max.X)
		}
	}

	// And name the failure rather than only reporting "not equal": before the fix
	// xMin landed on the left rule and xMax on the right one, so a crop 165px wide
	// came back spanning the whole framed area.
	for i, r := range framed {
		if r.BBox.Min.X <= 10+truleW {
			t.Errorf("line %d starts at x=%d, inside the left rule at x=10..%d",
				i, r.BBox.Min.X, 10+truleW-1)
		}
		if r.BBox.Max.X >= tw-10-truleW {
			t.Errorf("line %d ends at x=%d, inside the right rule", i, r.BBox.Max.X)
		}
	}
}

// TestRuleFreeCropExtentsAreUnchanged pins the no-op claim with numbers rather than
// asserting it in prose. On a page with no rules suppressPageRules returns the mask
// untouched, and the mask is the same `< 128` test over the same pixels the old
// img.At() scan read — so these are the values that scan produced too.
//
// Glyphs run x=100 .. 100+7*tpitch+tglyphW-1 = 251.
//
// The band measures exactly tband rows, and it used to measure tband+2. That changed
// on 2026-08-28 when boundary detection moved from the smoothed row profile to the
// raw one (see the comment in Segment): smoothing over a 3-wide window pulled the
// profile above the 0.05 gap threshold one row either side of the ink, so every band
// came back two rows taller than it is. padX is derived from the band height, so it
// went from ceil(42*0.15) = 7 to ceil(40*0.15) = 6 and each crop narrowed by one
// pixel per side. Measured, not assumed: Min.X moved from 93 to 94 and Max.X from
// 258 to 257 on this fixture.
func TestRuleFreeCropExtentsAreUnchanged(t *testing.T) {
	seg := NewLineSegmenter(10, 3)
	m, h := maskOf(4, 40, 8, false)
	lines, err := decodeAndSegment(seg, pngOf(m, h))
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 4 {
		t.Fatalf("segmented into %d lines, expected 4", len(lines))
	}

	x0, x1 := 100, 100+7*tpitch+tglyphW-1
	padX := int(math.Ceil(float64(tband) * 0.15))
	for i, r := range lines {
		if r.BBox.Min.X != x0-padX {
			t.Errorf("line %d starts at x=%d, expected %d", i, r.BBox.Min.X, x0-padX)
		}
		if r.BBox.Max.X != x1+padX {
			t.Errorf("line %d ends at x=%d, expected %d", i, r.BBox.Max.X, x1+padX)
		}
	}
}
