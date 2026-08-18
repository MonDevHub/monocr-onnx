package segmenter

import (
	"image"
	"image/color"
	"testing"
)

// This package had no tests until 2026-08-18, which is how monocr.go shipped
// for months calling it from the PDF path only. ReadImage fed whole pages to
// the model as one line; closing that made this package load-bearing for both
// paths, so the behaviour it promises is worth pinning down.

// page draws horizontal bars of ink on white, each `barH` tall, separated by
// `gapH` of whitespace: the simplest thing shaped like lines of text.
func page(w, barH, gapH, bars int) image.Image {
	h := bars*barH + (bars-1)*gapH
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.White)
		}
	}
	for b := 0; b < bars; b++ {
		top := b * (barH + gapH)
		for y := top; y < top+barH; y++ {
			// Inset horizontally so the bar is a band, not a full-bleed rectangle.
			for x := w / 10; x < w-w/10; x++ {
				img.Set(x, y, color.Black)
			}
		}
	}
	return img
}

func TestSegmentFindsEveryLine(t *testing.T) {
	// The defect this guards: three bars must not come back as one strip.
	const bars = 3
	got, err := NewLineSegmenter(10, 3).Segment(page(400, 30, 20, bars))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(got) != bars {
		t.Fatalf("got %d segments, want %d", len(got), bars)
	}
}

func TestSegmentsAreOrderedTopToBottom(t *testing.T) {
	// monocr.go joins the results with newlines and never sorts them, so the
	// reading order of a page is whatever this returns.
	got, err := NewLineSegmenter(10, 3).Segment(page(400, 30, 20, 4))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	for i := 1; i < len(got); i++ {
		if got[i].BBox.Min.Y < got[i-1].BBox.Min.Y {
			t.Fatalf("segment %d starts at y=%d, above segment %d at y=%d",
				i, got[i].BBox.Min.Y, i-1, got[i-1].BBox.Min.Y)
		}
	}
}

func TestSegmentsCarryTheirPixels(t *testing.T) {
	got, err := NewLineSegmenter(10, 3).Segment(page(400, 30, 20, 3))
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	for i, s := range got {
		if s.Img == nil {
			t.Fatalf("segment %d has no image", i)
		}
		b := s.Img.Bounds()
		if b.Dx() <= 0 || b.Dy() <= 0 {
			t.Fatalf("segment %d is empty: %v", i, b)
		}
	}
}

func TestBlankPageYieldsNoSegments(t *testing.T) {
	// Callers depend on this: both monocr.go paths read the whole image when
	// Segment returns nothing, which is what makes a single cropped line work.
	blank := image.NewRGBA(image.Rect(0, 0, 200, 200))
	for y := 0; y < 200; y++ {
		for x := 0; x < 200; x++ {
			blank.Set(x, y, color.White)
		}
	}

	got, err := NewLineSegmenter(10, 3).Segment(blank)
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("got %d segments on a blank page, want 0", len(got))
	}
}

func TestMinLineHeightDropsThinBands(t *testing.T) {
	// A 4px band is noise, not a line. With MinLineH at 40 nothing survives,
	// which is the knob monocr.go sets to 10 for both paths.
	img := page(400, 4, 20, 3)

	kept, err := NewLineSegmenter(2, 3).Segment(img)
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	dropped, err := NewLineSegmenter(40, 3).Segment(img)
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}

	if len(kept) == 0 {
		t.Fatal("a permissive MinLineH found nothing; the fixture is wrong, not the code")
	}
	if len(dropped) != 0 {
		t.Fatalf("MinLineH=40 kept %d bands of 4px", len(dropped))
	}
}
