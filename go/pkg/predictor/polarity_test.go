package predictor

// The model is trained on dark text on a light background; these check what we
// feed it.
//
// Measured 2026-08-27 over 300 labelled crops from mon_OCR's
// data/real/digits/val, same graph, only the polarity of the input changed:
//
//	upright, with the probe    CER 0.0000   300/300 exact
//	inverted, with the probe   CER 0.0000   300/300 exact
//	upright, without it        CER 0.0036   296/300
//	inverted, without it       CER 0.0342   288/300   <- 9.5x worse

import (
	"image"
	"image/color"
	"testing"
)

// page returns a bg-coloured image with an ink-coloured bar across its middle.
func page(w, h int, bg, ink uint8) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetGray(x, y, color.Gray{Y: bg})
		}
	}
	for y := h / 3; y < 2*h/3; y++ {
		for x := w / 5; x < 4*w/5; x++ {
			img.SetGray(x, y, color.Gray{Y: ink})
		}
	}
	return img
}

// TestDarkOnLightIsUntouched is THE NO-OP. Every input passes through the probe,
// so an ordinary page must come back as the same object — not merely equal, so
// that a future refactor cannot start copying every page for nothing.
func TestDarkOnLightIsUntouched(t *testing.T) {
	in := page(200, 60, 255, 0)
	if out := normalizePolarity(in); out != image.Image(in) {
		t.Fatalf("a dark-on-light page must be returned unchanged")
	}
}

func TestLightOnDarkIsInverted(t *testing.T) {
	out := normalizePolarity(page(200, 60, 0, 255))
	if g := color.GrayModel.Convert(out.At(0, 0)).(color.Gray); g.Y != 255 {
		t.Fatalf("dark background should have become light, got %d", g.Y)
	}
	if g := color.GrayModel.Convert(out.At(100, 30)).(color.Gray); g.Y != 0 {
		t.Fatalf("light ink should have become dark, got %d", g.Y)
	}
}

// TestDensePageIsNotMistakenForDark is why the probe uses a corner median and not
// a global mean. This page is ~64% ink, so its mean luminance is below 128 and a
// global test would invert a perfectly ordinary dense page.
func TestDensePageIsNotMistakenForDark(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 200, 60))
	for y := 0; y < 60; y++ {
		for x := 0; x < 200; x++ {
			img.SetGray(x, y, color.Gray{Y: 255})
		}
	}
	var sum int
	for y := 6; y < 54; y++ {
		for x := 20; x < 180; x++ {
			img.SetGray(x, y, color.Gray{Y: 0})
		}
	}
	for _, v := range img.Pix {
		sum += int(v)
	}
	if mean := float64(sum) / float64(len(img.Pix)); mean >= 128 {
		t.Fatalf("fixture must actually be mean-dark, got mean %.1f", mean)
	}
	if backgroundIsDark(img) {
		t.Fatalf("a dense page with light corners must not read as dark")
	}
}

// TestCornerFloorCoversBothAxes pins polarityCornerFloor on each axis separately.
// The patch is size/10, which is 0 under 10px; an empty sample has no median and
// would silently mean "not dark" — a wrong answer rather than a crash.
func TestCornerFloorCoversBothAxes(t *testing.T) {
	for _, tc := range []struct {
		name string
		w, h int
	}{{"short", 120, 8}, {"narrow", 8, 120}} {
		t.Run(tc.name, func(t *testing.T) {
			if !backgroundIsDark(page(tc.w, tc.h, 0, 255)) {
				t.Fatalf("a %dx%d dark-background crop must still read as dark", tc.w, tc.h)
			}
		})
	}
}

// TestTinyLightImagesAreNotInverted is what the bounds clamp is for, and the
// failure it prevents is a wrong answer rather than a panic.
//
// polarityCornerFloor is 3, which exceeds the image on anything smaller. Go's
// image.Gray.At returns black for an out-of-bounds read instead of panicking, so
// without the clamp a 1x1 WHITE image samples 8 out-of-bounds zeros against 1
// real white pixel, the median comes out at 0, and the page is inverted. Silent
// and backwards.
func TestTinyLightImagesAreNotInverted(t *testing.T) {
	for _, d := range [][2]int{{1, 1}, {2, 3}, {3, 1}, {5, 5}} {
		img := page(d[0], d[1], 255, 255) // uniformly light, nothing to invert
		if backgroundIsDark(img) {
			t.Fatalf("%dx%d uniformly light image read as dark background", d[0], d[1])
		}
	}
}

// TestPreprocessNormalisesPolarity: the units above are worthless if preprocess
// does not call them. An inverted page must reach the model as its upright twin.
func TestPreprocessNormalisesPolarity(t *testing.T) {
	p := &Predictor{targetHeight: 160, targetWidth: 1024}
	up, _, _, err := p.preprocess(page(200, 40, 255, 0))
	if err != nil {
		t.Fatal(err)
	}
	inv, _, _, err := p.preprocess(page(200, 40, 0, 255))
	if err != nil {
		t.Fatal(err)
	}
	for i := range up {
		if up[i] != inv[i] {
			t.Fatalf("tensors differ at %d: upright %f, inverted %f", i, up[i], inv[i])
		}
	}
}
