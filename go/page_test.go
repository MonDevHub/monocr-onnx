package monocr

import (
	"errors"
	"image"
	"image/color"
	"path/filepath"
	"testing"

	"github.com/MonDevHub/monocr-onnx/go/pkg/imageio"
	"github.com/MonDevHub/monocr-onnx/go/pkg/predictor"
	"github.com/MonDevHub/monocr-onnx/go/pkg/segmenter"
)

// fakeLines stands in for a Predictor. It returns errs[i] for the i-th call
// (nil once they run out) and records every image it was handed.
type fakeLines struct {
	errs []error
	seen []image.Image
}

func (f *fakeLines) Predict(img image.Image) (string, error) {
	f.seen = append(f.seen, img)
	if i := len(f.seen) - 1; i < len(f.errs) && f.errs[i] != nil {
		return "", f.errs[i]
	}
	return "line", nil
}

// Two dark bars on white: two lines for the segmenter.
func twoLinePage() image.Image {
	img := image.NewGray(image.Rect(0, 0, 400, 160))
	for y := 0; y < 160; y++ {
		for x := 0; x < 400; x++ {
			v := uint8(255)
			if x >= 20 && x < 380 && ((y >= 20 && y < 50) || (y >= 100 && y < 130)) {
				v = 0
			}
			img.SetGray(x, y, color.Gray{Y: v})
		}
	}
	return img
}

func TestPredictImageStillSkipsAnUnreadableLine(t *testing.T) {
	f := &fakeLines{errs: []error{errors.New("one bad crop")}}
	text, err := predictImage(f, twoLinePage())
	if err != nil {
		t.Fatalf("a single failed line should be skipped, got %v", err)
	}
	if len(f.seen) != 2 || text != "line" {
		t.Fatalf("got %q after %d calls, want the other line", text, len(f.seen))
	}
}

// A model returning NaN fails every line the same way. Skipping them all used
// to return an empty page with a nil error.
func TestPredictImageFailsOnABrokenModel(t *testing.T) {
	for _, broken := range []error{&predictor.OutputError{Msg: "NaN"}, &predictor.ContractError{Msg: "classes"}} {
		f := &fakeLines{errs: []error{broken, broken}}
		if _, err := predictImage(f, twoLinePage()); !errors.Is(err, broken) {
			t.Errorf("want %v returned, got %v", broken, err)
		}
	}
}

func TestReadPDFPageFailsOnABrokenModel(t *testing.T) {
	seg := segmenter.NewLineSegmenter(segMinLineHeight, segSmoothWindow)
	broken := &predictor.OutputError{Msg: "NaN"}
	if _, _, err := readPDFPage(&fakeLines{errs: []error{broken}}, seg, twoLinePage()); !errors.Is(err, broken) {
		t.Fatalf("want %v returned, got %v", broken, err)
	}
	text, ok, err := readPDFPage(&fakeLines{errs: []error{errors.New("one bad crop")}}, seg, twoLinePage())
	if err != nil || !ok || text != "line" {
		t.Fatalf("a single failed line should be skipped: %q %v %v", text, ok, err)
	}
}

// The page path flattens transparency before polarity and segmentation. Without
// it the (0,0,0,0) background reads black, the polarity probe inverts the page,
// and the dark text becomes near-white.
func TestThePagePathReadsATransparentImageAsDarkOnWhite(t *testing.T) {
	img, err := imageio.Load(filepath.Join("..", "data", "fixtures", "input", "alpha-text.png"))
	if err != nil {
		t.Fatal(err)
	}
	f := &fakeLines{}
	if _, err := predictImage(f, img); err != nil {
		t.Fatal(err)
	}
	darkest := uint8(255)
	for _, seen := range f.seen {
		b := seen.Bounds()
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if g := color.GrayModel.Convert(seen.At(x, y)).(color.Gray).Y; g < darkest {
					darkest = g
				}
			}
		}
	}
	if len(f.seen) == 0 || darkest > 64 {
		t.Fatalf("the model was handed no dark ink (darkest %d over %d images)", darkest, len(f.seen))
	}
}

// predictFile decodes the file itself, so it applies the EXIF orientation: the
// stored orient-6 fixture is 32x64 and must reach the model as the 64x32 image.
func TestPredictFileReadsAnExifRotatedFileUpright(t *testing.T) {
	f := &fakeLines{}
	if _, err := predictFile(f, filepath.Join("..", "data", "fixtures", "input", "orient-6.jpg")); err != nil {
		t.Fatal(err)
	}
	if len(f.seen) == 0 {
		t.Fatal("nothing reached the model")
	}
	for _, seen := range f.seen {
		if b := seen.Bounds(); b.Dx() < b.Dy() {
			t.Fatalf("the model was handed a %dx%d image: the file was read sideways", b.Dx(), b.Dy())
		}
	}
}
