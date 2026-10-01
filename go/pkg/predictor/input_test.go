package predictor

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/MonDevHub/monocr-onnx/go/pkg/imageio"
)

// What reaches the model, for the shared input fixtures. imageio's own tests
// cover the loaded pixels; these cover preprocess, which is where a line handed
// straight to Predict gets its transparency flattened.

var fixtures = filepath.Join("..", "..", "..", "data", "fixtures", "input")

var oriented = []string{
	"orient-1.jpg", "orient-2.jpg", "orient-3.jpg", "orient-4.jpg",
	"orient-5.jpg", "orient-6.jpg", "orient-7.jpg", "orient-8.jpg",
	"orient-6-be.jpg", "orient-6.png",
}

const orientTolerance = 8

func decodeFile(t *testing.T, path string) image.Image {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func loadFile(t *testing.T, path string) image.Image {
	t.Helper()
	img, err := imageio.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func TestTheModelInputIsTheSameForEveryOrientation(t *testing.T) {
	p := &Predictor{targetHeight: ExpectedInputHeight, targetWidth: DefaultInputWidth}
	want, _, _, err := p.preprocess(decodeFile(t, filepath.Join(fixtures, "upright.png")))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range oriented {
		got, _, _, err := p.preprocess(loadFile(t, filepath.Join(fixtures, name)))
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if d := got[i] - want[i]; d > orientTolerance/127.5 || d < -orientTolerance/127.5 {
				t.Fatalf("%s: input differs from the upright image's at %d by %f", name, i, d)
			}
		}
	}
}

func TestATransparentLineReachesTheModelAsDarkOnWhite(t *testing.T) {
	p := &Predictor{targetHeight: ExpectedInputHeight, targetWidth: DefaultInputWidth}
	want, _, _, _ := p.preprocess(decodeFile(t, filepath.Join(fixtures, "alpha-text.expected.png")))
	got, _, _, _ := p.preprocess(decodeFile(t, filepath.Join(fixtures, "alpha-text.png")))
	for i := range want {
		if d := got[i] - want[i]; d > 1.5/127.5 || d < -1.5/127.5 {
			t.Fatalf("input differs from the composited image's at %d by %f", i, d)
		}
	}
}
