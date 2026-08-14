package predictor

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/yalue/onnxruntime_go"
)

// The pinned model: input [1, 1, 128, width], output [1, sequence, 316].
const (
	pinnedClasses = 316
	pinnedCharLen = 315
)

func charsetOfLen(n int) string {
	// Leading U+0020, as the real charset has.
	return " " + strings.Repeat("x", n-1)
}

func TestCheckContractAcceptsThePinnedModel(t *testing.T) {
	if err := checkContract(pinnedCharLen, pinnedClasses, ExpectedInputHeight, "model.onnx"); err != nil {
		t.Fatalf("expected the pinned pair to pass, got %v", err)
	}
}

// The bundled charset used to be 225 characters against a 316-class model.
func TestCheckContractRejectsCharsetMismatch(t *testing.T) {
	err := checkContract(225, pinnedClasses, ExpectedInputHeight, "model.onnx")
	var ce *ContractError
	if !errors.As(err, &ce) {
		t.Fatalf("expected a ContractError for 225 characters vs 316 classes, got %v", err)
	}
	if !strings.Contains(err.Error(), "226") || !strings.Contains(err.Error(), "316") {
		t.Errorf("error should name both sides, got: %v", err)
	}
}

// TrimSpace eating the leading space is a one-character mismatch, and one
// character is enough to shift the whole decode.
func TestCheckContractRejectsOffByOneCharset(t *testing.T) {
	if err := checkContract(pinnedCharLen-1, pinnedClasses, ExpectedInputHeight, "model.onnx"); err == nil {
		t.Fatal("expected 314 characters vs 316 classes to be refused")
	}
}

// The artifact that used to sit behind `resolve/main` had a 64-pixel input.
func TestCheckContractRejectsHeightMismatch(t *testing.T) {
	err := checkContract(pinnedCharLen, pinnedClasses, 64, "stale.onnx")
	var ce *ContractError
	if !errors.As(err, &ce) {
		t.Fatalf("expected a ContractError for a 64-pixel input, got %v", err)
	}
}

func TestCheckContractRejectsEmptyCharset(t *testing.T) {
	if err := checkContract(0, pinnedClasses, ExpectedInputHeight, "model.onnx"); err == nil {
		t.Fatal("expected an empty charset to be refused")
	}
}

// A dynamic axis reports as -1; there is nothing to compare at load time, so
// the load passes and decode re-checks against the real output tensor.
func TestCheckContractSkipsDynamicAxes(t *testing.T) {
	if err := checkContract(pinnedCharLen, 0, 0, "dynamic.onnx"); err != nil {
		t.Fatalf("expected dynamic axes to defer the check, got %v", err)
	}
}

func TestStaticDim(t *testing.T) {
	shape := onnxruntime_go.NewShape(1, 1, 128, -1)
	if got := staticDim(shape, 2); got != 128 {
		t.Errorf("staticDim(shape, 2) = %d, want 128", got)
	}
	if got := staticDim(shape, 3); got != 0 {
		t.Errorf("dynamic axis should report 0, got %d", got)
	}
	if got := staticDim(shape, 9); got != 0 {
		t.Errorf("out-of-range axis should report 0, got %d", got)
	}
}

func syntheticLogits(seqLen, numClasses int) []float32 {
	preds := make([]float32, seqLen*numClasses)
	for i := range preds {
		preds[i] = float32(math.Sin(float64(i) * 0.37))
	}
	return preds
}

// The decode stride must come from the output tensor, never from the charset.
// Deriving it from the charset is what made ReadImage (TrimSpace, 225 classes)
// and ReadImages (raw, 226 classes) return two different strings for the same
// logits. Now a charset that disagrees with the tensor is refused outright.
func TestDecodeStrideComesFromTheTensor(t *testing.T) {
	const seqLen = 128
	preds := syntheticLogits(seqLen, pinnedClasses)
	shape := onnxruntime_go.NewShape(1, seqLen, pinnedClasses)

	good := &Predictor{charset: []rune(charsetOfLen(pinnedCharLen))}
	text, err := good.decode(preds, shape)
	if err != nil {
		t.Fatalf("the matching charset should decode, got %v", err)
	}
	if text == "" {
		t.Fatal("expected the synthetic logits to decode to something")
	}

	// The TrimSpace victim: one character short.
	trimmed := &Predictor{charset: []rune(charsetOfLen(pinnedCharLen - 1))}
	if _, err := trimmed.decode(preds, shape); err == nil {
		t.Fatal("a 314-character charset against a 316-class tensor must be refused, not decoded")
	}

	// The old bundled charset.
	stale := &Predictor{charset: []rune(charsetOfLen(225))}
	if _, err := stale.decode(preds, shape); err == nil {
		t.Fatal("a 225-character charset against a 316-class tensor must be refused, not decoded")
	}
}

func TestDecodeRejectsUnexpectedShapes(t *testing.T) {
	p := &Predictor{charset: []rune(charsetOfLen(pinnedCharLen))}
	preds := syntheticLogits(8, pinnedClasses)

	if _, err := p.decode(preds, onnxruntime_go.NewShape(8, pinnedClasses)); err == nil {
		t.Error("a 2-D output tensor must be refused")
	}
	if _, err := p.decode(preds, onnxruntime_go.NewShape(1, 0, pinnedClasses)); err == nil {
		t.Error("an empty sequence axis must be refused")
	}
	// Shape promises more values than the buffer holds.
	if _, err := p.decode(preds, onnxruntime_go.NewShape(1, 16, pinnedClasses)); err == nil {
		t.Error("a shape larger than the buffer must be refused")
	}
}

// CTC: index 0 is blank, repeats collapse, and index n maps to charset[n-1].
func TestDecodeCTCSemantics(t *testing.T) {
	charset := []rune("abc")
	p := &Predictor{charset: charset}
	numClasses := len(charset) + 1

	// Timesteps: a, a, blank, a, b, c
	argmax := []int{1, 1, 0, 1, 2, 3}
	preds := make([]float32, len(argmax)*numClasses)
	for t, want := range argmax {
		preds[t*numClasses+want] = 1
	}

	got, err := p.decode(preds, onnxruntime_go.NewShape(1, int64(len(argmax)), int64(numClasses)))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got != "aabc" {
		t.Fatalf("decode = %q, want %q", got, "aabc")
	}
}
