package predictor

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/yalue/onnxruntime_go"
)

// The pinned model: input [1, 1, 160, 1024], output [1, sequence, 277].
const (
	pinnedClasses = 277
	pinnedCharLen = 276
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
		t.Fatalf("expected a ContractError for 225 characters vs 277 classes, got %v", err)
	}
	if !strings.Contains(err.Error(), "226") || !strings.Contains(err.Error(), "277") {
		t.Errorf("error should name both sides, got: %v", err)
	}
}

// TrimSpace eating the leading space is a one-character mismatch, and one
// character is enough to shift the whole decode.
func TestCheckContractRejectsOffByOneCharset(t *testing.T) {
	if err := checkContract(pinnedCharLen-1, pinnedClasses, ExpectedInputHeight, "model.onnx"); err == nil {
		t.Fatal("expected 275 characters vs 277 classes to be refused")
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
	shape := onnxruntime_go.NewShape(1, 1, 160, -1)
	if got := staticDim(shape, 2); got != 160 {
		t.Errorf("staticDim(shape, 2) = %d, want 160", got)
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
		t.Fatal("a 275-character charset against a 277-class tensor must be refused, not decoded")
	}

	// The old bundled charset.
	stale := &Predictor{charset: []rune(charsetOfLen(225))}
	if _, err := stale.decode(preds, shape); err == nil {
		t.Fatal("a 225-character charset against a 277-class tensor must be refused, not decoded")
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
	// Shape promises fewer values than the buffer holds.
	if _, err := p.decode(preds, onnxruntime_go.NewShape(1, 4, pinnedClasses)); err == nil {
		t.Error("a shape smaller than the buffer must be refused")
	}
	// A batch of two used to decode the first item and drop the second. The
	// buffer holds both items, so only the batch check can refuse it.
	_, err := p.decode(preds, onnxruntime_go.NewShape(2, 4, pinnedClasses))
	if err == nil || !strings.Contains(err.Error(), "batch of 1") {
		t.Errorf("a batch other than 1 must be refused, got %v", err)
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

// The ONNX Runtime is a host shared library that no Go manifest can pin, so the
// only controls are: choose it explicitly, and record what actually loaded.
// These cover the choosing; RuntimeVersion covers the recording and needs a real
// library, so it is not exercised here.

func envReturning(pairs map[string]string) func(string) string {
	return func(k string) string { return pairs[k] }
}

func existsAmong(paths ...string) func(string) bool {
	set := make(map[string]bool, len(paths))
	for _, p := range paths {
		set[p] = true
	}
	return func(p string) bool { return set[p] }
}

func TestResolveSharedLibraryPathPrefersTheEnvironment(t *testing.T) {
	const custom = "/opt/ort-1.24.1/lib/libonnxruntime.dylib"
	got, err := resolveSharedLibraryPath(
		"darwin",
		envReturning(map[string]string{SharedLibraryPathEnv: custom}),
		existsAmong(custom, homebrewLibPath),
	)
	if err != nil {
		t.Fatalf("resolveSharedLibraryPath: %v", err)
	}
	if got != custom {
		t.Errorf("an explicit %s must win over the Homebrew default, got %q", SharedLibraryPathEnv, got)
	}
}

// Silently loading a different runtime than the one asked for is exactly the
// host-determined-version failure this is meant to prevent, so a set-but-missing
// path is an error rather than a fallthrough.
func TestResolveSharedLibraryPathRefusesAMissingExplicitPath(t *testing.T) {
	_, err := resolveSharedLibraryPath(
		"darwin",
		envReturning(map[string]string{SharedLibraryPathEnv: "/nope/libonnxruntime.dylib"}),
		existsAmong(homebrewLibPath),
	)
	if err == nil {
		t.Fatal("expected an error when the requested library does not exist")
	}
	if !strings.Contains(err.Error(), SharedLibraryPathEnv) {
		t.Errorf("error should name the variable, got: %v", err)
	}
}

func TestResolveSharedLibraryPathFallsBackToHomebrewOnDarwin(t *testing.T) {
	got, err := resolveSharedLibraryPath("darwin", envReturning(nil), existsAmong(homebrewLibPath))
	if err != nil {
		t.Fatalf("resolveSharedLibraryPath: %v", err)
	}
	if got != homebrewLibPath {
		t.Errorf("got %q, want the Homebrew fallback %q", got, homebrewLibPath)
	}
}

// With no override and no install at a known path, each platform gets the name
// its loader can find. Linux and macOS used to get "", which the wrapper turned
// into "onnxruntime.so" -- a name no official archive ships -- so the
// documented LD_LIBRARY_PATH setup loaded nothing.
func TestResolveSharedLibraryPathPerPlatform(t *testing.T) {
	cases := []struct {
		goos      string
		installed []string
		want      string
	}{
		{"linux", nil, "libonnxruntime.so"},
		{"linux", []string{homebrewLibPath}, "libonnxruntime.so"},
		{"freebsd", nil, "libonnxruntime.so"},
		{"darwin", []string{homebrewLibPath, intelHomebrewLibPath}, homebrewLibPath},
		{"darwin", []string{intelHomebrewLibPath}, intelHomebrewLibPath},
		{"darwin", nil, "libonnxruntime.dylib"},
		// The wrapper's Windows default, onnxruntime.dll, is the name the
		// release zips ship, so Windows still defers to it.
		{"windows", nil, ""},
	}
	for _, c := range cases {
		got, err := resolveSharedLibraryPath(c.goos, envReturning(nil), existsAmong(c.installed...))
		if err != nil {
			t.Fatalf("%s %v: %v", c.goos, c.installed, err)
		}
		if got != c.want {
			t.Errorf("%s with %v installed: got %q, want %q", c.goos, c.installed, got, c.want)
		}
	}
}

func TestTheEnvironmentStillWinsOnEveryPlatform(t *testing.T) {
	const custom = "/opt/ort/lib/libonnxruntime.so.1.24.1"
	for _, goos := range []string{"linux", "darwin", "windows"} {
		got, err := resolveSharedLibraryPath(goos,
			envReturning(map[string]string{SharedLibraryPathEnv: custom}),
			existsAmong(custom, homebrewLibPath))
		if err != nil || got != custom {
			t.Errorf("%s: got %q, %v; want %q", goos, got, err, custom)
		}
	}
}

// A NaN or an infinity in the logits must fail the read, not decode. The argmax
// compares with `>`, false for every NaN, so before this guard a NaN was skipped
// silently and +Inf simply won its timestep. On real input the pinned model's
// scores are finite; this only fires on a numeric failure.
func TestDecodeRefusesNonFiniteLogits(t *testing.T) {
	charset := []rune("abc")
	p := &Predictor{charset: charset}
	numClasses := len(charset) + 1
	for name, bad := range map[string]float32{
		"NaN":  float32(math.NaN()),
		"+Inf": float32(math.Inf(1)),
		"-Inf": float32(math.Inf(-1)),
	} {
		t.Run(name, func(t *testing.T) {
			argmax := []int{1, 0, 2}
			preds := make([]float32, len(argmax)*numClasses)
			for ts, want := range argmax {
				preds[ts*numClasses+want] = 1
			}
			preds[1*numClasses+2] = bad
			_, err := p.decode(preds, onnxruntime_go.NewShape(1, int64(len(argmax)), int64(numClasses)))
			var oe *OutputError
			if !errors.As(err, &oe) {
				t.Fatalf("expected an OutputError for %s, got %v", name, err)
			}
			if !strings.Contains(err.Error(), "non-finite") {
				t.Errorf("error should say what was wrong, got: %v", err)
			}
		})
	}
}
