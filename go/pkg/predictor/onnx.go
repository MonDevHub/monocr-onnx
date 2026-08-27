package predictor

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"runtime"
	"sort"

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// ExpectedInputHeight is the input height this binding preprocesses for.
//
// The charset, the input height and the classifier width are one contract. If
// they drift apart the model still runs and still returns text — it is just the
// wrong text, with no error anywhere. So this is declared here, checked against
// the graph in NewPredictor, and a disagreement refuses to load.
const ExpectedInputHeight = 160

// DefaultInputWidth is the padded canvas width fed to the model.
//
// This is a fallback, not a free choice. v3.5 was exported with only axis 0
// dynamic, so axis 3 is the literal integer 1024 and the graph runs at that
// width alone. This comment used to read "the model's width axis is dynamic;
// this is the binding's choice, not a model constraint" — true of v2
// ([1, 1, 128, width]), false since the move to d3d9d5e, and it is the stated
// reason checkContract validates height but not width.
const DefaultInputWidth = 1024

// ContractError reports a model artifact that disagrees with the charset or the
// input geometry this binding was built for. It is returned instead of running,
// because running would produce confident nonsense rather than an error.
type ContractError struct {
	Msg string
}

func (e *ContractError) Error() string { return "model contract violation: " + e.Msg }

type Predictor struct {
	session *onnxruntime_go.DynamicAdvancedSession
	charset []rune
	// targetHeight is taken from the model graph once the contract check has
	// confirmed it matches ExpectedInputHeight.
	targetHeight int
	targetWidth  int
	// numClasses is the model's classifier width, or 0 when the graph declares
	// that axis dynamic. Decoding always re-derives it from the output tensor.
	numClasses int
}

// checkContract compares the charset and geometry this binding holds against
// what the ONNX graph actually declares.
//
// modelClasses and modelHeight are 0 (or negative) when the graph leaves that
// axis dynamic, in which case there is nothing to compare and the check passes
// — decode() re-derives the class count from the real output tensor and fails
// there instead.
func checkContract(charsetLen, modelClasses, modelHeight int, modelPath string) error {
	if charsetLen == 0 {
		return &ContractError{Msg: "no charset supplied; cannot decode model output"}
	}
	if modelClasses > 0 {
		expected := charsetLen + 1
		if modelClasses != expected {
			return &ContractError{Msg: fmt.Sprintf(
				"charset/model mismatch.\n"+
					"  charset: %d characters -> expects %d classes (%d + CTC blank)\n"+
					"  model (%s): %d classes\n"+
					"Every index above the first divergence would decode to the wrong character.",
				charsetLen, expected, charsetLen, modelPath, modelClasses)}
		}
	}
	if modelHeight > 0 && modelHeight != ExpectedInputHeight {
		return &ContractError{Msg: fmt.Sprintf(
			"input height mismatch: this binding preprocesses to height %d but %s expects %d",
			ExpectedInputHeight, modelPath, modelHeight)}
	}
	return nil
}

// ONNX Runtime version facts for this binding.
//
// go.mod pins github.com/yalue/onnxruntime_go, but that is the cgo *wrapper* —
// the runtime itself is a shared library the host supplies, and no Go manifest
// can pin it. The three constants below are what can be said about it, and
// RuntimeVersion reports what was actually loaded.
const (
	// ORTAPIVersion is the C API revision the wrapper requests via
	// OrtGetApiBase()->GetApi(). onnxruntime_go v1.11.0 vendors headers with
	// ORT_API_VERSION 18. ONNX Runtime keeps GetApi backward compatible, so a
	// newer library serves this request fine; an older one returns NULL and
	// initialisation fails.
	ORTAPIVersion = 18

	// MinORTVersion is the oldest runtime that can answer GetApi(18) — the
	// release that introduced API 18. Below this, loading fails outright.
	MinORTVersion = "1.18.0"

	// TestedORTVersion is what this binding is developed and tested against.
	// It matches the Python and JS bindings, which pin onnxruntime 1.24.1.
	TestedORTVersion = "1.24.1"
)

// SharedLibraryPathEnv names the shared library to load, overriding every
// default. Set it to an absolute path to choose the runtime deliberately rather
// than inheriting whatever the host happens to have.
const SharedLibraryPathEnv = "MONOCR_ONNXRUNTIME_PATH"

// homebrewLibPath is the Apple-silicon Homebrew install location, used as a
// fallback on darwin when the environment variable is unset.
const homebrewLibPath = "/opt/homebrew/lib/libonnxruntime.dylib"

// loadedVersion is the version string of the ONNX Runtime that initEnvironment
// actually loaded, recorded once so errors and reports can name it. Empty until
// initialisation has been attempted.
var loadedVersion string

// RuntimeVersion returns the version of the ONNX Runtime shared library loaded
// into this process, or "" if no attempt has been made yet.
//
// This is the only pin available to a Go binding: the version cannot be
// declared up front, so it is read back and recorded. Include it in any report
// of a result — it identifies the runtime that produced it.
func RuntimeVersion() string { return loadedVersion }

// resolveSharedLibraryPath picks the shared library to hand to the wrapper.
// It returns "" to mean "say nothing and let the platform loader decide".
//
// Precedence is explicit request, then platform default, then the loader. An
// explicit request that does not exist is an error rather than a silent
// fallthrough: someone who set the variable is choosing a runtime, and quietly
// loading a different one is the failure this whole change exists to prevent.
func resolveSharedLibraryPath(goos string, getenv func(string) string, exists func(string) bool) (string, error) {
	if p := getenv(SharedLibraryPathEnv); p != "" {
		if !exists(p) {
			return "", fmt.Errorf("%s is set to %q but no file is there", SharedLibraryPathEnv, p)
		}
		return p, nil
	}
	if goos == "darwin" && exists(homebrewLibPath) {
		return homebrewLibPath, nil
	}
	return "", nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// initEnvironment brings up the ONNX Runtime environment once per process and
// records which runtime version answered.
func initEnvironment() error {
	if onnxruntime_go.IsInitialized() {
		// Something else in the process brought the runtime up — possibly with a
		// different library than we would have chosen. Record what is actually
		// loaded rather than reporting nothing.
		if loadedVersion == "" {
			loadedVersion = onnxruntime_go.GetVersion()
		}
		return nil
	}

	libPath, err := resolveSharedLibraryPath(runtime.GOOS, os.Getenv, fileExists)
	if err != nil {
		return err
	}
	if libPath != "" {
		onnxruntime_go.SetSharedLibraryPath(libPath)
	}

	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		// The wrapper records the library's version string before it checks
		// whether GetApi succeeded, so on a too-old runtime this names the
		// version that was rejected rather than leaving a bare error code.
		if v := onnxruntime_go.GetVersion(); v != "" {
			loadedVersion = v
			return fmt.Errorf(
				"failed to initialize ONNX Runtime: %v.\n"+
					"  loaded: %s (from %s)\n"+
					"  needed: >= %s, because this binding requests C API version %d\n"+
					"  tested against: %s\n"+
					"Set %s to an absolute path to choose a different library.",
				err, v, describeSource(libPath), MinORTVersion, ORTAPIVersion,
				TestedORTVersion, SharedLibraryPathEnv)
		}
		return fmt.Errorf(
			"failed to initialize ONNX Runtime: %v.\n"+
				"  looked for: %s\n"+
				"Install ONNX Runtime %s (macOS: `brew install onnxruntime`), or set %s "+
				"to the absolute path of libonnxruntime.dylib (macOS) / libonnxruntime.so (Linux).",
			err, describeSource(libPath), TestedORTVersion, SharedLibraryPathEnv)
	}

	loadedVersion = onnxruntime_go.GetVersion()
	return nil
}

// InitRuntime loads the ONNX Runtime shared library if it is not already
// loaded, so RuntimeVersion can report which one answered. NewPredictor does
// this itself; call it directly only to probe the runtime.
func InitRuntime() error { return initEnvironment() }

// describeSource names where the runtime was loaded from, for error messages.
func describeSource(libPath string) string {
	if libPath == "" {
		return "the system library path"
	}
	return libPath
}

// staticDim returns d[i] when it is a fixed positive size, and 0 when the axis
// is dynamic (ONNX reports those as -1) or the index is out of range.
func staticDim(d onnxruntime_go.Shape, i int) int {
	if i < 0 || i >= len(d) {
		return 0
	}
	if d[i] <= 0 {
		return 0
	}
	return int(d[i])
}

func NewPredictor(modelPath, charset string) (*Predictor, error) {
	if err := initEnvironment(); err != nil {
		return nil, err
	}

	// Read the real graph rather than assuming the tensor names, the input
	// height or the class count. All three have changed under this SDK before.
	inputs, outputs, err := onnxruntime_go.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model graph from %s: %v", modelPath, err)
	}
	if len(inputs) != 1 {
		return nil, &ContractError{Msg: fmt.Sprintf("expected 1 model input, %s has %d", modelPath, len(inputs))}
	}
	if len(outputs) != 1 {
		return nil, &ContractError{Msg: fmt.Sprintf("expected 1 model output, %s has %d", modelPath, len(outputs))}
	}

	inShape := inputs[0].Dimensions
	if len(inShape) != 4 {
		return nil, &ContractError{Msg: fmt.Sprintf(
			"expected a 4-D [batch, channel, height, width] input, %s declares %v", modelPath, inShape)}
	}
	outShape := outputs[0].Dimensions
	if len(outShape) != 3 {
		return nil, &ContractError{Msg: fmt.Sprintf(
			"expected a 3-D [batch, sequence, classes] output, %s declares %v", modelPath, outShape)}
	}

	modelHeight := staticDim(inShape, 2)
	modelClasses := staticDim(outShape, 2)
	charsetRunes := []rune(charset)

	if err := checkContract(len(charsetRunes), modelClasses, modelHeight, modelPath); err != nil {
		return nil, err
	}

	targetHeight := modelHeight
	if targetHeight == 0 {
		targetHeight = ExpectedInputHeight
	}

	options, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %v", err)
	}
	defer options.Destroy()

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	return &Predictor{
		session:      session,
		charset:      charsetRunes,
		targetHeight: targetHeight,
		targetWidth:  DefaultInputWidth,
		numClasses:   modelClasses,
	}, nil
}

func (p *Predictor) Close() error {
	if p.session != nil {
		return p.session.Destroy()
	}
	return nil
}

func (p *Predictor) Predict(img image.Image) (string, error) {
	inputData, h, w, err := p.preprocess(img)
	if err != nil {
		return "", err
	}

	shape := onnxruntime_go.NewShape(1, 1, int64(h), int64(w))
	inputTensor, err := onnxruntime_go.NewTensor(shape, inputData)
	if err != nil {
		return "", fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	inputValues := []onnxruntime_go.Value{inputTensor}
	outputValues := make([]onnxruntime_go.Value, 1)

	if err := p.session.Run(inputValues, outputValues); err != nil {
		return "", fmt.Errorf("inference failed: %v", err)
	}

	outputTensor := outputValues[0]
	if outputTensor == nil {
		return "", fmt.Errorf("output tensor is nil")
	}
	defer outputTensor.Destroy()

	outTensorFloat, ok := outputTensor.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return "", fmt.Errorf("unexpected output tensor type")
	}

	return p.decode(outTensorFloat.GetData(), outTensorFloat.GetShape())
}

// Polarity constants. The model is trained on dark text on a light background,
// and this binding never checked which it was given.
//
// Measured 2026-08-27 over 300 labelled crops from mon_OCR's
// data/real/digits/val, same graph, only the polarity of the input changed:
//
//	upright, with this probe    CER 0.0000   300/300 exact
//	inverted, with this probe   CER 0.0000   300/300 exact
//	upright, without it         CER 0.0036   296/300
//	inverted, without it        CER 0.0342   288/300   <- 9.5x worse
//
// Degradation rather than the total failure it might sound like, and cheap to
// close. Those crops are Myanmar digits on composited backgrounds, so the effect
// on full Mon text lines is unmeasured.
//
// A COPY of mon_OCR/src/monocr/utils.go's to_normalized_grayscale steps 1-3, not
// a shared module: these bindings ship independently. Step 4 of that function,
// background levelling, is not ported here and is what the 0.0036 upright row
// above costs.
const (
	polarityCornerFraction = 10
	polarityCornerFloor    = 3
	darkBackgroundMedian   = 128
)

// backgroundIsDark reports whether the four corner patches of img are dark
// enough that it is light-text-on-dark and needs inverting.
//
// Corner-median rather than a global mean: document corners are almost always
// background, so their median survives a dense, text-heavy page where a global
// mean is dragged toward the ink. A page 64% covered in ink has a mean below 128
// and must NOT be inverted.
func backgroundIsDark(img image.Image) bool {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return false
	}
	ch := h / polarityCornerFraction
	if ch < polarityCornerFloor {
		ch = polarityCornerFloor
	}
	cw := w / polarityCornerFraction
	if cw < polarityCornerFloor {
		cw = polarityCornerFloor
	}
	// The floor can exceed the image on a tiny crop; clamp so the sample is never
	// empty. An empty sample has no median, and "no opinion" would silently mean
	// "not dark" — a wrong answer rather than a crash.
	if ch > h {
		ch = h
	}
	if cw > w {
		cw = w
	}

	samples := make([]uint8, 0, 4*ch*cw)
	corners := [4][2]int{{0, 0}, {w - cw, 0}, {0, h - ch}, {w - cw, h - ch}}
	for _, c := range corners {
		for y := 0; y < ch; y++ {
			for x := 0; x < cw; x++ {
				g := color.GrayModel.Convert(img.At(b.Min.X+c[0]+x, b.Min.Y+c[1]+y)).(color.Gray)
				samples = append(samples, g.Y)
			}
		}
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
	n := len(samples)
	if n == 0 {
		return false
	}
	var median float64
	if n%2 == 1 {
		median = float64(samples[n/2])
	} else {
		median = (float64(samples[n/2-1]) + float64(samples[n/2])) / 2
	}
	return median < darkBackgroundMedian
}

// NormalizePolarity returns img as dark-text-on-light, inverting it when the
// background is dark. An already-correct image is returned unchanged, which is
// what makes this safe to run on every input.
func NormalizePolarity(img image.Image) image.Image {
	if !backgroundIsDark(img) {
		return img
	}
	b := img.Bounds()
	out := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			g := color.GrayModel.Convert(img.At(x, y)).(color.Gray)
			out.SetGray(x, y, color.Gray{Y: 255 - g.Y})
		}
	}
	return out
}

func (p *Predictor) preprocess(img image.Image) ([]float32, int, int, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, 0, 0, fmt.Errorf("cannot preprocess an empty image")
	}

	img = NormalizePolarity(img)

	targetHeight := p.targetHeight
	targetWidth := p.targetWidth

	scale := float64(targetHeight) / float64(height)
	newWidth := int(math.Round(float64(width) * scale))
	if newWidth > targetWidth {
		newWidth = targetWidth
	}
	if newWidth < 1 {
		newWidth = 1
	}

	// Resize using high quality resampling
	resized := image.NewGray(image.Rect(0, 0, newWidth, targetHeight))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Create white background canvas
	dst := image.NewGray(image.Rect(0, 0, targetWidth, targetHeight))
	draw.Draw(dst, dst.Bounds(), &image.Uniform{color.Gray{255}}, image.Point{}, draw.Src)
	draw.Draw(dst, image.Rect(0, 0, newWidth, targetHeight), resized, image.Point{}, draw.Over)

	// Normalize to [-1.0, 1.0]
	inputData := make([]float32, targetWidth*targetHeight)
	for i, v := range dst.Pix {
		inputData[i] = float32(v)/127.5 - 1.0
	}

	return inputData, targetHeight, targetWidth, nil
}

// decode runs CTC greedy decoding over the flat logits.
//
// The stride comes from the output tensor's own shape, never from the charset.
// Deriving it from the charset made any change to the model's class count
// silently reinterpret the whole buffer — and it made two entry points holding
// two different charsets return two different strings for the same image.
func (p *Predictor) decode(preds []float32, shape onnxruntime_go.Shape) (string, error) {
	if len(shape) != 3 {
		return "", &ContractError{Msg: fmt.Sprintf(
			"expected a 3-D [batch, sequence, classes] output tensor, got shape %v", shape)}
	}
	numClasses := int(shape[2])
	seqLen := int(shape[1])
	if numClasses <= 0 || seqLen <= 0 {
		return "", &ContractError{Msg: fmt.Sprintf("output tensor has an empty axis: shape %v", shape)}
	}
	if expected := len(p.charset) + 1; numClasses != expected {
		return "", &ContractError{Msg: fmt.Sprintf(
			"charset/model mismatch at decode time: charset has %d characters -> expects %d classes, tensor has %d",
			len(p.charset), expected, numClasses)}
	}
	if need := seqLen * numClasses; len(preds) < need {
		return "", &ContractError{Msg: fmt.Sprintf(
			"output tensor holds %d values, shape %v needs %d", len(preds), shape, need)}
	}

	var decoded []rune
	prevIdx := -1

	for t := 0; t < seqLen; t++ {
		maxVal := float32(math.Inf(-1))
		maxIdx := 0

		base := t * numClasses
		for c := 0; c < numClasses; c++ {
			if v := preds[base+c]; v > maxVal {
				maxVal = v
				maxIdx = c
			}
		}

		// Index 0 is the CTC blank; 1..N map onto charset[0..N-1].
		if maxIdx != 0 && maxIdx != prevIdx {
			decoded = append(decoded, p.charset[maxIdx-1])
		}
		prevIdx = maxIdx
	}

	return string(decoded), nil
}
