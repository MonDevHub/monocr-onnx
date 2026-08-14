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

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// ExpectedInputHeight is the input height this binding preprocesses for.
//
// The charset, the input height and the classifier width are one contract. If
// they drift apart the model still runs and still returns text — it is just the
// wrong text, with no error anywhere. So this is declared here, checked against
// the graph in NewPredictor, and a disagreement refuses to load.
const ExpectedInputHeight = 128

// DefaultInputWidth is the padded canvas width fed to the model. The model's
// width axis is dynamic; this is the binding's choice, not a model constraint.
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

// initEnvironment brings up the ONNX Runtime environment once per process.
func initEnvironment() error {
	if onnxruntime_go.IsInitialized() {
		return nil
	}
	if runtime.GOOS == "darwin" {
		// Homebrew's install location; harmless if absent, the runtime then
		// falls back to whatever the loader can find.
		libPath := "/opt/homebrew/lib/libonnxruntime.dylib"
		if _, err := os.Stat(libPath); err == nil {
			onnxruntime_go.SetSharedLibraryPath(libPath)
		}
	}
	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %v. Make sure libonnxruntime.dylib (macOS) or libonnxruntime.so (Linux) is in your library path", err)
	}
	return nil
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

func (p *Predictor) preprocess(img image.Image) ([]float32, int, int, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, 0, 0, fmt.Errorf("cannot preprocess an empty image")
	}

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
