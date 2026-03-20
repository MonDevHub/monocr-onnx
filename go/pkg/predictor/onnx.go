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
	"unicode/utf8"

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

type Predictor struct {
	session *onnxruntime_go.DynamicAdvancedSession
	charset string
}

func NewPredictor(modelPath, charset string) (*Predictor, error) {
	// Initialize ONNX Runtime environment if not already initialized
	// Note: SetSharedLibraryPath might be needed depending on system
	// For now we assume the default or system library is available
	if !onnxruntime_go.IsInitialized() {
		// Try to find libonnxruntime on macOS if not set
		if runtime.GOOS == "darwin" {
			// Common Homebrew path
			libPath := "/opt/homebrew/lib/libonnxruntime.dylib"
			if _, err := os.Stat(libPath); err == nil {
				onnxruntime_go.SetSharedLibraryPath(libPath)
			} else {
				// Fallback or check another location if needed
			}
		}

		if err := onnxruntime_go.InitializeEnvironment(); err != nil {
			// Check if we can find the library from JS SDK node_modules as a fallback
			return nil, fmt.Errorf("failed to initialize ONNX Runtime: %v. Make sure libonnxruntime.dylib (macOS) or libonnxruntime.so (Linux) is in your library path", err)
		}
	}

	options, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %v", err)
	}
	defer options.Destroy()

	inputs := []string{"input"}
	outputs := []string{"output"}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		inputs,
		outputs,
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	return &Predictor{
		session: session,
		charset: charset,
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

	// Correct usage of NewTensor based on original code and common usage
	// It seems NewTensor takes shape []int64, then data
	shape := []int64{1, 1, int64(h), int64(w)}
	inputTensor, err := onnxruntime_go.NewTensor(shape, inputData)
	if err != nil {
		return "", fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	// Run expects []Value, so we need to copy inputTensor into a []Value slice
	inputValues := []onnxruntime_go.Value{inputTensor}
	outputValues := make([]onnxruntime_go.Value, 1)

	err = p.session.Run(inputValues, outputValues)
	if err != nil {
		return "", fmt.Errorf("inference failed: %v", err)
	}

	outputTensor := outputValues[0]
	if outputTensor == nil {
		return "", fmt.Errorf("output tensor is nil")
	}
	// outputTensor is a Value, we need to assert it to Tensor to GetData
	defer outputTensor.Destroy()

	// Assuming output is float32 tensor
	// We need to type assert or use GetData() on the specific tensor type if generic
	// Let's assume output[0] is *Tensor[float32] which implements Value?
	// Actually NewDynamicAdvancedSession.Run returns []Value.
	// We might need to cast output[0] via interface check or assume it's Tensor[float32]

	// In original code: outputTensor := outputInfo[0]; preds := outputTensor.GetData()
	// But that was checking return of Run?

	// Let's check the type assertion
	outTensorFloat, ok := outputTensor.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return "", fmt.Errorf("unexpected output tensor type")
	}

	return p.decode(outTensorFloat.GetData()), nil
}

func (p *Predictor) preprocess(img image.Image) ([]float32, int, int, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	targetHeight := 128
	targetWidth := 1024
	
	scale := float64(targetHeight) / float64(height)
	newWidth := int(math.Round(float64(width) * scale))
	if newWidth > targetWidth {
		newWidth = targetWidth
	}

	// Resize using high quality resampling
	resized := image.NewGray(image.Rect(0, 0, newWidth, targetHeight))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Create white background canvas
	dst := image.NewGray(image.Rect(0, 0, targetWidth, targetHeight))
	draw.Draw(dst, dst.Bounds(), &image.Uniform{color.Gray{255}}, image.Point{}, draw.Src)
	draw.Draw(dst, image.Rect(0, 0, newWidth, targetHeight), resized, image.Point{}, draw.Over)

	// Normalize to [-1.0, 1.0]
	// (pix / 127.5) - 1.0
	inputData := make([]float32, targetWidth*targetHeight)
	for i, v := range dst.Pix {
		inputData[i] = float32(v)/127.5 - 1.0
	}

	return inputData, targetHeight, targetWidth, nil
}

func (p *Predictor) decode(preds []float32) string {
	decodedText := ""
	prevIdx := -1

	// numClasses = charset + blank
	numClasses := utf8.RuneCountInString(p.charset) + 1
	seqLen := len(preds) / numClasses

	// Charset array for lookup (runes)
	charsetRunes := []rune(p.charset)

	for t := 0; t < seqLen; t++ {
		maxVal := float32(-math.MaxFloat32)
		maxIdx := 0

		for c := 0; c < numClasses; c++ {
			val := preds[t*numClasses+c]
			if val > maxVal {
				maxVal = val
				maxIdx = c
			}
		}

		if maxIdx != 0 && maxIdx != prevIdx {
			// maxIdx 0 is blank
			// maxIdx 1..N maps to charset[0..N-1]
			charIdx := maxIdx - 1
			if charIdx < len(charsetRunes) {
				decodedText += string(charsetRunes[charIdx])
			}
		}
		prevIdx = maxIdx
	}

	return decodedText
}
