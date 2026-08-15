package monocr

import (
	_ "embed"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/MonDevHub/monocr-onnx/go/pkg/model"
	"github.com/MonDevHub/monocr-onnx/go/pkg/predictor"
	"github.com/MonDevHub/monocr-onnx/go/pkg/segmenter"
)

//go:embed charset.txt
var embeddedCharset string

// RuntimeVersion loads the ONNX Runtime shared library if it is not already
// loaded and reports its version.
//
// go.mod pins the cgo wrapper, not the runtime — the shared library comes from
// the host, so the version cannot be declared, only read back. Name it in any
// report of a result: it identifies the runtime that produced the text.
func RuntimeVersion() (string, error) {
	if err := predictor.InitRuntime(); err != nil {
		return "", err
	}
	return predictor.RuntimeVersion(), nil
}

// NormalizeCharset strips only line terminators.
//
// The charset's first character really is U+0020 — a space is one of the
// classes the model emits. strings.TrimSpace eats it, which drops the charset
// from 315 characters to 314 and shifts every index in the decode by one, so
// every character comes back as its neighbour. Trim newlines and nothing else.
func NormalizeCharset(charset string) string {
	return strings.Trim(charset, "\r\n")
}

// DefaultCharset is the charset compiled into this package. Every entry point
// resolves through here so no two of them can disagree.
func DefaultCharset() string {
	return NormalizeCharset(embeddedCharset)
}

// resolveModel returns the cached model path and the charset that belongs to
// it. The charset published alongside the pinned revision wins; the embedded
// copy is the offline fallback. Either way predictor.NewPredictor checks both
// against the graph before running anything.
func resolveModel() (modelPath, charset string, err error) {
	manager, err := model.NewManager()
	if err != nil {
		return "", "", err
	}

	modelPath, err = manager.GetModelPath()
	if err != nil {
		return "", "", err
	}

	charset = DefaultCharset()
	if published, err := manager.GetCharset(); err == nil {
		charset = NormalizeCharset(published)
	}
	return modelPath, charset, nil
}

// ReadImage recognizes text from an image file.
// It automatically downloads the model if not present.
func ReadImage(imagePath string) (string, error) {
	modelPath, charset, err := resolveModel()
	if err != nil {
		return "", err
	}

	return ReadImageWithModel(imagePath, modelPath, charset)
}

// ReadImages recognizes text from multiple image files.
func ReadImages(imagePaths []string) ([]string, error) {
	modelPath, charset, err := resolveModel()
	if err != nil {
		return nil, err
	}

	pred, err := predictor.NewPredictor(modelPath, charset)
	if err != nil {
		return nil, err
	}
	defer pred.Close()

	var results []string
	for _, path := range imagePaths {
		text, err := predictFile(pred, path)
		if err != nil {
			return nil, err
		}
		results = append(results, text)
	}
	return results, nil
}

// ReadImageWithAccuracy recognizes text and calculates accuracy against ground truth.
func ReadImageWithAccuracy(imagePath, groundTruth string) (string, float64, error) {
	text, err := ReadImage(imagePath)
	if err != nil {
		return "", 0, err
	}
	accuracy := calculateAccuracy(text, groundTruth)
	return text, accuracy, nil
}

// ReadImageWithModel allows specifying a custom model path and charset.
//
// The charset is normalized for line terminators only; it must otherwise be
// exactly the charset the model was trained with. A mismatch against the
// model's classifier width is refused rather than decoded.
func ReadImageWithModel(imagePath, modelPath, charset string) (string, error) {
	pred, err := predictor.NewPredictor(modelPath, NormalizeCharset(charset))
	if err != nil {
		return "", err
	}
	defer pred.Close()

	return predictFile(pred, imagePath)
}

func predictFile(pred *predictor.Predictor, imagePath string) (string, error) {
	f, err := os.Open(imagePath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return "", fmt.Errorf("failed to decode image: %v", err)
	}

	return pred.Predict(img)
}

// ReadPDF recognizes text from a PDF file (requires pdftoppm/poppler-utils).
func ReadPDF(pdfPath string) ([]string, error) {
	// Check for pdftoppm
	_, err := exec.LookPath("pdftoppm")
	if err != nil {
		return nil, fmt.Errorf("pdftoppm not found: please install poppler-utils")
	}

	modelPath, charset, err := resolveModel()
	if err != nil {
		return nil, err
	}

	return readPDFWithModel(pdfPath, modelPath, charset)
}

// ReadPDFs recognizes text from multiple PDF files.
func ReadPDFs(pdfPaths []string) ([][]string, error) {
	// Check for pdftoppm
	_, err := exec.LookPath("pdftoppm")
	if err != nil {
		return nil, fmt.Errorf("pdftoppm not found: please install poppler-utils")
	}

	modelPath, charset, err := resolveModel()
	if err != nil {
		return nil, err
	}

	var results [][]string
	for _, path := range pdfPaths {
		pages, err := readPDFWithModel(path, modelPath, charset)
		if err != nil {
			return nil, err
		}
		results = append(results, pages)
	}
	return results, nil
}

func readPDFWithModel(pdfPath, modelPath, charset string) ([]string, error) {
	// Create temp dir
	tempDir, err := os.MkdirTemp("", "monocr-go-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	// Convert PDF to images
	cmd := exec.Command("pdftoppm", "-png", "-r", "300", pdfPath, filepath.Join(tempDir, "page"))
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to convert PDF: %v", err)
	}

	// Read all generated images
	files, err := os.ReadDir(tempDir)
	if err != nil {
		return nil, err
	}

	pred, err := predictor.NewPredictor(modelPath, charset)
	if err != nil {
		return nil, err
	}
	defer pred.Close()

	seg := segmenter.NewLineSegmenter(10, 3)

	var results []string
	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".png") {
			imgPath := filepath.Join(tempDir, file.Name())

			// Open image for segmentation
			f, err := os.Open(imgPath)
			if err != nil {
				continue
			}
			img, _, err := image.Decode(f)
			f.Close()
			if err != nil {
				continue
			}

			// Segment lines
			lines, err := seg.Segment(img)
			if err != nil || len(lines) == 0 {
				// Fallback to full page prediction (single line assumption)
				text, err := pred.Predict(img)
				if err == nil {
					results = append(results, text)
				}
				continue
			}

			// Predict each line
			var pageLines []string
			for _, line := range lines {
				text, err := pred.Predict(line.Img)
				if err == nil {
					pageLines = append(pageLines, text)
				}
			}
			results = append(results, strings.Join(pageLines, "\n"))
		}
	}

	return results, nil
}

// Levenshtein distance calculation
func levenshtein(s1, s2 []rune) int {
	len1, len2 := len(s1), len(s2)
	column := make([]int, len1+1)

	for y := 1; y <= len1; y++ {
		column[y] = y
	}

	for x := 1; x <= len2; x++ {
		column[0] = x
		lastDiag := x - 1
		for y := 1; y <= len1; y++ {
			oldDiag := column[y]
			cost := 0
			if s1[y-1] != s2[x-1] {
				cost = 1
			}
			column[y] = min(column[y]+1, min(column[y-1]+1, lastDiag+cost))
			lastDiag = oldDiag
		}
	}
	return column[len1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func calculateAccuracy(pred, truth string) float64 {
	p := []rune(pred)
	t := []rune(truth)

	if len(t) == 0 {
		if len(p) == 0 {
			return 100.0
		}
		return 0.0
	}

	dist := levenshtein(p, t)
	maxLen := len(p)
	if len(t) > maxLen {
		maxLen = len(t)
	}

	if maxLen == 0 {
		return 100.0
	}

	return (1.0 - float64(dist)/float64(maxLen)) * 100.0
}
