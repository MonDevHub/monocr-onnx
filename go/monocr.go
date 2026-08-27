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

// Line segmentation parameters, shared by the image and PDF paths.
//
// They were the literals 10 and 3 at the one call site that segmented. Naming
// them is what stops the two paths drifting: they have to agree, or the same
// page read as a PNG and as a PDF comes back split differently.
const (
	segMinLineHeight = 10
	segSmoothWindow  = 3
)

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
// Lines are segmented and read top to bottom, joined with newlines, the same
// way the PDF path reads a page.
//
// NOTE (2026-08-16, gap 2 closed 2026-08-18): this did not segment at all. It
// fed the whole image to the model as one line, so a multi-line image was
// compressed into a single strip and decoded as one line, and the same page
// read as a PNG and as a PDF came back differently. Both paths now share
// segMinLineHeight and segSmoothWindow.
//
// The remaining gap: wide lines are SQUEEZED into the model canvas rather than
// cut into tiles at whitespace columns, which is what the Python binding and
// the web app do.
//
// This comment used to quote `v3.5 squeezed 0.1434 against tiled 0.0795` and
// conclude "this binding is still on the worse side of that". RETIRED
// 2026-08-22: that harness was never committed and the figures do not
// reproduce. Remeasured over 201 rendered lines, twice — Python arms and the
// Rust binding — in mon_OCR/eval/tiling-ab-2026-08-22.md, the answer is
// width-dependent: squeezing wins at 2 tiles, the two arms are level at 3, and
// tiling wins from 4 up. On a real book page at 150 dpi every line fitted one
// tile, so tiling never engaged at all.
//
// So squeezing is not a standing accuracy loss here; it is an unbounded one on
// unusually wide input, where tiling's downside stays bounded. Porting
// tile_line/cut_column from python/monocr_onnx/segmenter.py is still worth
// doing for that reason, and measuring it on this binding first is the point.
// ROADMAP 4.5.6.
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

// predictFile decodes an image and reads every line in it.
//
// It used to hand the whole image to the model as one line, so a page of text
// was compressed vertically into a single strip and decoded as one line. Only
// the PDF path segmented, which meant ReadImage("page.png") and
// ReadPDF("page.pdf") gave different answers for the same page. Segmenting here
// uses the same LineSegmenter with the same parameters as readPDFWithModel, so
// the two paths now agree.
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

	return predictImage(pred, img)
}

// predictImage reads every line of an already-decoded image, top to bottom.
//
// A page the segmenter finds no lines in is read whole rather than returning
// nothing, because a single cropped line is a legitimate input and produces
// zero segments. That matches readPDFWithModel.
func predictImage(pred *predictor.Predictor, img image.Image) (string, error) {
	seg := segmenter.NewLineSegmenter(segMinLineHeight, segSmoothWindow)

	// Polarity BEFORE segmentation, and this ordering is the point. The segmenter
	// treats dark as ink (segmenter.go's `< 128`), so handed a light-on-dark page it
	// segments the BACKGROUND and returns the gaps between lines. Inverting each
	// crop inside preprocess afterwards cannot recover a line never found.
	//
	// An audit caught this after the probe was added to preprocess alone. The probe
	// is idempotent -- once the corners are light a second call is a no-op -- so
	// both call sites are safe, and the per-crop one still covers ReadLine.
	img = predictor.NormalizePolarity(img)

	lines, err := seg.Segment(img)
	if err != nil || len(lines) == 0 {
		return pred.Predict(img)
	}

	var out []string
	for _, line := range lines {
		text, err := pred.Predict(line.Img)
		if err != nil {
			// One unreadable line must not lose the rest of the page. The PDF
			// path has always skipped and continued; this matches it.
			continue
		}
		if strings.TrimSpace(text) != "" {
			out = append(out, text)
		}
	}

	if len(out) == 0 {
		return "", nil
	}

	return strings.Join(out, "\n"), nil
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

	seg := segmenter.NewLineSegmenter(segMinLineHeight, segSmoothWindow)

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
			// Polarity BEFORE segmentation, and this ordering is the point. The segmenter
			// treats dark as ink (segmenter.go's `< 128`), so handed a light-on-dark page it
			// segments the BACKGROUND and returns the gaps between lines. Inverting each
			// crop inside preprocess afterwards cannot recover a line never found.
			//
			// An audit caught this after the probe was added to preprocess alone. The probe
			// is idempotent -- once the corners are light a second call is a no-op -- so
			// both call sites are safe, and the per-crop one still covers ReadLine.
			img = predictor.NormalizePolarity(img)

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
