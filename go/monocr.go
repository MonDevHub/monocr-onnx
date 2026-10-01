package monocr

import (
	_ "embed"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/MonDevHub/monocr-onnx/go/pkg/imageio"
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
// from 276 characters to 275 and shifts every index in the decode by one, so
// every character comes back as its neighbour. Trim newlines and nothing else.
//
// 315/314 here until 2026-08-27 — v2's counts, written by `0022277` when they
// were true. `6c6fc29` fixed this exact sentence in Rust's normalize_charset
// and stopped there; the Go and JS copies of it stayed a generation behind.
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
// Rust binding, in one A/B dated 2026-08-22 — the answer is
// width-dependent: squeezing wins at 2 tiles, the two arms are level at 3, and
// tiling wins from 4 up. On a real book page at 150 dpi every line fitted one
// tile, so tiling never engaged at all.
//
// So squeezing is not a standing accuracy loss here; it is an unbounded one on
// unusually wide input, where tiling's downside stays bounded. Porting
// tile_line/cut_column from python/monocr_onnx/segmenter.py is still worth
// doing for that reason, and measuring it on this binding first is the point.
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
func predictFile(pred linePredictor, imagePath string) (string, error) {
	// imageio.Load applies the EXIF Orientation tag, which image.Decode ignores.
	img, err := imageio.Load(imagePath)
	if err != nil {
		return "", err
	}

	return predictImage(pred, img)
}

// linePredictor is what the page paths need from a Predictor, so they can be
// tested without an ONNX session.
type linePredictor interface {
	Predict(img image.Image) (string, error)
}

// fatalLineError reports whether a line's error means the MODEL is broken
// rather than the line. The page paths skip a line that fails and read the
// rest, which is right for one bad crop and wrong for these: a model that
// returns NaN or disagrees with the charset fails every line the same way, and
// skipping them all returned an empty page with no error.
func fatalLineError(err error) bool {
	var oe *predictor.OutputError
	var ce *predictor.ContractError
	return errors.As(err, &oe) || errors.As(err, &ce)
}

// predictImage reads every line of an already-decoded image, top to bottom.
//
// A page the segmenter finds no lines in is read whole rather than returning
// nothing, because a single cropped line is a legitimate input and produces
// zero segments. That matches readPDFWithModel.
func predictImage(pred linePredictor, img image.Image) (string, error) {
	seg := segmenter.NewLineSegmenter(segMinLineHeight, segSmoothWindow)

	// Polarity BEFORE segmentation, and this ordering is the point. The segmenter
	// treats dark as ink (segmenter.go's `< 128`), so handed a light-on-dark page it
	// segments the BACKGROUND and returns the gaps between lines. Inverting each
	// crop inside preprocess afterwards cannot recover a line never found.
	//
	// An audit caught this after the probe was added to preprocess alone. The probe
	// is idempotent -- once the corners are light a second call is a no-op -- so
	// both call sites are safe, and the per-crop one still covers ReadLine.
	//
	// Transparency is flattened onto white first, for the same reason: the
	// segmenter reads a transparent background as black.
	img = predictor.NormalizePolarity(imageio.FlattenOnWhite(img))

	lines, err := seg.Segment(img)
	if err != nil || len(lines) == 0 {
		return pred.Predict(img)
	}

	var out []string
	for _, line := range lines {
		text, err := pred.Predict(line.Img)
		if err != nil {
			// One unreadable line must not lose the rest of the page. The PDF
			// path has always skipped and continued; this matches it. A broken
			// model is not one unreadable line, so that fails the read.
			if fatalLineError(err) {
				return "", err
			}
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
			img, err := imageio.Load(filepath.Join(tempDir, file.Name()))
			if err != nil {
				continue
			}

			text, ok, err := readPDFPage(pred, seg, img)
			if err != nil {
				return nil, err
			}
			if ok {
				results = append(results, text)
			}
		}
	}

	return results, nil
}

// readPDFPage reads one rendered page. ok is false when the page contributes
// nothing to the result: no lines were found and reading it whole failed.
//
// A failed line is skipped, as it always was, unless the failure says the model
// itself is broken (fatalLineError), which fails the whole read.
func readPDFPage(pred linePredictor, seg *segmenter.LineSegmenter, img image.Image) (string, bool, error) {
	// Polarity BEFORE segmentation, and this ordering is the point. The segmenter
	// treats dark as ink (segmenter.go's `< 128`), so handed a light-on-dark page it
	// segments the BACKGROUND and returns the gaps between lines. Inverting each
	// crop inside preprocess afterwards cannot recover a line never found.
	//
	// An audit caught this after the probe was added to preprocess alone. The probe
	// is idempotent -- once the corners are light a second call is a no-op -- so
	// both call sites are safe, and the per-crop one still covers ReadLine.
	img = predictor.NormalizePolarity(imageio.FlattenOnWhite(img))

	lines, err := seg.Segment(img)
	if err != nil || len(lines) == 0 {
		// Fallback to full page prediction (single line assumption)
		text, err := pred.Predict(img)
		if err != nil {
			if fatalLineError(err) {
				return "", false, err
			}
			return "", false, nil
		}
		return text, true, nil
	}

	var pageLines []string
	for _, line := range lines {
		text, err := pred.Predict(line.Img)
		if err != nil {
			if fatalLineError(err) {
				return "", false, err
			}
			continue
		}
		pageLines = append(pageLines, text)
	}
	return strings.Join(pageLines, "\n"), true, nil
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
