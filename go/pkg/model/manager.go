package model

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	// ModelRepo is the Hugging Face repository holding the ONNX artifact.
	ModelRepo = "janakhpon/monocr"

	// ModelRevision pins the artifact. `main` is a moving ref and the artifact
	// has already changed under it: the model served at one point had a 64-pixel
	// input and 225 output classes, the one served now has 128 and 316. A cache
	// keyed only on "the file exists" cannot tell those apart, so the revision is
	// part of the cache path.
	ModelRevision = "a51be11"

	ModelFilename   = "monocr.onnx"
	CharsetFilename = "charset.txt"
)

// ModelURL is the pinned download URL for the ONNX model.
var ModelURL = fmt.Sprintf("https://huggingface.co/%s/resolve/%s/onnx/%s", ModelRepo, ModelRevision, ModelFilename)

// CharsetURL is the charset that belongs to that exact revision. Fetching it
// from the same revision is the only way to be sure the two agree.
var CharsetURL = fmt.Sprintf("https://huggingface.co/%s/resolve/%s/onnx/%s", ModelRepo, ModelRevision, CharsetFilename)

type Manager struct {
	cacheDir string
}

func NewManager() (*Manager, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %v", err)
	}

	// Revision-scoped: a new pin is a new directory, so changing ModelRevision
	// invalidates the cache instead of silently reusing the previous artifact.
	cacheDir := filepath.Join(homeDir, ".monocr", "models", ModelRevision)
	return &Manager{cacheDir: cacheDir}, nil
}

// CacheDir is the directory this manager downloads into.
func (m *Manager) CacheDir() string { return m.cacheDir }

func (m *Manager) GetModelPath() (string, error) {
	modelPath := filepath.Join(m.cacheDir, ModelFilename)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Model %s not found at %s. Downloading...\n", ModelRevision, modelPath)
		if err := m.DownloadModel(); err != nil {
			return "", err
		}
	}

	return modelPath, nil
}

// GetCharset returns the charset published alongside the pinned model,
// downloading it if needed. It is preferred over the embedded copy because it
// is guaranteed to come from the same revision as the weights.
func (m *Manager) GetCharset() (string, error) {
	charsetPath := filepath.Join(m.cacheDir, CharsetFilename)

	if _, err := os.Stat(charsetPath); os.IsNotExist(err) {
		if err := download(CharsetURL, charsetPath); err != nil {
			return "", err
		}
	}

	data, err := os.ReadFile(charsetPath)
	if err != nil {
		return "", fmt.Errorf("failed to read cached charset: %v", err)
	}
	return string(data), nil
}

// DownloadModel fetches the pinned model and its charset into the cache.
func (m *Manager) DownloadModel() error {
	if err := os.MkdirAll(m.cacheDir, 0o755); err != nil {
		return fmt.Errorf("failed to create cache directory: %v", err)
	}

	destPath := filepath.Join(m.cacheDir, ModelFilename)
	if err := download(ModelURL, destPath); err != nil {
		return err
	}
	if err := download(CharsetURL, filepath.Join(m.cacheDir, CharsetFilename)); err != nil {
		return err
	}

	fmt.Printf("Model downloaded successfully to %s\n", destPath)
	return nil
}

// download writes url to dest via a temporary file, so an interrupted transfer
// never leaves a truncated artifact that the existence check would accept.
func download(url, dest string) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("failed to create cache directory: %v", err)
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download %s: %v", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download %s: status code %d", url, resp.StatusCode)
	}

	tmp, err := os.CreateTemp(filepath.Dir(dest), filepath.Base(dest)+".part-*")
	if err != nil {
		return fmt.Errorf("failed to create temporary file: %v", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return fmt.Errorf("failed to write %s: %v", dest, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("failed to close %s: %v", tmpName, err)
	}

	if err := os.Rename(tmpName, dest); err != nil {
		return fmt.Errorf("failed to move %s into place: %v", dest, err)
	}
	return nil
}
