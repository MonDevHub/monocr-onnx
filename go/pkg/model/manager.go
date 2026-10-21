package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
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
	// input and 225 output classes, a51be11 had 128 and 316, and the one served
	// now has 160 and 277. A cache
	// keyed only on "the file exists" cannot tell those apart, so the revision is
	// part of the cache path.
	ModelRevision = "d3d9d5e"

	ModelFilename   = "monocr.onnx"
	CharsetFilename = "charset.txt"

	// ModelSHA256 and CharsetSHA256 are the digests of each file at
	// ModelRevision, copied from python/monocr_onnx/model_manager.py
	// (`_SHA256`), which is the source of truth; manager_test.go fails if they
	// drift. Bump the revision and both digests together, never one alone.
	//
	// All four bindings share ~/.monocr/models/<revision>/, so a corrupt file
	// written by any of them would otherwise be reused by every one, forever.
	ModelSHA256   = "b95de1ea0e3dc99a5c31bea32e220835da801f683ed17091eaec9954c80a4c04"
	CharsetSHA256 = "edfd75f688e4155c64aeee0dbac755da0e7ba45a388a2d178a84190fb3d7e953"
)

// ModelURL is the pinned download URL for the ONNX model.
var ModelURL = fmt.Sprintf("https://huggingface.co/%s/resolve/%s/onnx/%s", ModelRepo, ModelRevision, ModelFilename)

// CharsetURL is the charset that belongs to that exact revision. Fetching it
// from the same revision is the only way to be sure the two agree.
var CharsetURL = fmt.Sprintf("https://huggingface.co/%s/resolve/%s/onnx/%s", ModelRepo, ModelRevision, CharsetFilename)

type Manager struct {
	cacheDir string

	// Where each file comes from and the digest it must have. Fields rather
	// than the constants read directly, so a test can point them at a local
	// server and a tiny fixture.
	modelURL, charsetURL       string
	modelSHA256, charsetSHA256 string
}

func NewManager() (*Manager, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %v", err)
	}

	// Revision-scoped: a new pin is a new directory, so changing ModelRevision
	// invalidates the cache instead of silently reusing the previous artifact.
	cacheDir := filepath.Join(homeDir, ".monocr", "models", ModelRevision)
	return &Manager{
		cacheDir:      cacheDir,
		modelURL:      ModelURL,
		charsetURL:    CharsetURL,
		modelSHA256:   ModelSHA256,
		charsetSHA256: CharsetSHA256,
	}, nil
}

// CacheDir is the directory this manager downloads into.
func (m *Manager) CacheDir() string { return m.cacheDir }

func (m *Manager) GetModelPath() (string, error) {
	modelPath := filepath.Join(m.cacheDir, ModelFilename)

	ok, err := cachedAndVerified(modelPath, m.modelSHA256)
	if err != nil {
		return "", err
	}
	if !ok {
		fmt.Printf("Model %s missing or corrupt at %s. Downloading...\n", ModelRevision, modelPath)
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

	ok, err := cachedAndVerified(charsetPath, m.charsetSHA256)
	if err != nil {
		return "", err
	}
	if !ok {
		if err := download(m.charsetURL, charsetPath, m.charsetSHA256); err != nil {
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
	if err := download(m.modelURL, destPath, m.modelSHA256); err != nil {
		return err
	}
	if err := download(m.charsetURL, filepath.Join(m.cacheDir, CharsetFilename), m.charsetSHA256); err != nil {
		return err
	}

	fmt.Printf("Model downloaded successfully to %s\n", destPath)
	return nil
}

// cachedAndVerified reports whether path exists with the wanted sha256, as the
// Python binding checks. A file that exists with any other digest -- a partial
// write, or one cached before digests were checked -- is removed, so the
// caller fetches it once; a second mismatch is an error from download.
//
// The digest is recomputed on every call rather than remembered in a sidecar
// marker. That is one read of the 46 MB model per load, small next to creating
// the ONNX session, and a marker could outlive a change to the file it vouches
// for.
func cachedAndVerified(path, want string) (bool, error) {
	got, err := fileSHA256(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("failed to read cached %s: %v", path, err)
	}
	if got == want {
		return true, nil
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return false, fmt.Errorf("failed to remove corrupt %s: %v", path, err)
	}
	return false, nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// download writes url to dest via a temporary file, so an interrupted transfer
// never leaves a truncated artifact that the existence check would accept. The
// sha256 is checked before the rename, so a corrupt transfer never reaches the
// cached path either.
func download(url, dest, wantSHA256 string) error {
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

	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmp, h), resp.Body); err != nil {
		tmp.Close()
		return fmt.Errorf("failed to write %s: %v", dest, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("failed to close %s: %v", tmpName, err)
	}

	if got := hex.EncodeToString(h.Sum(nil)); got != wantSHA256 {
		return fmt.Errorf(
			"checksum mismatch for %s from %s\n  expected sha256 %s\n  got      sha256 %s\n"+
				"The pinned revision is immutable, so this is a corrupted or intercepted download, not a new release",
			filepath.Base(dest), url, wantSHA256, got)
	}

	if err := os.Rename(tmpName, dest); err != nil {
		return fmt.Errorf("failed to move %s into place: %v", dest, err)
	}
	return nil
}
