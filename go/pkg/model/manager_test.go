package model

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
)

// `main` is a moving ref and the artifact has already changed under it — from a
// 64-pixel, 225-class model to a 128-pixel, 316-class one. Downloads must name
// a revision.
func TestDownloadURLsArePinned(t *testing.T) {
	for name, url := range map[string]string{"model": ModelURL, "charset": CharsetURL} {
		if strings.Contains(url, "/resolve/main/") {
			t.Errorf("%s URL still tracks the moving ref `main`: %s", name, url)
		}
		if !strings.Contains(url, "/resolve/"+ModelRevision+"/") {
			t.Errorf("%s URL is not pinned to %s: %s", name, ModelRevision, url)
		}
	}
}

// The charset has to come from the same revision as the weights, or the two can
// disagree without anything noticing.
func TestCharsetIsFetchedFromTheModelRevision(t *testing.T) {
	modelDir := strings.TrimSuffix(ModelURL, "/"+ModelFilename)
	charsetDir := strings.TrimSuffix(CharsetURL, "/"+CharsetFilename)
	if modelDir != charsetDir {
		t.Fatalf("model and charset come from different places:\n  %s\n  %s", modelDir, charsetDir)
	}
}

// The cache used to gate on file existence alone, so a cached artifact from an
// older revision was reused forever. Scoping the directory by revision makes a
// re-pin a cache miss.
func TestCacheDirIsScopedByRevision(t *testing.T) {
	m, err := NewManager()
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	if got := filepath.Base(m.CacheDir()); got != ModelRevision {
		t.Fatalf("cache directory is %q; expected it to end in the revision %q", m.CacheDir(), ModelRevision)
	}
}

func hexSHA256(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// fakeHub serves bodies by filename, records what was fetched, and stands in for
// Hugging Face so no test here touches the network.
type fakeHub struct {
	mu      sync.Mutex
	fetched []string
	bodies  map[string][]byte
}

func (h *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	name := filepath.Base(r.URL.Path)
	h.mu.Lock()
	h.fetched = append(h.fetched, name)
	h.mu.Unlock()
	body, ok := h.bodies[name]
	if !ok {
		http.NotFound(w, r)
		return
	}
	w.Write(body)
}

// testManager points a Manager at a temp cache and a local server, with tiny
// fixtures and their digests injected in place of the pinned ones.
func testManager(t *testing.T, served map[string][]byte, model, charset []byte) (*Manager, *fakeHub) {
	t.Helper()
	hub := &fakeHub{bodies: served}
	srv := httptest.NewServer(hub)
	t.Cleanup(srv.Close)
	return &Manager{
		cacheDir:      t.TempDir(),
		modelURL:      srv.URL + "/onnx/" + ModelFilename,
		charsetURL:    srv.URL + "/onnx/" + CharsetFilename,
		modelSHA256:   hexSHA256(model),
		charsetSHA256: hexSHA256(charset),
	}, hub
}

var (
	fakeModel   = []byte("fake onnx weights")
	fakeCharset = []byte("abc\n")
)

// python/monocr_onnx/model_manager.py is the source of truth for the digests.
// All bindings share one cache directory, so if their digests disagree, one of
// them rejects -- and deletes -- a file another one just verified.
func TestDigestsMatchThePythonBinding(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "..", "python", "monocr_onnx", "model_manager.py"))
	if err != nil {
		t.Skipf("python binding not alongside this module: %v", err)
	}
	for key, want := range map[string]string{"MODEL_FILENAME": ModelSHA256, "CHARSET_FILENAME": CharsetSHA256} {
		m := regexp.MustCompile(key + `:\s*"([0-9a-f]{64})"`).FindSubmatch(src)
		if m == nil {
			t.Fatalf("no %s digest in the Python model manager", key)
		}
		if string(m[1]) != want {
			t.Errorf("%s digest drifted from Python: Go has %s, Python has %s", key, want, m[1])
		}
	}
}

func TestEmbeddedCharsetHasThePinnedDigest(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "charset.txt"))
	if err != nil {
		t.Fatalf("read embedded charset: %v", err)
	}
	if got := hexSHA256(b); got != CharsetSHA256 {
		t.Fatalf("go/charset.txt has sha256 %s; the pin is %s", got, CharsetSHA256)
	}
}

// The cache used to accept any file that existed. A corrupt one must be
// deleted and fetched again, and a good one must not be.
func TestCorruptCachedModelIsRefetched(t *testing.T) {
	m, hub := testManager(t, map[string][]byte{ModelFilename: fakeModel, CharsetFilename: fakeCharset}, fakeModel, fakeCharset)
	modelPath := filepath.Join(m.cacheDir, ModelFilename)
	if err := os.WriteFile(modelPath, []byte("truncated or poisoned"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := m.GetModelPath()
	if err != nil {
		t.Fatalf("GetModelPath: %v", err)
	}
	if b, _ := os.ReadFile(got); string(b) != string(fakeModel) {
		t.Fatalf("cached model is %q after refetch; want the served bytes", b)
	}
	if len(hub.fetched) == 0 || hub.fetched[0] != ModelFilename {
		t.Fatalf("expected the corrupt model to be refetched, fetched %v", hub.fetched)
	}

	// A verified cache hit downloads nothing.
	hub.fetched = nil
	if _, err := m.GetModelPath(); err != nil {
		t.Fatalf("GetModelPath on a good cache: %v", err)
	}
	if len(hub.fetched) != 0 {
		t.Fatalf("a verified cache hit downloaded %v", hub.fetched)
	}
}

func TestCorruptCachedCharsetIsRefetched(t *testing.T) {
	m, _ := testManager(t, map[string][]byte{CharsetFilename: fakeCharset}, fakeModel, fakeCharset)
	if err := os.WriteFile(filepath.Join(m.cacheDir, CharsetFilename), []byte("xyz\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := m.GetCharset()
	if err != nil {
		t.Fatalf("GetCharset: %v", err)
	}
	if got != string(fakeCharset) {
		t.Fatalf("GetCharset returned %q; the corrupt cached copy was trusted", got)
	}
}

// A transfer with the wrong digest must fail loudly and never reach the cached
// path, where every binding would trust it from then on.
func TestDownloadWithWrongDigestIsRefused(t *testing.T) {
	wrong := []byte("same length, wrong!")
	m, _ := testManager(t, map[string][]byte{ModelFilename: wrong, CharsetFilename: fakeCharset}, fakeModel, fakeCharset)

	_, err := m.GetModelPath()
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("GetModelPath err = %v; want a checksum mismatch", err)
	}
	entries, _ := os.ReadDir(m.cacheDir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ModelFilename) {
			t.Fatalf("a refused download left %s in the cache", e.Name())
		}
	}
}
