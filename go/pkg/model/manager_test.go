package model

import (
	"path/filepath"
	"strings"
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
