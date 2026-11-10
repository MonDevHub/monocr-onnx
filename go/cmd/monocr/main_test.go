package main

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The extension match was case-sensitive, so IMG_0001.JPG from a camera was
// silently skipped.
func TestIsImageIgnoresCase(t *testing.T) {
	for name, want := range map[string]bool{
		"a.png": true, "a.PNG": true, "IMG_0001.JPG": true, "b.Jpeg": true,
		"notes.txt": false, "scan.pdf": false, "png": false,
	} {
		if got := isImage(name); got != want {
			t.Errorf("isImage(%q) = %v, want %v", name, got, want)
		}
	}
}

// batch used to exit 0 even when every file failed. runBatch returns the
// failure count the command exits on.
func TestRunBatchCountsFailures(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"good.png", "bad.JPG", "skip.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var read []string
	failed, err := runBatch(dir, func(path string) (string, error) {
		read = append(read, filepath.Base(path))
		if strings.HasPrefix(filepath.Base(path), "bad") {
			return "", errors.New("cannot decode")
		}
		return "text", nil
	}, io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if failed != 1 {
		t.Fatalf("runBatch reported %d failures, want 1", failed)
	}
	if strings.Join(read, ",") != "bad.JPG,good.png" {
		t.Fatalf("runBatch read %v, want the two images and not the text file", read)
	}
}
