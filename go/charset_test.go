package monocr

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"strings"
	"testing"
)

// The pinned revision (model.ModelRevision) exports 277 output classes:
// 276 characters plus the CTC blank at index 0. The charset was 315 characters
// (316 classes) until revision d3d9d5e moved the pin from the v2 network to
// v3.5.
//
// The first line said 316 until 2026-08-27, so the comment contradicted its own
// second line — 276 + 1 is 277 — and the const below it. `8da6178` rewrote the
// second line for v3.5 and left the first as `0022277` wrote it against v2, one
// line apart. Nothing failed, because only the const is compiled.
const pinnedCharsetLen = 276

func TestDefaultCharsetMatchesPinnedModel(t *testing.T) {
	got := len([]rune(DefaultCharset()))
	if got != pinnedCharsetLen {
		t.Fatalf("bundled charset has %d characters, the pinned model expects %d (%d classes minus the CTC blank)",
			got, pinnedCharsetLen, pinnedCharsetLen+1)
	}
}

// The charset's first character is U+0020. TrimSpace eats it, dropping 276 to
// 275 and shifting every index in the decode by one — the model still runs and
// still returns text, just the wrong text.
//
// Read "276 to 314" until 2026-08-27: `8da6178` updated the first number to
// v3.5's and not the second, leaving one subtraction with a foot in each model
// generation. The assertion below computes pinnedCharsetLen-1 and was right
// throughout.
func TestDefaultCharsetKeepsLeadingSpace(t *testing.T) {
	cs := []rune(DefaultCharset())
	if len(cs) == 0 {
		t.Fatal("bundled charset is empty")
	}
	if cs[0] != ' ' {
		t.Fatalf("charset[0] = %q, want U+0020", cs[0])
	}

	if trimmed := strings.TrimSpace(DefaultCharset()); len([]rune(trimmed)) != pinnedCharsetLen-1 {
		t.Fatalf("expected TrimSpace to drop exactly the leading space (%d -> %d), got %d",
			pinnedCharsetLen, pinnedCharsetLen-1, len([]rune(trimmed)))
	}
}

func TestNormalizeCharsetTrimsOnlyLineTerminators(t *testing.T) {
	cases := []struct{ in, want string }{
		{" abc", " abc"},
		{" abc\n", " abc"},
		{" abc\r\n", " abc"},
		{"\n abc\n", " abc"},
		{" abc ", " abc "}, // a trailing space is a class too
	}
	for _, c := range cases {
		if got := NormalizeCharset(c.in); got != c.want {
			t.Errorf("NormalizeCharset(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ReadImage used to pass strings.TrimSpace(embeddedCharset) while ReadImages
// passed the raw embed. Two charsets, two strides, two different strings for
// the same image through two documented entry points. Nothing but
// DefaultCharset may touch the raw embed.
func TestOnlyDefaultCharsetReadsTheRawEmbed(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatalf("parsing package source: %v", err)
	}

	for _, pkg := range pkgs {
		for path, file := range pkg.Files {
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Name.Name == "DefaultCharset" {
					continue
				}
				ast.Inspect(fn, func(n ast.Node) bool {
					id, ok := n.(*ast.Ident)
					if ok && id.Name == "embeddedCharset" {
						t.Errorf("%s: %s references embeddedCharset directly; go through DefaultCharset so every entry point sees the same charset",
							fset.Position(id.Pos()), fn.Name.Name)
					}
					return true
				})
			}
			_ = path
		}
	}
}
