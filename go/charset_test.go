package monocr

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"strings"
	"testing"
)

// The pinned revision (model.ModelRevision) exports 316 output classes:
// 315 characters plus the CTC blank at index 0.
const pinnedCharsetLen = 315

func TestDefaultCharsetMatchesPinnedModel(t *testing.T) {
	got := len([]rune(DefaultCharset()))
	if got != pinnedCharsetLen {
		t.Fatalf("bundled charset has %d characters, the pinned model expects %d (%d classes minus the CTC blank)",
			got, pinnedCharsetLen, pinnedCharsetLen+1)
	}
}

// The charset's first character is U+0020. TrimSpace eats it, dropping 315 to
// 314 and shifting every index in the decode by one — the model still runs and
// still returns text, just the wrong text.
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
