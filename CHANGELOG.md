# Changelog

All four bindings (Python, JavaScript, Go and Rust) share one model contract and
are versioned together. A release number means the same contract in every
language.

## 0.4.0 — 2026-09-04

**The JavaScript binding returned noise, and did so in every published version.**
`preprocess` builds its float canvas with `resizedBuffer[y * newWidth + x]`, one
byte per pixel. Re-wrapping a raw buffer loses the b-w colourspace `.grayscale()`
established — raw input carries no colourspace — so sharp's default sRGB output
applied and `.raw()` returned three interleaved channels. Every read landed on the
wrong byte and only the first third of each image was sampled at all.

Measured on a 925x1280 page: 55,680 bytes back where 160x116x1 is 18,560, exactly
3x. Published `monocr@0.3.2` returned **168 characters of noise** for that page
where the Python SDK returned 1,271 correct ones, and 5 characters for a line crop
where Python returned 77. Fixed with `.toColourspace('b-w')`, plus a guard that
throws when the buffer is not one byte per pixel, because the loop cannot tell a
three-channel buffer from a one-channel one — it just returns plausible-looking
text.

After the fix that page returns 1,178 characters of Mon text, 0.656 similar to
Python's. The remainder is the resampling-kernel divergence
`docs/CROSS_BINDING_PARITY.md` already records, not this defect.

Two tests were added. The suite was green through the whole bug: the existing
preprocessing test lifts the dimension arithmetic into a copy and never looked at
how many channels came back. Reverting the fix now fires the guard rather than
returning text.


**Breaking, Python only: the CLI is now `monocr-onnx`.** It was `monocr`, which is
also the command the separate `monocr` package installs. Both declared the same
`console_scripts` name, so an environment with both had one silently shadow the
other by install order — and the shadowed one was usually this package, whose
documented `monocr pdf` therefore ran a CLI with no `pdf` command. `monocr-download`
is likewise now `monocr-onnx-download`.

Nothing about the model, the charset or the Python API changed. Same contract,
same weights, same revision.

- **Corrected a false claim in the Python README.** It advertised "one API for
  images and PDFs: the same call shape for a line, a page and a document".
  `MonOCR.predict` opens its argument as an image and raises
  `PIL.UnidentifiedImageError` on a PDF. PDFs go through the module-level
  `read_pdf`, which was documented nowhere despite being the only PDF entry point.
- **Documented that PDFs need poppler.** `read_pdf` shells out to `pdftoppm`
  through `pdf2image`. Nothing said so; verified by running it with poppler off
  the PATH, which raises a `RuntimeError` that at least names poppler.
- **Added what-to-know notes**: the 46 MB first-run download, measured
  throughput, and that this package claims no accuracy figure of its own.
- **`monocr --version` reported 0.1.0 on the JavaScript CLI.** The number was a
  literal in `bin/monocr.js` and had not moved since 0.1.0, so the published
  0.3.0 and 0.3.2 both answered two minor versions behind. It now reads
  `package.json`.

Version parity kept across all four bindings even though the fix is Python-only,
because the changelog's contract is that one number means one contract everywhere.

## 0.3.2 — 2026-09-03

No change to the model, the charset or any API. This release exists to carry two
text corrections onto the registry pages, which serve whatever was uploaded with
a release and never re-read the repository.

- **One description across PyPI, npm and crates.io.** The three said two
  different things and neither said what a caller gets. All three now read
  "On-device Mon (mnw) OCR, powered by ONNX Runtime", taken from this
  repository's own opening line.
- **Removed an aside about an unrelated language** from the crate-level doc
  comment in `rust/src/lib.rs`, which is what docs.rs renders. It was already
  gone from the Rust and Go READMEs; this was the copy that kept shipping. A CI
  job now fails if authored text names that language again.

Version parity restored. 0.3.1 was a Rust-only emergency republish — the `ort`
dependency was declared as a caret range over a pre-release, so 0.3.0 could not
be compiled by anyone who depended on it — and it left Rust one number ahead of
the other three. All four are 0.3.2.

## 0.3.0 — 2026-08-27

**Breaking against everything currently installed.** The registries are two
generations behind this release, so upgrading changes results on every input.

### If you have 0.1.x installed, read this

The published artifacts (npm `monocr@0.1.5` from 2026-02-14, and PyPI
`monocr-onnx==0.1.0`) target a model that no longer exists. Measured by reading
the published tarball and wheel, not inferred:

| | published 0.1.x | this release |
|---|---|---|
| input height | **64** | **160** |
| input width | 1024 | 1024, static |
| charset | **225 characters** | **276 characters** |
| output classes | — | 277 (276 + CTC blank at 0) |
| normalisation | `pixel / 255` (Python only; the JS binding already did `/127.5 - 1.0`) | `pixel / 127.5 - 1.0` |
| model revision | pre-v3.5 | `d3d9d5e` |

A 225-character charset against a 277-class graph shifts every decoded index, so
0.1.x does not merely score worse — it returns the wrong characters. There is no
compatibility shim and there should not be one: the two are different models.

Two corrections to the first version of this entry, both measured from the
published artifacts rather than assumed:

- The `/255` normalisation is **Python-only**. npm `monocr@0.1.5` already computed
  `(pixel / 127.5) - 1.0` (`src/monocr.js:83`). Saying both bindings shared the
  old formula was wrong; only the geometry and the charset were common to them.
- The runtime charset in 0.1.x is **224**, not the 225 the file contains. Both
  bindings load it with a bare strip (`f.read().strip()` in `predictor.py:20`,
  `.trim()` in `monocr.js:36`), and the charset's first character is U+0020, a
  class the model emits. Measured on the shipped file: 225 raw, 225 after
  stripping only line terminators, **224** after a bare strip. That is a second,
  independent index shift on top of the wrong charset size.

`0.2.0` and `0.2.1` were tagged (`go/v0.2.0`, `js/v0.2.1`, `python/v0.2.1`) and
never released to npm or PyPI. Nothing is skipped by going straight to 0.3.0;
those numbers describe commits no user could install. Go is the exception — Go
modules resolve by tag, so `go/v0.2.0` has been live and correct since
2026-08-15.

### The contract, as verified in this tree

All four charset files are byte-identical: 276 characters, sha256 beginning
`edfd75f688e4155c`, first character U+0020 — a space is one of the classes the
model emits, which is why every binding strips only line terminators from the
charset file and never calls a bare `trim`.

### Fixed since the 0.2.1 tags

- **Polarity is normalised before segmentation, not after.** The probe had been
  placed in the per-crop preprocessing step. Every segmenter treats dark as ink,
  so on a light-on-dark page the segmenter was finding the gaps *between* lines
  and inverting the crops afterwards could not recover a line that was never
  found. Fixed at page level in the Python, JavaScript and Go bindings.
- **Printed page rules are suppressed before the smear** (step 3.5). A printed
  border adds a constant ink floor to every row, and once that floor clears the
  gap threshold no row inside the frame reads as a gap, so a whole page collapses
  into one band. Measured on real book pages, this recovers lines that previously
  returned nothing.
- **JavaScript crop dimensions are read from the resulting buffer** rather than
  from the pre-crop metadata, which reported the wrong size for a cropped image.
- Corrected comments that still described the v2 model: the width axis is
  **static**, not dynamic (the v3.5 export declares only axis 0 dynamic), and the
  charset figures in `normalize_charset`, `NormalizeCharset`, `readCharset` and
  the Go/Python/JS charset tests said 314/315/316 where the value is 276/277.

### Known gaps, stated rather than left to be discovered

- `check_contract` (Rust) and `checkContract` (Go) validate charset, class count
  and height, but **not width**. The Python binding reads axis 3 off the graph;
  Rust, Go and JavaScript hardcode 1024. A re-export at a different width would
  fail with an opaque shape error from inside ONNX Runtime rather than a named
  contract error.
- The Rust crate has never been published to crates.io.
- **Correction:** this entry first said `cargo test` "could not be run on the
  release machine — a linker limitation". It can. Command Line Tools 21 removed
  the clang 17 directory the linker searches while the runtime sits in the clang
  21 one, so
  `RUSTFLAGS="-C link-arg=-L/Library/Developer/CommandLineTools/usr/lib/clang/21/lib/darwin"`
  runs the whole suite: 28 lib tests and 14 doc-tests, 0 failures. The binding is
  verified by execution, not only by `cargo check` and `clippy`.

## 0.2.1 — 2026-08-16 (tagged, never released)

## 0.2.0 — 2026-08-15

Moved all four bindings to the v3.5 model at `d3d9d5e`: height 128 → 160,
charset 315 → 276 characters, 316 → 277 classes. Released for Go only.

## 0.1.5 and earlier — 2026-02

The 64-pixel-height generation and the 225-character charset (224 at runtime,
after a bare strip eats the leading U+0020). `pixel / 255` in the Python binding;
the JS binding already used `pixel / 127.5 - 1.0`. Live on npm and PyPI until
this release.
