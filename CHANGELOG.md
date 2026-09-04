# Changelog

All four bindings (Python, JavaScript, Go and Rust) share one model contract and
are versioned together. A release number means the same contract in every
language.

## Unreleased

### Corrected: the floors were already on main, and they have never run

**Recorded 2026-09-03. This entry replaces one written the same day that was wrong in
three of its four factual claims, and the way it went wrong is worth more than what it
got wrong.**

The previous entry said *"Three of the four surfaces have them and this repository's Rust
binding does not"*, and offered as proof: *"Verified 2026-09-03: `git show
origin/main:rust/src/segmenter.rs` contains no `CASES_MIN`."*

**All four floors are on `origin/main`** — `rust/src/segmenter.rs:954-957`,
`TILING_CASES_MIN` 14, `TILING_PROBES_MIN` 3, `RULE_CASES_MIN` 23, `MERGE_CASES_MIN` 18 —
merged on **2026-08-31** (PR #17), a commit dated 2026-08-30. So `bac90d1`, which that
entry told the reader to merge, was already merged before the entry was written.

**Why the check passed while being false: `origin/main` is a local ref, and it had not
been fetched.** It was eight commits stale. `git show origin/main:<path>` reads the
remote-tracking branch, not the remote, so it answers a question about this machine's last
fetch and looks exactly like an answer about the remote. A `git fetch` first would have
cost nothing. **A verification that does not touch the network cannot certify a remote.**

### The real defect, which the wrong entry hid

**The floors exist and have never executed in CI.** `.github/workflows/test.yml:117` runs
`cargo fmt --check` before `cargo test` at `:209`, and the committed tree does not
satisfy rustfmt: `df1f09b` (2026-08-29, the commit that wired `merge_runs` to the shared
fixture) left an `assert_eq!` at `rust/src/segmenter.rs:2067` wider than `fn_call_width`.
The Rust job has therefore died at the format step, on this branch and on `main`, since
2026-08-29 — so the four floors landed, were merged, and guarded nothing.

MEASURED 2026-09-03: `cargo fmt --check` exits 1 on the tree as committed. Fixed in this
commit by running `cargo fmt`; the suite then passes 56 lib tests and 14 doc-tests with
`cargo clippy --all-targets` clean.

That is the second time in this file's history that a gate was present and unreachable,
and the pattern is the one worth carrying: **a floor added below a step that always fails
is indistinguishable from no floor at all**, and nothing reports the difference, because
the job's failure is attributed to formatting.

### What was true in the old entry, and remains open

**Python, JavaScript and Go do not read the shared fixtures at all.** `grep -rn
'segmentation-fixtures'` matches only `rust/src/segmenter.rs` and
`.github/workflows/test.yml`; the other three bindings assert hand-written literals.
Python is additionally the oracle those cases were generated from, so it cannot
corroborate them. F-69 is the recorded cost of a half-ported segmenter: 45 of 145 pages
with over 40% of bands decoding to nonsense at 300 DPI, invisible at 150 because no
fixture drew more than one band height at one scale.

**The cross-repo checkout is pinned here and not on `main`.** `test.yml:183` carries
`ref: dc5c8ae…` on this branch; `git show origin/main:.github/workflows/test.yml` has
`repository: MonDevHub/monocr` with no `ref:`, so main's Rust suite validates against
whatever sits on that repository's default branch. The reciprocal gap is also still open:
`monocr-monorepo`'s `origin/main` checks out `MonDevHub/monocr-onnx` with no `ref:`.

**To close:** merge this branch so the pin and the formatting fix reach `main`; put
Python, JS and Go on the shared fixtures with floors from the start; and pin the
monorepo's checkout of this repository.

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
