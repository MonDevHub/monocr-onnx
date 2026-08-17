# Cross-binding parity — measured 2026-08-14 against v2, not re-run since

Four bindings run the same ONNX graph with the same charset. They do **not** produce
identical text.

This file records the measurement rather than the intention, because "the bindings are
aligned" was asserted in several READMEs and had never been run across more than one
image.

> **Every number below is stale.** It was measured against `a51be11`, which had 316
> classes at H=128. On 2026-08-15 all four bindings moved to `d3d9d5e`: 277 classes,
> H=160, a 276-character charset. The network changed underneath this file and the run has
> not been repeated. Treat the agreement counts, the per-image character counts and
> the specific `U+1060` / `U+102F` / `U+1046` diagnoses as history.
>
> The causal claim below, that the divergence comes from four different resampling
> kernels, is an argument about the code rather than about the artifact, so it
> plausibly survives the model change. It has not been re-checked either.
>
> There is also a **second axis this file never covered**: Python tiles a wide line
> into canvas-width pieces, while JS, Go and Rust squeeze it into the window. On v3.5
> that gap was measured upstream at CER 0.1434 squeezed against 0.0795 tiled — larger
> than any disagreement recorded here. Go's `ReadImage` goes further and does not
> segment at all, so a multi-line page is read as one strip; only its PDF path
> segments. See the notes in `go/monocr.go` and `python/monocr_onnx/predictor.py`.

## What was measured

All four bindings, on all seven images in `data/images/`, against the revision-pinned
model `a51be11` (316 classes, H=128) with the 315-character charset they shared at the
time.

There is **no ground truth** for these images. They carry no labels, and the two whose
filenames match mon_OCR's generator (`000028.jpg`, `000029.jpg`) are *not* the same
bytes as the files behind those labels — that corpus was regenerated. So this measures
**agreement**, not accuracy. Four implementations agreeing is evidence the decode path
is consistent; it is not evidence that the text is right.

## Result: agreement on 5 of 7

| Image | Python | JS | Go |
|---|---|---|---|
| `000028.jpg` | identical | identical | identical |
| `000029.jpg` | identical | identical | identical |
| `pdf_screenshot.png` | identical | identical | identical |
| `test_0006_h61.png` | identical | identical | identical |
| `test_0011_h30.png` | identical | identical | identical |
| `test_0005_h71.png` | 78 chars | 78 chars | **77, drops the final `U+1060`** |
| `test_0012_h86.png` | 78 chars | **77** | **77**, both dropping the final `U+102F` |

Rust additionally reads `U+1046` (Myanmar digit six) on `test_0011_h30.png` where the
other three read `U+0030` (ASCII zero).

**Every disagreement is a single trailing character**, and in each case it is a Mon
diacritic or digit rather than a decorative mark: `U+1060 MYANMAR CONSONANT SIGN MON
MEDIAL LA`, `U+102F MYANMAR VOWEL SIGN U`. Dropping one is a real transcription error,
not a formatting difference.

## Cause: four resampling kernels, two of them the wrong family

The training pipeline resizes with `cv2.INTER_LINEAR`
(`mon_OCR/src/monocr/utils.py`, `resize_and_pad`). The bindings do not agree with it or
with each other:

| Binding | Resampler | Family | Matches training |
|---|---|---|---|
| Python | PIL `BILINEAR` | bilinear | yes |
| Rust | `FilterType::Triangle` | bilinear | yes |
| Go | `draw.CatmullRom` | **bicubic** | no |
| JS | sharp default (`fit: 'fill'`) | **lanczos3** | no |

Different kernels produce slightly different pixels, which produce slightly different
logits, which flip the argmax on the final timestep where CTC is least certain. That is
why the divergence appears at the end of a line and nowhere else.

## Not fixed, and why

Aligning every binding on bilinear is the principled default — it is what the model was
trained through, and train/serve preprocessing skew is a real defect independent of
which output happens to be nicer. But it **cannot be validated here**: with no labelled
image in this repository, there is no way to show that changing a kernel improves
accuracy rather than merely changing which line loses its last diacritic.

So the sequence is: obtain a labelled set, then align the kernels and measure. Doing it
in the other order swaps an unmeasured disagreement for an unmeasured agreement.

## Reproducing

Run each binding over `data/images/` and diff. Python, JS and Go emit one line per
image; the Rust CLI writes a `.txt` beside each input instead, and emits multiple lines
for the page image, so its output needs aligning per-image rather than by line number.

## What this does not cover

- **Accuracy.** No ground truth exists here. See above.
- **The segmenter.** These are pre-cropped lines. Every binding also carries its own
  line segmenter, and those diverge further — see `mon_OCR/src/monocr/segmenter.py`,
  which records that seven implementations exist and that three of fourteen constants
  survive across all of them.
- **iOS and Android.** The apps in `monocr-monorepo` are a separate pair of
  implementations again, and were not part of this run.
