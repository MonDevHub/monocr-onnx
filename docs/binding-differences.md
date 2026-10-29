# Where the bindings differ

The four bindings do not yet return the same text for the same page. This page
records how far apart they are, why, and why the gap is not closed yet.

## Measured agreement

Measured 2026-10-01 on the page path — `read_image` / `predict`, the call the
quick starts use — over the seven images in [`data/images/`](../data/images/):
Python, JavaScript and Go return identical text on **1 of 7**. Pairwise, Python
matches JavaScript on 1, Python matches Go on 1, and JavaScript matches Go on 3.
Rust matches Python on 3. The widest gap is `000028.jpg`, where Python and Rust
return 154 characters and JavaScript and Go return 55 and 61.

**That is agreement, not accuracy.** These images have no ground truth, and four
implementations reading the same wrong thing would agree perfectly.

## Why they differ

Two things compound: four different image-resampling kernels, two of them the
wrong family, and line segmenters that are not configured alike.

- **Resampling.** Training resizes bilinearly, as Python (OpenCV `INTER_LINEAR`)
  and Rust (`FilterType::Triangle`) do. Go uses `draw.CatmullRom`, which is
  bicubic, and JavaScript uses sharp's default, lanczos3.
- **Binarisation.** Python binarises adaptively; the other three use a flat
  threshold at 128.
- **Line gaps.** Python sets its gap threshold at 0.02 of the profile's maximum
  with a smoothing window of 5; the other three use 0.05 of the non-zero mean and
  a window of 3.
- **Wide lines.** Python and Rust cut a line too wide for the model into tiles;
  JavaScript and Go squeeze it.

Smaller input differences are in [image-input.md](image-input.md).

## Why the kernels are not aligned yet

On purpose. Bilinear everywhere is the right default, but with no labelled image
in this repository nothing can show that switching Go and JavaScript improves
accuracy rather than changing which line loses its last diacritic. A labelled
set comes first; then align the kernels and measure. Any such change follows the
A/B rule in [CONTRIBUTING.md](../CONTRIBUTING.md).
