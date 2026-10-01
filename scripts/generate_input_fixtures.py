#!/usr/bin/env python3
"""Generate the image-loading fixtures under data/fixtures/input/.

All four bindings read these files, so a binding that loads an image
differently from the others fails its own test instead of returning different
text. Run from the repository root:

    uv run --project python python scripts/generate_input_fixtures.py

Only Pillow and NumPy are needed. The output is deterministic, so rerunning the
script on an unchanged checkout leaves `git status` clean.

What is generated, and what each binding's tests assert about it:

upright.png
    A 64x32 greyscale image, dark strokes on white, that looks different under
    each of the eight EXIF orientations (no mirror or rotation of it equals
    another). It is the reference picture.

orient-1.jpg ... orient-8.jpg
    One greyscale JPEG per EXIF orientation tag. Each file stores the pixels
    the tag says to transform, so that applying the tag gives `upright.png`
    back. A loader that honours the tag returns the upright picture for all
    eight; one that ignores it returns a mirrored or rotated picture for 2-8,
    and for 5-8 a 32x64 one.

orient-6-be.jpg
    Orientation 6 again, with a big-endian ("MM") TIFF header. Pillow writes
    little-endian ("II"), and the Go and Rust bindings parse the tag by hand,
    so both byte orders are covered.

orient-6.png
    Orientation 6 in a PNG eXIf chunk, the other place a phone or editor puts
    the tag. Lossless, so it must match `upright.png` exactly.

    The JPEGs are quality 100 and greyscale, so the only difference from
    `upright.png` is a few levels of rounding; tests compare with a tolerance
    far below the difference a wrong orientation makes.

alpha-text.png, alpha-text.expected.png
    An RGBA PNG: fully transparent (0,0,0,0) background, opaque dark strokes,
    and one stroke at alpha 128. The expected file is the greyscale result of
    compositing it onto white: background 255, opaque ink unchanged, the
    half-transparent stroke about halfway to white. Converting to grey without
    compositing turns the background black, which is what the bindings did.

alpha-opaque.png
    `upright.png` as RGBA with alpha 255 everywhere. Compositing must leave it
    byte-identical to its plain greyscale conversion.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "input"

WIDTH, HEIGHT = 64, 32
INK = 24

# Pillow transpose that produces the STORED pixels for each tag, i.e. the
# inverse of the transform the tag asks a viewer to apply. Tags 2, 3, 4, 5 and
# 7 are their own inverse; 6 and 8 are each other's.
STORE = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def upright() -> Image.Image:
    """Strokes that make every one of the eight orientations distinguishable."""
    a = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    a[4:28, 4:8] = INK  # tall bar on the left
    a[4:8, 4:24] = INK  # top arm, long
    a[14:18, 4:16] = INK  # middle arm, short: an "F"
    a[20:28, 32:36] = INK  # a short bar, low in the middle
    a[4:12, 48:60] = INK  # a block, top right
    a[24:28, 44:60] = INK  # a base line, bottom right
    return Image.fromarray(a, mode="L")


def exif_orientation(tag: int, big_endian: bool = False) -> bytes:
    """A minimal APP1 payload: TIFF header, IFD0 with the Orientation tag only."""
    e = ">" if big_endian else "<"
    tiff = (b"MM" if big_endian else b"II") + struct.pack(e + "HI", 42, 8)
    # One entry: tag 0x0112, type SHORT (3), count 1, value left-justified.
    tiff += struct.pack(e + "H", 1)
    tiff += struct.pack(e + "HHI", 0x0112, 3, 1) + struct.pack(e + "HH", tag, 0)
    tiff += struct.pack(e + "I", 0)  # no next IFD
    return b"Exif\x00\x00" + tiff


def save_jpeg(img: Image.Image, path: Path, tag: int, big_endian: bool = False) -> None:
    img.save(path, "JPEG", quality=100, subsampling=0, exif=exif_orientation(tag, big_endian))
    check_upright(path)


def check_upright(path: Path) -> None:
    """Self-check: Pillow's own EXIF transpose must give the reference back."""
    check = ImageOps.exif_transpose(Image.open(path)).convert("L")
    ref = np.asarray(upright(), dtype=np.int16)
    got = np.asarray(check, dtype=np.int16)
    assert got.shape == ref.shape, (path.name, got.shape)
    assert int(np.abs(got - ref).max()) <= 8, path.name


def alpha_text() -> tuple[Image.Image, Image.Image]:
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)  # (0,0,0,0) everywhere
    rgba[6:26, 6:10] = (INK, INK, INK, 255)
    rgba[6:10, 6:30] = (INK, INK, INK, 255)
    rgba[16:20, 6:22] = (INK, INK, INK, 255)
    rgba[6:26, 40:44] = (0, 0, 0, 128)  # half-transparent black stroke

    rgb = rgba[..., :3].astype(np.float64)
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    flat = np.rint(rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)
    expected = Image.fromarray(flat, mode="RGB").convert("L")
    return Image.fromarray(rgba, mode="RGBA"), expected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    up = upright()
    up.save(OUT / "upright.png", optimize=False)

    for tag, op in STORE.items():
        stored = up if op is None else up.transpose(op)
        save_jpeg(stored, OUT / f"orient-{tag}.jpg", tag)
    save_jpeg(up.transpose(STORE[6]), OUT / "orient-6-be.jpg", 6, big_endian=True)
    # PNG's eXIf chunk holds the bare TIFF structure, without the "Exif\0\0"
    # prefix a JPEG APP1 segment carries.
    up.transpose(STORE[6]).save(OUT / "orient-6.png", exif=exif_orientation(6)[6:])
    check_upright(OUT / "orient-6.png")

    rgba, expected = alpha_text()
    rgba.save(OUT / "alpha-text.png")
    expected.save(OUT / "alpha-text.expected.png")

    up.convert("RGBA").save(OUT / "alpha-opaque.png")

    for p in sorted(OUT.iterdir()):
        print(p.relative_to(OUT.parent.parent.parent))


if __name__ == "__main__":
    main()
