//! Reading an image file the way it is displayed.
//!
//! Two things the `image` crate (0.24) leaves to the caller, and this crate
//! used to leave undone:
//!
//! - **EXIF orientation.** `image::open` returns the stored pixels. A phone
//!   photo stored sideways, with an Orientation tag of 2-8 saying how to show
//!   it, was read mirrored or rotated. The tag is read here from a JPEG APP1
//!   segment or a PNG eXIf chunk and applied.
//! - **Transparency.** `to_luma8` drops the alpha channel, so a transparent
//!   background stored as (0, 0, 0, 0) — the common encoding — read as black,
//!   and dark text on it vanished. The web and mobile apps flatten onto white
//!   before reading; this does the same.
//!
//! An opaque image with no tag takes exactly the path it always took, so its
//! pixels are unchanged. Only files this crate decodes are oriented; the other
//! three bindings make the same split.

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use std::path::Path;

/// Decode `path`, apply its EXIF orientation, and convert it to greyscale,
/// compositing onto white first if any pixel is not fully opaque.
pub(crate) fn load_grey(path: &Path) -> Result<GrayImage> {
    let img = image::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    // Read separately so the decode stays `image::open`, format chosen exactly
    // as before. A file that cannot be re-read has no tag to apply.
    let orientation = std::fs::read(path)
        .map(|data| exif_orientation(&data))
        .unwrap_or(1);
    Ok(to_grey(&apply_orientation(img, orientation)))
}

/// The EXIF Orientation tag (1-8) of an encoded JPEG or PNG, or 1 when there is
/// none or it cannot be read. A malformed tag is treated as absent rather than
/// as an error, so a file that decoded before still does.
pub(crate) fn exif_orientation(data: &[u8]) -> u16 {
    let tiff = if data.starts_with(&[0xFF, 0xD8]) {
        jpeg_exif(data)
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        png_exif(data)
    } else {
        None
    };
    tiff.and_then(tiff_orientation)
        .filter(|o| (1..=8).contains(o))
        .unwrap_or(1)
}

/// The TIFF structure inside the first APP1 "Exif" segment.
fn jpeg_exif(data: &[u8]) -> Option<&[u8]> {
    let mut i = 2;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            return None;
        }
        let marker = data[i + 1];
        match marker {
            0xFF => {
                i += 1; // fill byte
                continue;
            }
            0x01 | 0xD0..=0xD7 => {
                i += 2; // no length field
                continue;
            }
            0xDA | 0xD9 => return None, // start of scan, end of image
            _ => {}
        }
        let n = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        if n < 2 || i + 2 + n > data.len() {
            return None;
        }
        let seg = &data[i + 4..i + 2 + n];
        if marker == 0xE1 && seg.starts_with(b"Exif\0\0") {
            return Some(&seg[6..]);
        }
        i += 2 + n;
    }
    None
}

/// The contents of the eXIf chunk: a bare TIFF structure, no "Exif\0\0" prefix.
fn png_exif(data: &[u8]) -> Option<&[u8]> {
    let mut i = 8;
    while i + 12 <= data.len() {
        let n = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        let typ = &data[i + 4..i + 8];
        // Subtracted rather than added, so a length near 2^32 cannot overflow a
        // 32-bit usize past the bounds check.
        if n > data.len() - i - 12 {
            return None;
        }
        if typ == b"eXIf" {
            return Some(&data[i + 8..i + 8 + n]);
        }
        if typ == b"IDAT" || typ == b"IEND" {
            return None; // eXIf must precede IDAT
        }
        i += 12 + n;
    }
    None
}

/// Tag 0x0112 from IFD0.
fn tiff_orientation(t: &[u8]) -> Option<u16> {
    if t.len() < 8 {
        return None;
    }
    let big = match &t[..2] {
        b"II" => false,
        b"MM" => true,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = [*t.get(o)?, *t.get(o + 1)?];
        Some(if big {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = [*t.get(o)?, *t.get(o + 1)?, *t.get(o + 2)?, *t.get(o + 3)?];
        Some(if big {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = u32_at(4)? as usize;
    // Bounded first, so the offsets below cannot overflow a 32-bit usize. An
    // offset inside the 8-byte header is refused, as the Go parser does.
    if ifd < 8 || ifd > t.len() {
        return None;
    }
    let count = u16_at(ifd)? as usize;
    for k in 0..count {
        let e = ifd + 2 + 12 * k;
        if u16_at(e)? != 0x0112 {
            continue;
        }
        // The standard type is SHORT (3). Some writers use LONG (4), and
        // Pillow and libvips both accept it, so this does too.
        return match u16_at(e + 2)? {
            3 => u16_at(e + 8),
            4 => u32_at(e + 8).and_then(|v| u16::try_from(v).ok()),
            _ => None,
        };
    }
    None
}

/// Transform `img` the way EXIF Orientation tag `orientation` says to display
/// it. Tag 1, or any value outside 2-8, returns `img` untouched. Pixels are
/// moved, never resampled, and the pixel type is kept.
pub(crate) fn apply_orientation(img: DynamicImage, orientation: u16) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img,
    }
}

/// Greyscale, composited onto white when any pixel is not fully opaque.
///
/// Only an image that actually has a transparent pixel is composited. An
/// opaque one — including RGBA with alpha 255 everywhere — takes `to_luma8`
/// directly, as it always did, so its bytes are unchanged.
pub(crate) fn to_grey(img: &DynamicImage) -> GrayImage {
    if !img.color().has_alpha() {
        return img.to_luma8();
    }
    let rgba = img.to_rgba8();
    if rgba.pixels().all(|p| p[3] == u8::MAX) {
        return img.to_luma8();
    }
    let mut flat = RgbImage::new(rgba.width(), rgba.height());
    for (out, p) in flat.pixels_mut().zip(rgba.pixels()) {
        let a = p[3] as u32;
        for c in 0..3 {
            out[c] = ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        }
    }
    DynamicImage::ImageRgb8(flat).to_luma8()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monocr::{preprocess_line, DEFAULT_INPUT_WIDTH, EXPECTED_INPUT_HEIGHT};
    use ndarray::Array4;
    use std::path::PathBuf;

    fn model_input(img: &GrayImage) -> Array4<f32> {
        preprocess_line(img, EXPECTED_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH)
    }

    // The fixtures are shared by all four bindings and generated by
    // scripts/generate_input_fixtures.py; its docstring describes each one.
    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../data/fixtures/input")
            .join(name)
    }

    const ORIENTED: [&str; 10] = [
        "orient-1.jpg",
        "orient-2.jpg",
        "orient-3.jpg",
        "orient-4.jpg",
        "orient-5.jpg",
        "orient-6.jpg",
        "orient-7.jpg",
        "orient-8.jpg",
        "orient-6-be.jpg",
        "orient-6.png",
    ];

    // JPEG rounding on these block-aligned quality-100 fixtures is a few levels
    // at most; a wrong orientation differs by 36 or more on average, or changes
    // the shape.
    const TOLERANCE: i32 = 8;

    fn max_diff(a: &GrayImage, b: &GrayImage) -> i32 {
        a.as_raw()
            .iter()
            .zip(b.as_raw())
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap_or(0)
    }

    fn old_path(path: &Path) -> GrayImage {
        image::open(path).expect("decodes").to_luma8()
    }

    #[test]
    fn every_exif_orientation_loads_upright() {
        let up = old_path(&fixture("upright.png"));
        for name in ORIENTED {
            let got = load_grey(&fixture(name)).expect("loads");
            assert_eq!(got.dimensions(), up.dimensions(), "{name}");
            assert!(max_diff(&got, &up) <= TOLERANCE, "{name}");
        }
    }

    /// Guards the fixtures: decoding without the tag gives a different picture.
    #[test]
    fn the_fixtures_are_not_upright_without_the_tag() {
        let up = old_path(&fixture("upright.png"));
        for name in &ORIENTED[1..] {
            let raw = old_path(&fixture(name));
            assert!(
                raw.dimensions() != up.dimensions() || max_diff(&raw, &up) > TOLERANCE,
                "{name} decodes upright without its tag"
            );
        }
    }

    #[test]
    fn exif_orientation_reads_each_tag_and_byte_order() {
        for n in 1..=8u16 {
            let data = std::fs::read(fixture(&format!("orient-{n}.jpg"))).unwrap();
            assert_eq!(exif_orientation(&data), n);
        }
        for (name, want) in [
            ("orient-6-be.jpg", 6),
            ("orient-6.png", 6),
            ("upright.png", 1),
            ("alpha-text.png", 1),
        ] {
            let data = std::fs::read(fixture(name)).unwrap();
            assert_eq!(exif_orientation(&data), want, "{name}");
        }
    }

    /// A malformed or truncated header is read as "no tag", never as a panic.
    #[test]
    fn exif_orientation_tolerates_garbage() {
        let data = std::fs::read(fixture("orient-6.jpg")).unwrap();
        for n in 0..data.len().min(200) {
            exif_orientation(&data[..n]);
        }
        for junk in [
            &b""[..],
            &[0xFF],
            &[0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF],
            b"\x89PNG\r\n\x1a\n\xff\xff\xff\xff",
        ] {
            assert_eq!(exif_orientation(junk), 1);
        }
    }

    /// A minimal TIFF header with one IFD0 entry for tag 0x0112.
    fn tiff(big: bool, typ: u16, value: u32) -> Vec<u8> {
        let p16 = |v: u16| {
            if big {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let p32 = |v: u32| {
            if big {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let mut b = Vec::new();
        b.extend_from_slice(if big { b"MM" } else { b"II" });
        b.extend_from_slice(&p16(42));
        b.extend_from_slice(&p32(8));
        b.extend_from_slice(&p16(1));
        b.extend_from_slice(&p16(0x0112));
        b.extend_from_slice(&p16(typ));
        b.extend_from_slice(&p32(1));
        if typ == 3 {
            b.extend_from_slice(&p16(value as u16));
            b.extend_from_slice(&[0, 0]);
        } else {
            b.extend_from_slice(&p32(value));
        }
        b.extend_from_slice(&p32(0));
        b
    }

    #[test]
    fn tiff_orientation_accepts_short_and_long() {
        for big in [false, true] {
            assert_eq!(tiff_orientation(&tiff(big, 3, 6)), Some(6));
            assert_eq!(tiff_orientation(&tiff(big, 4, 8)), Some(8));
            assert_eq!(tiff_orientation(&tiff(big, 4, 1 << 31)), None);
            assert_eq!(tiff_orientation(&tiff(big, 7, 6)), None);
        }
        // An IFD offset near 2^32 must be refused, not wrapped.
        let mut b = tiff(false, 3, 6);
        b[4..8].copy_from_slice(&0xFFFF_FFFEu32.to_le_bytes());
        assert_eq!(tiff_orientation(&b), None);
        // An IFD offset inside the header is refused.
        let mut b = tiff(false, 3, 6);
        b[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(tiff_orientation(&b), None);
    }

    /// The walk has to step over fill bytes and an APP1 that is not Exif (XMP
    /// is the common one) to reach the Exif segment behind them.
    #[test]
    fn jpeg_exif_skips_fill_bytes_and_other_app1_segments() {
        let data = std::fs::read(fixture("orient-6.jpg")).unwrap();
        let xmp = b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta/>";
        let mut b = data[..2].to_vec(); // SOI
        b.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xE1]); // fill bytes, then APP1
        b.extend_from_slice(&((xmp.len() + 2) as u16).to_be_bytes());
        b.extend_from_slice(xmp);
        b.extend_from_slice(&data[2..]);
        assert_eq!(exif_orientation(&b), 6);
    }

    #[test]
    fn the_model_input_is_the_same_for_every_orientation() {
        let want = model_input(&old_path(&fixture("upright.png")));
        for name in ORIENTED {
            let got = model_input(&load_grey(&fixture(name)).unwrap());
            let d = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(d <= TOLERANCE as f32 / 127.5, "{name}: {d}");
        }
    }

    #[test]
    fn transparency_is_composited_onto_white() {
        let want = old_path(&fixture("alpha-text.expected.png"));
        let got = load_grey(&fixture("alpha-text.png")).unwrap();
        assert_eq!(got.get_pixel(0, 0)[0], 255);
        assert!(max_diff(&got, &want) <= 1, "{}", max_diff(&got, &want));
    }

    /// Guards the fixture: the conversion this replaces turns it black.
    #[test]
    fn without_compositing_the_background_is_black() {
        assert_eq!(old_path(&fixture("alpha-text.png")).get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn a_transparent_line_reaches_the_model_as_dark_on_white() {
        let want = model_input(&old_path(&fixture("alpha-text.expected.png")));
        let got = model_input(&load_grey(&fixture("alpha-text.png")).unwrap());
        let d = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d <= 1.5 / 127.5, "{d}");
    }

    /// The old path was `image::open(path)?.to_luma8()`.
    #[test]
    fn opaque_untagged_input_is_byte_identical_to_the_old_path() {
        let images = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/images");
        let mut paths: Vec<PathBuf> = std::fs::read_dir(images)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        paths.sort();
        assert!(paths.len() >= 7);
        for f in ["upright.png", "alpha-opaque.png", "orient-1.jpg"] {
            paths.push(fixture(f));
        }
        for p in paths {
            assert_eq!(
                load_grey(&p).unwrap().as_raw(),
                old_path(&p).as_raw(),
                "{}",
                p.display()
            );
        }
    }

    #[test]
    fn opaque_rgba_is_not_altered_by_compositing() {
        let a = load_grey(&fixture("alpha-opaque.png")).unwrap();
        let b = old_path(&fixture("upright.png"));
        assert_eq!(a.as_raw(), b.as_raw());
    }

    /// Two lines of blobs; on a (0, 0, 0, 0) background when `transparent`.
    fn blob_page(transparent: bool) -> DynamicImage {
        let ink = |x: u32, y: u32| {
            [30u32, 110].iter().any(|&top| {
                (top..top + 40).contains(&y)
                    && (60..60 + 20 * 24).contains(&x)
                    && (x - 60) % 24 < 14
            })
        };
        if transparent {
            DynamicImage::ImageRgba8(image::RgbaImage::from_fn(600, 200, |x, y| {
                image::Rgba([0, 0, 0, if ink(x, y) { 255 } else { 0 }])
            }))
        } else {
            DynamicImage::ImageLuma8(GrayImage::from_fn(600, 200, |x, y| {
                image::Luma([if ink(x, y) { 0 } else { 255 }])
            }))
        }
    }

    /// `LineSegmenter::segment` decodes a path itself, so it reads it the same
    /// way. A transparent background used to read as black, which its `< 128`
    /// threshold calls ink.
    #[test]
    fn the_segmenter_reads_a_transparent_page_as_dark_on_white() {
        let dir = tempfile::tempdir().unwrap();
        let (opaque, transparent) = (dir.path().join("o.png"), dir.path().join("t.png"));
        blob_page(false).save(&opaque).unwrap();
        blob_page(true).save(&transparent).unwrap();
        let seg = crate::segmenter::LineSegmenter::new(10, 3);
        let boxes = |p: &Path| -> Vec<(u32, u32, u32, u32)> {
            seg.segment(p)
                .unwrap()
                .iter()
                .map(|l| (l.bbox.x, l.bbox.y, l.bbox.w, l.bbox.h))
                .collect()
        };
        let want = boxes(&opaque);
        assert_eq!(want.len(), 2);
        assert_eq!(boxes(&transparent), want);
    }
}
