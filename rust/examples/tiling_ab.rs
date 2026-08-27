//! Squeezing against tiling, measured on THIS pipeline.
//!
//! `mon_OCR/docs/ROADMAP.md` item 4.5.6 requires that the tiling direction be
//! re-measured on a port before it is trusted, and is explicit about why: "the
//! app segmenters are not the Python one." Until this example existed, no port
//! had ever been measured — the numbers in the doc comment on `predict_page`
//! came from an uncommitted Python harness.
//!
//! This reads a directory of pre-rendered line images with a `labels.txt`
//! (produced by `mon_OCR/scripts/tiling_ab.py --dump-dir`), runs each image
//! through this crate twice — once tiling, once squeezing — and reports the
//! character error rate of each arm. Reading the same images the Python harness
//! scored is the point: it isolates the pipeline as the variable rather than
//! re-rendering and changing two things at once.
//!
//! Usage:
//!     cargo run --release --example tiling_ab -- /tmp/wide-lines

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use monocr_onnx::MonOcr;

/// Grapheme-cluster CER would be the right metric, matching
/// `mon_OCR/src/monocr/metrics.py`. Rust has no grapheme segmenter in this
/// crate's dependency set, so this uses `char`-level edit distance and says so:
/// the two differ on Mon, where a base plus its stacked marks is several chars
/// and one grapheme. The comparison between arms stays valid because both arms
/// are scored the same way; only the absolute rate is not comparable to the
/// Python report.
fn char_cer(pred: &str, reference: &str) -> f64 {
    let a: Vec<char> = pred.chars().collect();
    let b: Vec<char> = reference.chars().collect();
    if b.is_empty() {
        return if a.is_empty() { 0.0 } else { 1.0 };
    }

    let mut prev: Vec<usize> = (0..=a.len()).collect();
    let mut cur = vec![0usize; a.len() + 1];
    for (j, bc) in b.iter().enumerate() {
        cur[0] = j + 1;
        for (i, ac) in a.iter().enumerate() {
            let cost = usize::from(ac != bc);
            cur[i + 1] = (cur[i] + 1).min(prev[i + 1] + 1).min(prev[i] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[a.len()] as f64 / b.len() as f64
}

struct Row {
    tiles: usize,
    cer_tiled: f64,
    cer_squeezed: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .context("usage: tiling_ab <dir with line_*.png and labels.txt>")?
        .into();

    let labels_path = dir.join("labels.txt");
    let labels_raw = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("cannot read {}", labels_path.display()))?;

    let mut labels: Vec<(PathBuf, String)> = Vec::new();
    for line in labels_raw.lines() {
        let Some((name, text)) = line.split_once('\t') else {
            continue;
        };
        labels.push((dir.join(name), text.to_string()));
    }
    if labels.is_empty() {
        anyhow::bail!(
            "{} contained no tab-separated entries",
            labels_path.display()
        );
    }
    eprintln!("{} labelled lines", labels.len());

    // Null baseline, asserted before any real number is produced. If the metric
    // arithmetic is wrong every rate below is wrong in the same direction, and
    // nothing else in this example would reveal it.
    assert_eq!(
        char_cer("", "abc"),
        1.0,
        "empty prediction must score exactly 1.0"
    );
    assert_eq!(
        char_cer("abc", "abc"),
        0.0,
        "identity must score exactly 0.0"
    );
    eprintln!("null baseline ok");

    // Two sessions rather than rebuilding one per arm: the flag is fixed at build
    // time, and reloading the graph 240 times would dominate the runtime.
    let mut tiled = MonOcr::builder().tile_wide_lines(true).build().await?;
    let mut squeezed = MonOcr::builder().tile_wide_lines(false).build().await?;

    let mut rows: Vec<Row> = Vec::new();
    for (i, (path, truth)) in labels.iter().enumerate() {
        let t = tiled.predict_single_line(path).await?;
        let s = squeezed.predict_single_line(path).await?;

        // Recovered from the geometry: a tiled read reports the union of its
        // tiles, so width over the window width is the tile count.
        let img = image::open(path)?.to_luma8();
        let (w, h) = img.dimensions();
        let scaled = (w as f64 * (160.0 / h as f64)) as u32;
        let tiles = ((scaled as f64) / 1024.0).ceil().max(1.0) as usize;

        rows.push(Row {
            tiles,
            cer_tiled: char_cer(&t.text, truth),
            cer_squeezed: char_cer(&s.text, truth),
        });

        if (i + 1) % 25 == 0 {
            eprintln!("  {}/{}", i + 1, labels.len());
        }
    }

    let mean =
        |f: fn(&Row) -> f64, rs: &[Row]| -> f64 { rs.iter().map(f).sum::<f64>() / rs.len() as f64 };
    let m_sq = mean(|r| r.cer_squeezed, &rows);
    let m_ti = mean(|r| r.cer_tiled, &rows);

    println!("\nn                {}", rows.len());
    println!("squeezed CER     {m_sq:.4}");
    println!("tiled CER        {m_ti:.4}");
    println!("ratio sq/tiled   {:.2}x", m_sq / m_ti);
    println!(
        "tiled better on  {}/{}",
        rows.iter().filter(|r| r.cer_tiled < r.cer_squeezed).count(),
        rows.len()
    );

    // The band table is the finding; the aggregate depends entirely on the width
    // mix of whatever sample was handed in.
    let mut bands: BTreeMap<usize, Vec<&Row>> = BTreeMap::new();
    for r in &rows {
        bands.entry(r.tiles).or_default().push(r);
    }
    println!("\n tiles     n   squeezed     tiled    ratio");
    for (band, sub) in &bands {
        if sub.len() < 3 {
            continue;
        }
        let s_m = sub.iter().map(|r| r.cer_squeezed).sum::<f64>() / sub.len() as f64;
        let t_m = sub.iter().map(|r| r.cer_tiled).sum::<f64>() / sub.len() as f64;
        println!(
            " {band:>5}  {:>4}   {s_m:>8.4}  {t_m:>8.4}  {:>6.1}x",
            sub.len(),
            s_m / t_m
        );
    }
    Ok(())
}
