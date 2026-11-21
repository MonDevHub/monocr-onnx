use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use monocr_onnx::{page_text, LineResult, MonOcrBuilder};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "monocr")]
#[command(about = "Mon OCR CLI - Convert images/PDFs containing Mon text to plain text", long_about = None)]
struct Args {
    /// Input file (image: PNG, JPG, BMP, etc. or PDF)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file path (default: <input>.txt or <input>.json)
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Output format
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Replace the output file if it already exists
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    /// Plain text output
    Text,
    /// JSON output with line details and bounding boxes
    Json,
}

#[derive(Serialize)]
struct JsonOutput {
    pub success: bool,
    pub pages: Vec<PageOutput>,
}

#[derive(Serialize)]
struct PageOutput {
    pub page: usize,
    pub lines: Vec<LineOutput>,
    pub full_text: String,
}

#[derive(Serialize)]
struct LineOutput {
    pub text: String,
    pub bbox: BboxOutput,
}

#[derive(Serialize)]
struct BboxOutput {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase() == "pdf")
        .unwrap_or(false)
}

/// Refuse an output path that would destroy a file the user has.
///
/// The default output is the input with its extension swapped, so `monocr
/// scan.png` silently replaced a `scan.txt` already beside it, and `monocr
/// notes.txt` would write its result over the input it was reading. An
/// existing file is replaced only with `--force`; the input never is.
fn check_output(input: &Path, output: &Path, force: bool) -> Result<()> {
    if let (Ok(a), Ok(b)) = (input.canonicalize(), output.canonicalize()) {
        if a == b {
            anyhow::bail!(
                "Output {:?} is the input file; pass --output to choose another path",
                output
            );
        }
    }
    if output.exists() && !force {
        anyhow::bail!(
            "Output file already exists: {:?}. Pass --force to replace it, or --output to choose another path",
            output
        );
    }
    Ok(())
}

fn get_default_output(input: &Path, format: &OutputFormat) -> PathBuf {
    let ext = match format {
        OutputFormat::Text => "txt",
        OutputFormat::Json => "json",
    };
    let mut output = input.to_path_buf();
    output.set_extension(ext);
    output
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("Input file: {:?}", args.input);
        println!("Format: {:?}", args.format);
    }

    if !args.input.exists() {
        anyhow::bail!("Input file does not exist: {:?}", args.input);
    }

    let output_path = args
        .output
        .unwrap_or_else(|| get_default_output(&args.input, &args.format));

    // Before the model loads, so a refusal costs nothing.
    check_output(&args.input, &output_path, args.force)?;

    if args.verbose {
        println!("Output file: {:?}", output_path);
    }

    if args.verbose {
        println!("Initializing OCR model...");
    }

    let mut ocr = MonOcrBuilder::new()
        .build()
        .await
        .context("Failed to build OCR engine")?;

    let is_pdf_file = is_pdf(&args.input);

    if args.verbose {
        if is_pdf_file {
            println!("Processing PDF file...");
        } else {
            println!("Processing image file...");
        }
    }

    // Keep the per-line results, not just the joined text: the JSON output
    // reports each line's bounding box, and re-deriving lines by splitting the
    // page text throws that geometry away.
    let pages: Vec<Vec<LineResult>> = if is_pdf_file {
        ocr.predict_pdf(&args.input).await?
    } else {
        vec![ocr.predict_page(&args.input).await?]
    };

    match args.format {
        OutputFormat::Text => {
            let mut file = File::create(&output_path).context("Failed to create output file")?;
            for (i, lines) in pages.iter().enumerate() {
                if i > 0 {
                    writeln!(file, "\n--- Page {} ---\n", i + 1)?;
                }
                writeln!(file, "{}", page_text(lines))?;
            }
        }
        OutputFormat::Json => {
            let json_output = build_json_output(&pages);
            let json_str =
                serde_json::to_string_pretty(&json_output).context("Failed to serialize JSON")?;
            let mut file = File::create(&output_path).context("Failed to create output file")?;
            file.write_all(json_str.as_bytes())?;
        }
    }

    if args.verbose {
        println!("OCR completed successfully!");
    }

    println!("Output written to: {:?}", output_path);

    Ok(())
}

/// Serialize the OCR results, keeping each line's real geometry.
///
/// Every bbox used to be reported as all zeroes even though the library returns
/// real coordinates, which made the JSON format useless for anything that needs
/// to point back at the page. For a line that was tiled, the bbox is the union of
/// its tiles, so it still describes the line the text came from.
fn build_json_output(pages: &[Vec<LineResult>]) -> JsonOutput {
    let pages: Vec<PageOutput> = pages
        .iter()
        .enumerate()
        .map(|(i, lines)| {
            let line_outputs: Vec<LineOutput> = lines
                .iter()
                .filter(|line| !line.text.trim().is_empty())
                .map(|line| LineOutput {
                    text: line.text.clone(),
                    bbox: BboxOutput {
                        x: line.bbox.x,
                        y: line.bbox.y,
                        width: line.bbox.w,
                        height: line.bbox.h,
                    },
                })
                .collect();

            PageOutput {
                page: i + 1,
                lines: line_outputs,
                full_text: page_text(lines),
            }
        })
        .collect();

    JsonOutput {
        success: true,
        pages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_existing_output_is_refused_without_force() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("scan.png");
        let output = dir.path().join("scan.txt");
        std::fs::write(&input, b"x").unwrap();
        std::fs::write(&output, b"the user's own notes").unwrap();

        let err = check_output(&input, &output, false).unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
        assert!(check_output(&input, &output, true).is_ok());
    }

    #[test]
    fn a_new_output_is_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("scan.png");
        std::fs::write(&input, b"x").unwrap();
        assert!(check_output(&input, &dir.path().join("scan.txt"), false).is_ok());
    }

    #[test]
    fn the_input_is_never_the_output_even_with_force() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("notes.txt");
        std::fs::write(&input, b"x").unwrap();
        let output = get_default_output(&input, &OutputFormat::Text);

        assert!(check_output(&input, &output, true).is_err());
    }
}
