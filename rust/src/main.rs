use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use monocr_onnx::MonOcrBuilder;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "monocr")]
#[command(about = "Mongolian OCR CLI - Convert images/PDFs with Mongolian characters to text", long_about = None)]
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

fn is_pdf(path: &PathBuf) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase() == "pdf")
        .unwrap_or(false)
}

fn get_default_output(input: &PathBuf, format: &OutputFormat) -> PathBuf {
    let ext = match format {
        OutputFormat::Text => "txt",
        OutputFormat::Json => "json",
    };
    let mut output = input.clone();
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

    let result: Vec<String> = if is_pdf_file {
        ocr.read_pdf(&args.input).await?
    } else {
        vec![ocr.read_image(&args.input).await?]
    };

    match args.format {
        OutputFormat::Text => {
            let mut file = File::create(&output_path).context("Failed to create output file")?;
            for (i, page_text) in result.iter().enumerate() {
                if i > 0 {
                    writeln!(file, "\n--- Page {} ---\n", i + 1)?;
                }
                writeln!(file, "{}", page_text)?;
            }
        }
        OutputFormat::Json => {
            let json_output = build_json_output(&result, &args.input, is_pdf_file);
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

fn build_json_output(pages_text: &[String], _input: &PathBuf, _is_pdf: bool) -> JsonOutput {
    let pages: Vec<PageOutput> = pages_text
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let lines: Vec<LineOutput> = text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line_text| LineOutput {
                    text: line_text.to_string(),
                    bbox: BboxOutput {
                        x: 0,
                        y: 0,
                        width: 0,
                        height: 0,
                    },
                })
                .collect();

            PageOutput {
                page: i + 1,
                lines,
                full_text: text.clone(),
            }
        })
        .collect();

    JsonOutput {
        success: true,
        pages,
    }
}
