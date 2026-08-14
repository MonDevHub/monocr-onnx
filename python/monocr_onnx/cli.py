import click
from pathlib import Path
from .ocr import read_image, read_pdf
from .model_manager import ModelManager

@click.group()
def main():
    """MonOCR Command Line Interface"""
    pass

@main.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', default=None, help='Path to ONNX model (optional)')
@click.option('--charset', '-c', default=None, help='Path to charset file (optional)')
def image(image_path, model, charset):
    """Recognize text from an image file."""
    try:
        text = read_image(image_path, model, charset)
        click.echo(text)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)

@main.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--model', '-m', default=None, help='Path to ONNX model (optional)')
@click.option('--charset', '-c', default=None, help='Path to charset file (optional)')
def pdf(pdf_path, model, charset):
    """Recognize text from a PDF file."""
    try:
        pages = read_pdf(pdf_path, model, charset)
        for i, text in enumerate(pages):
            click.echo(f"--- Page {i+1} ---")
            click.echo(text)
            click.echo()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)

@main.command()
@click.option('--force', '-f', is_flag=True, help='Force re-download')
def download(force):
    """Download model files to local cache."""
    manager = ModelManager()
    if force:
        manager.clear_cache()

    try:
        # Both, so the charset always comes from the same revision as the weights.
        manager.get_model_path()
        manager.get_charset_path()
        click.echo(f"Model is ready ({manager.REPO_ID}@{manager.REVISION}): {manager.cache_dir}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)

    legacy = manager.legacy_cache_files()
    if legacy:
        click.echo(
            "Note: unpinned files from an older release are still in the cache and "
            "are no longer read. They may be a different network entirely. Safe to delete:",
            err=True,
        )
        for path in legacy:
            click.echo(f"  {path}", err=True)

@main.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--model', '-m', default=None, help='Path to ONNX model')
@click.option('--charset', '-c', default=None, help='Path to charset')
def batch(directory, model, charset):
    """Process all images in a directory."""
    path = Path(directory)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    files = [f for f in path.iterdir() if f.suffix.lower() in extensions]
    
    if not files:
        click.echo("No images found in directory.")
        return

    from .predictor import MonOCR
    ocr = MonOCR(model, charset)
    
    for file in sorted(files):
        click.echo(f"Processing {file.name}...", err=True)
        try:
            text = ocr.predict(file)
            click.echo(f"--- {file.name} ---")
            click.echo(text)
            click.echo()
        except Exception as e:
             click.echo(f"Failed to process {file.name}: {e}", err=True)

if __name__ == '__main__':
    main()
