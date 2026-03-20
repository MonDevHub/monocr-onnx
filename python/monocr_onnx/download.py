import argparse
import requests
import os
from pathlib import Path

MODEL_URLS = {
    "onnx": "https://huggingface.co/janakh/monocr/resolve/main/onnx/monocr.onnx",
    "tflite": "https://huggingface.co/janakh/monocr/resolve/main/tflite/monocr.tflite",
    "charset": "https://huggingface.co/janakh/monocr/resolve/main/charset.txt",
}

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Done.")

def main():
    parser = argparse.ArgumentParser(description="Download MonOCR models")
    parser.add_argument("--dest", type=str, default="model", help="Destination directory")
    args = parser.parse_args()
    
    dest_path = Path(args.dest)
    
    for name, url in MODEL_URLS.items():
        ext = ".txt" if name == "charset" else f".{name}"
        file_dest = dest_path / f"monocr{ext}" if name != "charset" else dest_path / "charset.txt"
        download_file(url, file_dest)

if __name__ == "__main__":
    main()
