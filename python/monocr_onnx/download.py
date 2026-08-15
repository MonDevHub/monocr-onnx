"""
``monocr-download`` — fetch the pinned model artifacts into a local directory.

Every URL here used to point at ``huggingface.co/janakh/monocr``. The org is
``janakhpon``; ``janakh`` returns 401, so the console script shipped in 0.1.0
fails on all three files. The ``tflite/`` directory is also gone — the pinned
revision is the commit that deleted it — so that entry is not resurrected here.
"""

import argparse
from pathlib import Path

from .model_manager import ModelManager


def main():
    manager = ModelManager()
    parser = argparse.ArgumentParser(
        description=(
            f"Download the MonOCR ONNX model and its charset "
            f"({manager.REPO_ID}@{manager.REVISION})"
        )
    )
    parser.add_argument("--dest", type=str, default="model", help="Destination directory")
    args = parser.parse_args()

    dest_path = Path(args.dest)

    # Go through the manager so both files come from the same pinned revision
    # and are checksummed, then copy them out to the requested directory.
    for filename in (manager.MODEL_FILENAME, manager.CHARSET_FILENAME):
        cached = manager.ensure(filename)
        target = dest_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != cached.resolve():
            target.write_bytes(cached.read_bytes())
        print(f"{filename} -> {target}")


if __name__ == "__main__":
    main()
