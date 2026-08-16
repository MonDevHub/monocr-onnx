import importlib.resources
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .model_manager import ModelManager
from .segmenter import LineSegmenter, tile_line

# The input height this binding targets, and the height the pinned artifact was
# traced at. It is checked against the ONNX graph at load; the graph wins for
# preprocessing when it declares a static height.
#
# This is the v3.5 network: H=160, 276 characters, 277 classes,
# MobileNetV3-Large + SE + 2xBiLSTM + attention + CTC. It replaced v2 (H=128,
# 315 characters) at revision d3d9d5e on 2026-08-15. Anything still pinned to
# a51be11 gets v2 and must keep the old numbers — the two are different
# networks, not two versions of one, which is why the expectation is
# cross-checked against the artifact rather than trusted on its own.
EXPECTED_INPUT_HEIGHT = 160

# Width of the training canvas, and — since v3.5 — of the graph itself.
#
# v2 left the width axis dynamic, so a crop of any width could be fed directly.
# v3.5 declares a static 1024 and accepts nothing else: attention fixes the
# sequence length when the graph is traced, so a wider declaration would be a
# promise the artifact cannot keep. Resize and pad to this width before calling.
# Used only when the graph does not declare a static width.
DEFAULT_INPUT_WIDTH = 1024


class ModelContractError(RuntimeError):
    """
    Raised when the model artifact and the charset describe different output spaces.

    The charset, the input height and the classifier width are one contract. If
    they drift apart the model still runs and still returns text — it is just
    the wrong text, with no error anywhere. Fail closed instead.
    """


def _read_charset(text):
    """
    Turn charset file bytes into the index-ordered character string.

    ``.strip("\\n\\r")`` and not ``.strip()``. The charset really does begin
    with U+0020 — a space is a character the model predicts — and a bare strip
    eats it, dropping 276 characters to 275 and shifting every index in
    ``idx2char`` by one. Every decoded glyph then comes out as its neighbour.
    The sibling ``monocr`` package shipped exactly this defect: measured
    2026-08-15, all 315 of its decodable indices returned the wrong character.
    """
    return text.strip("\n\r")


def _static_dim(shape, axis):
    """Return shape[axis] when it is a concrete int, else None (dynamic axis)."""
    if shape is None or len(shape) <= axis:
        return None
    value = shape[axis]
    return value if isinstance(value, int) else None


class MonOCR:
    def __init__(self, model_path=None, charset_path=None):
        manager = None
        if model_path is None:
            manager = ModelManager()
            model_path = manager.get_model_path()

        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.segmenter = LineSegmenter()

        self.charset, charset_source = self._load_charset(charset_path, manager)

        # Read the contract off the graph, not off constants. A constant only
        # says what this code believes; the graph says what will actually run.
        num_classes = _static_dim(self.session.get_outputs()[0].shape, -1)
        model_height = _static_dim(self.session.get_inputs()[0].shape, 2)
        model_width = _static_dim(self.session.get_inputs()[0].shape, 3)

        self._check_charset_contract(num_classes, charset_source)
        self._check_height_contract(model_height)

        self.num_classes = num_classes
        self.input_height = model_height or EXPECTED_INPUT_HEIGHT
        self.input_width = model_width or DEFAULT_INPUT_WIDTH

        self.idx2char = {i + 1: c for i, c in enumerate(self.charset)}
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_charset(self, charset_path, manager):
        """Return (charset, human-readable source) for the contract error message."""
        if charset_path:
            text = Path(charset_path).read_text(encoding="utf-8")
            return _read_charset(text), str(charset_path)

        # Prefer the charset published alongside the pinned weights: same
        # revision, so the two cannot disagree. Falling back to the bundled copy
        # keeps offline and pinned-model-path use working.
        if manager is not None:
            try:
                path = manager.get_charset_path()
                return _read_charset(path.read_text(encoding="utf-8")), str(path)
            except Exception as e:
                # The bundled copy is byte-identical to the pinned one, so this
                # is safe — but say so rather than swallowing it, because a
                # silent switch of charset source is how the two drift apart.
                warnings.warn(
                    f"could not fetch the pinned charset ({e}); using the bundled copy",
                    RuntimeWarning,
                    stacklevel=3,
                )

        try:
            ref = importlib.resources.files("monocr_onnx") / "charset.txt"
            return _read_charset(ref.read_text(encoding="utf-8")), "bundled charset.txt"
        except Exception as e:
            charset_file = Path(__file__).parent / "charset.txt"
            if charset_file.exists():
                return (
                    _read_charset(charset_file.read_text(encoding="utf-8")),
                    str(charset_file),
                )
            raise RuntimeError(f"Charset file not found. Error: {e}")

    def _check_charset_contract(self, num_classes, source):
        """
        CTC reserves index 0 for blank, so num_classes == len(charset) + 1.

        A mismatch means every index above the first divergence decodes to the
        wrong character, and the run looks perfectly healthy while it does it.
        """
        if not self.charset:
            raise ModelContractError(
                f"no charset available for {self.model_path} (looked in {source})"
            )
        if num_classes is None:
            # Nothing to compare against. Decoding still refuses out-of-range
            # indices, so this degrades to a late error rather than a silent one.
            return
        expected = len(self.charset) + 1
        if num_classes != expected:
            raise ModelContractError(
                f"charset/model mismatch.\n"
                f"  charset ({source}): {len(self.charset)} characters -> expects "
                f"num_classes={expected}\n"
                f"  model ({self.model_path}): emits {num_classes} classes -> implies "
                f"{num_classes - 1} characters\n"
                f"These are different output spaces, so every index above the first "
                f"divergence would decode to the wrong character. Either the charset "
                f"belongs to a different model, or the cached model predates a "
                f"charset change — clear ~/.monocr/models and re-download."
            )

    def _check_height_contract(self, model_height):
        """Verify the height this binding preprocesses to is the height the model was traced at."""
        if model_height is None:
            return  # dynamic height axis — nothing to check
        if model_height != EXPECTED_INPUT_HEIGHT:
            raise ModelContractError(
                f"input height mismatch: {self.model_path} expects height="
                f"{model_height} but this binding targets {EXPECTED_INPUT_HEIGHT}. "
                f"A different height means a different network, not a resize."
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def preprocess(self, img):
        """
        Scale a line crop to the model's input height and right-pad it with white.

        Mirrors ``resize_and_pad`` in the training pipeline: aspect ratio is
        preserved while the height-scaled width still fits the canvas, then the
        image is squeezed horizontally rather than truncated, and the result is
        normalised to [-1, 1].

        The old code divided by 255, producing [0, 1]. Every pixel then reached
        the network offset by +1.0 and at half the intended scale — the network
        was trained on white=+1.0 / black=-1.0 and was being fed white=1.0 /
        black=0.0. The JS, Go and Rust bindings all use /127.5 - 1.0.
        """
        if img.mode != "L":
            img = img.convert("L")

        if img.height == 0 or img.width == 0:
            return None

        target_h, target_w = self.input_height, self.input_width
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        scale = target_h / h
        new_w = max(1, min(int(w * scale), target_w))
        resized = cv2.resize(arr, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        # White canvas (255 = background), matching the padding used in training.
        canvas = np.full((target_h, target_w), 255.0, dtype=np.float32)
        canvas[:, :new_w] = resized

        normalized = canvas / 127.5 - 1.0
        return normalized[np.newaxis, np.newaxis, :, :]

    def decode(self, preds):
        """Greedy CTC decode: drop blanks (index 0) and collapse repeats."""
        decoded_text = []
        prev_idx = -1

        for idx in preds:
            idx = int(idx)
            if idx != 0 and idx != prev_idx:
                if idx not in self.idx2char:
                    # Only reachable when the graph's class axis is dynamic, so
                    # the load-time contract check had nothing to compare. The
                    # old code did `.get(idx, "")` here and dropped the
                    # character on the floor — the model's output was being
                    # edited to fit the charset, silently.
                    raise ModelContractError(
                        f"class index {idx} is outside the {len(self.charset)}-character "
                        f"charset. The model and the charset do not match."
                    )
                decoded_text.append(self.idx2char[idx])
            prev_idx = idx

        return "".join(decoded_text)

    def predict_line(self, img):
        if isinstance(img, (str, Path)):
            img = Image.open(img)

        input_data = self.preprocess(img)
        if input_data is None:
            return ""

        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        preds = np.argmax(outputs[0], axis=2)[0]  # Batch size 1

        return self.decode(preds)

    def predict_page(self, img_path):
        """
        Segment page into lines and predict each line.

        A line wider than the canvas after being scaled to the model height is
        cut into canvas-width tiles at whitespace columns, read separately, and
        rejoined with no separator. Squeezing it into the canvas instead breaks
        the aspect ratio the model was trained on.

        Which of the two is better is a property of the network, not of the
        method. MEASURED 2026-08-15 with one harness over 240 rendered Mon lines
        wide enough to need the choice, median 3 model windows each, swapping
        only the graph:

            v2     squeezed 0.0676   tiled 0.0758    CER, squeezing better
            v3.5   squeezed 0.1434   tiled 0.0795    CER, tiling better

        This package pins v3.5, so it tiles. Anything repinned to v2 at a51be11
        should not: re-measure before changing the pin, because the direction
        flips.

        Rendered lines rather than photographed pages, so this is a preview and
        not an evaluation; the two arms differ only in the choice under test.
        """
        if isinstance(img_path, (str, Path)):
            img = Image.open(img_path)
        else:
            img = img_path

        lines = self.segmenter.segment(img)

        if not lines:
            # Segmentation found no horizontal bands. Either the page is blank
            # or it is a single line the projection profile did not split, so
            # try it as one line before giving up.
            return self.predict_line(img)

        return "\n".join(self._read_line_tiled(line["img"]) for line in lines)

    def _read_line_tiled(self, crop):
        """Read one line, tiling it first if it will not fit the model window.

        Tiles join with no separator: the cut falls inside a word by
        construction, since cut_column searches for the *quietest* column rather
        than a word boundary. A space here would insert one that is not there.
        """
        tiles = tile_line(crop, self.input_height, self.input_width)
        if len(tiles) == 1:
            return self.predict_line(tiles[0])
        return "".join(self.predict_line(tile) for tile in tiles)

    def predict(self, img_path):
        # Alias for backward compatibility or ease of use
        return self.predict_page(img_path)
