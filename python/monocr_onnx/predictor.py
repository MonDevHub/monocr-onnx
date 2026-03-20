import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
import importlib.resources
from .model_manager import ModelManager
from .segmenter import LineSegmenter

class MonOCR:
    def __init__(self, model_path=None, charset_path=None):
        if model_path is None:
            manager = ModelManager()
            model_path = manager.get_model_path()
            
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.segmenter = LineSegmenter()
        
        if charset_path:
            with open(charset_path, 'r', encoding='utf-8') as f:
                self.charset = f.read().strip()
        else:
            # Load bundled charset
            try:
                # Python 3.9+ resource loading
                ref = importlib.resources.files('monocr_onnx') / 'charset.txt'
                with ref.open('r', encoding='utf-8') as f:
                    self.charset = f.read().strip()
            except Exception as e:
                # Fallback for older python or loose script usage
                charset_file = Path(__file__).parent / 'charset.txt'
                if charset_file.exists():
                     with open(charset_file, 'r', encoding='utf-8') as f:
                        self.charset = f.read().strip()
                else:
                    raise RuntimeError(f"Charset file not found. Error: {e}")

    def preprocess(self, img):
        if img.mode != 'L':
            img = img.convert('L')
        
        target_height = 128
        # Handle small images
        if img.height == 0:
             return None
             
        aspect_ratio = img.width / img.height
        target_width = int(target_height * aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
        
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0) # Add channel dim
        img_arr = np.expand_dims(img_arr, axis=0) # Add batch dim
        return img_arr

    def decode(self, preds):
        idx2char = {i + 1: c for i, c in enumerate(self.charset)}
        decoded_text = []
        prev_idx = -1
        
        # Simple Greedy Decoding
        for idx in preds:
            if idx != 0 and idx != prev_idx:
                decoded_text.append(idx2char.get(idx, ""))
            prev_idx = idx
            
        return "".join(decoded_text)

    def predict_line(self, img):
        if isinstance(img, (str, Path)):
            img = Image.open(img)
            
        input_data = self.preprocess(img)
        if input_data is None:
            return ""
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        outputs = self.session.run([output_name], {input_name: input_data})
        preds = np.argmax(outputs[0], axis=2)[0] # Batch size 1
        
        return self.decode(preds)

    def predict_page(self, img_path):
        """
        Segment page into lines and predict each line.
        """
        if isinstance(img_path, (str, Path)):
            img = Image.open(img_path)
        else:
            img = img_path

        lines = self.segmenter.segment(img)
        results = []
        
        if not lines:
            # Fallback: maybe it IS a single line?
            # Or return empty string?
            # Let's try to predict as single line if segmentation failed but image is smallish?
            # actually segmenter should handle single line if it has enough height
            # If segmentation returns nothing, it means empty page or very noisy.
            # But let's try single line prediction as fallback for robustness
            text = self.predict_line(img)
            if text:
                return text
            return ""

        for line in lines:
            text = self.predict_line(line['img'])
            results.append(text)
            
        return "\n".join(results)

    def predict(self, img_path):
        # Alias for backward compatibility or ease of use
        return self.predict_page(img_path)
