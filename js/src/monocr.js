const ort = require('onnxruntime-node');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const LineSegmenter = require('./segmenter');
const ModelManager = require('./model-manager');

class MonOCR {
    constructor(modelPath = null, charsetPath = null) {
        this.modelPath = modelPath;
        this.charsetPath = charsetPath;
        this.session = null;
        this.charset = "";
        this.segmenter = new LineSegmenter();
        this.modelManager = new ModelManager();
        
        // Metadata
        this.targetHeight = 128;
        this.targetWidth = 1024;
    }

    async init() {
        if (this.session) return;
        
        // Ensure model exists
        if (!this.modelPath) {
            this.modelPath = await this.modelManager.ensureModel();
        }
        
        // Use bundled charset if not provided
        if (!this.charsetPath) {
            this.charsetPath = path.join(__dirname, 'charset.txt');
        }
        
        this.session = await ort.InferenceSession.create(this.modelPath);
        this.charset = fs.readFileSync(this.charsetPath, 'utf-8').trim();
    }

    /**
     * Replicates Python's resize_and_pad:
     * 1. Resize height to 64, maintain aspect ratio.
     * 2. Pad width to 1024 (white background).
     * 3. Normalize to [-1, 1].
     */
    async preprocess(imageSource) {
        let sharpImg;
        if (typeof imageSource.metadata === 'function') {
            sharpImg = imageSource;
        } else {
            sharpImg = sharp(imageSource);
        }
        const metadata = await sharpImg.metadata();
        
        // Scale to target height
        const scale = this.targetHeight / metadata.height;
        const newWidth = Math.min(this.targetWidth, Math.round(metadata.width * scale));

        // Create the grayscale resized image
        const resizedBuffer = await sharpImg
            .grayscale()
            .resize({
                height: this.targetHeight,
                width: newWidth,
                fit: 'fill'
            })
            .raw()
            .toBuffer();

        // Create target canvas (1024 width, white background = 255)
        const totalSize = this.targetHeight * this.targetWidth;
        const canvas = new Float32Array(totalSize);

        // Fill canvas with resized image and normalize to [-1.0, 1.0]
        // (pix / 127.5) - 1.0
        for (let y = 0; y < this.targetHeight; y++) {
            for (let x = 0; x < this.targetWidth; x++) {
                const canvasIdx = y * this.targetWidth + x;
                if (x < newWidth) {
                    const imgIdx = y * newWidth + x;
                    const pixelValue = resizedBuffer[imgIdx];
                    canvas[canvasIdx] = (pixelValue / 127.5) - 1.0;
                } else {
                    // Padding is white (255)
                    canvas[canvasIdx] = (255 / 127.5) - 1.0; // 1.0
                }
            }
        }
        
        return new ort.Tensor('float32', canvas, [1, 1, this.targetHeight, this.targetWidth]); 
    }

    /**
     * CTC Greedy Decoding
     * Ignores blank (0) and contracts repeats.
     */
    decode(outputTensor) {
        const data = outputTensor.data;
        const dims = outputTensor.dims; // [Batch, Time, Classes]
        const numClasses = dims[2];
        const sequenceLength = dims[1];

        const idx2char = {};
        for (let i = 0; i < this.charset.length; i++) {
            idx2char[i + 1] = this.charset[i];
        }

        let decodedText = "";
        let prevIdx = -1;

        for (let t = 0; t < sequenceLength; t++) {
             let maxVal = -Infinity;
             let maxIdx = 0;
             for (let c = 0; c < numClasses; c++) {
                 const val = data[t * numClasses + c];
                 if (val > maxVal) {
                     maxVal = val;
                     maxIdx = c;
                 }
             }
             
             // CTC logic: 0 is blank, ignore repeats
             if (maxIdx !== 0 && maxIdx !== prevIdx) {
                 decodedText += idx2char[maxIdx] || "";
             }
             prevIdx = maxIdx;
        }

        return decodedText;
    }

    async predictLine(imageSource) {
        if (!this.session) await this.init();
        
        const inputTensor = await this.preprocess(imageSource);
        const feeds = {};
        feeds[this.session.inputNames[0]] = inputTensor;
        
        const results = await this.session.run(feeds);
        const outputTensor = results[this.session.outputNames[0]];
        
        return this.decode(outputTensor);
    }

    /**
     * Processes full page: segments into lines and predicts each.
     */
    async predictPage(imagePath) {
        const lines = await this.segmenter.segment(imagePath);
        const results = [];
        
        for (const line of lines) {
            const text = await this.predictLine(line.img);
            results.push({
                text,
                bbox: line.bbox
            });
        }
        
        return results;
    }
}

module.exports = MonOCR;
