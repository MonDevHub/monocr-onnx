const sharp = require('sharp');

class LineSegmenter {
    /**
     * @param {number} minLineH Minimum height of a line to be considered valid.
     * @param {number} smoothWindow Smoothing window for projection profile.
     */
    constructor(minLineH = 10, smoothWindow = 3) {
        this.minLineH = minLineH;
        this.smoothWindow = smoothWindow;
    }

    /**
     * Segment a document image into text lines.
     * @param {string|Buffer} imagePath Path to image or Buffer.
     * @returns {Promise<Array<{img: sharp.Sharp, bbox: {x: number, y: number, w: number, h: number}}>>}
     */
    async segment(imagePath) {
        const image = sharp(imagePath);
        const { width, height } = await image.metadata();
        
        // 1. Get raw grayscale data for thresholding
        const grayBuffer = await image
            .grayscale()
            .raw()
            .toBuffer();

        // 2. Simple Adaptive-ish Thresholding
        // Since we don't have CV2's adaptiveThreshold easily, we'll do a simple threshold 
        // or just use sharp's threshold if we can get the mask.
        // Actually, to replicate Horizontal Projection, we need the sum of "text" pixels.
        // We'll treat dark pixels (< 128) as text (since background is white).
        const binary = new Uint8Array(grayBuffer.length);
        const hist = new Float32Array(height).fill(0);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                // Threshold: 128 is a safe bet for black text on white paper.
                // Inverted so text is "high" (1) and background is 0.
                if (grayBuffer[idx] < 128) {
                    binary[idx] = 1;
                    hist[y]++;
                } else {
                    binary[idx] = 0;
                }
            }
        }

        // 3. Smoothing projection profile
        let smoothedHist = hist;
        if (this.smoothWindow > 1) {
            smoothedHist = new Float32Array(height);
            const half = Math.floor(this.smoothWindow / 2);
            for (let i = 0; i < height; i++) {
                let sum = 0;
                let count = 0;
                for (let j = i - half; j <= i + half; j++) {
                    if (j >= 0 && j < height) {
                        sum += hist[j];
                        count++;
                    }
                }
                smoothedHist[i] = sum / count;
            }
        }

        // 4. Gap Detection
        const nonZeroVals = smoothedHist.filter(v => v > 0);
        if (nonZeroVals.length === 0) return [];

        const meanDensity = nonZeroVals.reduce((a, b) => a + b, 0) / nonZeroVals.length;
        const gapThreshold = meanDensity * 0.05;

        const results = [];
        let start = null;

        for (let y = 0; y < height; y++) {
            const isText = smoothedHist[y] > gapThreshold;
            if (isText && start === null) {
                start = y;
            } else if (!isText && start !== null) {
                const end = y;
                if (end - start >= this.minLineH) {
                    await this._extractLine(image, grayBuffer, width, height, start, end, results);
                }
                start = null;
            }
        }

        if (start !== null && (height - start) >= this.minLineH) {
            await this._extractLine(image, grayBuffer, width, height, start, height, results);
        }

        return results;
    }

    async _extractLine(image, grayBuffer, width, height, rStart, rEnd, results) {
        // Find horizontal bounds within this vertical strip
        let xMin = width;
        let xMax = 0;
        let hasPixels = false;

        for (let y = rStart; y < rEnd; y++) {
            for (let x = 0; x < width; x++) {
                if (grayBuffer[y * width + x] < 128) {
                    if (x < xMin) xMin = x;
                    if (x > xMax) xMax = x;
                    hasPixels = true;
                }
            }
        }

        if (!hasPixels) return;

        // Relative padding based on line height
        const hRaw = rEnd - rStart;
        const padY = Math.ceil(hRaw * 0.20);
        const padX = Math.ceil(hRaw * 0.15);
        const y1 = Math.max(0, rStart - padY);
        const y2 = Math.min(height, rEnd + padY);
        const x1 = Math.max(0, xMin - padX);
        const x2 = Math.min(width, xMax + padX);

        const w = x2 - x1;
        const h = y2 - y1;

        // Crop the line
        const crop = image.clone().extract({ left: x1, top: y1, width: w, height: h });
        
        results.push({
            img: crop,
            bbox: { x: x1, y: y1, w, h }
        });
    }
}

module.exports = LineSegmenter;
