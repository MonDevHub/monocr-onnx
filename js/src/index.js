const MonOCR = require('./monocr');
const { ModelContractError } = require('./monocr');
const ModelManager = require('./model-manager');
const { MODEL_REVISION } = require('./model-manager');
const { calculateAccuracy } = require('./utils');

module.exports = {
    MonOCR,
    ModelManager,
    // Thrown at load when the model and the charset disagree. Catch it if you
    // would rather degrade than decode with a mismatched vocabulary; do not
    // swallow it, because what it prevents is confident wrong text.
    ModelContractError,
    MODEL_REVISION,
    calculateAccuracy,
    read_image,
    read_images,
    read_pdf,
    read_pdfs,
    read_image_with_accuracy
};

/**
 * Read text from an image file
 * @param {string} imagePath - Path to image file
 * @param {string} modelPath - Path to ONNX model (optional)
 * @param {string} charsetPath - Path to charset file (optional)
 * @returns {Promise<string>} Recognized text
 */
async function read_image(imagePath, modelPath = null, charsetPath = null) {
    const ocr = new MonOCR(modelPath, charsetPath);
    await ocr.init();
    const results = await ocr.predictPage(imagePath);
    return results.map(r => r.text).join('\n');
}

/**
 * Read text from multiple image files
 * @param {string[]} imagePaths - Array of paths to image files
 * @param {string} modelPath - Path to ONNX model (optional)
 * @param {string} charsetPath - Path to charset file (optional)
 * @returns {Promise<string[]>} Array of recognized text
 */
async function read_images(imagePaths, modelPath = null, charsetPath = null) {
    const ocr = new MonOCR(modelPath, charsetPath);
    await ocr.init();
    
    const results = [];
    for (const path of imagePaths) {
        const pageResults = await ocr.predictPage(path);
        results.push(pageResults.map(r => r.text).join('\n'));
    }
    return results;
}

/**
 * Read text from a PDF file
 * @param {string} pdfPath - Path to PDF file
 * @param {string} modelPath - Path to ONNX model (optional, auto-downloads if not provided)
 * @param {string} charsetPath - Path to charset file (optional, auto-downloads if not provided)
 * @returns {Promise<string[]>} Array of text per page
 */
async function read_pdf(pdfPath, modelPath = null, charsetPath = null) {
    const fs = require('fs');
    const os = require('os');
    const path = require('path');
    const { exec } = require('child_process');
    const util = require('util');
    const execPromise = util.promisify(exec);
    
    // Check for pdftoppm
    try {
        await execPromise('pdftoppm -v');
    } catch (e) {
        throw new Error('pdftoppm not found: please install poppler-utils (brew install poppler on macOS, or sudo apt install poppler-utils on Linux)');
    }

    // Initialize OCR
    const ocr = new MonOCR(modelPath, charsetPath);
    await ocr.init();
    
    // Create temp directory
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'monocr-js-'));
    
    try {
        // Convert PDF to images using pdftoppm
        // pdftoppm -png -r 300 input.pdf output_prefix
        const outputPrefix = path.join(tempDir, 'page');
        await execPromise(`pdftoppm -png -r 300 "${pdfPath}" "${outputPrefix}"`);

        // Read generated images
        const files = fs.readdirSync(tempDir).sort(); // Sort to ensure page order? pdftoppm uses -1, -2 etc.
        // Actually pdftoppm numbers are zero-padded usually? No, "page-1.png", "page-10.png" might sort wrong alphabetically.
        // pdftoppm output format: prefix-1.png, prefix-2.png ...
        // We should sort naturally.
        
        const imageFiles = files.filter(f => f.endsWith('.png')).sort((a, b) => {
             // Extract numbers
             const numA = parseInt(a.match(/-(\d+)\.png$/)[1]);
             const numB = parseInt(b.match(/-(\d+)\.png$/)[1]);
             return numA - numB;
        });

        if (imageFiles.length === 0) {
            throw new Error("Failed to convert PDF: No images generated");
        }

        const pages = [];
        
        for (const file of imageFiles) {
            const imgPath = path.join(tempDir, file);
            const results = await ocr.predictPage(imgPath);
            const pageText = results.map(r => r.text).join('\n');
            pages.push(pageText);
        }
        
        return pages;
    } catch (err) {
        throw new Error(`Failed to process PDF: ${err.message}`);
    } finally {
        // Cleanup
        try {
            fs.rmSync(tempDir, { recursive: true, force: true });
        } catch (e) {
            // ignore cleanup error
        }
    }
}

/**
 * Read text from multiple PDF files
 * @param {string[]} pdfPaths - Array of paths to PDF files
 * @param {string} modelPath - Path to ONNX model (optional)
 * @param {string} charsetPath - Path to charset file (optional)
 * @returns {Promise<string[][]>} Array of arrays of text per page
 */
async function read_pdfs(pdfPaths, modelPath = null, charsetPath = null) {
    const results = [];
    for (const path of pdfPaths) {
        const pages = await read_pdf(path, modelPath, charsetPath);
        results.push(pages);
    }
    return results;
}

/**
 * Read text from an image with accuracy measurement
 * @param {string} imagePath - Path to image file
 * @param {string} groundTruth - Expected text for accuracy calculation
 * @param {string} modelPath - Path to ONNX model (optional, auto-downloads if not provided)
 * @param {string} charsetPath - Path to charset file (optional, auto-downloads if not provided)
 * @returns {Promise<{text: string, accuracy: number}>}
 */
async function read_image_with_accuracy(imagePath, groundTruth, modelPath = null, charsetPath = null) {
    const text = await read_image(imagePath, modelPath, charsetPath);
    const accuracy = calculateAccuracy(text, groundTruth);
    return { text, accuracy: parseFloat(accuracy) };
}
