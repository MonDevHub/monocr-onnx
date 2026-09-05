#!/usr/bin/env node

const { program } = require('commander');
const { read_image, read_pdf } = require('../src/index');
const fs = require('fs');
const path = require('path');

// Read from the manifest rather than repeating it. This literal said '0.1.0'
// through 0.3.0 and 0.3.2, so `monocr --version` reported a version two minors
// behind the package the user had installed.
const { version } = require('../package.json');

program
    .name('monocr')
    .description('Mon language OCR using ONNX Runtime')
    .version(version);

program
    .command('image <path>')
    .description('Recognize text from an image file')
    .option('-m, --model <path>', 'Path to ONNX model (optional, auto-downloads)')
    .option('-c, --charset <path>', 'Path to charset file (optional)')
    .action(async (imagePath, options) => {
        try {
            const text = await read_image(imagePath, options.model, options.charset);
            console.log(text);
        } catch (err) {
            console.error('Error:', err.message);
            process.exit(1);
        }
    });

program
    .command('pdf <path>')
    .description('Recognize text from a PDF file')
    .option('-m, --model <path>', 'Path to ONNX model (optional, auto-downloads)')
    .option('-c, --charset <path>', 'Path to charset file (optional)')
    .action(async (pdfPath, options) => {
        try {
            const pages = await read_pdf(pdfPath, options.model, options.charset);
            pages.forEach((pageText, i) => {
                console.log(`--- Page ${i + 1} ---`);
                console.log(pageText);
                console.log();
            });
        } catch (err) {
            console.error('Error:', err.message);
            process.exit(1);
        }
    });

program
    .command('batch <directory>')
    .description('Process all images in a directory')
    .option('-m, --model <path>', 'Path to ONNX model (optional, auto-downloads)')
    .option('-c, --charset <path>', 'Path to charset file (optional)')
    .option('-o, --output <path>', 'Output file for results (optional)')
    .action(async (directory, options) => {
        try {
            const files = fs.readdirSync(directory)
                .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
                .sort();
            
            const results = [];
            
            for (const file of files) {
                const filePath = path.join(directory, file);
                console.error(`Processing: ${file}...`);
                
                try {
                    const text = await read_image(filePath, options.model, options.charset);
                    results.push({ file, text, success: true });
                } catch (err) {
                    results.push({ file, error: err.message, success: false });
                }
            }
            
            if (options.output) {
                fs.writeFileSync(options.output, JSON.stringify(results, null, 2));
                console.error(`Results written to ${options.output}`);
            } else {
                console.log(JSON.stringify(results, null, 2));
            }
        } catch (err) {
            console.error('Error:', err.message);
            process.exit(1);
        }
    });

program
    .command('download')
    .description('Download model files to local cache')
    .action(async () => {
        try {
            const { MonOCR } = require('../src/index');
            const ocr = new MonOCR();
            await ocr.modelManager.downloadModel();
        } catch (err) {
            console.error('Error:', err.message);
            process.exit(1);
        }
    });

program.parse();
