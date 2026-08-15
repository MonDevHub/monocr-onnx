// Smoke script: run the OCR over the sample images in this repository.
//
//   node example.js
//
// The model is fetched from the pinned Hugging Face revision on first run and
// cached under ~/.monocr/models/<revision>/.

const { MonOCR, MODEL_REVISION } = require('./src/index');
const path = require('path');
const fs = require('fs');

async function runExample() {
    const imagesDir = path.join(__dirname, '..', 'data', 'images');

    console.log('--- MonOCR JavaScript Example ---');
    console.log(`Model revision: ${MODEL_REVISION}`);

    const monocr = new MonOCR();
    await monocr.init();
    console.log(`Model: ${monocr.modelPath}`);
    console.log(`Charset: ${[...monocr.charset].length} characters\n`);

    if (!fs.existsSync(imagesDir)) {
        console.error(`Images directory not found: ${imagesDir}`);
        return;
    }

    const files = fs.readdirSync(imagesDir)
        .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
        .sort();

    console.log(`Found ${files.length} images.\n`);

    for (const file of files) {
        const filePath = path.join(imagesDir, file);
        console.log(`Processing: ${file}...`);

        try {
            // For general images, we use predictPage which handles multiple lines
            const results = await monocr.predictPage(filePath);

            if (results.length === 0) {
                console.log('  [No text detected]');
            } else {
                results.forEach((res, i) => {
                    console.log(`  Line ${i + 1}: "${res.text}"`);
                });
            }
        } catch (err) {
            console.error(`  Error processing ${file}:`, err.message);
        }
        console.log('-'.repeat(40));
    }
}

runExample().catch(err => {
    console.error('Fatal Error:', err);
    process.exitCode = 1;
});
