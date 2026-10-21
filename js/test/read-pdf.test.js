// read_pdf hands the caller's path to pdftoppm as an argument, never to a shell.
// Through a shell, a file named `$(touch x).pdf` would run `touch x`.
//
// A fake pdftoppm on PATH records the arguments it was given and writes one
// empty page, and the model is stubbed, so this runs offline and with or
// without poppler installed. POSIX only: the fake is a shell script.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const FAKE_PDFTOPPM = `#!/bin/sh
if [ "$1" = "-v" ]; then
    echo "pdftoppm version 0.0.0 (fake)" >&2
    exit 0
fi
printf '%s\\n' "$@" > "$MONOCR_TEST_ARGV"
: > "$5-1.png"
`;

test('read_pdf passes a hostile path to pdftoppm without a shell', {
    skip: process.platform === 'win32' && 'the fake pdftoppm is a POSIX shell script',
}, async (t) => {
    const MonOCR = require('../src/monocr');
    const { read_pdf } = require('../src/index');

    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'monocr-js-inject-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

    const bin = path.join(dir, 'bin');
    fs.mkdirSync(bin);
    fs.writeFileSync(path.join(bin, 'pdftoppm'), FAKE_PDFTOPPM, { mode: 0o755 });

    const argvLog = path.join(dir, 'argv.txt');
    const marker = path.join(dir, 'pwned');
    const pdfPath = path.join(dir, `doc$(touch ${marker})\`touch ${marker}\`.pdf`);

    const oldPath = process.env.PATH;
    const oldLog = process.env.MONOCR_TEST_ARGV;
    process.env.PATH = `${bin}${path.delimiter}${oldPath}`;
    process.env.MONOCR_TEST_ARGV = argvLog;
    t.after(() => {
        process.env.PATH = oldPath;
        if (oldLog === undefined) delete process.env.MONOCR_TEST_ARGV;
        else process.env.MONOCR_TEST_ARGV = oldLog;
    });

    t.mock.method(MonOCR.prototype, 'init', async function () { this.session = {}; });
    t.mock.method(MonOCR.prototype, 'predictPage', async () => [{ text: 'page one' }]);

    const pages = await read_pdf(pdfPath);

    assert.equal(fs.existsSync(marker), false, 'the path was run by a shell');
    const argv = fs.readFileSync(argvLog, 'utf8').split('\n').slice(0, -1);
    assert.equal(argv.length, 5);
    assert.deepEqual(argv.slice(0, 3), ['-png', '-r', '300']);
    assert.equal(argv[3], pdfPath, 'the PDF path reached pdftoppm unaltered');
    assert.equal(path.basename(argv[4]), 'page');
    assert.match(path.basename(path.dirname(argv[4])), /^monocr-js-/);
    assert.deepEqual(pages, ['page one']);
    // The temp dir is removed after the pages are read.
    assert.equal(fs.existsSync(path.dirname(argv[4])), false);
});
