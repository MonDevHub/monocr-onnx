const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');

const MonOCR = require('../src/monocr');
const { PINNED, writeCharset, BUNDLED_CHARSET } = require('./helpers');

// The bug this file exists for: monocr 0.1.5 bundled a 225-character charset
// against a 316-class model. Nothing compared the two, so every Mon character
// decoded to the wrong glyph and every index past 224 decoded to nothing.

test('bundled charset has one character per model class, minus the CTC blank', () => {
    const charset = MonOCR.readCharset(BUNDLED_CHARSET);
    assert.equal([...charset].length, PINNED.numChars);
    assert.equal([...charset].length + 1, PINNED.numClasses);
});

test('bundled charset is byte-identical to the pinned revision', () => {
    // Fetched from
    // huggingface.co/janakhpon/monocr/resolve/a51be11/onnx/charset.txt
    assert.equal(fs.statSync(BUNDLED_CHARSET).size, PINNED.charsetBytes);
});

test('bundled charset starts with U+0020 and keeps it after loading', () => {
    // The file really does begin with a space, and that space is class index 1.
    const raw = fs.readFileSync(BUNDLED_CHARSET, 'utf-8');
    assert.equal(raw.codePointAt(0), 0x20);

    const loaded = MonOCR.readCharset(BUNDLED_CHARSET);
    assert.equal(loaded.codePointAt(0), 0x20, 'the leading space was eaten by the loader');
    assert.equal(loaded.length, raw.length, 'loading must not change the character count');
});

test('readCharset strips newlines but not the leading space', () => {
    // A bare .trim() passes every assertion above except this one: it eats
    // U+0020 and shifts every class index by one.
    const file = writeCharset(' abc\n\n');
    assert.equal(MonOCR.readCharset(file), ' abc');
});

test('readCharset strips CRLF line endings too', () => {
    const file = writeCharset(' abc\r\n');
    assert.equal(MonOCR.readCharset(file), ' abc');
});

test('bundled charset has no duplicate characters', () => {
    // Two indices mapping to one character means one of them is unreachable and
    // the model was trained against a vocabulary this file does not describe.
    const chars = [...MonOCR.readCharset(BUNDLED_CHARSET)];
    assert.equal(new Set(chars).size, chars.length);
});
