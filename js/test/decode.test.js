const test = require('node:test');
const assert = require('node:assert/strict');

const MonOCR = require('../src/monocr');
const { ModelContractError } = require('../src/monocr');
const { PINNED, fakeSession, StubOCR, BUNDLED_CHARSET } = require('./helpers');

/**
 * Build a logits tensor that argmaxes to `indices`, one class index per
 * timestep, in the [1, T, C] layout the model emits.
 */
function logitsFor(indices, numClasses) {
    const data = new Float32Array(indices.length * numClasses).fill(-10);
    indices.forEach((idx, t) => {
        data[t * numClasses + idx] = 10;
    });
    return { data, dims: [1, indices.length, numClasses] };
}

/** A MonOCR holding a small synthetic charset, for decode-only assertions. */
function ocrWithCharset(charset) {
    const ocr = new StubOCR(fakeSession({ classes: [...charset].length + 1 }), BUNDLED_CHARSET);
    ocr.charset = charset;
    return ocr;
}

test('class index i decodes to charset[i - 1]; index 0 is the CTC blank', () => {
    const ocr = ocrWithCharset(' abc');
    // 1 -> ' ', 2 -> 'a', 3 -> 'b', 4 -> 'c'
    assert.equal(ocr.decode(logitsFor([2, 3, 4], 5)), 'abc');
    assert.equal(ocr.decode(logitsFor([1, 2], 5)), ' a');
    assert.equal(ocr.decode(logitsFor([0, 0, 0], 5)), '');
});

test('CTC contracts repeats and blanks separate them', () => {
    const ocr = ocrWithCharset(' abc');
    assert.equal(ocr.decode(logitsFor([2, 2, 2, 3, 3], 5)), 'ab');
    // A blank between two identical labels keeps both.
    assert.equal(ocr.decode(logitsFor([2, 0, 2], 5)), 'aa');
});

test('the leading space is a decodable character, not padding', async () => {
    // Class 1 in the real charset is U+0020. Losing it to a .trim() would shift
    // every later index down by one, which decodes as fluent nonsense.
    const ocr = new StubOCR(fakeSession(), BUNDLED_CHARSET);
    await ocr.init();
    assert.equal(ocr.decode(logitsFor([1], PINNED.numClasses)), ' ');
});

test('the last class index decodes to the last character, not to nothing', async () => {
    // 0.1.5 mapped only 1..224 and swallowed everything above it with `|| ""`.
    const ocr = new StubOCR(fakeSession(), BUNDLED_CHARSET);
    await ocr.init();
    const last = PINNED.numClasses - 1; // 315
    const expected = [...ocr.charset][PINNED.numChars - 1];
    assert.equal(ocr.decode(logitsFor([last], PINNED.numClasses)), expected);
    assert.notEqual(expected, undefined);
});

test('the real charset decodes real Mon codepoints, not Latin ones', async () => {
    const ocr = new StubOCR(fakeSession(), BUNDLED_CHARSET);
    await ocr.init();
    const chars = [...ocr.charset];
    // Every Mon/Myanmar character in the charset must round-trip through its
    // own index. With the 225-character charset these all landed elsewhere.
    //
    // 89 at revision d3d9d5e, down from 130 at a51be11. The v3.5 charset is
    // deliberately narrower — 276 characters against v2's 315 — so this floor
    // moved with it. It is a floor, not the count: raising it to exactly 89
    // would make a charset addition fail for no reason.
    const myanmar = chars
        .map((ch, i) => ({ ch, idx: i + 1 }))
        .filter(({ ch }) => ch.codePointAt(0) >= 0x1000 && ch.codePointAt(0) <= 0x109f);
    assert.ok(myanmar.length > 80, `expected a Myanmar block, found ${myanmar.length}`);
    for (const { ch, idx } of myanmar) {
        assert.equal(ocr.decode(logitsFor([idx], PINNED.numClasses)), ch);
    }
});

test('decode refuses a tensor whose class axis disagrees with the charset', async () => {
    // Catches what init() cannot: a model with a dynamic class axis.
    const ocr = new StubOCR(fakeSession(), BUNDLED_CHARSET);
    await ocr.init();
    assert.throws(() => ocr.decode(logitsFor([1, 2, 3], 225)), (err) => {
        assert.ok(err instanceof ModelContractError);
        assert.match(err.message, /225 classes/);
        return true;
    });
});

test('MonOCR preprocesses to the height the pinned model expects', () => {
    const ocr = new MonOCR();
    assert.equal(ocr.targetHeight, PINNED.inputHeight);
    assert.equal(ocr.targetWidth, 1024);
});
