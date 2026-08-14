const test = require('node:test');
const assert = require('node:assert/strict');

const MonOCR = require('../src/monocr');
const { ModelContractError, assertModelContract } = require('../src/monocr');
const { PINNED, fakeSession, StubOCR, writeCharset, BUNDLED_CHARSET } = require('./helpers');

const CHARSET_315 = MonOCR.readCharset(BUNDLED_CHARSET);

test('a matching model and charset load', async () => {
    const ocr = new StubOCR(fakeSession(), BUNDLED_CHARSET);
    await ocr.init();
    assert.equal([...ocr.charset].length, PINNED.numChars);
    assert.equal(ocr.targetHeight, PINNED.inputHeight);
});

test('init refuses a model whose class count does not match the charset', async () => {
    // The exact 0.1.5 failure: 316-class model, 225-character charset.
    const shortCharset = writeCharset(CHARSET_315.slice(0, 225));
    const ocr = new StubOCR(fakeSession({ classes: PINNED.numClasses }), shortCharset);

    await assert.rejects(() => ocr.init(), (err) => {
        assert.ok(err instanceof ModelContractError, `expected ModelContractError, got ${err.name}`);
        assert.match(err.message, /316 output classes/);
        assert.match(err.message, /225 characters/);
        return true;
    });
    assert.equal(ocr.session, null, 'a refused model must not be left loaded');
});

test('init refuses the old 225-class model against the current charset', async () => {
    // The other direction: a stale cached artifact from before the retrain.
    const ocr = new StubOCR(fakeSession({ classes: 225, height: 64 }), BUNDLED_CHARSET);
    await assert.rejects(() => ocr.init(), ModelContractError);
});

test('init refuses a model whose input height is not what preprocessing produces', async () => {
    const ocr = new StubOCR(fakeSession({ height: 64 }), BUNDLED_CHARSET);
    await assert.rejects(() => ocr.init(), (err) => {
        assert.ok(err instanceof ModelContractError);
        assert.match(err.message, /height 64/);
        assert.match(err.message, /128/);
        return true;
    });
});

test('a charset one character short is refused, not silently accepted', async () => {
    // What `.trim()` produced: 314 characters against 316 classes.
    const trimmed = writeCharset(CHARSET_315.trim());
    const ocr = new StubOCR(fakeSession(), trimmed);
    await assert.rejects(() => ocr.init(), (err) => {
        assert.match(err.message, /314 characters/);
        return true;
    });
});

test('assertModelContract refuses a session whose metadata cannot be read', () => {
    assert.throws(
        () => assertModelContract({ inputMetadata: [], outputMetadata: [] }, CHARSET_315, 128),
        ModelContractError
    );
});

test('assertModelContract skips a dynamic class axis rather than guessing', () => {
    // Nothing to compare against; `decode()` re-checks the real dims instead.
    const session = fakeSession();
    session.outputMetadata[0].shape = [1, 'sequence', 'classes'];
    assert.doesNotThrow(() => assertModelContract(session, CHARSET_315, 128));
});
