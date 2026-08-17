const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ModelManager = require('../src/model-manager');
const { MODEL_REVISION, REMOTE_FILES } = require('../src/model-manager');
const { PINNED } = require('./helpers');

// No test in this file touches the network. They assert the shape of the URLs
// and the cache, which is where the drift happened.

test('downloads are pinned to an immutable revision, never to main', () => {
    const mgr = new ModelManager();
    assert.equal(MODEL_REVISION, PINNED.revision);
    assert.equal(mgr.baseUrl, `https://huggingface.co/janakhpon/monocr/resolve/${PINNED.revision}`);
    assert.doesNotMatch(mgr.baseUrl, /\/resolve\/main/);
});

test('the model URL points at the right org and the onnx/ path', () => {
    // 0.1.5 shipped a second downloader aimed at `janakh/monocr` (no such org)
    // and at `/resolve/main/monocr.onnx` (no such path -- the file lives under
    // `onnx/`). Both 404.
    const mgr = new ModelManager();
    const model = REMOTE_FILES.find((f) => f.label === 'model');
    assert.equal(
        `${mgr.baseUrl}/${model.remote}`,
        `https://huggingface.co/janakhpon/monocr/resolve/${PINNED.revision}/onnx/monocr.onnx`
    );
});

test('the charset comes from the same revision as the weights', () => {
    const mgr = new ModelManager();
    const charset = REMOTE_FILES.find((f) => f.label === 'charset');
    assert.equal(charset.remote, 'onnx/charset.txt');
    assert.equal(charset.bytes, PINNED.charsetBytes);
    // Same base URL for both, so they cannot come from different commits.
    assert.equal(new Set(REMOTE_FILES.map(() => mgr.baseUrl)).size, 1);
});

test('the cache path carries the revision, so a bump is a cache miss', () => {
    const a = new ModelManager('d3d9d5e');
    const b = new ModelManager('deadbee');
    assert.equal(path.basename(a.cacheDir), 'd3d9d5e');
    assert.notEqual(a.getModelPath(), b.getModelPath());
    assert.notEqual(a.getCharsetPath(), b.getCharsetPath());
});

test('a file of the wrong size is not a cache hit', (t) => {
    // `fs.existsSync` alone cannot tell a finished download from a truncated one
    // or from a different revision's artifact left in place.
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'monocr-home-'));
    t.mock.method(os, 'homedir', () => home);

    const mgr = new ModelManager();
    mgr.ensureCacheDir();
    assert.equal(mgr.hasModel(), false);

    for (const spec of REMOTE_FILES) {
        fs.writeFileSync(path.join(mgr.cacheDir, spec.local), Buffer.alloc(spec.bytes - 1));
    }
    assert.equal(mgr.hasModel(), false, 'a short file must not count as cached');

    for (const spec of REMOTE_FILES) {
        fs.writeFileSync(path.join(mgr.cacheDir, spec.local), Buffer.alloc(spec.bytes));
    }
    assert.equal(mgr.hasModel(), true);
});

test('ensureModel returns both artifacts from the cache without downloading', async (t) => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'monocr-home-'));
    t.mock.method(os, 'homedir', () => home);

    const mgr = new ModelManager();
    mgr.ensureCacheDir();
    for (const spec of REMOTE_FILES) {
        fs.writeFileSync(path.join(mgr.cacheDir, spec.local), Buffer.alloc(spec.bytes));
    }

    mgr.downloadModel = async () => {
        throw new Error('ensureModel downloaded despite a complete cache');
    };

    const artifacts = await mgr.ensureModel();
    assert.equal(artifacts.revision, PINNED.revision);
    assert.equal(artifacts.modelPath, mgr.getModelPath());
    assert.equal(artifacts.charsetPath, mgr.getCharsetPath());
});
