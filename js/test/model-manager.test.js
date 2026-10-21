const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const fs = require('fs');
const https = require('https');
const os = require('os');
const path = require('path');
const { Readable } = require('stream');

const ModelManager = require('../src/model-manager');
const { MODEL_REVISION, REMOTE_FILES } = require('../src/model-manager');
const { PINNED } = require('./helpers');

// No test in this file touches the network. They assert the shape of the URLs
// and the cache, which is where the drift happened.

const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

/**
 * Tiny stand-ins for the pinned files: same shape as REMOTE_FILES, but a few
 * bytes each with their digests injected, so the integrity check runs for real
 * without a 46 MB fixture or a download.
 */
function fakeFiles() {
    const model = Buffer.from('fake onnx weights');
    const charset = Buffer.from('abc\n');
    return {
        contents: { 'monocr.onnx': model, 'charset.txt': charset },
        specs: [
            { remote: 'onnx/monocr.onnx', local: 'monocr.onnx', bytes: model.length, sha256: sha256(model), label: 'model' },
            { remote: 'onnx/charset.txt', local: 'charset.txt', bytes: charset.length, sha256: sha256(charset), label: 'charset' },
        ],
    };
}

/** Answer `https.get` with `body` and status 200, once per call, offline. */
function serve(t, bodyFor) {
    const urls = [];
    t.mock.method(https, 'get', (url, _opts, cb) => {
        urls.push(url);
        const body = bodyFor(url);
        const res = Readable.from([body]);
        res.statusCode = 200;
        res.headers = { 'content-length': String(body.length) };
        process.nextTick(() => cb(res));
        return { on() { return this; } };
    });
    return urls;
}

function managerIn(t) {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'monocr-home-'));
    t.after(() => fs.rmSync(home, { recursive: true, force: true }));
    t.mock.method(os, 'homedir', () => home);
    const mgr = new ModelManager();
    mgr.ensureCacheDir();
    return mgr;
}

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
    const mgr = managerIn(t);
    const { specs, contents } = fakeFiles();
    mgr.files = specs;
    assert.equal(mgr.hasModel(), false);

    for (const spec of specs) {
        fs.writeFileSync(path.join(mgr.cacheDir, spec.local), contents[spec.local].subarray(1));
    }
    assert.equal(mgr.hasModel(), false, 'a short file must not count as cached');

    for (const spec of specs) {
        fs.writeFileSync(path.join(mgr.cacheDir, spec.local), contents[spec.local]);
    }
    assert.equal(mgr.hasModel(), true);
});

test('the pinned digests are the ones the Python binding verifies', () => {
    // python/monocr_onnx/model_manager.py is the source of truth. All bindings
    // share one cache directory, so if their digests disagree, one of them
    // rejects -- and deletes -- a file another one just verified.
    const py = fs.readFileSync(
        path.join(__dirname, '..', '..', 'python', 'monocr_onnx', 'model_manager.py'),
        'utf-8'
    );
    for (const spec of REMOTE_FILES) {
        const key = spec.label === 'model' ? 'MODEL_FILENAME' : 'CHARSET_FILENAME';
        const m = py.match(new RegExp(`${key}:\\s*"([0-9a-f]{64})"`));
        assert.ok(m, `no ${key} digest found in the Python model manager`);
        assert.equal(spec.sha256, m[1], `${spec.label} digest drifted from Python`);
    }
});

test('the bundled charset has the pinned charset digest', () => {
    const charset = REMOTE_FILES.find((f) => f.label === 'charset');
    const bundled = fs.readFileSync(path.join(__dirname, '..', 'src', 'charset.txt'));
    assert.equal(sha256(bundled), charset.sha256);
});

test('a right-sized file with the wrong bytes is not a cache hit', (t) => {
    // The size check alone accepted this: same length, different content.
    const mgr = managerIn(t);
    const { specs, contents } = fakeFiles();
    mgr.files = specs;
    for (const spec of specs) {
        fs.writeFileSync(path.join(mgr.cacheDir, spec.local), contents[spec.local]);
    }
    assert.equal(mgr.hasModel(), true, 'control: the right bytes are a cache hit');

    const model = specs[0];
    fs.writeFileSync(path.join(mgr.cacheDir, model.local), Buffer.alloc(model.bytes, 0x41));
    assert.equal(mgr.hasModel(), false, 'same size, wrong digest must not count as cached');
});

test('a corrupt cached file is deleted and fetched again', async (t) => {
    const mgr = managerIn(t);
    const { specs, contents } = fakeFiles();
    mgr.files = specs;
    fs.writeFileSync(path.join(mgr.cacheDir, 'monocr.onnx'), Buffer.alloc(specs[0].bytes, 0x41));
    fs.writeFileSync(path.join(mgr.cacheDir, 'charset.txt'), contents['charset.txt']);

    const urls = serve(t, (url) => contents[path.basename(url)]);
    t.mock.method(console, 'log', () => {});
    await mgr.ensureModel();

    assert.deepEqual(urls.map((u) => path.basename(u)), ['monocr.onnx'], 'only the corrupt file is refetched');
    assert.deepEqual(fs.readFileSync(mgr.getModelPath()), contents['monocr.onnx']);
});

test('a download with the wrong digest is refused before it reaches the cache', async (t) => {
    // Right size, wrong bytes: only the digest can tell. The file must not be
    // renamed into place, or every binding would trust it from then on.
    const mgr = managerIn(t);
    const { specs } = fakeFiles();
    mgr.files = specs;
    serve(t, (url) => Buffer.alloc(specs.find((s) => s.local === path.basename(url)).bytes, 0x41));
    t.mock.method(console, 'log', () => {});

    await assert.rejects(mgr.ensureModel(), (err) => {
        assert.match(err.message, /Checksum mismatch for model/);
        assert.ok(err.message.includes(`manually into ${mgr.cacheDir}.`), err.message);
        return true;
    });
    assert.equal(fs.existsSync(mgr.getModelPath()), false, 'a bad download must not be cached');
    assert.equal(fs.existsSync(`${mgr.getModelPath()}.part`), false, 'the temp file must be removed');
});

test('ensureModel returns both artifacts from the cache without downloading', async (t) => {
    const mgr = managerIn(t);
    const { specs, contents } = fakeFiles();
    mgr.files = specs;
    for (const spec of specs) {
        fs.writeFileSync(path.join(mgr.cacheDir, spec.local), contents[spec.local]);
    }

    mgr.downloadModel = async () => {
        throw new Error('ensureModel downloaded despite a complete cache');
    };

    const artifacts = await mgr.ensureModel();
    assert.equal(artifacts.revision, PINNED.revision);
    assert.equal(artifacts.modelPath, mgr.getModelPath());
    assert.equal(artifacts.charsetPath, mgr.getCharsetPath());
});
