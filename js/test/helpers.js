const fs = require('fs');
const os = require('os');
const path = require('path');

const MonOCR = require('../src/monocr');

/**
 * The pinned artifact these tests describe.
 *
 * Read off the real model at `janakhpon/monocr@d3d9d5e` with:
 *
 *   node -e "require('onnxruntime-node').InferenceSession.create('monocr.onnx')
 *     .then(s => console.log(JSON.stringify([s.inputMetadata, s.outputMetadata])))"
 *
 *   input  float32 [1, 1, 160, 1024]
 *   logits float32 [1, "sequence", 277]
 */
const PINNED = {
    revision: 'd3d9d5e',
    inputHeight: 160,
    numClasses: 277,
    numChars: 276,
    charsetBytes: 556,
};

/**
 * A stand-in for an ONNX Runtime session: metadata only, no model file and no
 * network. The contract check reads exactly these fields off a real session.
 */
function fakeSession({ height = PINNED.inputHeight, classes = PINNED.numClasses } = {}) {
    return {
        inputNames: ['input'],
        outputNames: ['logits'],
        inputMetadata: [
            { name: 'input', isTensor: true, type: 'float32', shape: [1, 1, height, 'width'] },
        ],
        outputMetadata: [
            { name: 'logits', isTensor: true, type: 'float32', shape: [1, 'sequence', classes] },
        ],
        async run() {
            throw new Error('fakeSession cannot run inference');
        },
    };
}

/**
 * MonOCR with the session swapped for a fake. `modelPath` is set so the model
 * manager is never consulted, which keeps these tests offline.
 */
class StubOCR extends MonOCR {
    constructor(session, charsetPath) {
        super('/nonexistent/monocr.onnx', charsetPath);
        this._stubSession = session;
    }

    async _loadSession() {
        return this._stubSession;
    }
}

/** Write a charset file into a per-test temp dir and return its path. */
function writeCharset(contents) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'monocr-test-'));
    const file = path.join(dir, 'charset.txt');
    fs.writeFileSync(file, contents, 'utf-8');
    return file;
}

const BUNDLED_CHARSET = path.join(__dirname, '..', 'src', 'charset.txt');

module.exports = { PINNED, fakeSession, StubOCR, writeCharset, BUNDLED_CHARSET };
