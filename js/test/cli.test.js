// `monocr batch` exits non-zero when a file fails.
//
// It used to record each failure in the JSON results and still exit 0, so a
// script or CI step checking the status saw success even when every file had
// failed. Here every file fails because the model path does not exist, which
// needs no network and no model.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const CLI = path.join(__dirname, '..', 'bin', 'monocr.js');

test('batch exits non-zero when every file fails', (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'monocr-js-cli-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    fs.writeFileSync(path.join(dir, 'a.png'), 'not an image');

    const run = spawnSync(process.execPath, [
        CLI, 'batch', dir,
        '--model', path.join(dir, 'missing.onnx'),
        '--charset', path.join(__dirname, '..', 'src', 'charset.txt'),
    ], { encoding: 'utf8', env: { ...process.env, HOME: dir } });

    const results = JSON.parse(run.stdout);
    assert.equal(results.length, 1);
    assert.equal(results[0].success, false);
    assert.equal(run.status, 1, `exit status ${run.status}; stderr:\n${run.stderr}`);
});
