const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');
const crypto = require('crypto');

/**
 * The exact Hugging Face revision this release decodes.
 *
 * It used to be `main`, which is a moving ref: the artifact behind it has
 * already been replaced at least once (57.8 MB / 225 classes / height 64 ->
 * 26.4 MB / 316 classes / height 128), so two users running the same published
 * npm version could get two different networks. A commit is immutable; the
 * charset is fetched from the same one, so weights and vocabulary cannot come
 * from different versions of the repository.
 *
 * Bumping this is a deliberate act: change the revision, re-check the class
 * count and input height against `js/test/`, and update the bundled
 * `src/charset.txt` in the same commit.
 */
const MODEL_REVISION = 'd3d9d5e';
const MODEL_REPO = 'janakhpon/monocr';

/**
 * Files fetched from the pinned revision. `bytes` is the size Hugging Face
 * reports for that revision; a file on disk of any other size is a partial or
 * corrupted download, not a cache hit.
 *
 * `sha256` is the digest of each file at that revision, copied from
 * `python/monocr_onnx/model_manager.py` (`_SHA256`), which is the source of
 * truth; `test/model-manager.test.js` fails if the two drift. Bump the revision
 * and both digests together, never one alone.
 *
 * All four bindings share `~/.monocr/models/<revision>/`, so a corrupt file
 * written by any of them would otherwise be reused by every one, forever.
 */
const REMOTE_FILES = [
    {
        remote: 'onnx/monocr.onnx',
        local: 'monocr.onnx',
        bytes: 46247040,
        sha256: 'b95de1ea0e3dc99a5c31bea32e220835da801f683ed17091eaec9954c80a4c04',
        label: 'model',
    },
    {
        remote: 'onnx/charset.txt',
        local: 'charset.txt',
        bytes: 556,
        sha256: 'edfd75f688e4155c64aeee0dbac755da0e7ba45a388a2d178a84190fb3d7e953',
        label: 'charset',
    },
];

/** Hex sha256 of a file, read in chunks so a 46 MB model is not held twice. */
function sha256File(filePath) {
    const hash = crypto.createHash('sha256');
    const fd = fs.openSync(filePath, 'r');
    try {
        const buf = Buffer.allocUnsafe(1 << 20);
        let n;
        while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
            hash.update(buf.subarray(0, n));
        }
    } finally {
        fs.closeSync(fd);
    }
    return hash.digest('hex');
}

class ModelManager {
    constructor(revision = MODEL_REVISION) {
        this.revision = revision;

        // Cache directory in the user's home, keyed by revision.
        //
        // The old layout was a flat `~/.monocr/models/monocr.onnx` guarded by a
        // bare `fs.existsSync`, which cannot see that the file behind it came
        // from a different revision -- a stale artifact stays "cached" forever.
        // Putting the revision in the path makes a revision change a cache miss
        // by construction.
        this.rootDir = path.join(os.homedir(), '.monocr', 'models');
        this.cacheDir = path.join(this.rootDir, this.revision);

        this.baseUrl = `https://huggingface.co/${MODEL_REPO}/resolve/${this.revision}`;
        this.modelFileName = 'monocr.onnx';
        this.charsetFileName = 'charset.txt';

        // The specs this instance fetches and verifies. An instance field so a
        // test can inject a tiny file and its digest without touching the network.
        this.files = REMOTE_FILES;
    }

    /**
     * Ensure cache directory exists
     */
    ensureCacheDir() {
        if (!fs.existsSync(this.cacheDir)) {
            fs.mkdirSync(this.cacheDir, { recursive: true });
        }
    }

    /**
     * Get local path for the model
     */
    getModelPath() {
        return path.join(this.cacheDir, this.modelFileName);
    }

    /**
     * Get local path for the charset that shipped with these weights
     */
    getCharsetPath() {
        return path.join(this.cacheDir, this.charsetFileName);
    }

    /**
     * True when a cached file is present, the right size, *and* has the pinned
     * sha256 -- the same check the Python binding makes.
     *
     * Size is checked first because it is free and catches the common failure,
     * an interrupted download. The digest catches the rest: a same-size file
     * with different bytes, or one written by another binding into the shared
     * cache before that binding verified anything.
     *
     * The digest is recomputed on every check rather than remembered in a
     * sidecar marker. That costs one read of the 46 MB model per load (a
     * fraction of a second, small next to creating the ONNX session), and a
     * marker could be left standing over a file that changed after it was
     * written.
     */
    hasFile(spec) {
        const target = path.join(this.cacheDir, spec.local);
        if (!fs.existsSync(target)) return false;
        if (fs.statSync(target).size !== spec.bytes) return false;
        return sha256File(target) === spec.sha256;
    }

    /**
     * Check if the pinned revision is fully cached
     */
    hasModel() {
        return this.files.every((spec) => this.hasFile(spec));
    }

    /**
     * Download a file from HuggingFace.
     *
     * Writes to `<dest>.part` and renames on success, so an interrupted or
     * short download never occupies the cached path.
     */
    async downloadFile(url, destPath, expectedBytes, label, expectedSha256) {
        const tmpPath = `${destPath}.part`;

        await new Promise((resolve, reject) => {
            const file = fs.createWriteStream(tmpPath);
            let settled = false;
            const fail = (err) => {
                if (settled) return;
                settled = true;
                file.destroy();
                fs.unlink(tmpPath, () => reject(err));
            };

            const request = (requestUrl, redirectsLeft) => {
                https.get(requestUrl, { headers: { 'User-Agent': 'monocr-npm' } }, (response) => {
                    if ([301, 302, 307, 308].includes(response.statusCode)) {
                        if (redirectsLeft <= 0) {
                            response.resume();
                            fail(new Error(`Too many redirects fetching ${url}`));
                            return;
                        }
                        let redirectUrl = response.headers.location;
                        if (!redirectUrl.startsWith('http')) {
                            const originalUrl = new URL(requestUrl);
                            redirectUrl = `${originalUrl.protocol}//${originalUrl.host}${redirectUrl}`;
                        }
                        response.resume();
                        request(redirectUrl, redirectsLeft - 1);
                    } else if (response.statusCode === 200) {
                        const totalSize = parseInt(response.headers['content-length'], 10);
                        let downloadedSize = 0;

                        response.on('data', (chunk) => {
                            downloadedSize += chunk.length;
                            if (totalSize && process.stdout.isTTY) {
                                const progress = ((downloadedSize / totalSize) * 100).toFixed(1);
                                process.stdout.write(`\r  Downloading ${label}: ${progress}% (${(downloadedSize / 1024 / 1024).toFixed(2)} MB)`);
                            }
                        });
                        response.on('error', fail);

                        response.pipe(file);

                        file.on('finish', () => {
                            file.close((err) => {
                                if (err) return fail(err);
                                if (process.stdout.isTTY && totalSize) process.stdout.write('\n');
                                if (settled) return;
                                settled = true;
                                resolve();
                            });
                        });
                    } else {
                        response.resume();
                        fail(new Error(`Failed to download ${url}: HTTP ${response.statusCode}`));
                    }
                }).on('error', fail);
            };

            file.on('error', fail);
            request(url, 5);
        });

        // Every failure below removes the temp file, not only the mismatch.
        try {
            const actualBytes = fs.statSync(tmpPath).size;
            if (typeof expectedBytes === 'number' && actualBytes !== expectedBytes) {
                throw new Error(
                    `Downloaded ${label} is ${actualBytes} bytes, expected ${expectedBytes} ` +
                    `at revision ${this.revision}. Refusing to cache it.`
                );
            }

            // Verified before the rename, so a corrupt file never reaches the cached
            // path where every binding would trust it.
            if (typeof expectedSha256 === 'string') {
                const actual = sha256File(tmpPath);
                if (actual !== expectedSha256) {
                    throw new Error(
                        `Checksum mismatch for ${label} from ${url}\n` +
                        `  expected sha256 ${expectedSha256}\n` +
                        `  got      sha256 ${actual}\n` +
                        `Revision ${this.revision} is immutable, so this is a corrupted or ` +
                        'intercepted download, not a new release. Refusing to cache it.\n' +
                        'Retry; if it persists, a proxy may be rewriting the download: ' +
                        `fetch ${url} manually into ${path.dirname(destPath)}.`
                    );
                }
            }

            fs.renameSync(tmpPath, destPath);
        } catch (err) {
            fs.rmSync(tmpPath, { force: true });
            throw err;
        }
    }

    /**
     * Download the model and the charset that belongs to it
     */
    async downloadModel() {
        this.ensureCacheDir();
        this.warnAboutLegacyCache();

        console.log(`Downloading monocr model from HuggingFace (${MODEL_REPO} @ ${this.revision})...`);
        console.log(`Cache directory: ${this.cacheDir}`);

        let fetched = 0;
        for (const spec of this.files) {
            if (this.hasFile(spec)) continue;
            // Present but wrong: a partial write, or a file that predates the
            // digest check. Remove it and fetch once; a second mismatch throws
            // from downloadFile rather than looping.
            const dest = path.join(this.cacheDir, spec.local);
            fs.rmSync(dest, { force: true });
            await this.downloadFile(
                `${this.baseUrl}/${spec.remote}`,
                dest,
                spec.bytes,
                spec.label,
                spec.sha256
            );
            fetched += 1;
        }

        console.log(fetched === 0 ? '✓ Already cached.' : '✓ Model downloaded successfully!');
    }

    /**
     * Point at the flat pre-revision cache if it is still lying around. It is
     * dead weight now (~58 MB) and it belongs to a network this code cannot
     * decode, but it is the user's file, so say so rather than delete it.
     */
    warnAboutLegacyCache() {
        const legacy = path.join(this.rootDir, this.modelFileName);
        try {
            if (fs.existsSync(legacy) && fs.statSync(legacy).isFile()) {
                console.warn(
                    `Note: ${legacy} is from an older, unpinned download and is no longer used. ` +
                    'You can delete it.'
                );
            }
        } catch (err) {
            // Never let a cache-hygiene notice break a download.
        }
    }

    /**
     * Get the cached artifacts, downloading if needed.
     *
     * @returns {Promise<{modelPath: string, charsetPath: string, revision: string}>}
     */
    async ensureModel() {
        if (!this.hasModel()) {
            await this.downloadModel();
        }
        return {
            modelPath: this.getModelPath(),
            charsetPath: this.getCharsetPath(),
            revision: this.revision,
        };
    }
}

module.exports = ModelManager;
module.exports.MODEL_REVISION = MODEL_REVISION;
module.exports.MODEL_REPO = MODEL_REPO;
module.exports.REMOTE_FILES = REMOTE_FILES;
module.exports.sha256File = sha256File;
