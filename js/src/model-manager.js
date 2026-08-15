const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');

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
const MODEL_REVISION = 'a51be11';
const MODEL_REPO = 'janakhpon/monocr';

/**
 * Files fetched from the pinned revision. `bytes` is the size Hugging Face
 * reports for that revision; a file on disk of any other size is a partial or
 * corrupted download, not a cache hit.
 */
const REMOTE_FILES = [
    { remote: 'onnx/monocr.onnx', local: 'monocr.onnx', bytes: 26355440, label: 'model' },
    { remote: 'onnx/charset.txt', local: 'charset.txt', bytes: 674, label: 'charset' },
];

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
     * True when a cached file is present *and* the right size. Size alone is not
     * integrity, but it does catch the failure this cache actually suffers:
     * an interrupted download left behind as a permanent, silently truncated
     * "cache hit".
     */
    hasFile(spec) {
        const target = path.join(this.cacheDir, spec.local);
        if (!fs.existsSync(target)) return false;
        return fs.statSync(target).size === spec.bytes;
    }

    /**
     * Check if the pinned revision is fully cached
     */
    hasModel() {
        return REMOTE_FILES.every((spec) => this.hasFile(spec));
    }

    /**
     * Download a file from HuggingFace.
     *
     * Writes to `<dest>.part` and renames on success, so an interrupted or
     * short download never occupies the cached path.
     */
    async downloadFile(url, destPath, expectedBytes, label) {
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

        const actualBytes = fs.statSync(tmpPath).size;
        if (typeof expectedBytes === 'number' && actualBytes !== expectedBytes) {
            fs.unlinkSync(tmpPath);
            throw new Error(
                `Downloaded ${label} is ${actualBytes} bytes, expected ${expectedBytes} ` +
                `at revision ${this.revision}. Refusing to cache it.`
            );
        }

        fs.renameSync(tmpPath, destPath);
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
        for (const spec of REMOTE_FILES) {
            if (this.hasFile(spec)) continue;
            await this.downloadFile(
                `${this.baseUrl}/${spec.remote}`,
                path.join(this.cacheDir, spec.local),
                spec.bytes,
                spec.label
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
