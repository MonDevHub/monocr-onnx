# Publishing MonOCR (JS) with pnpm

Quick reference for publishing new versions using `pnpm`.

## Prerequisites

- **Install**: `npm install -g pnpm`
- **Login**: `pnpm login`

## Steps

1. **Verify State**

   ```bash
   pnpm install
   pnpm approve-builds  # Important for sharp/canvas binaries
   pnpm test            # Must pass: it checks the model/charset contract
   ```

   If `src/charset.txt` or the pinned `MODEL_REVISION` changed, both move in the
   same commit. The tests assert they describe one model; do not publish red.

2. **Commit Changes**

   pnpm requires a clean git working tree by default.

   ```bash
   git add .
   git commit -m "chore: release prep"
   ```

3. **Bump Version**

   ```bash
   pnpm version patch   # 0.1.0 -> 0.1.1
   # or
   pnpm version minor   # 0.1.0 -> 0.2.0
   ```

4. **Publish**

   ```bash
   pnpm publish --access public
   ```

   _Tip: If you must publish with uncommitted changes (not recommended), use `--no-git-checks`._

5. **Push Tags**

   ```bash
   git push origin main --tags
   ```

## Files

Run `pnpm pack` to see exactly what will be uploaded (tarball).
It should include:

- `src/`
- `bin/`
- `README.md`
- `LICENSE`
- `package.json`

(Model files are excluded).

## Full Guide

See [docs/publishing.md](../docs/publishing.md) for complete details on all SDKs.
