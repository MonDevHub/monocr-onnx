# Releasing the bindings

All four bindings share one model contract and one version number. A release
means the same contract in every language, so they go out together or the number
stops meaning anything.

## Why this file exists

`0.2.0` and `0.2.1` were bumped, tagged (`go/v0.2.0`, `js/v0.2.1`,
`python/v0.2.1`) and **never published**. For six months `npm install monocr` and
`pip install monocr-onnx` delivered a 64px-height, 225-character-charset build
while this tree, its READMEs and its tests all described 160px and 276
characters. Nothing failed, because nothing checked. A tag is not a release.

## Preconditions

Credentials are not on the dev machine by default. Check first:

```bash
npm whoami                  # ENEEDAUTH means not logged in
ls ~/.pypirc                # or set UV_PUBLISH_TOKEN / TWINE_* in the environment
```

## 1. Verify before building

```bash
cd go     && gofmt -l . && go vet ./... && go test ./...
cd ../js  && npm test
cd ../python && .venv/bin/python3 -m pytest -q
cd ../rust && cargo fmt --check && cargo check --all-targets && cargo clippy --all-targets
```

`cargo test` cannot link on macOS with Command Line Tools only (`ld: library
'clang_rt.osx' not found`). If it does not run, say so in the release notes rather
than implying Rust was tested.

## 2. Build

```bash
cd python && uv build --out-dir ../dist
cd ../js  && npm pack --pack-destination ../dist
```

## 3. Verify the ARTIFACTS, not the tree

This is the step whose absence caused the six-month gap. A green test suite says
nothing about what got packed — `.npmignore`, `include`/`exclude` in
`pyproject.toml` and a stale `dist/` can all ship something other than what you
tested.

```bash
cd dist && mkdir -p verify && cd verify
unzip -oq ../monocr_onnx-*.whl -d whl
tar xzf ../monocr-*.tgz

# charset must be 276 chars, sha256 edfd75f688e4155c…, first char U+0020
python3 -c "
import hashlib
for f in ('whl/monocr_onnx/charset.txt','package/src/charset.txt'):
    b=open(f,'rb').read(); s=b.decode().strip('\n\r')
    print(f, len(s), hashlib.sha256(b).hexdigest()[:16], hex(ord(s[0])))"

# geometry and revision
grep -hoE 'EXPECTED_INPUT_HEIGHT = [0-9]+|DEFAULT_INPUT_WIDTH = [0-9]+' whl/monocr_onnx/predictor.py
grep -hoE 'targetHeight = [0-9]+|targetWidth = [0-9]+' package/src/monocr.js
grep -hoE 'REVISION = "[a-f0-9]+"' whl/monocr_onnx/model_manager.py

# the superseded normalisation must be ABSENT. Check grep's own exit status:
# `grep … | sed …` returns sed's status, so an `|| echo clean` fallback after a
# pipeline never fires and tests nothing.
for f in whl/monocr_onnx/predictor.py package/src/monocr.js; do
  grep -qE '/[[:space:]]*255' "$f" && echo "STALE /255 in $f" || echo "clean: $f"
done

# and the package version, from ^Version: — not Wheel-Version, which is 2.5
grep -E '^Version:' whl/monocr_onnx-*.dist-info/METADATA
```

Expected: 276 / `edfd75f688e4155c` / `0x20`, height 160, width 1024, revision
`d3d9d5e`, no `/255`, and the version you intended.

## 4. Publish

Irreversible. A PyPI version number can never be reused even after deletion, and
npm `unpublish` closes after 72 hours.

```bash
cd python && uv publish --token "$PYPI_TOKEN" ../dist/monocr_onnx-0.3.0*
cd ../js  && npm publish ../dist/monocr-0.3.0.tgz --access public
cd ../rust && cargo publish            # never published; expect a first-owner flow
```

Go needs no registry step — modules resolve by tag, so pushing `go/v0.3.0` is the
release.

## 5. Push the tags

```bash
git push origin python/v0.3.0 js/v0.3.0 rust/v0.3.0 go/v0.3.0
```

## 6. Confirm from outside

Do not trust the publish command's own output.

```bash
curl -s https://registry.npmjs.org/monocr | python3 -c "import json,sys;print(json.load(sys.stdin)['dist-tags'])"
curl -s https://pypi.org/pypi/monocr-onnx/json | python3 -c "import json,sys;print(json.load(sys.stdin)['info']['version'])"
```

Then re-run step 3 against the *downloaded* artifact, not the local one. That is
the only check that would have caught the 0.1.x situation.
