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

## How a release actually happens

**Pushing a tag. That is the whole mechanism.** `.github/workflows/release-python.yml`
fires on `python/v*` and `release-js.yml` on `js/v*`; each runs the tests, verifies the
built artifact, and publishes. Go needs no workflow at all, because the module proxy
resolves `go/v*` straight from the tag.

Both publish by **trusted publishing (OIDC)**. There is no `PYPI_TOKEN` and no
`NPM_TOKEN` anywhere in this repository, by design: `id-token: write` mints a short-lived
credential per run, and on npm it also attaches a provenance attestation tying the tarball
to the run and the commit.

**Corrected 2026-09-03.** This section used to open "Credentials are not on the dev
machine by default" and tell you to check `npm whoami` and `~/.pypirc`. That describes a
manual local publish, which is not how either binding ships and has not been since these
workflows were written. It is plausibly why `0.3.0` has been tagged and unpublished:
the runbook asked for a login, the workflows wanted a tag, and the two never met.

## The one-time registry setup, which is the actual blocker

**Nothing here is a GitHub permission.** Both workflows already declare what they need:

```yaml
permissions:
  contents: read
  id-token: write
```

`id-token: write` is the one that matters and it is already there. So there is nothing to
grant on this repository, and no secret to add. **What is missing is a one-time
configuration on each registry, and it can only be done by a human with owner rights on
the package.**

### PyPI — `monocr-onnx`

Recorded in `release-python.yml`'s header and reproduced here so it lives in the runbook
too:

> Project `monocr-onnx` → **Publishing** → *Add a GitHub publisher*
> - owner: `MonDevHub`
> - repository: `monocr-onnx`
> - workflow: `release-python.yml`
> - environment: *(blank)*

Requires **Owner** on the PyPI project. Without it the upload is refused and PyPI says so
plainly, which is the good failure — a refused upload is better than a wrong one.

Note this is an *existing* project, not a first publish: `monocr-onnx 0.1.0` has been on
PyPI since 2026-02-14. So it needs a normal publisher entry rather than a pending one.

### npm — `monocr`

The workflow mints an OIDC token for audience `npm:registry.npmjs.org` and runs
`npm publish --provenance --access public`. It needs a trusted publisher configured for
the package `monocr` pointing at `MonDevHub/monocr-onnx` and `release-js.yml`, and it
needs **npm ≥ 11.5.1**, which the runner's bundled npm lags — the workflow installs a
newer one itself.

Requires **owner** on the npm package. `monocr@0.1.5` already exists, so again this is an
existing package.

**Confirm the exact UI path against npm's current documentation before following it.**
Unlike the PyPI settings above, `release-js.yml` does not record npm's menu path, and npm
has moved it before. The workflow's own diagnostics are the thing to trust: it asks the
registry whether a trusted publisher is configured and fails with a named error rather
than a generic 403, so a misconfiguration is legible from the run log.

### If OIDC is refused at the org level

`id-token: write` can be denied by an organisation policy regardless of what the workflow
declares. The symptom is specific and the workflow already checks for it:
`ACTIONS_ID_TOKEN_REQUEST_URL` unset, which it reports as *"this job has no
id-token: write permission and trusted publishing cannot work"*. If that fires, the fix is
an organisation Actions setting and not a change here.

## Publishing from a laptop instead

**Prefer the tag.** The workflows run the tests and verify the artifact before uploading,
and the reason that matters is in the section below: a sibling repository published without
running anything and shipped a decoder that mapped every index to the wrong character.

If a local publish is genuinely necessary, then credentials are needed and are not on the
machine by default:

```bash
npm whoami                  # ENEEDAUTH means not logged in
ls ~/.pypirc                # or set UV_PUBLISH_TOKEN / TWINE_* in the environment
```

A local publish gets **no provenance attestation**, so the tarball cannot be traced to a
commit afterwards. Treat it as the fallback it is, and follow §1 to §3 below by hand.

## 1. Verify before building

```bash
cd go     && gofmt -l . && go vet ./... && go test ./...
cd ../js  && npm test
cd ../python && .venv/bin/python3 -m pytest -q
cd ../rust && cargo fmt --check && cargo check --all-targets && cargo clippy --all-targets
```

`cargo test` fails to link on macOS with `ld: library 'clang_rt.osx' not found`.
This is **not** a missing-Xcode problem and it is fixable: Command Line Tools 21
removed the clang 17 directory the linker searches, while the runtime itself is
present under the current version. Point at it:

```bash
export RUSTFLAGS="-C link-arg=-L/Library/Developer/CommandLineTools/usr/lib/clang/21/lib/darwin"
cargo test        # 28 lib tests + 14 doc-tests
```

Adjust `21` to whatever `ls /Library/Developer/CommandLineTools/usr/lib/clang/`
reports. Earlier releases recorded the Rust binding as "unverified by execution"
on the strength of this error; that was avoidable.

If it still does not run, say so in the release notes rather than implying Rust
was tested.

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
