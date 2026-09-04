# MonOCR Release Runbook

> [!IMPORTANT]
> **Superseded by [`../RELEASING.md`](../RELEASING.md). Read that first.**
>
> Marked 2026-09-03, when three release runbooks were found in this repository saying
> different things. This one omits the artifact-verification step that `RELEASING.md`
> records as the cause of the original six-month publish gap: `0.2.0` and `0.2.1` were
> tagged and never published while the registries served a 64px, 225-character build.
>
> It also predates trusted publishing. Releases happen by pushing a `python/v*` or
> `js/v*` tag, with no local credentials; `RELEASING.md` carries the one-time registry
> setup that is the real precondition.
>
> Kept rather than deleted because the per-binding smoke tests below are still useful.
> Use it for those and nothing else.

> **Context**: We use a manual release process. Automation is great, but for this stage of the project, we want full control over exactly what bits assume the `latest` tag.
>
> **Golden Rule**: If it's not in `main`, it doesn't exist. If it's not tested, it's broken.

## Pre-Flight Checks

Before you even think about publishing, verify your environment matches our constraints.

| Runtime     | Requirement     | Check Commmand     |
| :---------- | :-------------- | :----------------- |
| **Python**  | `3.11` (Strict) | `python --version` |
| **Node.js** | `20+`           | `node -v`          |
| **pnpm**    | `9+` (Strict)   | `pnpm -v`          |
| **Go**      | `1.23+`         | `go version`       |

---

## Python (`monocr-onnx`)

We use `uv` because it's fast and correct. Don't use `setup.py`.

### 1. Build & verify

Clean slate first. We don't want stale artifacts.

```bash
cd python
rm -rf dist/
uv build
```

### 2. Smoke Test

Install the wheel we just built in a fresh venv to ensure we didn't miss a file in `include`.

```bash
uv venv .test-venv
source .test-venv/bin/activate
uv pip install dist/*.whl
python -c "import monocr_onnx; print(monocr_onnx.__version__)"
deactivate
rm -rf .test-venv
```

_(If that failed, fix your `pyproject.toml`)_

### 3. Ship it

```bash
uv publish
```

---

## JavaScript (`monocr`)

We use `pnpm`. It enforces strict dependency boundaries which saves us from "it works on my machine" phantom dependency issues.

### 1. Hard Reset

Ensure dependencies are exactly what the lockfile says.

```bash
cd js
pnpm install --frozen-lockfile
pnpm approve-builds # Build binaries (sharp, onnxruntime)
```

### 2. Verify

Run the batch example. It's our de-facto integration test.

```bash
node ../examples/js/batch-example.js
```

### 3. Version & Release

**Crucial**: `pnpm publish` checks for a clean git state. Commit your version bump _before_ running this.

```bash
pnpm version patch # or minor
git commit -am "chore(js): bump version"
pnpm publish --access public
```

---

## Go (`github.com/MonDevHub/monocr-onnx/go`)

Go modules are just git tags. But we have a mono-repo, so tags are namespaced.

### 1. Tidy Up

Never ship a `go.mod` that hasn't been tidied.

```bash
cd go
go mod tidy
go test -v ./...
```

### 2. Tag It

The tag **must** look like `go/vX.Y.Z`. If you tag it `vX.Y.Z`, Go won't find the module in the subdirectory.

```bash
# assuming you're in root
git tag go/v0.1.1
git push origin go/v0.1.1
```

### 3. Verify

Ask the proxy to index it immediately.

```bash
GOPROXY=proxy.golang.org go list -m github.com/MonDevHub/monocr-onnx/go@v0.1.1
```
