# Contributing

Bug reports and pull requests are welcome. The four bindings (Python, JavaScript, Go, Rust) share one
version and one model contract; see [README.md § How it works](README.md#how-it-works) and
[§ Limitations](README.md#limitations) for where they still differ, and [RELEASING.md](RELEASING.md)
for how a release is cut.

## Running the tests

These are the commands CI runs; each line starts from the repository root:

```bash
cd python && pip install . pytest && pytest tests/
cd js     && pnpm install --frozen-lockfile && pnpm test
cd go     && test -z "$(gofmt -l .)" && go vet ./... && go test ./...
cd rust   && cargo fmt --check && cargo test
```

The Rust tests read three shared fixtures from [MonDevHub/monocr](https://github.com/MonDevHub/monocr),
set by `MONOCR_TILING_FIXTURE`, `MONOCR_RULE_FIXTURE` and `MONOCR_MERGE_FIXTURE`. Unset, each defaults to
`shared/segmentation-fixtures/` in a sibling clone named `monocr-monorepo` beside this repository. On
macOS, `cargo test` may fail to link with `clang_rt.osx` not found; [RELEASING.md](RELEASING.md#1-verify-before-building)
has the fix.

## Segmentation and model changes: A/B first, ship only if better

A change that can alter the text a binding returns ships only with an A/B comparison, and
only if it is better. That covers the segmenter or its constants, tiling, preprocessing, decoding,
and a new model revision.

- **Two arms, same inputs.** A is the current code, B is the change. Nothing else differs between
  them: same images, same model revision, same settings.
- **Decide what counts as better before running.** Name the primary metric (usually CER, or for
  segmentation, lines found, missed and merged) and the guard metrics, and state the result that
  would ship B.
- **Ship only if B wins the primary metric and loses no guard metric.** A tie or an unclear result
  does not ship. If the change is still worth keeping, it goes behind an option that is off by
  default.
- **Put the result in the pull request:** the inputs, both arms' numbers, any per-bucket split (for
  example by line width), the exact command, and what the comparison doesn't cover.
- **Keep the bindings in step.** A change to one binding's segmentation says whether the others
  get the same change, and updates [README.md § Limitations](README.md#limitations) if they now
  differ. A change to tiling in `python/monocr_onnx/segmenter.py` also regenerates
  `shared/segmentation-fixtures/tiling-cases.json` in MonDevHub/monocr, which is generated from this
  binding's `tile_line`.
