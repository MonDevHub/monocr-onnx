# The model contract

Every binding loads the same model at the same revision and checks it before
decoding, so a mismatch fails loudly instead of returning wrong text.

## Pinning and cache

The model is pinned to revision `d3d9d5e` of
[`janakhpon/monocr`](https://huggingface.co/janakhpon/monocr), not to `main`, and
cached under `~/.monocr/models/<revision>/`.
[`scripts/download_models.sh`](../scripts/download_models.sh) fetches the same
revision by hand.

## Checks before decoding

- **Charset.** Each binding ships the matching charset and refuses to decode if
  the two disagree: CTC reserves index 0 for the blank, so a model over N
  characters must emit N + 1 classes. Without that check a mismatch returns
  well-formed Mon text that is wrong, with no error and no lookup miss.
- **Non-finite output.** Logits holding a NaN or an infinity raise
  `ModelOutputError` (Go: `*predictor.OutputError`).
- **Wrong shape.** Logits that are not `[1, sequence, N + 1]` raise
  `ModelContractError` (Go: `*predictor.ContractError`).

Both output checks fail rather than decode into a blank or wrong line.

## Names per binding

The pipeline in the [README](../README.md#how-it-works) uses the Python and
JavaScript class names; JavaScript's page method is `predictPage`. Rust spells
the engine `MonOcr`; Go uses `model.Manager`, `segmenter.LineSegmenter` and
`predictor.Predictor`.
