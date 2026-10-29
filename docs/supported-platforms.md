# Supported platforms

All four bindings run on the CPU and are built for Linux, Windows and
Apple-silicon macOS. CI runs on Linux only.

## Intel Macs

Intel Macs are narrower:

- **JavaScript:** `onnxruntime-node` ships no Intel-macOS binary.
- **Rust:** the crate's build has no prebuilt ONNX Runtime to link there; see
  [`rust/README.md`](../rust/README.md) for pointing it at your own build.
- **Go:** neither Microsoft's 1.24.1 release nor Homebrew has a prebuilt
  Intel-macOS library; see [`go/README.md`](../go/README.md).
- **Python:** still installs, on onnxruntime 1.23.2 (the last release with an
  Intel-macOS wheel) and Python 3.11 to 3.13.

