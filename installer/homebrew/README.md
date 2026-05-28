# Homebrew tap — sparrow-engine

This directory holds the source-of-truth Homebrew formulas for the
`sparrow-engine` + `sparrow-engine-gpu` CLI binaries (RP-17). The
formulas live here, not in a separate tap repo, so they version with
the rest of the codebase.

## What ships

- `sparrow-engine.rb` — CPU formula. macOS arm64 + brew-Linux x86_64. Ships
  the `spe` binary with bundled `libonnxruntime`.
- `sparrow-engine-gpu.rb` — GPU formula. brew-Linux x86_64 only. Ships
  `spe-gpu` with bundled `libonnxruntime` + ORT CUDA provider sidecars.
  Adds a wrapper script that auto-discovers all 8 required CUDA sidecar
  SONAMEs (`libcudnn.so.9`, `libnvjpeg.so.12`, `libnvrtc.so.12`,
  `libcudart.so.12`, `libcublas.so.12` + `libcublasLt.so.12`,
  `libcurand.so.10`, `libcufft.so.11`) — 7 pip packages, since `libcublas`
  and `libcublasLt` ship together in `nvidia-cublas-cu12` — from common
  host locations at startup (no `LD_LIBRARY_PATH` manual setup required).

Both formulas point at the GH Release tarballs produced by RP-4
(`.github/workflows/release.yml § build-cli-*` + `publish-cli-release-assets`).

## End-user UX (post-tap-publish)

CPU (works on Apple Silicon Macs + brew-Linux):
```bash
brew tap microsoft/sparrow-engine
brew install sparrow-engine
spe device                  # {"device":"cpu"}
spe detect /path/to/photo.jpg --model MDV6-yolov10-e --print
```

GPU (Linux x86_64 + NVIDIA CUDA only):
```bash
brew install sparrow-engine-gpu
spe-gpu device              # {"device":"cuda:0"} when host has cuDNN
spe-gpu detect /path/to/photos/ --model MDV6-yolov10-e --recursive --print
```

Both binaries coexist cleanly — install one, the other, or both. They
share the model cache at `~/.sparrow-engine/models/`.

## The wrapper script (GPU only)

`brew install sparrow-engine-gpu` generates `bin/spe-gpu` as a small
POSIX shell wrapper (not a symlink) that auto-discovers all 8 hard-
required CUDA sidecar SONAMEs (across 7 pip packages — see table below)
before `exec`'ing the real binary at
`libexec/bin/spe-gpu`. The discovery loop covers, per library, the same
9 candidate locations (pip wheel under `~/.sparrow-engine/cuda-sidecars`,
PyTorch / TensorFlow / JAX bundles, system CUDA, apt, HPC, RHEL). The
wrapper eliminates the `LD_LIBRARY_PATH` setup production users would
otherwise need.

Required libs (matches `probe_gpu_quality.sh:144-150`):

| pip package                     | library            |
|---------------------------------|--------------------|
| `nvidia-cudnn-cu12`             | `libcudnn.so.9`    |
| `nvidia-nvjpeg-cu12`            | `libnvjpeg.so.12`  |
| `nvidia-cuda-nvrtc-cu12`        | `libnvrtc.so.12`   |
| `nvidia-cuda-runtime-cu12`      | `libcudart.so.12`  |
| `nvidia-cublas-cu12`            | `libcublas.so.12` + `libcublasLt.so.12` (same dir) |
| `nvidia-curand-cu12`            | `libcurand.so.10`  |
| `nvidia-cufft-cu12`             | `libcufft.so.11`   |

Override the search via `SPARROW_ENGINE_CUDA_LIB_DIR=/some/path
spe-gpu …` — the wrapper skips auto-discovery for all 8 SONAMEs when this
env var is set. Brew rewrites the wrapper on every `(re)install`; do
NOT edit it in place.

Full caveats block (with the 9 location list and the 7-package /
8-SONAME table) appears at the end of `brew install` output, or run
`brew info sparrow-engine-gpu`.

## Bootstrapping the tap repo (one-time, operator action — DONE 2026-05-27)

The tap is live at https://github.com/microsoft/homebrew-sparrow-engine
with both formulas pinned to v0.1.10. Procedure if cutting fresh:

1. Cut the release: `git tag vX.Y.Z && git push origin vX.Y.Z` — CI runs
   `publish-cli-release-assets` and attaches CPU + GPU tarballs + sha256
   sidecars to the GH Release.
2. Fetch the SHA256 sidecars:
   ```bash
   gh release download vX.Y.Z --repo microsoft/Pytorch-Wildlife \
     --pattern '*.sha256' --dir /tmp/sha
   ```
3. Copy this directory's formulas to the tap repo, substituting the
   `REPLACE_WITH_*_sha256` placeholders for the real SHA256s.
4. Commit + push to `microsoft/homebrew-sparrow-engine` `main`.
5. Smoke-install on a macOS arm64 host (CPU) and a brew-Linux + NVIDIA
   host (GPU):
   ```bash
   brew update
   brew install sparrow-engine sparrow-engine-gpu
   spe device         # {"device":"cpu"}
   spe-gpu device     # {"device":"cuda:0"}
   ```

## Per-release bump (after bootstrap)

Each subsequent release: fetch new `.sha256` files, substitute, commit
to the tap repo on a feature branch, fast-forward merge to `main`,
push. Automatable via a small helper script (deferred — see
`sparrow-engine-dev/docs/ideas.md § RP-17 follow-ups` if/when needed).

## Why not just submit to homebrew-core?

`homebrew-core` has acceptance criteria (notable, maintained, widely
used). Pre-public-release, sparrow-engine doesn't meet them yet. The
custom tap is the bridge until the project is established enough to
warrant a core submission. Migration from custom-tap → core later is
straightforward (formula source code is the same; just submit the
file via PR to homebrew/homebrew-core).
