# Sparrow Engine

A Rust ML inference engine for camera-trap and bioacoustic data.
Drop-in for MegaDetector v6, DeepFaune, HerdNet, OWL-T, SpeciesNet, and
MD_AudioBirds_V1; model-agnostic via TOML manifests.

## Quickstart

Clone the repo and run the install wrapper (CWD = repo root):

```bash
# Linux / macOS
bash installer/sparrow-engine-install.sh
```

```powershell
# Windows PowerShell
installer\sparrow-engine-install.ps1
```

The wrapper probes hardware once, picks the right CPU or GPU build, and
installs the matching CLI binary plus the Python wheel into `~/.sparrow_engine/`.
Pass `--flavor cpu` or `--flavor gpu` to skip the probe. Pass `--docker`
to install the HTTP-server image instead.

System prerequisites for GPU: NVIDIA driver ≥550.x, CUDA 12.6 runtime,
and **cuDNN ≥9.10** (cuDNN 9.8 has a Conv-engine bug on sm_89).

See [`docs/user-manual.md`](docs/user-manual.md) for the full install
reference (per-platform commands, air-gapped path, exit-code catalog
0–14, troubleshooting).

## Architecture

Sparrow Engine is engine-only: it loads ONNX models and runs inference.
Annotation, training, data versioning, model registry, drift detection,
and deployment orchestration live in sibling repos.

Core invariants:

- ONNX for all models (vision + audio)
- NCHW layout mandatory
- Normalized bbox `[0,1]` at all public API boundaries
- TOML manifests (one per model)
- NMS in the ONNX graph, never in the Sparrow Engine
- `Engine` is a singleton (ORT is process-global)

## Documentation

- [`docs/user-manual.md`](docs/user-manual.md) — User manual: install,
  CLI (`spe`), Python (`import sparrow_engine`), HTTP API server,
  HTTP SDK, native DLL (C ABI), models + TOML manifests, the Phase 4
  inference-log / drift / provenance surface, cold-start + lazy load,
  gotchas + edge cases, performance characteristics, Sparrow Studio
  integration.

## License

See [`LICENSE`](LICENSE).
