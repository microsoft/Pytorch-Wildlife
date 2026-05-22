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

---

> 📖 **[Read the full user manual →](docs/user-manual.md)**
>
> One document covering install, CLI (`spe`), Python wheel (`import sparrow_engine`), HTTP API server, HTTP SDK, native DLL (C ABI), TOML model manifests, the Phase 4 inference-log / drift / provenance surface, cold-start + lazy load, gotchas + edge cases, performance characteristics, and Sparrow Studio integration.

---

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

## License

See [`LICENSE`](LICENSE).
