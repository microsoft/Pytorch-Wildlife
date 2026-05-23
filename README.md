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

## Model zoo

Sparrow Engine doesn't ship the ONNX model weights in the repo. They live in a public Zenodo record so the repo stays small and operators can pull just the models they need.

**Zenodo DOI**: [10.5281/zenodo.20351248](https://doi.org/10.5281/zenodo.20351248) (v0.2.0) — concept DOI [10.5281/zenodo.20348978](https://doi.org/10.5281/zenodo.20348978) always resolves to the latest version.

Download all 15 models to `./models/`:

```bash
bash scripts/download_models.sh
```

Or just specific models:

```bash
bash scripts/download_models.sh MDV6-yolov10-e SpeciesNet-Crop
bash scripts/download_models.sh --list          # list available model IDs
bash scripts/download_models.sh --dest /custom/path
```

Point Sparrow Engine at the directory:

```bash
export SPARROW_ENGINE_MODELS_DIR=$(realpath ./models)
spe models list                                 # confirms catalog discovery
spe detect --model MDV6-yolov10-e --print image.jpg
```

The downloader verifies SHA-256 per model, is idempotent (skip-if-present unless `--force`), and unpacks into the layout Sparrow Engine expects (`<dir>/<model_id>/manifest.toml` + `model.onnx` + `labels.txt`).

### Per-model licensing

This is a **multi-license bundle** — each model ships under its own upstream license:

| License | Models |
|---------|--------|
| Ultralytics AGPL-3.0 | MDV6 × 2, MDV5a, deepfaune-yolo8s, european / NA / sub-Saharan mammals |
| CC-BY-NC-SA 4.0 | Deepfaune-Europe, Deepfaune-New-England |
| AGPL-3.0 + CC-BY-NC-SA 4.0 (intersection) | deepfaune-yolo8s (also YOLO) |
| Apache 2.0 | SpeciesNet-Crop, perch-v2 |
| MIT | AI4G-Amazon-V2, AI4G-Serengeti, OWL, HerdNet |

Open each `models/<model_id>/LICENSE.md` after download for the canonical terms. **Commercial users of YOLO-based detectors** should obtain an [Ultralytics Enterprise License](https://www.ultralytics.com/license).

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
