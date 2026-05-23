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

### Per-model catalog

This is a **multi-license bundle** — each model ships under its own upstream license. Open each `models/<model_id>/LICENSE.md` after download for the canonical terms.

The catalog splits into four families (detectors, heatmap detectors, classifiers, audio). All detectors emit bounding boxes via in-graph NMS; all classifiers consume crops produced by an upstream detector.

#### Bounding-box detectors

| Model ID | Resolution | Classes | ONNX | License |
|---|---|---|---|---|
| `MDV6-yolov10-c` | 640 × 640 | 3 (animal / person / vehicle) | 9 MB | Ultralytics AGPL-3.0 |
| `MDV6-yolov10-e` | 1280 × 1280 | 3 (animal / person / vehicle) | 113 MB | Ultralytics AGPL-3.0 |
| `Species_Net_MDV5a` | 1280 × 1280 | 3 (animal / person / vehicle) | 535 MB | Ultralytics AGPL-3.0 |
| `deepfaune-yolo8s` | 960 × 960 | 3 (MD-style) | 43 MB | AGPL-3.0 ∩ CC-BY-NC-SA 4.0 |
| `european_mammals` | 640 × 480 | 31 | 113 MB | Ultralytics AGPL-3.0 |
| `north_american_mammals` | 640 × 480 | 14 | 113 MB | Ultralytics AGPL-3.0 |
| `sub_saharan` | 640 × 480 | 35 | 113 MB | Ultralytics AGPL-3.0 |

- MegaDetector v6 (`MDV6-yolov10-c` / `-e`) is the recommended default detector — `-c` for speed, `-e` for accuracy.
- `Species_Net_MDV5a` is the legacy v5a detector; kept for projects validated against v5a outputs.
- `deepfaune-yolo8s` is the DeepFaune detector stage, designed to pair with `Deepfaune-Europe` / `Deepfaune-New-England` classifiers.
- `european_mammals` / `north_american_mammals` / `sub_saharan` are the AI for Good Lab regional YOLO detectors (multi-species per region).

#### Heatmap-based detectors

| Model ID | Resolution | Classes | ONNX | License |
|---|---|---|---|---|
| `HerdNet_General_Dataset_2022` | 512 × 512 | 6 species + background | 70 MB | MIT |
| `OWL` | 512 × 512 (tiled) | 1 (animal) | 114 MB | MIT |

- `HerdNet_General_Dataset_2022` counts large African mammals (elephants, antelopes, zebras, etc.) in low-altitude aerial / drone imagery.
- `OWL` does tiled detection of small wildlife in large camera-trap or aerial scenes; converts heatmap peaks to fixed-size boxes.

#### Image classifiers (consume crops from a detector)

| Model ID | Crop | Classes | ONNX | License |
|---|---|---|---|---|
| `Deepfaune-Europe` | 182 × 182 | 34 | 1.2 GB | CC-BY-NC-SA 4.0 |
| `Deepfaune-New-England` | 182 × 182 | 24 | 1.2 GB | CC-BY-NC-SA 4.0 |
| `SpeciesNet-Crop` | 480 × 480 | 2498 | 214 MB | Apache 2.0 |
| `AI4G-Amazon-V2` | 224 × 224 | 36 | 90 MB | MIT |
| `AI4G-Serengeti` | 224 × 224 | 10 | 43 MB | MIT |

- `Deepfaune-Europe` / `Deepfaune-New-England` are the DeepFaune classifier stage for European and New England (NA) mammals.
- `SpeciesNet-Crop` is Google's SpeciesNet classifier; pairs downstream of a detector (e.g. MDv6).
- `AI4G-Amazon-V2` and `AI4G-Serengeti` are AI for Good Lab regional classifiers for Amazon-basin and Serengeti / East African species.

#### Audio detectors / classifiers

| Model ID | Input window | Classes | ONNX | License |
|---|---|---|---|---|
| `MD_AudioBirds_V1` | 1 s @ 48 kHz, mel spectrogram (0.3 s stride) | 1 (bird vs no-bird) | 81 MB | MIT |
| `perch-v2` | 5 s @ 32 kHz raw audio | 14795 | 391 MB | Apache 2.0 |

- `MD_AudioBirds_V1` is the sparrow-engine default audio detector — a lightweight binary bird-vs-no-bird model used in benchmarks and Phase 4.x manual tests. Sliding-window mel-spectrogram front-end (Slaney mel scale + Slaney filter norm). Ships in the v0.1.0 Zenodo bundle as FP32; the FP16 conversion path is in `sparrow-engine/tools/convert_fp16.py` and is parity-verified against the FP32 reference (Phase 3.8 Step 2 post-STRETCH audit, 2026-05-05).
- `perch-v2` is Google Perch 2, a global bird-vocalisation classifier (Conformer encoder) with an in-graph mel front-end. Takes 160000-sample windows of raw audio; emits softmax over 14795 classes (birds + non-bird FSD50K labels).

#### License summary

- **Ultralytics AGPL-3.0** (7 models): MDv6 × 2, MDv5a, the 3 AI4G regional YOLOs, plus `deepfaune-yolo8s` (which also intersects CC-BY-NC-SA 4.0).
- **CC-BY-NC-SA 4.0** (3 models): `deepfaune-yolo8s`, `Deepfaune-Europe`, `Deepfaune-New-England`.
- **Apache 2.0** (2 models): `SpeciesNet-Crop`, `perch-v2`.
- **MIT** (5 models): `AI4G-Amazon-V2`, `AI4G-Serengeti`, `OWL`, `HerdNet_General_Dataset_2022`, `MD_AudioBirds_V1`.

**Commercial users of YOLO-based detectors** should obtain an [Ultralytics Enterprise License](https://www.ultralytics.com/license).

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

---

## Internal development

This is the **public** sparrow-engine repo. It carries the shipping code, the install wrapper, models, and one user-facing manual.

Dev/AI artifacts — design rounds, research notes, audit-fix / doc-fix / `/implement` skill rounds, inquisitor reports, scope ledgers, prompt logs, agent instructions, plan / changelog / lessons / ideas — live in the **internal dev companion** repo (`zhmiao/sparrow-engine-dev`), NOT here. See that repo's `docs/design/architecture.md § Internal dev companion convention` for the full rule.
