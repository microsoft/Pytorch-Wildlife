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

| Model ID | Task | Input | Output | ONNX | License | Notes |
|---|---|---|---|---|---|---|
| `MDV6-yolov10-c` | Detector | 640×640 RGB, letterbox | bboxes, in-graph NMS (animal / person / vehicle) | 9 MB | Ultralytics AGPL-3.0 | MegaDetector v6, compact / fast variant. |
| `MDV6-yolov10-e` | Detector | 1280×1280 RGB, letterbox | bboxes, in-graph NMS (animal / person / vehicle) | 113 MB | Ultralytics AGPL-3.0 | MegaDetector v6, highest-accuracy variant. |
| `Species_Net_MDV5a` | Detector | 1280×1280 RGB, letterbox | bboxes (animal / person / vehicle) | 535 MB | Ultralytics AGPL-3.0 | Legacy MegaDetector v5a; kept for projects validated against v5a outputs pre-v6. |
| `deepfaune-yolo8s` | Detector | 960×960 RGB, letterbox | bboxes (3 cls, MD-style) | 43 MB | AGPL-3.0 ∩ CC-BY-NC-SA 4.0 | DeepFaune detector stage; pairs with `Deepfaune-Europe` / `Deepfaune-New-England` classifiers. |
| `Deepfaune-Europe` | Classifier | 182×182 RGB crop | softmax, 34 cls | 1.2 GB | CC-BY-NC-SA 4.0 | DeepFaune classifier for European mammals. Downstream of a detector. |
| `Deepfaune-New-England` | Classifier | 182×182 RGB crop | softmax, 24 cls | 1.2 GB | CC-BY-NC-SA 4.0 | DeepFaune classifier for New England (NA) mammals. Downstream of a detector. |
| `HerdNet_General_Dataset_2022` | Heatmap detector | 512×512 RGB, resize | point detections, 6 species (+ bg) | 70 MB | MIT | Counts large African mammals (elephants, antelopes, zebras, etc.) in low-altitude aerial / drone imagery. |
| `OWL` | Heatmap detector | 512×512 RGB, resize, tiled | point detections → fixed-size boxes (animal) | 114 MB | MIT | Tiled detection of small wildlife in large camera-trap / aerial scenes. |
| `SpeciesNet-Crop` | Classifier | 480×480 RGB crop | softmax, 2498 cls | 214 MB | Apache 2.0 | Google SpeciesNet species classifier; pairs downstream of a detector (e.g. MDv6). |
| `AI4G-Amazon-V2` | Classifier | 224×224 RGB crop | softmax, 36 cls | 90 MB | MIT | Amazon-basin species, Microsoft AI for Good Lab. |
| `AI4G-Serengeti` | Classifier | 224×224 RGB crop | softmax, 10 cls | 43 MB | MIT | Serengeti / East African species, Microsoft AI for Good Lab. |
| `european_mammals` | Detector | 640×480 RGB, letterbox | bboxes, in-graph NMS (31 cls) | 113 MB | Ultralytics AGPL-3.0 | AI for Good Lab regional YOLO. |
| `north_american_mammals` | Detector | 640×480 RGB, letterbox | bboxes, in-graph NMS (14 cls) | 113 MB | Ultralytics AGPL-3.0 | AI for Good Lab regional YOLO. |
| `sub_saharan` | Detector | 640×480 RGB, letterbox | bboxes, in-graph NMS (35 cls) | 113 MB | Ultralytics AGPL-3.0 | AI for Good Lab regional YOLO. |
| `perch-v2` | Audio classifier | 5 s @ 32 kHz raw audio (160000 samples; in-graph mel front-end) | softmax, 14795 cls (birds + non-bird FSD50K) | 391 MB | Apache 2.0 | Google Perch 2 global bird-vocalisation classifier (Conformer encoder). |

**License summary**: Ultralytics AGPL-3.0 (7 models: MDv6 × 2, MDv5a, 3 regional YOLOs, plus `deepfaune-yolo8s` which intersects with CC-BY-NC-SA 4.0) · CC-BY-NC-SA 4.0 (3 models: `deepfaune-yolo8s`, `Deepfaune-Europe`, `Deepfaune-New-England`) · Apache 2.0 (2 models: `SpeciesNet-Crop`, `perch-v2`) · MIT (4 models: `AI4G-Amazon-V2`, `AI4G-Serengeti`, `OWL`, `HerdNet`).

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
