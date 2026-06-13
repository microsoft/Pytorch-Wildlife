# Sparrow Engine

A Rust ML inference engine for camera-trap and bioacoustic data.
Drop-in for MegaDetector v6, DeepFaune, HerdNet, OWL-T, SpeciesNet, and
MD_AudioBirds_V1; model-agnostic via TOML manifests.

## Quickstart

### Easiest: Homebrew (macOS arm64 / brew-Linux x86_64)

```bash
brew tap microsoft/sparrow-engine
brew install sparrow-engine            # CPU; works on macOS arm64 + brew-Linux x86_64
brew install sparrow-engine-gpu        # GPU; brew-Linux x86_64 + NVIDIA only

spe device                              # {"device":"cpu"}  or  {"device":"cuda:0"}

# One-time: download a model from the Zenodo bundle (brew doesn't ship models)
mkdir -p ~/.sparrow-engine/models && cd ~/.sparrow-engine/models
curl -fLO https://zenodo.org/records/20674275/files/MDV6-yolov10-e.zip
unzip -q MDV6-yolov10-e.zip && rm MDV6-yolov10-e.zip
cd -

spe detect /path/to/photos --model MDV6-yolov10-e --recursive --export-format megadet --export-output detections.json
```

Both formulas can coexist (separate binaries `spe` + `spe-gpu`; shared model cache at `~/.sparrow-engine/models/`). The example above pulls MegaDetector v6 (general camera-trap detection); see the [Model zoo](#model-zoo) section below for the other 17 models in the Zenodo bundle (image classifiers, audio detectors, overhead-imagery detectors). See `docs/user-manual.md §2.4` for the other install paths.

#### GPU host prerequisites

The `sparrow-engine-gpu` formula ships ~256 MB of `libonnxruntime` + ORT CUDA provider sidecars, but it does **NOT** bundle NVIDIA's runtime libraries (NVIDIA's license forbids redistribution). The host must provide:

| Library | Apt package (Ubuntu/Debian) | pip wheel (no root) | Why |
|---|---|---|---|
| NVIDIA driver ≥550.x | `nvidia-driver-550` (or newer) | — (kernel module; host-only) | GPU access |
| CUDA runtime 12.6 | `nvidia-cuda-toolkit` brings it | `nvidia-cuda-runtime-cu12` | `libcudart.so.12` |
| **cuDNN ≥9.10** (9.8 has Conv bug on sm_89) | `nvidia-cudnn` | `nvidia-cudnn-cu12` | `libcudnn.so.9` — convolutions |
| cuBLAS | bundled with CUDA toolkit | `nvidia-cublas-cu12` | matrix multiplications |
| cuRAND | bundled with CUDA toolkit | `nvidia-curand-cu12` | rand sampling (some models) |
| cuFFT | bundled with CUDA toolkit | `nvidia-cufft-cu12` | audio FFT (MD_AudioBirds_V1) |
| nvJPEG | bundled with CUDA toolkit | `nvidia-nvjpeg-cu12` | GPU JPEG decode |

After installing the libraries (system or pip), the brew-installed `spe-gpu` wrapper auto-discovers them from common host locations — no `LD_LIBRARY_PATH` setup needed for production users. Search order (first hit wins):

1. `SPARROW_ENGINE_CUDA_LIB_DIR` (user override; honored as-is)
2. `~/.sparrow-engine/cuda-sidecars/lib/python*/site-packages/nvidia/*/lib` (the convention if you used pip sidecars)
3. `/usr/lib/python3/dist-packages/torch/lib` (Lambda Stack / system PyTorch — cuDNN comes bundled)
4. `/usr/local/cuda/lib64` (NVIDIA CUDA toolkit)
5. `/usr/lib/x86_64-linux-gnu` (Ubuntu apt nvidia-cudnn)

Full table + remediation appears in `brew info sparrow-engine-gpu`. Quick all-pip install (no root) for a fresh host:

```bash
uv venv ~/.sparrow-engine/cuda-sidecars --python 3.11
~/.sparrow-engine/cuda-sidecars/bin/pip install \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 \
    nvidia-cufft-cu12 nvidia-nvjpeg-cu12 nvidia-cuda-runtime-cu12
```

Verify with `spe-gpu device` — `{"device":"cuda:0"}` means good, any dlopen error in the output names the missing library.

### Alternative install paths

If brew isn't right for your environment (server distro without brew-Linux, Windows, etc.), the install wrapper handles probe-and-install for Linux / macOS / Windows:

```bash
# Linux / macOS — clone the repo and run from its root
bash installer/sparrow-engine-install.sh
```

```powershell
# Windows PowerShell — clone the repo and run from its root
installer\sparrow-engine-install.ps1
```

The wrapper probes hardware once, picks the right CPU or GPU build, and
installs the matching CLI binary plus the Python wheel into `~/.sparrow-engine/`.
Pass `--flavor cpu` or `--flavor gpu` to skip the probe. Pass `--docker`
to install the HTTP-server image instead.

System prerequisites for GPU: NVIDIA driver ≥550.x, CUDA 12.6 runtime,
and **cuDNN ≥9.10** (cuDNN 9.8 has a Conv-engine bug on sm_89).

### Python package only (PyPI)

If you only want the Python wheel — no CLI, no Docker image — install
straight from PyPI. Both wheels target CPython ≥ 3.11 (`cp311-abi3`), so
make sure your venv runs Python 3.11 or newer.

**With `uv` (recommended)**:

```bash
uv venv --python 3.11
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# CPU
uv pip install sparrow-engine

# GPU (Linux x86_64 only; requires CUDA 12.6 runtime on the host)
uv pip install sparrow-engine-gpu
```

`uv venv` does not ship `pip` inside the venv by default, so use `uv pip
install` (uv's pip-compatible wrapper) instead of bare `pip install`.
Calling `pip install …` after `source activate` falls back to the system
pip, which usually targets the wrong Python version and fails with
`No matching distribution found`.

**With stdlib `venv`**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# CPU
pip install sparrow-engine

# GPU (Linux x86_64 only; requires CUDA 12.6 runtime on the host)
pip install sparrow-engine-gpu
```

Both wheels import as `sparrow_engine`. Never install both into the same
environment. Check the installed version with
`python -c "import sparrow_engine; print(sparrow_engine.__version__)"`.
See [§6 of the user manual](docs/user-manual.md#6-python-package--sparrow-engine)
for the full API surface and GPU sidecar options.

### Docker image (server deployments)

Sparrow Engine ships as a self-contained HTTP server in two Docker flavors. Both expose `/v1/detect`, `/v1/classify`, `/v1/detect_audio`, `/healthz`, `/openapi.json` on port 8080.

| Image | Size | GPU |
|---|---|---|
| `zhongqimiao/sparrow-engine-server:latest` | ~170 MB | CPU only |
| `zhongqimiao/sparrow-engine-server-gpu:latest` | ~3.7 GB | CUDA 12 + cuDNN bundled; requires NVIDIA Container Toolkit on the host |

Three install paths. **Option A** is the simplest; **B** + **C** remain for offline operators and the absolute-latest-source case.

**Option A — `docker pull` from Docker Hub** (RP-35, 2026-06-05; published on every prod tag via `release.yml`):

```bash
# CPU image (~61 MB compressed, ~170 MB extracted)
docker pull zhongqimiao/sparrow-engine-server:latest
docker pull zhongqimiao/sparrow-engine-server:v0.1.17        # version pin (recommended for prod)

# GPU image (~2.2 GB compressed, ~3.7 GB extracted)
docker pull zhongqimiao/sparrow-engine-server-gpu:latest
docker pull zhongqimiao/sparrow-engine-server-gpu:v0.1.17
```

Public repos (anonymous pull, no Docker Hub login required):

- https://hub.docker.com/r/zhongqimiao/sparrow-engine-server
- https://hub.docker.com/r/zhongqimiao/sparrow-engine-server-gpu

Heads-up: anonymous Docker Hub pulls are rate-limited (100 pulls / 6 hr / source IP). For CI behind shared NAT, `docker login` with a free Docker Hub account lifts the limit to 200/6 hr.

**Option B — download pre-built tarballs from Zenodo** (offline / air-gapped). Uses the sparrow companion repo's downloader script which knows the current Zenodo record + expected SHA-256 digests:

```bash
git clone https://github.com/Clamps251/sparrow.git
cd sparrow
./scripts/download_sparrow_engine_images.sh                 # CPU + GPU
./scripts/download_sparrow_engine_images.sh --cpu-only       # CPU only (~43 MB compressed)
./scripts/download_sparrow_engine_images.sh --gpu-only       # GPU only (~1.5 GB compressed)
```

The script verifies SHA-256 + `docker load`s + retags as `sparrow-engine-server[-gpu]:sparrow-combined`. **Caveat**: the Zenodo record is refreshed manually per release, not on every commit, so the published tarballs may lag the latest source by one or more releases.

**Option C — build from source** (~10 min the first time; cached layers on subsequent builds; always reflects the current source tree):

```bash
git clone --branch sparrow-engine-dev https://github.com/microsoft/Pytorch-Wildlife.git
cd Pytorch-Wildlife/sparrow-engine
docker build -f docker/Dockerfile.cpu -t sparrow-engine-server:sparrow-combined .
docker build -f docker/Dockerfile.gpu -t sparrow-engine-server-gpu:sparrow-combined .  # GPU
```

**Run the server** (after any of the three options). The container expects models mounted read-only at `/models`:

```bash
# CPU (Option A pull)
docker run -d --rm --name sparrow-engine -p 8080:8080 \
  -v $HOME/.sparrow-engine/models:/models:ro \
  -e SPARROW_ENGINE_DEVICE=cpu \
  zhongqimiao/sparrow-engine-server:latest

# GPU (requires NVIDIA Container Toolkit on the host)
docker run -d --rm --name sparrow-engine-gpu -p 8080:8080 --gpus all \
  -v $HOME/.sparrow-engine/models:/models:ro \
  -e SPARROW_ENGINE_DEVICE=cuda:0 \
  zhongqimiao/sparrow-engine-server-gpu:latest

# Verify
curl -fsS http://localhost:8080/healthz
curl -fsS http://localhost:8080/openapi.json | jq '.paths | keys'
```

**Or use the bundled `docker-compose.yml`** (resource limits, healthcheck, log rotation, read-only filesystem all pre-configured):

```bash
cd Pytorch-Wildlife/sparrow-engine/docker
docker compose --profile cpu up -d        # CPU
docker compose --profile gpu up -d        # GPU
docker compose --profile cpu logs -f      # tail logs
docker compose --profile cpu down         # stop
```

The Compose file mounts `${SPARROW_ENGINE_MODEL_DIR:-./models}` read-only into the container; set the env var or place models at `sparrow-engine/docker/models/` before bringing the stack up. Models can also be downloaded via the [Model zoo](#model-zoo) section below.

For full HTTP API documentation, request shapes, response schemas, and operator-grade env-var reference: [§7 of the user manual](docs/user-manual.md#7-http-api-server--sparrow-engine-server).

---

> 📖 **[Read the full user manual →](docs/user-manual.md)**
>
> One document covering install, CLI (`spe`), Python wheel (`import sparrow_engine`), HTTP API server, HTTP SDK, native DLL (C ABI), TOML model manifests, the Phase 4 inference-log / drift / provenance surface, cold-start + lazy load, gotchas + edge cases, performance characteristics, and Sparrow Studio integration.

---

## Model zoo

Sparrow Engine doesn't ship the ONNX model weights in the repo. They live in a public Zenodo record so the repo stays small and operators can pull just the models they need.

**Zenodo DOI**: [10.5281/zenodo.20360316](https://doi.org/10.5281/zenodo.20360316) (v0.4.0) — concept DOI [10.5281/zenodo.20348978](https://doi.org/10.5281/zenodo.20348978) always resolves to the latest version.

Download all 16 models to `~/.sparrow-engine/models/` (the default model dir read by `spe`, `sparrow-engine-server`, and the Python wheel):

```bash
bash scripts/download_models.sh
```

Or just specific models:

```bash
bash scripts/download_models.sh MDV6-yolov10-e SpeciesNet-Crop
bash scripts/download_models.sh --list          # list available model IDs
bash scripts/download_models.sh --dest /custom/path
```

Point Sparrow Engine at the directory (only needed if you used `--dest`; the default location is auto-detected):

```bash
# Default path (auto-detected — env var only needed if you want to be explicit):
export SPARROW_ENGINE_MODEL_DIR=$(realpath ~/.sparrow-engine/models)
# Custom path (required if you used `--dest /opt/sparrow-models`):
export SPARROW_ENGINE_MODEL_DIR=/opt/sparrow-models
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
| `orca-detector-dclde2026-v1` | 3 s @ 24 kHz, mel spectrogram (1.5 s stride) | 1 (Orca vs rest) | 43 MB | MIT |
| `orca-ecotype-dclde2026-v1` | 3 s @ 24 kHz raw audio (in-graph mel) | 5 (SRKW / TKW / SAR / NRKW / OKW) | 48 MB | MIT |

- `MD_AudioBirds_V1` is the sparrow-engine default audio detector — a lightweight binary bird-vs-no-bird model used in benchmarks and Phase 4.x manual tests. Sliding-window mel-spectrogram front-end (Slaney mel scale + Slaney filter norm). Ships in the v0.5.0 Zenodo bundle (DOI [10.5281/zenodo.20563673](https://doi.org/10.5281/zenodo.20563673)) as FP32; the FP16 conversion path is in `sparrow-engine/tools/convert_fp16.py` and is parity-verified against the FP32 reference (Phase 3.8 Step 2 post-STRETCH audit, 2026-05-05).
- `perch-v2` is Google Perch 2, a global bird-vocalisation classifier (Conformer encoder) with an in-graph mel front-end. Takes 160000-sample windows of raw audio; emits softmax over 14795 classes (birds + non-bird FSD50K labels).
- `orca-detector-dclde2026-v1` + `orca-ecotype-dclde2026-v1` are a two-stage killer-whale cascade from the [DCLDE 2026 challenge](https://github.com/microsoft/orcas_dclde2026). Stage 1 screens 3-s windows for orcas (3-class NonBio/Bio/Orca classifier exposed as a binary Orca-vs-rest sigmoid at the engine boundary). Stage 2 classifies the Orca-positive windows into 5 Pacific Northwest ecotypes (Southern Resident / Transient / Southern Alaska Residents / Northern Resident / Offshore), with temperature scaling (T=5.4254) baked into the ONNX so the engine's softmax output is calibrated. Both stages **require sparrow-engine ≥ v0.1.16** because they use the RP-27 `fill_highfreq` engine opt-in to match the upstream training pipeline on under-sampled hydrophone audio (most field hydrophones cap at 16 kHz). Cascade usage and the Stage 2 abstention threshold (0.94 → `Unassigned_KW`) are documented in each model's `MODEL_CARD.md`.

#### License summary

- **Ultralytics AGPL-3.0** (7 models): MDv6 × 2, MDv5a, the 3 AI4G regional YOLOs, plus `deepfaune-yolo8s` (which also intersects CC-BY-NC-SA 4.0).
- **CC-BY-NC-SA 4.0** (3 models): `deepfaune-yolo8s`, `Deepfaune-Europe`, `Deepfaune-New-England`.
- **Apache 2.0** (2 models): `SpeciesNet-Crop`, `perch-v2`.
- **MIT** (7 models): `AI4G-Amazon-V2`, `AI4G-Serengeti`, `OWL`, `HerdNet_General_Dataset_2022`, `MD_AudioBirds_V1`, `orca-detector-dclde2026-v1`, `orca-ecotype-dclde2026-v1`.

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
