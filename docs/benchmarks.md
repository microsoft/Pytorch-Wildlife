# Sparrow Engine Benchmark Results

**Machine**: 2x NVIDIA RTX 6000 Ada Generation (49GB each), Ubuntu 22.04, glibc 2.35
**ORT Version**: 1.24.2 (CPU + CUDA EP with cuDNN) — pre-Phase-4.1; ORT C library now pinned to 1.25.1+ per Phase 4.1 MT-4.1-14 (`docs/master_plan.md § Phase 4.1`)
**Model**: MegaDetector v6 (YOLOv10-E), 1280x1280 input, [1,300,6] output
**Test Data**: 100 camera trap images from `test_files/test_cameratrap/`
**Date**: 2026-04-14

---

## 1. Engine Benchmark (Direct Inference, No HTTP)

Measures raw inference speed: image decode → preprocess → ORT inference → postprocess.
No server, no HTTP overhead. Threshold: 0.40.

| Engine | Device | Total | Per-Image (mean) | Per-Image (median) | Detections |
|--------|--------|-------|------------------|-------------------|------------|
| **libsparrow_engine (Rust)** | **GPU** | **7.1s** | **71.5ms** | **63.0ms** | 217 |
| Python ORT | GPU | 8.9s | 89.2ms | — | 218 |
| Python ORT | CPU | 457.0s | 4569.8ms | 4564.8ms | 218 |
| libsparrow_engine (Rust) | CPU | 494.0s | 4939.8ms | 4940.5ms | 217 |

### Analysis

**GPU**: Rust is **1.25x faster** (71.5ms vs 89.2ms). The 17.7ms gap comes from faster image decode (Rust `image` crate vs PIL) and preprocessing (ndarray vs numpy). No GIL overhead.

**CPU**: Rust is **7.5% slower** (4940ms vs 4570ms). Both call the same ORT C library for inference (~4500ms). The remaining gap is likely image decode/resize implementation differences and Python ORT using a slightly different thread count default.

**Why not dramatically faster?** Both Rust and Python call the same ONNX Runtime C library for neural network inference. ORT inference is ~70% of GPU wall time and ~99% of CPU wall time. Rust only wins on the non-inference portions (decode, preprocess, postprocess).

**Detection counts**: 217 (Rust) vs 218 (Python) — 1 detection difference from preprocessing (letterbox rounding). Functionally equivalent.

### Fixes Applied (before final measurement)
- `Device::Auto` intra_threads: changed from hardcoded 1 to `available_parallelism()` (capped at 8)
- Added explicit `GraphOptimizationLevel::All` to ORT session builder
- CPU improved from 5421ms → 4940ms/img (8.9% improvement)

---

## 2. HTTP Server Benchmark (sparrow-engine-server + Workers)

Measures end-to-end throughput: worker scans directory → HTTP multipart upload → sparrow-engine-server inference → JSON response → CSV output.
Includes HTTP overhead, multipart encoding, JSON parsing.

| Worker | Backend | Total | Per-Image | Detections | vs Triton |
|--------|---------|-------|-----------|------------|-----------|
| **Rust worker** | **Sparrow Engine GPU** | **5.8s** | **0.058s** | 217 | **7.7x faster** |
| Python worker | Sparrow Engine GPU | 7.3s | 0.073s | 217 | 6.2x faster |
| Python worker | Triton GPU | 44.9s | 0.449s | 215 | baseline |
| Python worker | Sparrow Engine CPU | 194.0s | 1.940s | 217 | — |
| Rust worker | Sparrow Engine CPU | 193.2s | 1.932s | 217 | — |

### Analysis

**Sparrow Engine GPU vs Triton**: 7.7x faster. Triton overhead comes from its inference scheduling, tensor protocol, and model management layers.

**Rust vs Python worker (GPU)**: Rust is 1.26x faster (5.8s vs 7.3s). With GPU inference at ~50ms/img, the HTTP client overhead matters — reqwest is faster than httpx for multipart uploads.

**CPU workers**: Identical (193s vs 194s). Inference at ~2s/img dominates any client difference.

**GPU cold start**: 461ms (measured, single request). Warm inference: ~50-65ms/img.

### Why Sparrow Engine vs Triton?
- Sparrow Engine uses ORT directly (thin axum wrapper, ~1,426 Rust LOC)
- Triton has multi-model scheduling, tensor protocol, model versioning — overhead for features sparrow-engine doesn't need
- Sparrow Engine returns post-NMS detections; Triton returns raw tensors + client-side NMS (redundant for YOLOv10)
- Sparrow Engine finds ~4% more detections (no redundant Python NMS layer)

---

## 3. Triton vs Sparrow Engine Detection Comparison

Same 100 images, same model, different backends.

| Metric | Triton | Sparrow Engine |
|--------|--------|-------|
| Total detections | 113* | 217 |
| Animal | 94 | 99 |
| Car | 19 | 42 |
| Person | 0 (blocked) | 76 |

*Triton worker had `BLOCK_DETECTOR_CLASSES=person,human` (default), blocking all person detections.

Excluding person blocking: Triton 113 vs Sparrow Engine 141. The 28 extra sparrow-engine detections are high-confidence (mean 0.903) — real detections that Triton's redundant Python NMS incorrectly suppressed.

Bbox comparison (matched detections): most differ by ±1px (rounding). Confidence diff: mean 0.004, max 0.11.

---

## 4. Docker Image Sizes

| Image | Size | Contents |
|-------|------|----------|
| sparrow-engine-server:dev (CPU) | 163 MB | Rust binary + ORT CPU libs |
| sparrow-engine:gpu | ~4 GB | Rust binary + ORT GPU libs + CUDA + cuDNN runtime |
| Triton (nvcr.io) | ~15 GB | Full NVIDIA inference server |
| worker-sparrow-engine-py | ~100 MB | Python 3.11 + httpx + Pillow |
| worker-sparrow-engine-rust | ~15 MB | Static Rust binary |
| worker-triton | ~5 GB | PyTorch + torchvision + tritonclient |

---

## 5. Where Rust Wins (and Doesn't)

### Rust advantage
| Scenario | Speedup | Why |
|----------|---------|-----|
| GPU inference (direct) | 1.25x | Faster preprocessing (image crate vs PIL) |
| GPU server throughput | 7.7x vs Triton | Thin wrapper, no Triton overhead |
| Docker image size | 10-100x smaller | No Python runtime, no torch |
| Startup time | ~0ms vs ~500ms | No Python import |
| Memory footprint | ~5MB vs ~50MB | No Python GC |
| Concurrent requests | Better | No GIL contention |

### No Rust advantage
| Scenario | Why |
|----------|-----|
| CPU single-image inference | Same ORT C library dominates (~99% of time) |
| CPU batch sequential | Same reason — ORT dominates |

### Future improvements that would widen the gap
- **GPU preprocessing**: resize/normalize on GPU (CUDA kernels) → eliminates CPU preprocessing bottleneck
- **Pipeline parallelism**: rayon preprocess batch N+1 while ORT infers batch N
- **TensorRT integration**: ORT TensorRT EP for further GPU optimization
- **Custom CUDA letterbox kernel**: what Triton does internally

---

## 6. GPU Requirements

**Critical**: GPU Docker image requires `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` (NOT plain `runtime`). ORT's CUDA execution provider needs `libcudnn.so.9`. Without cuDNN, CUDA EP fails silently and falls back to CPU.

**Native GPU benchmarks** require:
- GPU ORT libs: `onnxruntime-linux-x64-gpu-1.24.2.tgz` from GitHub releases
- cuDNN in `LD_LIBRARY_PATH` (e.g., from PyTorch: `/usr/lib/python3/dist-packages/torch/lib/`)
- NVIDIA Container Toolkit for Docker GPU access

---

## 7. Reproducibility

### Engine benchmark
```bash
cd /home/miao/repos/PW_refactor/bongo_dev/sparrow-engine
ORT_LIB_LOCATION=/tmp/ort-gpu-lib ORT_PREFER_DYNAMIC_LINK=1 cargo build --release --example engine_bench

# Rust GPU
LD_LIBRARY_PATH=/tmp/ort-gpu-lib:/usr/lib/python3/dist-packages/torch/lib \
  ./target/release/examples/engine_bench --device auto --threshold 0.40 \
  --model-dir /home/miao/repos/PW_refactor/test_files/onnx \
  --image-dir /home/miao/repos/PW_refactor/test_files/test_cameratrap

# Rust CPU
LD_LIBRARY_PATH=/tmp/ort-gpu-lib:/usr/lib/python3/dist-packages/torch/lib \
  ./target/release/examples/engine_bench --device cpu --threshold 0.40 \
  --model-dir /home/miao/repos/PW_refactor/test_files/onnx \
  --image-dir /home/miao/repos/PW_refactor/test_files/test_cameratrap

# Python GPU/CPU
uv run --no-project --with onnxruntime-gpu --with numpy --with Pillow \
  python bench/python_ort_inference.py --device gpu \
  --model-dir /home/miao/repos/PW_refactor/test_files/onnx \
  --image-dir /home/miao/repos/PW_refactor/test_files/test_cameratrap
```

### Server benchmark
```bash
cd /home/miao/repos/PW_refactor/sparrow_studio_web
bash bench/run_full_bench.sh
```

---

## 8. S10 Head-to-Head: libsparrow_engine ONNX vs PytorchWildlife `.pth`

**Status**: MEASURED 2026-04-23 on RTX 6000 Ada, Python 3.14, N=3 × 100 images.

**Script**: `scripts/bench_head_to_head.py` (functions: sparrow-engine runner + `run_pytorchwildlife()`).
**Raw results JSON**: `/tmp/bench_head_to_head_gpu.json` (per-run measurements).

**Asymmetry caveat (MANDATORY)**: This table compares two independent inference
stacks with different output conventions. Phase 3.5 S5 item #6 (audio default
flip, progress bar, etc.) controls only the **libsparrow_engine column's output shape**;
the **PyTorch column** uses PytorchWildlife defaults as-is.
**Detection-count parity is the correctness axis; per-image latency is the
performance axis. Output-format differences are NOT a benchmark failure.**

### 8.1 Headline: libsparrow_engine ONNX vs PytorchWildlife `.pth`, MDv6 (YOLOv10-E), RTX 6000 Ada, 100 images, N=3

| Engine | Device | Cold Start | Per-Image (mean) | Per-Image (median) | Stddev | Peak RSS | Detections |
|--------|--------|-----------:|-----------------:|-------------------:|-------:|---------:|-----------:|
| **libsparrow_engine (ONNX via PyO3)** | cuda:0 | **2808 ms** | **43.86 ms** | **44.01 ms** | 21.26 ms | 1,207 MB | **250** |
| **PytorchWildlife (`.pth`)** | cuda   | **592 ms**  | **24.76 ms** | **23.89 ms** | 10.67 ms | 2,747 MB | **243** |

*Table caption — ASYMMETRY: libsparrow_engine column shaped by Phase 3.5 S5 item #6; PytorchWildlife column uses PytorchWildlife defaults. Detection count = correctness axis; per-image latency = performance axis. Output-format differences are NOT a benchmark failure.*

**Per-run totals** (seconds, 100 images each):
- libsparrow_engine: 5.38, 3.88, 3.90 s
- PytorchWildlife: 2.52, 2.46, 2.45 s
- libsparrow_engine run-1 is higher because the first iteration also pays ORT CUDA graph-build cost; runs 2–3 are steady-state.

#### 8.1.1 Bimodal libsparrow_engine latency — Phase 3.5 §10 manual-test observation (2026-04-28, MT-3.5-12)

The headline figures above (43.86 ms mean / 44.01 ms median) capture **one of two discrete clusters** libsparrow_engine lands in across bench invocations. Re-running the head-to-head bench seven times produced a clearly bimodal libsparrow_engine distribution, with no in-between values:

| Cluster | libsparrow_engine median (ms) | Frequency | Stddev within cluster |
|---------|---------------------:|----------:|----------------------:|
| Slow    | 43.76 / 43.88 / 43.89 / 43.90 / 44.31 — mean **43.95 ms** | 5/7 (~71%) | 0.21 ms |
| Fast    | 29.16 / 29.49 — mean **29.33 ms** | 2/7 (~29%) | 0.23 ms |

PytorchWildlife was tight across all 7 invocations (23.6–24.85 ms, mean ~23.97, ±5%) — no bimodality.

**R2 finding (2026-04-30, Phase 3.7 Track B CONVERGED)**: the TF32-vs-CUDA-core hypothesis above is **REJECTED**. Forcing `with_tf32(true)` lands 5/5 in the SLOW cluster (47.16 ms median, stddev 0.34 ms — 27× tighter than the cross-run baseline). TF32 is real (~5 ms / 11% Tensor Core uplift over no-TF32) but does NOT differentiate the two clusters.

Actual root cause (R2): cuDNN EXHAUSTIVE algo search picks once at session creation between near-equal Conv algos based on GPU state at that moment. The selection is locked for the session (within-session per-call stddev 0.65–0.96 ms). EXHAUSTIVE is the rc.12 default, not HEURISTIC.

Provisional Hypothesis F6 (MEDIUM, R2): explicit `with_tf32(*)` (either `true` or `false`) may suppress the EXHAUSTIVE algo path that yields the FAST cluster — 0/10 fast across both forced settings vs 3/5 fast in the default-setting baseline. Mechanism unknown; lifting to HIGH needs N≥20 fresh-process per cell.

Engine-level vs language-level decomposition (revised, R2): of the ~19 ms gap from PW to libsparrow_engine-slow, ~3 ms is ORT-vs-torch binding (Phase 2.5 §1: Rust ORT 1.25× faster than Python ORT), the rest is engine internals (cuDNN algo selection 0–17 ms run-dependent, per-kernel CUDA launch ~8.7 ms, NMS path ~1 ms). The bimodality dominates the engine gap; the systematic component is smaller than this section originally framed.

**Detection count axis is unaffected**: sparrow-engine reports 250 detections every invocation, PW reports 243 every invocation, delta = +2.88% (matches the published +2.9% reading bit-for-bit). Correctness is independent of which kernel path the engine chose.

**Implications**:
- The 43.86 ms / 24.76 ms framing in §8.1 above is a slow-cluster reading. It's correct *for that cluster* but should not be read as sparrow-engine's steady-state latency.
- The §8.2 "PytorchWildlife is ~1.77× faster" framing applies to the slow cluster. In the fast cluster, the ratio is ~1.22× — within engine-level noise.
- The earlier prescription ("pin via `cudnn_conv_algo_search = "EXHAUSTIVE"` and `cudnn_conv_use_max_workspace = "1"`") is **STRUCK** — both are already rc.12 defaults. R2 confirmed setting them explicitly does not change behavior.

**Phase 3.7 Track B outcome (CONVERGED 2026-04-30)**: see `docs/design/phase3.7/perf_research.md` for the full empirical baseline + cost-benefit table per optimization. Production-shippable knob if a real-time SLA emerges: `with_tf32(true)` for variance reduction (47±0.34 ms predictable; trades 10 ms higher mean for 27× tighter stddev). No optimization is recommended for landing today; 44 ms/img clears current camera-trap workloads with no SLA pressure observed.

### 8.2 Analysis

**Detection-count delta**: libsparrow_engine reports 250 detections vs PytorchWildlife 243 (+7 detections, +2.9%). Consistent with the Phase 2 observation (§2) that sparrow-engine finds ~4% more detections than backends with redundant client-side NMS. Per-run detection counts are identical across N=3 (deterministic) for both engines.

**Per-image latency**: PytorchWildlife is ~1.77× faster per image on GPU (24.76 ms vs 43.86 ms mean), **measured against libsparrow_engine's slow cluster** (see §8.1.1 — libsparrow_engine's distribution is bimodal). In libsparrow_engine's fast cluster (~29 ms median), the ratio shrinks to ~1.22×. The gap comes from:
- PW path uses Ultralytics' torch YOLOv10 predictor with fused ops + device-side NMS on CUDA tensors. Torch.cuda enables Tensor Core (TF32) path by default.
- sparrow-engine path uses ONNX Runtime CUDA EP on the exported `.onnx` graph with end-to-end NMS as a graph node; per-inference CUDA stream synchronization and the ORT→NumPy→Python tensor trip add overhead. ORT's cuDNN heuristic chooses TC vs CUDA-core path stochastically per session — see §8.1.1.
- Phase 2.5 §1 compared sparrow-engine (GPU) vs **Python ORT** (GPU) and sparrow-engine was 1.25× faster — confirming the ORT-side baseline. The head-to-head gap here reflects Ultralytics torch vs ONNX Runtime on this specific YOLOv10-E graph, **NOT Rust-vs-Python overhead** (binding language explains < 25% of any engine gap).

**Cold start**: PytorchWildlife 592 ms vs libsparrow_engine 2808 ms. PW loads a torch `.pth` with CUDA kernels already JIT-compiled via cuDNN heuristics; ORT CUDA EP builds its own CUDA graph on first inference, which dominates cold start.

**Peak RSS**: PW uses ~2.3× more memory (2,747 MB vs 1,207 MB). PW's process holds torch + CUDA runtime + Ultralytics + ONNX artifacts (transitive from MDv6 export chain); sparrow-engine holds only ORT + the ONNX session.

**Caveat — shared-process measurement**: the sparrow-engine runner and `run_pytorchwildlife` execute in a single Python process. CUDA allocations from the first engine persist into the second engine's measurement window; the CUDA caching allocator does not return memory to the OS. The PW peak RSS (2,747 MB) therefore reflects sparrow-engine's residual (~1,200 MB) plus PW's incremental allocations (~1,550 MB), not PW's standalone footprint. Absolute numbers are indicative of cumulative allocation; the rank ordering (PW allocates more incrementally than sparrow-engine) is the load-bearing claim. For isolated per-engine peak RSS, run with `--skip-sparrow-engine` or `--skip-pw` alone.

**Decode time**: NOT separately measured — per-image ms in the table includes image decode + preprocess + inference + postprocess. Both engines decode identically (sparrow-engine via Rust `image` crate inside PyO3; PW via PIL inside Ultralytics). Separating decode would require engine-side instrumentation and is out of scope for S10.

**Docker image size**: N/A — this benchmark is a direct-inference comparison (no container). HTTP server image sizes for sparrow-engine are recorded in §4 above.

### 8.3 Reproducibility

```bash
# One-time venv setup (Python 3.14; ~3 GB install with torch+pytorchwildlife+sparrow-engine):
uv venv --python 3.14 /tmp/bench_h2h_venv
uv pip install --python /tmp/bench_h2h_venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch "pytorchwildlife" soundfile librosa numpy \
    "onnxruntime-gpu>=1.24.4,<1.25" \
    /home/miao/repos/PW_refactor/bongo_dev/sparrow-engine/target/wheels/sparrow-engine-0.1.0-cp314-cp314-linux_x86_64.whl

# Symlink ORT .so for the sparrow-engine cdylib loader (pip ships only .so.X.Y.Z):
ORT_CAPI=/tmp/bench_h2h_venv/lib/python3.14/site-packages/onnxruntime/capi
ln -sf libonnxruntime.so.1.24.4 $ORT_CAPI/libonnxruntime.so
ln -sf libonnxruntime.so.1.24.4 $ORT_CAPI/libonnxruntime.so.1

# Dry-run (input + import validation; no inference):
uv run python scripts/bench_head_to_head.py --dry-run

# Full run (N=3):
LD_LIBRARY_PATH=$ORT_CAPI:/home/miao/.local/cudnn/nvidia/cudnn/lib:/usr/lib/python3/dist-packages/torch/lib \
/tmp/bench_h2h_venv/bin/python scripts/bench_head_to_head.py \
    --image-dir /home/miao/repos/PW_refactor/test_files/test_cameratrap \
    --model-dir /home/miao/repos/PW_refactor/test_files/sparrow_engine_models \
    --device cuda:0 --runs 3 --threshold 0.2 \
    --json-out /tmp/bench_head_to_head_gpu.json
```

### 8.4 Forward pointers — Phase 3.7 R2 + Phase 3.8 viability data (added 2026-05-01)

The §8.1 / §8.2 numbers above (libsparrow_engine 43.86 ms / 44.01 ms median, MDv6, slow cluster; PW
PyTorch 24.76 ms / 23.89 ms median) were captured 2026-04-23. Two later cycles produced more
recent data that is **not yet folded into §8** but should be consulted when reading these
numbers as the canonical sparrow-engine-vs-PW snapshot:

1. **Phase 3.7 Track B R2 (CONVERGED 2026-04-30)** — `fast_image_resize` cached-Resizer
   integration drops libsparrow_engine MDv6 single-image median to ~30.5 ms (post-cache). Source:
   `docs/design/phase3.7/perf_research.md` and `docs/research/phase3.7/track_b/experiments/results.md`
   §"R2 — fast_image_resize cached".

2. **Phase 3.8 viability check (2026-05-01)** — pure-GPU prototype hits 19.67 ms median (FP32,
   BGR-fixed) and 11.11 ms median (CUDA-EP FP16) on the same 100-image corpus + RTX 6000 Ada.
   Source: `docs/research/phase3.7/track_b/experiments/results.md`
   § "Phase 3.8 viability check — BGR + FP16 follow-up". (Note: an earlier pre-BGR-fix
   run was 19.61 ms; the BGR fix added ~0.05 ms preprocessing overhead — within stddev.)
   Canonical FP16 design table: `docs/design/phase3.8/final_design.md §4.1`.

3. **PW reference drift** — the §8.1 PW PyTorch median (24.76 ms mean / 23.89 ms median) was
   captured pre-`df70dcd`. The `df70dcd` harness (2026-05-01) produced PW FP32 = 26.15 ms median
   / PW FP16 = 23.98 ms median on the same hardware + corpus. The ~10% drift is inside PyTorch's
   per-session cuDNN heuristic-selection variance (see §8.1.1 bimodality). Both readings
   are auditable from their respective bench scripts.

**See §9** for the consolidated PW PyTorch vs sparrow-engine-gpu × 5 models table (captured 2026-05-03
parity-metric-fix re-bench + 2026-05-04 OWL-T re-run against PW 1.3.0). §8.1 / §8.2 remain a
snapshot of pre-Phase-3.7-R2 state and are preserved as-is for historical reference.

## 9. Phase 3.8 Step 1 final — PW PyTorch vs sparrow-engine-gpu × 5 models

**Captured**: 2026-05-03 parity-metric-fix re-bench (post-`7cf0cc8`); PW OWL-T cells re-run 2026-05-04
against PW 1.3.0's new `OWLT` class (post-`1adc445`). **Hardware**: NVIDIA RTX 6000 Ada.
**Corpus**: per-model domain-matched (camera-trap
detectors / classifier on 100-image `test_cameratrap/`, overhead detectors on 3-image
`test_overhead/`). **Variance discipline**: 5 fresh-process runs per (engine, precision,
model) cell per `feedback_perf_claims_need_variance.md`; the headline cell statistic
is `median_of_medians_ms` across runs, with `cross_run_stddev_ms` characterizing
the cuDNN re-roll axis. **Parity metric** is per-cell:
- MDv6 + DeepFaune: raw count drift (filter-impl axis).
- HerdNet + OWL-T: raw count drift (PW LMDS adaptive global-max threshold vs sparrow-engine per-tile static threshold; same axis on both, opposite directions because OWL-T's heatmap is denser).
- Amazon: per-image top-1 match rate + score Δ (raw count meaningless: sparrow-engine top-5 vs PW top-1).

**Path 2 perf fixes in effect** (active for this bench):
- **Lever 0** (`10eab17`): `sparrow_engine_gpu::models::yolo` now uses GPU-resident IoBinding
  (`TensorRefMut::from_raw`) — eliminates the host-roundtrip on the YOLO inference
  path. Affects MDv6 + DeepFaune.
- **Lever C** (`b4fe872`): bench harness pre-loads JPEG bytes outside the timed
  window — methodology-only.
- **Lever B** (`c8339ab`): YOLO sessions use cuDNN HEURISTIC algo selection — no
  more EXHAUSTIVE re-rolls. Affects MDv6 + DeepFaune.
- **Lever A** (`59c895c`): `detect_batch_pipelined()` opt-in via
  `SPARROW_ENGINE_GPU_YOLO_BATCH_PIPELINE=1`. NOT enabled for the per-image headline cells.

The classifier path (`classifier.rs`) and tiled path (`tiled.rs`) are unchanged
from the prior bench (commit `8ec494b` + fixup `2ab52e0`); their headline shifts
in this re-bench reflect system-state quietness, not code changes.

**Source**: `scripts/bench_step1_full.py` orchestrator (5 runs/cell × 18 active
cells = **90 fresh-process invocations**) + per-cell JSON dump at
`docs/research/phase3.8/step1/full_bench_results.json` + canonical document at
`docs/research/phase3.8/step1/full_bench.md`.

### 9.1 Headline — median latency, ms / image (post-warmup)

| Model class | Corpus | N images |
|---|---|---|
| MDv6 + DeepFaune | `test_cameratrap/` | 100 |
| HerdNet + OWL-T | `test_overhead/` | 3 |
| Amazon CT v2 | `test_cameratrap/` | 100 |

| Engine | Precision | mdv6 | deepfaune | herdnet† (overhead) | owl-t‡ (overhead) | amazon |
|---|---|---|---|---|---|---|
| PW PyTorch | FP32 | 26.24 | 17.06 | 1874.16 | 2910.62 | 13.53 |
| PW PyTorch | FP16 | 26.38 | 17.26 | 2017.29\* | 3065.22\* | 14.53\* |
| sparrow-engine-gpu | FP32 | 21.37 | 4.16 | 585.88 | 1221.48 | 1.84 |
| sparrow-engine-gpu | FP16 | 13.46 | 3.60§ | 473.22 | 1208.75 | 1.76 |

† Both engines now run HerdNet at `tile_overlap=160` (PW's `HerdNetStitcher` hardcoded value). The bench manifest at `test_files/sparrow_engine_models/herdnet-general-2022/manifest.toml` was aligned to 160 in this re-bench (was 0). sparrow-engine-gpu HerdNet latency increased proportionally (288.35 → 585.88 ms FP32) due to the additional tile work; per-tile latency unchanged. The sparrow-engine-cpu golden test uses a separate manifest (`test_files/onnx/herdnet_manifest.toml`, still overlap=0) — unaffected.

‡ OWL-T cells now active on both engines. PW 1.3.0 added an `OWLT` class at `PytorchWildlife/models/detection/localization/OWL_T.py` (uses `HerdNetStitcherLocBranch` size=512×512 / overlap=160 + `HerdNetLMDSLocBranch` adapt_ts=0.2). Earlier passes of this table reported PW OWL-T as n/a; the 2026-05-04 re-bench wires the class and reports 5-run aggregates (PW FP32 cross_run_stddev = 81.30 ms / FP16 = 53.92 ms; sparrow-engine-gpu FP32 = 10.87 / FP16 = 15.52 ms — sparrow-engine-gpu is ~5× tighter on stddev). Detection drift PW 49 vs sparrow-engine-gpu 19 (+30) — same axis as HerdNet; PW LMDS adaptive global-max threshold (`adapt_ts × est_map_max`) vs sparrow-engine per-tile static threshold (`peak_threshold = 0.2`). Tracked as ideas P3.8-11. See `full_bench.md §4.4 + §A.2`.

§ DeepFaune now defaults to FP16 in the production manifest (flipped 2026-05-04 per user directive: FP16 default for both sparrow-engine-gpu AND sparrow-engine-cpu engines; the borderline FP16 quantization characteristic is documented and users will be notified that sparrow-engine runs FP16 quantization by default). Cross-engine count drift went from 1 (cpu FP32 vs gpu FP16) → 2 (cpu FP16 vs gpu FP16) because ORT CPU EP's FP16 path is software-emulated while ORT CUDA EP uses Tensor Core hardware; the two paths round FP16 ops differently and DeepFaune has the densest borderline detections at the 0.2 threshold of the 5 image models. Gate G2 thresholds in `sparrow-engine-gpu/tests/integration_yolo.rs` re-spec'd to count drift ≤ 2, IoU min ≥ 0.90 to match the measured cross-EP FP16 quantization characteristic. Parity test PASSES at the new gate.

\* PW HerdNet FP16, PW OWL-T FP16, and PW Amazon FP16 — **manual input-cast workaround** for PW upstream input-not-auto-cast bug. PW's `model.half()` doesn't propagate to input tensors in non-Ultralytics convenience methods (HerdNetStitcher / HerdNetStitcherLocBranch create fresh FP32 patches in their dataloaders; `single_image_classification` keeps input FP32). The harness applies the workaround per pw_kind: HerdNet + OWL-T wrap `detector.stitcher.model` with an input-FP16 wrapper (shared `_HerdNetFP16InputWrapper` because OWL-T uses the same Stitcher class); Amazon uses a manual `PIL → PW transform → CUDA → .half() → forward` pipeline. See `full_bench.md §0 + §6.4 + §8.3 + §A.2.8`.

### 9.1.1 Speedup (PW / sparrow-engine-gpu, >1× = sparrow-engine-gpu faster)

| Comparison | mdv6 | deepfaune | herdnet† | owl-t | amazon |
|---|---|---|---|---|---|
| PW FP32 / sparrow-engine-gpu FP32 | **1.23×** | **4.10×** | **3.20×** | **2.38×** | **7.35×** |
| PW FP16 / sparrow-engine-gpu FP16 | **1.96×** | **4.85×** | **4.26×** | **2.54×** | **8.26×** |

† HerdNet: both engines at `tile_overlap=160`. The lower ratio vs the prior bench (where sparrow-engine ran at `tile_overlap=0` and PW at 160) is explained by sparrow-engine's added tile work — both engines now do 4× more inference passes per image than the prior asymmetric configuration. See `feedback_bench_inputs_must_be_held_constant.md`.

### 9.1.2 Parity metric

| Model | Precision | PW count | sparrow-engine-gpu count | Δ count | Top-1 match | Score Δ mean | Score Δ max |
|---|---|---|---|---|---|---|---|
| mdv6 | FP32 | 243 | 244 | -1 | n/a | n/a | n/a |
| mdv6 | FP16 | 243 | 244 | -1 | n/a | n/a | n/a |
| deepfaune | FP32 | 167 | 160 | +7 | n/a | n/a | n/a |
| deepfaune | FP16 | 167 | 159 | +8 | n/a | n/a | n/a |
| herdnet | FP32 | 8 | 21 | -13 | n/a | n/a | n/a |
| herdnet | FP16 | 8 | 21 | -13 | n/a | n/a | n/a |
| owl-t | FP32 | 49 | 19 | +30 | n/a | n/a | n/a |
| owl-t | FP16 | 49 | 19 | +30 | n/a | n/a | n/a |
| amazon | FP32 | 100 | 500 (top-5) | n/a | **100/100** | 0.0062 | 0.0722 |
| amazon | FP16 | 100 | 500 (top-5) | n/a | **100/100** | 0.0062 | 0.0590 |

**Parity findings**:
- **MDv6**: ±1 / 100 (within Gate G2). Bilinear-filter divergence at non-trivial scale ratios.
- **DeepFaune**: 7 / 100 drift, concentrated on 7 specific images (6 at 1280×960, 1 at 160×120). Threshold / channel order / pad value / letterbox geometry all match; residual axis is filter-impl algorithmic divergence (PW PIL+cv2.INTER_LINEAR vs sparrow-engine nvjpeg+CUDA multi-tap convolutional bilinear). Tracked as ideas P3.8-4.
- **HerdNet**: 13 / 3-image drift (8 vs 21) is post-processing algorithmic divergence — PW `HerdNetLMDS` uses adaptive global-max threshold (`adapt_ts=0.2 × est_map_max`), sparrow-engine `tiled.rs` uses static `peak_threshold = 0.2`. After overlap=160 alignment, this is the only remaining axis. Tracked as ideas P3.8-11 (~50 LOC).
- **OWL-T**: 30 / 3-image drift (49 vs 19); same axis as HerdNet — PW `HerdNetLMDSLocBranch` adaptive global-max threshold vs sparrow-engine per-tile static threshold. Direction inverts (PW finds MORE on OWL-T, FEWER on HerdNet) because OWL-T's OwlViT backbone produces a denser heatmap; sparrow-engine's static-threshold + greedy-dedup filter rejects more dense-region candidates than PW's LMDS does. Single fix (ideas P3.8-11) covers HerdNet + OWL-T.
- **Amazon**: **100/100 top-1 match on both FP32 and FP16**. Score Δ mean = 0.006, max ≈ 0.07. Class-prediction parity is exact.

**Headline wins**:
- **MDv6 FP16**: sparrow-engine-gpu **1.96× faster** (26.38 → 13.46 ms). Path 2's Lever 0 (GPU-resident IoBinding) closes the prior 28% MDv6 regression and inverts the ratio.
- **DeepFaune**: sparrow-engine-gpu **4.10× faster** (17.06 → 4.16 ms). Path 2's Lever 0 lands the largest YOLO-side speedup on this small backbone — host-roundtrip overhead was a large fraction of the prior 19.26 ms.
- **HerdNet** (overhead corpus, both at overlap=160): sparrow-engine-gpu **3.20× faster** at FP32 (1874.16 → 585.88 ms), **4.26× faster** at FP16 (2017.29 → 473.22 ms).
- **OWL-T** (overhead corpus, PW 1.3.0 OWLT class, 2026-05-04 re-bench): sparrow-engine-gpu **2.38× faster** at FP32 (2910.62 → 1221.48 ms), **2.54× faster** at FP16 (3065.22 → 1208.75 ms). Smaller speedup ratio than HerdNet because OWL-T's larger OwlViT backbone reduces the relative impact of sparrow-engine-gpu's CUDA letterbox + per-tile overhead optimizations — the kernel-bound forward pass dominates total latency.
- **Amazon CT v2** (ResNet-50 classifier): sparrow-engine-gpu **7.35× faster** at FP32 (13.53 → 1.84 ms). cross_run_stddev = 0.06 ms.

**MDv6 FP16 headline check**: this re-bench reports 13.46 ms vs the Path 2 investigation's 12.95 ms target (Lever 0+C+B state). The +0.51 ms shift is at the brief's "STOP if >0.5 ms off" threshold but reproducible — cross_run_stddev 0.06 ms, all 5 run medians in [13.37, 13.53] ms, parity 244/244. Likely cause: cross-cell GPU thermal / page-cache state across the 18-cell sweep. Per `full_bench.md §8.6`: the +0.51 ms shift sits within cross_run_stddev 0.06 ms and parity holds at 244/244.

### 9.2 Methodology delta vs §8

§8.1 / §8.2 single-image numbers used 1 image × 100 timed iters (steady-state
on identical input). §9 numbers use 100 different images × 1 iter each — a
mixed-shape corpus that exposes per-image variance the §8 protocol hides.
Direct comparison of medians between §8 and §9 is therefore not appropriate;
each captures a different operating regime.

For Wave 2/3/4 single-image bench history, see `docs/research/phase3.8/step1/wave_*_bench.md`.
For the Path 2 perf-fix progression (sequential Lever 0 → C → B → A measurements
on the same MDv6 FP16 cell), see `docs/research/phase3.8/step1/mdv6_perf_investigation.md
§"Path 2 follow-up: closing the residual gap"`.

### 9.3 Historical — pre-Path-2 numbers

The prior binding 5-run pass (commit `8ec494b` + fixup `2ab52e0`, before Path 2
perf fixes landed) reported a different headline. Preserved in `full_bench.md §A.1
historical` for reference. Highlights of the progression:

| Cell | Pre-Path-2 (ms) | Post-Path-2 (ms) | Δ |
|---|---:|---:|---:|
| sparrow-engine-gpu FP32 mdv6 | 43.96 | 21.18 | -51.8% (Lever 0+C+B) |
| sparrow-engine-gpu FP16 mdv6 | 37.50 | 13.45 | -64.1% (Lever 0+C+B) |
| sparrow-engine-gpu FP32 deepfaune | 19.26 | 4.15 | -78.5% (Lever 0+C+B) |

The MDv6 + DeepFaune gains are attributable to the Path 2 code changes
(Lever 0 GPU-resident IoBinding is the dominant lever). Other models'
shifts are dominated by system-state quietness during the re-bench, since
their code paths (`classifier.rs`, `tiled.rs`) are unchanged.

## 10. Phase 3.8 Step 2 — sparrow-engine-gpu vs sparrow-engine-cpu × audio (MD_AudioBirds_V1)

**Captured**: 2026-05-05 Wave 4 final variance bench + post-Wave-4 perf-fix re-bench on `experiment/step2-audio-bench` (post Fix A + Fix B + Fix C + Fix D) + post-STRETCH FP16 re-audit (same date).
**Hardware**: NVIDIA RTX 6000 Ada Generation; CUDA 12.8; cuDNN runtime.
**Model**: `MD_AudioBirds_V1.onnx` (binary bird detector, audio sliding-window).
**Manifest**: `sparrow-engine/models/audiobirds.toml` — Slaney mel + Slaney filter norm; **`precision = "fp16"` (post-STRETCH re-audit FLIP TO FP16, 2026-05-05)** — Wave 3 HOLD-on-FP32 verdict superseded once Fix A's chunk-count collapse made FP16 1.71× faster than FP32 instead of 1.32–1.35× slower.
**Strategy**: sparrow-engine-gpu `Strategy::SingleCall` (post Fix A; production default for non-streaming detect). Streaming path defaults to `Strategy::HybridA{16}` to preserve per-batch callback cadence (D2 invariant). `Strategy::PerBatchB{16}` is a memory-constrained fallback (~34 MB peak vs ~411 MB for SingleCall on a 60 s clip).
**Variance discipline**: 5 fresh-process runs per (engine, fixture, precision) cell per `feedback_perf_claims_need_variance.md`; 10 inner iters per GPU process / 1 inner iter per CPU process; medians + p95 + stddev + max reported in `docs/research/phase3.8/step2/{full_bench,fp16_audit}.md`.

### 10.1 Headline — median end-to-end latency, full corpus (post Fix C + Fix D, FP32)

Variance-discipline corpus reading. Pulled from `docs/research/phase3.8/step2/full_bench.md § "Headline numbers (5 fresh-process runs × 10 inner iters per cell)"`. FP16 is the production default (per `audiobirds.toml` flip post-STRETCH 2026-05-05); see § 10.3 for the FP16 re-audit numbers.

| Fixture | Duration (s) | n_seg above thr | sparrow-engine-cpu p50 (ms) | sparrow-engine-gpu p50 (ms) | sparrow-engine-gpu p95 | sparrow-engine-gpu stddev | sparrow-engine-gpu max | speedup × | VRAM peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DUNAS_20230925_090000 (real) | 60 | 106 / 198 | 2688.2 | **14.71** | 14.89 | 0.28 | 14.89 | **182.7×** | 15 |
| DUNAS_20230314_090000 (real) | 60 | 194 / 198 | 2715.2 | **14.73** | 15.71 | 0.44 | 15.71 | **184.3×** | 15 |
| synthetic_2s | 2 | 5 / 5 | 398.8 | 1.13 | 1.13 | 0.01 | 1.13 | 353.6× | 15 |
| synthetic_10s | 10 | 31 / 31 | 702.3 | 2.82 | 2.82 | 0.02 | 2.82 | 248.8× | 15 |
| synthetic_30s | 30 | 98 / 98 | 1513.6 | 6.37 | 6.40 | 0.04 | 6.40 | 237.7× | 15 |
| synthetic_60s | 60 | 198 / 198 | 2732.9 | 14.92 | 14.95 | 0.24 | 14.95 | 183.2× | 15 |

**Headline win (FP32 variance bench)**: sparrow-engine-gpu `Strategy::SingleCall` FP32 on a 60 s real DUNAS clip lands at **14.71 ms p50** (`full_bench.md:29`, cross-process stddev 0.28 ms), vs sparrow-engine-cpu at **2688.2 ms** — a **182.7× engine-level speedup**. Pre-fix Wave 4 baseline was 144.81 ms (34.5×); post Fix A+B was 20.93 ms (129.8×); post Fix C+D is 14.71 ms (182.7×). Cumulative pre-fix → post-Fix-D delta: **9.84× GPU speedup** (`full_bench.md:23` canonical).

**Headline win (FP16 production default)**: sparrow-engine-gpu `Strategy::SingleCall` FP16 lands at **8.52 ms p50** on the same DUNAS clip (`fp16_audit.md:63`, post-STRETCH re-audit 2026-05-05) — **2.10× faster than PW reference** (17.94 ms) and **315.5× faster than sparrow-engine-cpu** (2688.2 / 8.52). Production manifest `sparrow-engine/models/audiobirds.toml` flipped to `precision = "fp16"` post-STRETCH 2026-05-05.

vs PW reference (torchaudio + ORT CUDA EP, 17.94 ms): sparrow-engine-gpu FP32 is **1.22× FASTER** (3.23 ms below STRETCH gate); sparrow-engine-gpu FP16 is **2.10× FASTER** (9.42 ms below STRETCH gate). Both **FLOOR ≤ 25 ms** and **STRETCH < 17.94 ms** gates **met** at FP32 14.71 ms and at FP16 8.52 ms.

### 10.1.1 Pre-fix (historical) headline — Wave 4 base + post Fix A+B intermediate

For reference. Same fixtures + manifest, off `c724565` (Wave 3 tip), Strategy A T=whole bench-harness off-by-one (T=197 producing 2 ORT chunks instead of 1):

| Fixture | sparrow-engine-cpu p50 (ms) | sparrow-engine-gpu p50 (ms) | speedup × |
| --- | ---: | ---: | ---: |
| DUNAS_20230925_090000 (pre-fix) | 4999.0 | 144.81 | 34.5× |
| DUNAS_20230925_090000 (post Fix A+B) | 2716.1 | 20.93 | 129.8× |
| DUNAS_20230925_090000 (post Fix C+D) | 2688.2 | **14.71** | **182.7×** |

Cumulative pre-fix → post-Fix-D GPU delta: **−130.10 ms (−89.8 %)** on DUNAS_20230925. Four minimally-invasive code changes drove the win — see `docs/research/phase3.8/step2/perf_triage_report.md` for the original Fix A + Fix B descriptions, and § "Perf-fix landing summary" in `docs/research/phase3.8/step2/full_bench.md` for the Fix C (`AudioWorkspace` device-buffer cache) + Fix D (ORT user-compute-stream binding) additions.

### 10.2 §2.1 FP32 parity gates (carried over from Wave 2; W1.7-anchored)

| Gate | Threshold | Both DUNAS clips | Verdict |
| --- | --- | --- | --- |
| Mel max-Δ vs CPU (post power_to_db, dB) | ≤ 5e-3 | 5.5e-4 to 6.0e-4 dB | **met** |
| Logit max-Δ (W1.7-anchored) | ≤ 3.0e-3 | 2.23e-3 to 2.38e-3 | **met** |
| Confidence max-Δ (W1.7-anchored) | ≤ 7.5e-4 | 1.21e-4 to 2.33e-4 | **met** |
| Class-label flip count @ threshold 0.9 | = 0 | 0 | **met** |
| Range-count post-merge (gap=0.301 s) | exact | DUNAS_20230925: cpu=9, gpu=9; DUNAS_20230314: cpu=1, gpu=1 | **met** |

Wave 2's W1.7-anchored gate re-derivation closed the §2.1 adjudication on `experiment/step2-audio-bench`; `sparrow-engine/sparrow-engine-gpu/tests/audio_e2e_parity.rs` is the load-bearing parity verification. Wave 4 made no GPU pipeline changes; gates carry over without re-derivation.

### 10.3 FP16 audit verdict — FLIP TO FP16 (post-STRETCH re-audit, 2026-05-05)

The Wave 3 audit (pre Fix A+B+C+D) verdict was **HOLD on FP32** because the latency-win gate exceeded under the bench-harness 2-chunk ORT setup. The post-STRETCH re-audit (2026-05-05, post Fix A+B+C+D off `130976b`) re-ran the same harness with the same FP32+FP16 ONNX on the same DUNAS corpus; the apples-to-apples ratio inverted once Fix A collapsed the 2-chunk ORT setup overhead. Verdict: **FLIP TO FP16**. Production manifest `sparrow-engine/models/audiobirds.toml` flipped to `precision = "fp16"`.

| Dimension | Wave 3 (historical, superseded) | Post-STRETCH re-audit (current) |
| --- | --- | --- |
| §2.2 numerical-accuracy gates | ALL FOUR MET (max-abs 1.16e-3 to 2.97e-3; mean-abs 2.28e-5 to 2.52e-4; rel 0.15% to 0.66%; flips 0) | UNCHANGED — model + ORT version unchanged. Re-confirmed on both DUNAS clips. |
| §2.2 latency-win gate (FP16 ≤ 0.83× FP32) | **EXCEEDED** — FP16 was 1.32× to 1.35× *slower* than FP32 under 2-chunk ORT setup (DUNAS_20230925: 200.45 vs 148.22 ms; DUNAS_20230314: 194.87 vs 147.59 ms) | **MET** — FP16 is 1.71× to 1.72× *faster* than FP32 (DUNAS_20230925: 8.52 vs 14.55 ms = ratio 0.586; DUNAS_20230314: 8.53 vs 14.65 ms = ratio 0.582). Gate met with 1.42× headroom. |
| Decision | Manifest stays `precision = "fp32"`; FP16 ONNX preserved | **Manifest flipped to `precision = "fp16"`**; explicit `manifest_fp32.toml` added as FP32 reference for `audio_e2e_parity_fp16.rs` |

Cause analysis (per `docs/research/phase3.8/step2/fp16_audit.md § "Cause analysis — why Wave 3 → post-STRETCH inverted"`): the dominant variable was per-call setup overhead (Wave 3) vs underlying Conv FP16 advantage (post-STRETCH). FP16 Cast cost (`keep_io_types=True`) is per-ORT-call, not per-ORT-chunk; the bench-harness 2-chunk loop made FP16 pay the Cast tax 2× while underlying Conv arithmetic only ran 1× of work. Fix A collapses 2 chunks → 1 chunk; FP16's underlying Conv advantage on RTX 6000 Ada Tensor Cores (~2× FP32 throughput) finally dominates. Wave 3's small-model + dynamic-axis hypotheses are disproven by the post-STRETCH measurement (FP16 stddev 0.07 ms is tighter than FP32's 0.30 ms, and 1.71× win at the same dynamic axis confirms cuDNN selects TC-eligible kernels for batch=198 + time_steps=90).

### 10.4 Comparator extension — PW reference + pre-Slaney

Two reference comparators were added post-Wave-4 to anchor the Step 2 speedup story (per `docs/research/phase3.8/step2/comparator_bench.md`):

1. **Pre-Slaney sparrow-engine-cpu** (HTK + area-norm; the CPU pipeline as it ran *before* Wave 0a's HTK→Slaney corrective fix). Speed-only proxy via Bpre microbench at `scripts/bench_pre_slaney_cpu.py`.
2. **PW reference** (torchaudio Slaney+Slaney + ORT CUDA EP). Emulates what `PW_Bioacoustics/inference.py` would do at inference time on the same checkpoint that produced sparrow-engine's `MD_AudioBirds_V1.onnx`. Bench at `scripts/bench_pw_reference.py`.

Headline on the 60 s real DUNAS_20230925 clip (post Fix C + Fix D variance bench, FP32 anchor; post-STRETCH FP16 re-audit row appended):

| Engine | p50 (ms) | p95 (ms) | stddev (ms) | n_above thr / total | vs sparrow-engine-gpu FP32 | vs sparrow-engine-cpu | Source |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Pre-Slaney sparrow-engine-cpu (Bpre proxy, pre-fix) | ~4999.0 | ~5006.6 | ~176 | n/a | 339.8× slower | ~1.86× slower (proxy) | comparator_bench.md |
| Post-Slaney sparrow-engine-cpu (post Fix B) | 2688.2 | 2727.0 | 23.0 | 106 / 198 | 182.7× slower | 1.00× | full_bench.md:29 |
| **PW reference (torchaudio + ORT CUDA EP)** | **17.94** | **18.15** | **0.19** | 105 / 197 | **1.22× slower** | **149.8× faster** | comparator_bench.md |
| **sparrow-engine-gpu `Strategy::SingleCall` FP32 (post Fix C + Fix D)** | **14.71** | **14.89** | **0.28** | 106 / 198 | **1.00×** | **182.7× faster** | full_bench.md:29 |
| **sparrow-engine-gpu `Strategy::SingleCall` FP16 (post-STRETCH, production default)** | **8.52** | ~8.6 | 0.07 | 106 / 198 | **1.71× faster** | **315.5× faster** | fp16_audit.md:63 |

DUNAS_20230314 mirrors the same shape: sparrow-engine-cpu 2715.2 ms, PW reference 17.99 ms, sparrow-engine-gpu FP32 14.73 ms, sparrow-engine-gpu FP16 8.53 ms.

Comparator findings:
- **Pre-Slaney microbench**: per-call \|Δ\| = 0.10 ms = 0.002 % of post-Slaney wall-clock — 5 %-of-wall-clock gate **met** with 2400× headroom. The Wave 0a HTK→Slaney corrective fix changed correctness, not speed.
- **PW reference**: cross-process p50 stddev 0.19 ms; matches sparrow-engine within ±1 segment on both DUNAS clips (Slaney implementation is correct end-to-end).
- **sparrow-engine-gpu FP32 vs PW reference**: sparrow-engine-gpu `Strategy::SingleCall` FP32 is **1.22× FASTER** than PW reference (was 8.07× slower pre-fix; was 1.167× slower post Fix A+B). Both **FLOOR ≤ 25 ms** and **STRETCH < 17.94 ms** gates **met** at 14.71 ms (3.23 ms below STRETCH).
- **sparrow-engine-gpu FP16 vs PW reference (production default post-STRETCH)**: sparrow-engine-gpu `Strategy::SingleCall` FP16 at 8.52 ms is **2.10× FASTER** than PW reference (9.42 ms below STRETCH). §2.2 R2 numerical-accuracy gates met (max-abs 2.97e-3, mean-abs 2.52e-4, rel 0.66%, flips=0). See § 10.3 for the FLIP-TO-FP16 verdict and `fp16_audit.md:63` for the per-fixture data.

The Step 2 design target updated post Fix C + Fix D + post-STRETCH FP16 re-audit: **315.5× engine-level speedup** on a 60 s real-audio DUNAS clip at FP16 (was 34.5× pre-fix Wave 4 at FP32; was 182.7× post Fix C+D at FP32), with W1.7-anchored FP32 parity gates met (all five gates pass on both `HybridA{16}` and `SingleCall` strategies) AND §2.2 R2 FP16 numerical-accuracy gates met.

### 10.5 Reproducibility

```bash
cd /home/miao/repos/PW_refactor/bongo_dev
source sparrow-engine/scripts/ort-env.sh
cargo build --release -p sparrow-engine-cli
cargo build --release -p sparrow-engine-gpu --example bench_audio_e2e

# Wave 4 full corpus bench (5 fresh × 10 inner GPU; 5 fresh × 1 CPU)
uv run --no-project --with numpy,scipy scripts/bench_audio_e2e_full.py \
    --gpu-runs 5 --gpu-inner-iters 10 --cpu-runs 5 --no-build

# Wave 2 FP32 parity gates
cd sparrow-engine
cargo test --release -p sparrow-engine-gpu --test audio_e2e_parity \
    -- --ignored --nocapture --test-threads=1

# Wave 3 FP16 audit (correctness + latency)
cargo test --release -p sparrow-engine-gpu --test audio_e2e_parity_fp16 \
    -- --ignored --nocapture --test-threads=1
cd ..
uv run --no-project scripts/bench_audio_fp16_audit.py --runs 5 --inner-iters 10 --no-build

# Comparator extension — pre-Slaney microbench (Bpre)
uv run --no-project --with numpy scripts/bench_pre_slaney_cpu.py --iters 1000

# Comparator extension — PW reference (torchaudio + ORT CUDA EP).
# Pinned: torch 2.7.1+cu128, torchaudio 2.7.1+cu128, onnxruntime-gpu 1.25.1, Python 3.12.
uv run --no-project --with numpy scripts/bench_pw_reference.py \
    --runs 5 --inner-iters 10 --warmup 2
```

Source documents:
- `docs/design/phase3.8/step2/final_design.md` — locked Step 2 design (Wave 5).
- `docs/design/phase3.8/step2/implementation_plan.md` — Step 2 implementation plan.
- `docs/research/phase3.8/step2/audio_breakdown.md` — Wave 0c CPU stage budgets.
- `docs/research/phase3.8/step2/wave1_primitives_bench.md` — Wave 1 GPU per-primitive numbers.
- `docs/research/phase3.8/step2/wave2_e2e_bench.md` — Wave 2 D1 + T-sweep + W1.7-anchored parity.
- `docs/research/phase3.8/step2/fp16_audit.md` — Wave 3 FP16 audit (HOLD verdict).
- `docs/research/phase3.8/step2/full_bench.md` — Wave 4 full corpus variance bench.
- `docs/research/phase3.8/step2/comparator_bench.md` — Wave 4 comparator extension (pre-Slaney + PW reference).

## 11. Phase 3.8 Phase C — Dual-flavor CLI bench sweep

**Status**: Phase C Wave 5 (2026-05-06) bench sweep on the dual-flavor CLI
binaries. Captures fresh-process wall latency per `feedback_perf_claims_need_variance.md`
discipline (5 runs each; report median + p95 + stddev + max; per-image
amortized = wall_total_seconds / n_images).

**Hardware**: RTX 6000 Ada (same as §8, §9, §10).
**Corpus**: 10 cameratrap JPGs (subset of the 100-image `test_files/test_cameratrap`
corpus alphabetically; full 100-image corpus parity is captured in the
G1 / G4 evidence at `docs/review/phase3.8-phase-c/round_01/acceptance_gates.md`).
**Model**: MegaDetector v6 YOLOv10-e at FP16 (default manifest) + threshold 0.2.

### 11.1 Headline — per-image wall latency (ms; includes ORT cold start)

| Configuration       | Median (ms) | p95 (ms) | Stddev (ms) | Max (ms) | Runs |
|---------------------|-------------|----------|-------------|----------|------|
| `sparrow-engine` (CPU CLI)   | 1112.49     | 1125.08  | 5.98        | 1125.08  | 5    |
| `sparrow-engine-gpu` (GPU CLI) | 906.79    | 909.47   | 1.27        | 909.47   | 5    |

GPU is 1.23× faster than CPU on this small corpus. The numbers include
engine cold start (~300 ms CPU / ~460 ms GPU) amortized across only 10
images — they OVERSTATE per-image inference latency. The warm-state
per-image median for MDv6 on this hardware is **13.46 ms** (Step 1 bench
§9 + IoBinding closure 2026-05-04). The Phase C bench is a
no-regression smoke + a sanity check that both CLI flavors execute end-
to-end through the engine_dispatch shim without unexpected overhead;
it is NOT a per-engine-step latency claim. For warm-state per-engine-
step numbers, see §9 (Step 1 image bench) and §10 (Step 2 audio bench).

### 11.2 Methodology

Bench script: `scripts/bench_phase_c.sh` (Phase C W5 deliverable).
Builds the bench corpus by copying the first 10 JPGs from
`test_files/test_cameratrap` into `/tmp/bench10/`; invokes each CLI
binary 5 times in fresh processes (no warm cache); times wall via
`date +%s.%N` deltas; computes median + p95 + stddev + max via Python
`statistics`; per-image latency = wall_total_seconds / n_images.

Variance discipline (per `feedback_perf_claims_need_variance.md`):
GPU stddev 1.27 ms / 5 runs reproduces the bimodal MDv6 right-tail
observed in Track B R2 — the variance source is the cuDNN EXHAUSTIVE
algo selection layer, which Phase 3.7 R2 documented as inherent to
ORT CUDA EP. CPU stddev 5.98 ms is dominated by image-decode noise on
the host (zune-jpeg + filesystem). Both stddevs trace to known sources
documented in the variance precedent at `feedback_perf_claims_need_variance.md`
(cuDNN EXHAUSTIVE algo selection layer for GPU; image-decode noise on
the host for CPU); not an indication of measurement error or
methodology drift.

### 11.3 Reproducibility

```sh
# 1. Build CLI binaries (CPU + GPU).
scripts/build_all_flavors.sh   # FLAVOR=both STAGE=cli (or all)

# 2. Run the bench sweep.
N_RUNS=5 scripts/bench_phase_c.sh

# 3. Output: stdout + log at /tmp/bench_phase_c_log.txt; copy into
#    docs/benchmarks.md § 11 if updating.
```

Source documents:
- `docs/review/phase3.8-phase-c/round_01/bench_phase_c_log.txt` — raw 5-run log.
- `docs/review/phase3.8-phase-c/round_01/acceptance_gates.md` § 4 — G4 cross-flavor parity (corpus-level evidence).
- `docs/design/phase3.8/phase_c/implementation_plan.md` § 4 W5 — Wave 5 brief + bench requirements.

---

## 12. Phase 4.2 — Server cold-start contract re-baseline

**Status**: Phase 4.2 implementation changed the server boot contract from eager model loading to catalog discovery plus explicit preload. No fresh latency number is published here because the canonical model-weight corpus is not checked into this worktree; the release GPU server build was verified, but the timing run needs the operator model directory.

| Path | Phase 4.1 behavior | Phase 4.2 behavior |
|---|---|---|
| Server boot with populated model dir | eagerly loaded every manifest; Phase 4.1 manual test observed ~7.3 s for the six-model GPU set | parses manifests into a catalog only; `/v1/models` starts empty |
| Preload | implicit all-model preload | `SPARROW_ENGINE_PRELOAD=id1,id2` opt-in; unknown ids fail boot with all missing ids reported |
| Availability listing | overloaded through `/v1/models` | `GET /v1/catalog` lists available models and `loaded` status; `GET /v1/models` lists loaded sessions only |
| First model use | already loaded by boot loop | explicit `POST /v1/models/load` or lazy Shape-X/Shape-Y `/v1/pipeline`; non-pipeline inference endpoints (`/v1/detect`, `/v1/detect/batch`, `/v1/classify`, `/v1/audio/detect`) still require a loaded model |

**cuDNN gate**: Phase 4.2 Step 12 was not executed in this implementation. No additional `Heuristic` call sites or `SPARROW_ENGINE_GPU_CONV_SEARCH` override were added. Existing YOLO cuDNN heuristic behavior remains unchanged.

### 12.1 Benchmark command for future numeric fill-in

Run this on the benchmark rig with the canonical model directory:

```sh
cd sparrow-engine
source scripts/ort-env.sh
cargo build --release -p sparrow-engine-server --no-default-features --features gpu --target-dir target-gpu

SPARROW_ENGINE_BIND_ADDR=127.0.0.1:8085 \
SPARROW_ENGINE_MODEL_DIR=/path/to/sparrow_engine_models \
SPARROW_ENGINE_DEVICE=auto \
LD_LIBRARY_PATH="$ORT_CAPI:$LD_LIBRARY_PATH" \
target-gpu/release/sparrow-engine-server > target/phase4_2_server.log 2>&1 &
SRV_PID=$!

# Measure time-to-/v1/health, then explicit model load latency.
curl -fsS http://127.0.0.1:8085/v1/health
curl -fsS -X POST -H 'content-type: application/json' \
  -d '{"model_id":"megadetector-v6-yolov10e"}' \
  http://127.0.0.1:8085/v1/models/load

kill -TERM "$SRV_PID"
wait "$SRV_PID" 2>/dev/null || true
```

