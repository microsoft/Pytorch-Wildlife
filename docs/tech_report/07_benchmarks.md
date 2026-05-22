# Benchmarks

All sparrow-engine performance numbers, with hardware, workload, and methodology. Raw logs in `appendices/benchmark_logs/`.

## Hardware baseline

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 6000 Ada Generation (49 GB each, sm_89) |
| CPU | same workstation as GPU |
| OS | Ubuntu 22.04 |
| glibc | 2.35 |
| ORT | 1.24.2 (CPU + CUDA EP with cuDNN) — pre-Phase-4.1; ORT C library now pinned to 1.25.1+ per Phase 4.1 MT-4.1-14 (`docs/master_plan.md § Phase 4.1`) |
| cuDNN | 9.10+ (installed in `~/.local/cudnn`; avoids 9.8 Conv bug) |
| CUDA | 12.6.3 |

All benchmarks below run on this single workstation unless otherwise noted. Multi-GPU scaling not yet measured.

## Model under test

MegaDetector v6 (YOLOv10-E). Input 1280×1280. Output `[1, 300, 6]` (cx, cy, w, h, conf, class) with in-graph NMS. Detection confidence threshold: 0.40.

## Test data

100 camera-trap images from `test_files/test_cameratrap/`. Mixed resolution (typical ~2000×1500), mixed content (animals + empty + vehicles).

## Phase 2 — HTTP end-to-end benchmark

Measures: worker process scans directory → HTTP multipart upload → sparrow-engine-server inference → JSON response → CSV output file. Includes HTTP overhead, multipart encoding, JSON parsing.

| Worker | Backend | Total | Per-image | Detections | vs Triton |
|--------|---------|-------|-----------|------------|-----------|
| **Rust worker** | **Sparrow Engine GPU** | **5.8 s** | **58 ms** | 217 | **7.7× faster** |
| Python worker | Sparrow Engine GPU | 7.3 s | 73 ms | 217 | 6.2× faster |
| Python worker | Triton GPU | 44.9 s | 449 ms | 215 | baseline |
| Python worker | Sparrow Engine CPU | 194.0 s | 1.94 s | 217 | n/a |
| Rust worker | Sparrow Engine CPU | 193.2 s | 1.93 s | 217 | n/a |

Source: `docs/benchmarks.md:47-51`. Date: 2026-04-14.

### Reading the numbers

- **Sparrow Engine GPU vs Triton GPU: 7.7× faster.** Triton's overhead comes from its inference scheduling, tensor protocol (gRPC), and model-management layers on top of raw ORT. Sparrow Engine's axum shell passes the request straight to libsparrow_engine with minimal middleware.
- **Rust worker vs Python worker (GPU): 1.26×.** 5.8 s vs 7.3 s. At ~50 ms/image inference, HTTP client overhead matters. `reqwest` (Rust) is faster than `httpx` (Python) for multipart uploads.
- **Rust vs Python worker (CPU): identical.** 193.2 s vs 194.0 s. Inference at ~2 s/image dominates any client difference.
- **Detection count: Sparrow Engine 217 vs Triton 215.** Sparrow Engine returns ~4 extra detections (~2%) because Triton applied a second NMS pass after the model's in-graph NMS. See § 06 gotchas — detection parity.

### Cold start

461 ms on RTX 6000 Ada (measured, single request). Includes container warm state, first-request model load, first inference.

Source: `docs/master_plan.md:89`, measured during Phase 2 handoff.

## Phase 2.5 — Direct inference (no HTTP)

Measures raw inference speed: image decode → preprocess → ORT inference → postprocess. No HTTP, no multipart. Threshold: 0.40.

| Engine | Device | Total | Per-image (mean) | Per-image (median) | Detections |
|--------|--------|-------|------------------|--------------------|------------|
| **libsparrow_engine (Rust)** | **GPU** | **7.1 s** | **71.5 ms** | **63.0 ms** | 217 |
| Python ORT | GPU | 8.9 s | 89.2 ms | — | 218 |
| Python ORT | CPU | 457.0 s | 4569.8 ms | 4564.8 ms | 218 |
| libsparrow_engine (Rust) | CPU | 494.0 s | 4939.8 ms | 4940.5 ms | 217 |

Source: `docs/benchmarks.md:18-21`. Date: 2026-04-14.

### Reading the numbers

- **GPU: Rust is 1.25× faster than Python ORT (71.5 ms vs 89.2 ms).** The 17.7 ms gap comes from faster image decode (Rust `image` crate vs PIL) and preprocessing (`ndarray` vs `numpy`). No GIL. ORT inference itself is the same C library in both cases — roughly 70% of GPU wall time.
- **CPU: Rust is 7.5% slower (4940 ms vs 4570 ms).** Both call the same ORT C library for inference (~4500 ms in both). Gap comes from image decode / resize implementation differences and Python ORT using a slightly different default thread count. CPU is ~99% ORT inference wall time — Rust wins only on the 1%.
- **Mean vs median (Rust GPU).** Mean 71.5 ms vs median 63.0 ms. The 8.5 ms gap is warm-up effects on the first few images. Median is the better number for steady-state throughput.
- **Detection count.** 217 (Rust) vs 218 (Python). 1-detection difference is letterbox rounding in preprocessing (Rust rounds padding top/left; PIL distributes symmetrically).

### Pre-benchmark fixes that landed

- `Device::Auto` intra-threads: changed from hardcoded 1 to `available_parallelism()` (capped at 8).
- Added explicit `GraphOptimizationLevel::All` to ORT session builder.
- CPU improved from 5421 ms → 4940 ms/image (8.9% improvement) after these fixes.

## Pipeline benchmark (detect + classify)

`spe pipeline --detector mdv6 --classifier speciesnet` on 100 images, GPU, post cuDNN 9.21 upgrade.

| Metric | Value |
|--------|-------|
| Total | 7.7 s |
| Per-image | 76 ms |

Source: `~/.claude/projects/.../memory/project_tech_report_notes.md`. Date: Phase 3 manual testing (2026-04-20 GPU retest).

Note: pipeline GPU hits MT-17 intermittent heap corruption at process exit (~20-33% reproduction). Inference results are correct when the process doesn't crash. See § 06 gotchas — MT-17.

## Docker image size

| Image | Base | Compressed | Uncompressed |
|-------|------|------------|--------------|
| Sparrow Engine CPU | `debian:bookworm-slim` | 163 MB | ~420 MB |
| Sparrow Engine GPU | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` | ~1.4 GB | ~4 GB |
| Rust worker | minimal Debian | 15 MB | ~50 MB |
| Python worker | `python:3.12-slim` + sparrow-engine-client | ~200 MB | — |

For comparison, the research synthesis projected Python v1 Docker image at ~650 MB (CPU only). Sparrow Engine CPU is ~25% the size.

## CLI binary size

`spe` binary distributed via GitHub Releases, static ORT linked in: ~35 MB. No runtime dependencies.

For comparison, pip installing `sparrow-engine-python` brings in `onnxruntime` (CPU wheel: ~70 MB) plus `pillow` plus the sparrow-engine wheel — total env ~150 MB. Static binary wins on portability; pip win on size when users already have Python.

## Test counts over the project

| Date | Event | libsparrow_engine | sparrow-engine-cli | sparrow-engine-server lib | sparrow-engine-python | sparrow-engine-client | Total |
|------|-------|----------|-----------|-------------------|--------------|--------------|-------|
| Phase 2 ship | — | 80 P/Invoke + ? | — | 23 | — | — | ~100 |
| Phase 2.5 ship | — | 115 | 31 | 23 | 20 | — | 189 |
| Phase 3 ship | — | 163 | 38 | 23 | 20 | — | 244 |
| Phase 3 final audit-fix R5 CONVERGED | 2026-04-20 | 171 | 45 | 2 (+ 6+8+9 integration ignored) | 24 | 21 | 263 |
| Post `serial_test` retrofit | 2026-04-20 (commit 7fed112) | 173 | 45 | 2 | 24 | 21 | 265 |

sparrow-engine-server lib tests dropped from 23 to 2 because Phase 3 audit-fix separated lib unit tests (2) from integration tests that require a live server (6+8+9 = 23, all `#[ignore]`'d by default, run separately). Integration tests still exist.

## Golden-output tolerance

Reference outputs generated by libsparrow_engine itself (see § 04, D-v4-9). Tolerance:

| Metric | Vision | Audio |
|--------|--------|-------|
| Bbox coordinate | ±0.005 (normalized) | n/a |
| Confidence | ±0.12 | ±0.25 |

The audio tolerance is wider because audio models use float64 for intermediate mel computations internally while ONNX-exported graphs run float32. The f32/f64 gap accounts for the wider tolerance. Source: `docs/master_plan.md:70`.

## What is not yet measured

- **Multi-GPU scaling.** sparrow-engine-server has not been benchmarked across 2+ GPUs. Single-GPU saturation expected at ~60 QPS for MDv6 + warm cache; 2-GPU should be ~2× if sparrow-engine-server dispatches correctly.
- **Concurrent HTTP load.** sparrow-engine-server's axum + tokio runtime supports concurrent inference requests (session `Mutex` serializes per model, not globally). Not stress-tested under high concurrency.
- **Memory footprint under load.** RSS of sparrow-engine-server with the full Sparrow Studio Local catalog loaded vs a single detector model. Relevant for edge deployment.
- **Jetson / ARM64.** sparrow-engine has not been tested on NVIDIA Jetson or generic ARM64. ORT supports both but needs a test pass.
- **CPU models with AVX-512.** Intel Sapphire Rapids and AMD Zen 4 have AVX-512; ORT uses it when available. Current CPU benchmark numbers are from a Zen 2 or similar. Newer CPUs should be faster.

These are candidate Phase 3.7 Track B (perf research) or Phase 4+ follow-ups depending on user-priority signals. Phase 3.5 closed 2026-04-28 without folding any of these in; MT-3.5-12 (bimodal libsparrow_engine latency) is the only Phase 3.5 perf item that carried forward, into Phase 3.7 Track B.

## Reproducibility

All Phase 2 and Phase 2.5 numbers above used:

- **ORT 1.24.2** with `onnxruntime-gpu` pip wheel (not standalone ORT binary)
- **Fixed cuDNN 9.10+** via `~/.local/cudnn`
- **`--test-threads=1`** for test runs to avoid engine-singleton races (now replaced by `serial_test` tagging)
- **Same 100-image test set** across configurations (`test_files/test_cameratrap/`)

Raw logs in `appendices/benchmark_logs/` (to be populated from `docs/benchmarks.md` snapshots).

## Confidence

**Confidence**: HIGH
- Factual accuracy: HIGH — every number traced to `docs/benchmarks.md` or `docs/master_plan.md` or `project_tech_report_notes.md`
- Completeness: MEDIUM — covers what has been measured; explicit "not yet measured" list for the rest
- Freshness: HIGH — 2026-04-29 (R11 forward-pointer correction at L141: Phase 3.5/4 → Phase 3.7 Track B / Phase 4+; post-serial_test retrofit test counts unchanged)

## References

- `docs/benchmarks.md` — primary benchmark log
- `docs/master_plan.md` § Benchmark Results — Phase 2 and Phase 2.5 summaries
- `~/.claude/projects/.../memory/project_tech_report_notes.md` — pipeline GPU benchmark + tech-report numbers
- `commit 7fed112` — serial_test retrofit (test count +2)
- `commit 09ee0aa` — tree state at which the 265-test total was verified
