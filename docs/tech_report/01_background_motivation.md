# Background and Motivation

Why sparrow-engine exists, what it replaces, and what constraints shaped the design.

## The pre-sparrow-engine system

Before sparrow-engine, AI-for-biodiversity inference at Microsoft's AI for Good Lab ran on two stacks.

**Python stack — PytorchWildlife / CameraTraps**. A PyTorch-coupled deep-learning framework (`CameraTraps/PytorchWildlife` repository, under `/home/miao/repos/PW_refactor/CameraTraps/`). Shipped as a pip package. Used directly by scripts and notebooks for detection and classification. The ONNX-export path existed but was secondary — PyTorch was the primary inference backend. Deployment was Docker image ~650 MB with a 4.3 s cold start and a ~3.2 s end-to-end pipeline for the typical startup + preprocess + single-image inference path (`docs/research/v2/round_05/research_v2_final_synthesis.md:59`, Subonis and Pistek RegNet benchmarks).

**Server stack — Triton on GPU**. NVIDIA Triton Inference Server hosted the ONNX models for Sparrow Studio Web worker deployments. Sparrow workers spoke the Triton gRPC/HTTP protocol. End-to-end HTTP benchmark: 449 ms per image on RTX 6000 Ada for 100 camera-trap images (`docs/benchmarks.md:49`). That wall-clock includes Triton's inference scheduling, tensor protocol serialization, and model-management layers on top of raw ORT inference.

## What drove the rewrite

### Performance cost of the Python stack

Three specific symptoms pushed the team off the Python implementation:

| Metric | Python v1 | Rust v2 target | Ratio |
|--------|-----------|----------------|-------|
| Docker image (CPU) | ~650 MB | ~70–200 MB | 3–13× smaller |
| Cold start | ~4.3 s | ~348 ms | 12× |
| Total pipeline (startup + preprocess + infer, single image) | 3214 ms | 91.4 ms | 2.87× |
| Isolated inference speed | ~85 ms | ~85 ms | 0% diff (same ORT C API) |

Source: `docs/research/v2/round_05/research_v2_final_synthesis.md:51-64`, Subonis + Pistek benchmarks.

Note the last row. **The rewrite does not target faster raw inference.** Both stacks end up in the same ONNX Runtime C library. The wins come from everything around inference: image decode, preprocessing, cold start, memory, GIL elimination, and deployment size.

### GIL contention in concurrent deployments

Python's Global Interpreter Lock blocks meaningful concurrent inference when multiple workers share one process. In Sparrow Studio Web, workers scale horizontally and each needs its own Python interpreter — inefficient compared to a Rust worker that runs all models in one process with thread-level concurrency. `docs/research/v2/round_05/research_v2_final_synthesis.md:63` documents the GIL issue under "Concurrent throughput".

### Overhead cost of the Triton server

Triton is a general-purpose inference server. For a workload with a small number of sparrow-engine-owned models and a single consumer pattern (multipart image upload, JSON response), Triton's generality imposes overhead that shows up in benchmarks:

| Configuration | Per-image (end-to-end HTTP) | Source |
|---------------|-----------------------------|--------|
| Sparrow Engine GPU + Rust worker | 58 ms | `docs/benchmarks.md:47` |
| Sparrow Engine GPU + Python worker | 73 ms | `docs/benchmarks.md:48` |
| Triton GPU (baseline) | 449 ms | `docs/benchmarks.md:49` |

Both sparrow-engine worker types also return ~4% more detections than the Triton baseline. Root cause: Triton's pipeline applied a second NMS pass after the model's in-graph NMS, deleting legitimate boxes that survived the first pass. This is both a performance loss and a correctness loss that sparrow-engine sidesteps by putting NMS in the ONNX graph and never running it again in library code.

### Sparrow Studio integration friction

Sparrow Studio Local is a cross-platform .NET desktop application (currently Windows, with macOS and Linux support in progress via the Avalonia UI port). Shipping a Python runtime inside a .NET product pulls in a Python installer, managed-unmanaged interop, and an environment teams cannot reliably lock down. A native dynamic library consumable via C# P/Invoke avoids that cost and matches how other ML libraries ship on each platform (onnxruntime, OpenCV, tensorflow-lite — all available as native libraries on Windows, macOS, and Linux).

Sparrow Studio Web wanted a container it could treat as a black-box HTTP service. Triton delivered that but at the overhead cost above, and with a protocol heavier than the workload needed.

### Codebase divergence across consumers

The Python stack embedded preprocessing and postprocessing inside library code. Sparrow Studio Local reimplemented portions in C#. Sparrow Studio Web had its own glue in workers. Three partial reimplementations of the same preprocessing code existed across consumers. Any correctness fix had to land three times. The v4 design explicitly removed this divergence by moving all preprocessing and postprocessing into libsparrow_engine behind TOML manifests. See `04_design_decisions.md` § libsparrow_engine owns pre/post.

## What the replacement needed to be

From v2 research synthesis (§ "Project Vision") and v4 design report:

1. **Model-agnostic core library.** Not wildlife-specific in API surface. Release as an independent open-source repository so external biodiversity researchers and field biologists can use it without Sparrow Studio.
2. **Self-describing models.** No vendored architecture code in the library. Models ship as ONNX files plus TOML manifests that fully describe pre/post.
3. **Single inference runtime.** ONNX everywhere, vision and audio. No TFLite fallback, no PyTorch at inference time.
4. **Multiple consumer surfaces from one codebase.** C DLL for Sparrow Studio Local; HTTP service for Sparrow Studio Web; Rust CLI for researcher workflows; Python PyO3 bindings for notebook and scripting use; Python HTTP client for remote Sparrow Studio Web users.
5. **Offline-capable.** All models downloadable, SHA-256-verified, work without internet after first download (Phase 3: `catalog.rs` with in-manifest `onnx_sha256`, `onnx_size_bytes`).
6. **Conservation-first UX.** 90% of users are ecologists, not software engineers. `pip install sparrow-engine` must feel native; `spe detect` must work with no configuration.

Sources: `docs/research/v2/round_05/research_v2_final_synthesis.md` § 1.1–1.3, `docs/design/v4/libsparrow_engine/design_report.md`.

## Why Rust, not C++

Same ONNX Runtime C API underneath. Rust was chosen over C++ for four reasons that showed up in the v2 research (`docs/research/v2/round_05/research_v2_final_synthesis.md:78`, subagent_cpp_alternatives.md):

- **Memory safety.** Preprocessing is non-trivial — letterbox, normalization, NCHW packing, audio mel-spectrogram. Rust's ownership model removes entire classes of bugs that would have had to be caught by CI.
- **Tooling.** cargo, clippy, rust-analyzer give immediate feedback loops. C++ tooling (CMake, CTest, various lint stacks) is fragmented.
- **Web and CLI ecosystem.** `axum` for HTTP, `clap` for CLI, `ort` for ONNX Runtime, `hound` + `realfft` + `rubato` for audio, `image` for vision preprocessing. C++ equivalents exist but integration is heavier.
- **FFI story.** `cbindgen` and `csbindgen` generate C and C# headers from Rust source. Shipping a DLL for Sparrow Studio Local is straightforward.

## Why not stay with PyTorch and accept slower inference

PyTorch at inference time pulled in a ~1 GB dependency, made cold-start latency dominate for short-running workloads, and tied the library to a specific PyTorch major version. ONNX decouples model definition from the inference runtime. The library onboards any ONNX model that conforms to the manifest schema — current set includes the full Sparrow Studio Local model catalog (MegaDetector variants, DeepFaune, HerdNet, OWL-T, SpeciesNet, audio classifiers) plus any additional ONNX model a user drops into `{model_dir}/`. All models route through the same `Engine` API, all as `.onnx` files.

## Constraints that shaped the design

- **Cross-platform desktop target for Sparrow Studio Local (Windows today, macOS/Linux via Avalonia port in progress).** Rules out Linux-only dependencies. Both sparrow-engine builds use Rust's cross-platform story plus ORT's native support across Windows, macOS, and Linux.
- **ORT CUDA EP bugs with NHWC.** ORT issues #27912 and #12288 force NCHW layout. See `06_gotchas_and_constraints.md` § NCHW mandate.
- **ORT Environment is process-global.** Drives the engine-singleton decision.
- **glibc 2.35 on the dev Ubuntu 22.04 workstation; ORT static lib requires glibc 2.38+.** Drives the dynamic-linking path via `scripts/ort-env.sh` for tests and local builds; the static lib is used only for the shipped CLI binary and in Docker containers with newer base images.
- **cuDNN 9.8 has a Conv-engine bug with asymmetric padding on sm_89.** PyTorch and TensorFlow wheels bundle 9.8 by default. Sparrow Engine requires standalone `nvidia-cudnn-cu12>=9.10` to avoid the bug on RTX 6000 Ada. See `06_gotchas_and_constraints.md` § cuDNN 9.8.

## Phase 0 artifacts that survived the rewrite

The rewrite kept:

- **Model set.** Models onboarded to sparrow-engine are models PytorchWildlife already supported; sparrow-engine's design is model-agnostic so adding a new model is a manifest + ONNX file, not a code change.
- **Golden reference outputs.** Sparrow Engine regenerates them using libsparrow_engine itself, but the original PytorchWildlife-generated references were the starting point for cross-checking correctness during Phase 1 (`bbox ±0.005`, `confidence ±0.12` vision / `±0.25` audio — the float32/float64 precision gap).
- **MegaDetector v1.5 JSON export format.** Verified from upstream `run_detector_batch.py`. Phase 3 export uses the same schema: `[x_min, y_min, width, height]` normalized [0, 1].
- **Algorithmic choices.** Audio preprocessing (`n_fft=1024`, `hop=512`, `n_mels=224`, `sr=48000`, 1.0 s windows with 0.3 s stride), detection thresholds, pipeline crop-and-classify logic — all ported directly from the Python stack.

The Python stack was not wrong. It was the right first iteration. The rewrite reflects having enough data on where it hurt and enough consumer variety to justify a rewrite in a language that handles the deployment story better.

**Confidence**: HIGH
- Factual accuracy: HIGH — all metrics cited to `docs/research/v2/round_05/` synthesis, `docs/benchmarks.md`, and v4 design
- Completeness: HIGH — covers Python-era, Triton-era, consumer-friction, and the positive case for the replacement
- Freshness: HIGH — 2026-04-21

## References

- `docs/research/v2/round_05/research_v2_final_synthesis.md` — full research synthesis that drove v2 design
- `docs/design/v2/round_04/definitive_design_v2.md` — the design that became libsparrow_engine
- `docs/design/v4/libsparrow_engine/design_report.md` — v4 design report (libsparrow_engine + Sparrow Studio integration)
- `docs/benchmarks.md` — all benchmark numbers cited above
- `/home/miao/repos/PW_refactor/CameraTraps/` — source Python repository
- ORT bug references: #27912 (Conv SafeInt overflow with NHWC), #12288 (CUDA EP NHWC dynamic shapes)
