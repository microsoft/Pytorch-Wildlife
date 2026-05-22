# Executive Summary

Sparrow Engine is a generic, model-agnostic inference library for AI-for-biodiversity workloads. It is written in Rust, runs ONNX models (vision and audio) on CPU or GPU via ONNX Runtime, and is intended to ship as an independent open-source repository.

At Microsoft's AI for Good Lab, sparrow-engine is the inference backbone for two downstream products: Sparrow Studio Local (desktop application — currently Windows, being ported to macOS and Linux via Avalonia; consumes sparrow-engine as a native DLL via C# P/Invoke) and Sparrow Studio Web (server deployments — consumes sparrow-engine via a Docker HTTP service). The same library also ships as a Rust CLI (`spe`) and as a Python package (`sparrow-engine-python`, PyO3 direct bindings) so external researchers and field biologists can use it without either Sparrow Studio product. Sparrow Engine replaces the Python-era PytorchWildlife pipeline and the Triton-based server infrastructure with a single-codebase, single-runtime design.

## What sparrow-engine is

- **Core library**: `libsparrow_engine` (Rust, `rlib` + optional `cdylib` via `ffi` feature). Model-agnostic — onboards the full Sparrow Studio Local model catalog (MegaDetector variants, DeepFaune, HerdNet, OWL-T, SpeciesNet, audio classifiers) plus any additional ONNX model added via TOML manifest, no code change required.
- **Runtime**: ONNX Runtime (`ort` crate, v2.0.0-rc.12) for all models, vision and audio. No TFLite, no PyTorch at inference time. (ORT C library 1.25.1+ post-Phase-4.1 MT-4.1-14 pin per `docs/master_plan.md § Phase 4.1`.)
- **Workspace** (pre-Phase-3.8-Phase-A — see `docs/design/architecture.md` for the post-2026-05-02 7-crate state): four crates. `libsparrow_engine` (core) + `sparrow-engine-server` (axum HTTP shell) + `sparrow-engine-cli` (`spe` binary) + `sparrow-engine-python` (PyO3 0.25 bindings). Plus `sparrow-engine-client` (a separate pure-Python HTTP SDK) for Sparrow Web consumers. A 5th workspace crate (`sparrow-utils`) is planned for Phase 4+ to host stateless utilities (per Phase 3.7 Track A).
- **5-component architecture** (Phase 3.7 Track A, ratified 2026-04-29): sparrow-engine is the engine in a 5-component system — Sparrow Studio (annotation/GUI) + sparrow-engine (this repo) + `sparrow-data` (sibling, data substrate) + `fine-tuning repo` (colleague's repo) + `sparrow-ops` (sibling, model registry/drift/CI). Sparrow Engine stays engine-only. Canonical reference: `docs/design/architecture.md`.
- **Consumers**: C# via P/Invoke (Sparrow Studio Local), Docker HTTP (Sparrow Studio Web, three worker types), `spe` CLI (single binary, ~35MB static ORT, GitHub Releases), Python (PyO3 direct bindings or HTTP client).
- **Data path**: TOML manifests drive preprocessing and postprocessing; libsparrow_engine owns all pre/post. Models ship normalized bboxes at the public API boundary. NMS lives in the ONNX graph, never in libsparrow_engine.

## Headline results

| Workload | Hardware | Per-image | vs Triton |
|----------|----------|-----------|-----------|
| Sparrow Engine GPU + Rust worker, 100 camera-trap images (end-to-end over HTTP) | RTX 6000 Ada | 58 ms | **7.7× faster** |
| Sparrow Engine GPU + Python worker, 100 images (end-to-end over HTTP) | RTX 6000 Ada | 73 ms | 6.2× faster |
| Triton GPU baseline (end-to-end over HTTP) | RTX 6000 Ada | 449 ms | 1.0× |
| Sparrow Engine CPU (end-to-end over HTTP) | same CPU as Triton | ~1.94 s | n/a |

All four rows above are end-to-end HTTP benchmarks: worker scans directory → HTTP multipart upload → inference → JSON response → CSV output. The 449 ms Triton baseline includes Triton's inference scheduling, tensor protocol, and model-management layers; raw ORT inference inside Triton is substantially faster than that wall-clock figure.

Raw inference (no HTTP) comparison for the same workload: libsparrow_engine Rust GPU 71.5 ms mean (63.0 ms median) vs Python + ORT GPU 89.2 ms — Rust is 1.25× faster on the non-inference portion (image decode + preprocess + postprocess). ORT inference itself is ~70% of GPU wall time and is the same C library in both cases.

Cold start: 461 ms on RTX 6000 Ada (measured, single request).

Detection parity: sparrow-engine returns ~4% **more** detections than Triton, attributable to removal of a redundant NMS pass that Triton applied after the model's in-graph NMS.

Pipeline (detect + classify) on GPU: 76 ms/image for 100 images after the cuDNN 9.21 upgrade.

## Validation status

- **265/265 automated tests** passing across 5 packages (4 Rust crates + `sparrow-engine-client` pure-Python SDK) — 173 libsparrow_engine + 45 sparrow-engine-cli + 2 sparrow-engine-server lib + 24 sparrow-engine-python + 21 sparrow-engine-client at `dev@09ee0aa`. (Subsequent phases through Phase 4.4 (2026-05-13) brought test counts to 343+ post-Phase-4; see `docs/master_plan.md § Phase 4` Verification gate and `docs/changelog.md` for per-phase counts.)
- **69/69 manual checks** passed across two sessions (CPU 2026-04-18, GPU retest 2026-04-20); 17 issues + 5 decisions logged in `docs/review/phase3-testing/manual_testing_logs.md`.
- **Phase 2.5 consumer audit**: 4 rounds, 9 fixes, CONVERGED R4.
- **Phase 3 final audit-fix**: 5 rounds, 28 fixes, CONVERGED R5. Two genuinely new findings (sparrow-engine-client SRV1 wire-compat break, COCO label_id collision) were surfaced by the inquisitor's independent scan *after* the first-round reviewer missed them — evidence that the iterative approach pays for itself.

## Phase status

| Phase | What | Status |
|-------|------|--------|
| 0 | PytorchWildlife (Python era, CameraTraps repo) | Archived |
| 1 | libsparrow_engine inference engine (vision + audio, model-agnostic, full Sparrow Studio Local catalog onboarded) | Complete |
| 2 | Docker HTTP service (initial thirteen-route API, later expanded to the current 15 endpoint/method surface; CPU + GPU images, `sparrow-engine-client` SDK, Rust worker) | Complete |
| 2.5 | `spe` CLI + `sparrow-engine-python` package (8 MVP functions, PyO3 direct bindings) | Complete |
| 3 | Utilities + local model catalog (hash, day/night, viz, stats, export, catalog; 9 FFI exports) | Complete |
| 3.5 | 12 planned enhancements across 11 sections in 5 waves. W1 (2026-04-22): MT-17 GPU mitigation, debug+release CI matrix, Sparrow summary parity, pytorchwildlife-compat shim. W2 (2026-04-23): viz subtype dispatch (MT-9), CLI output hygiene, Python progress + tracing bridge. W3 (2026-04-23): S4/S9/S10. W4 audit-fix CONVERGED 2026-04-23; W5 clean-room install + audit-fix CONVERGED 2026-04-24. | Complete (2026-04-28; manual-test sign-off; MT-3.5-12 bimodal libsparrow_engine latency carried to Phase 3.7 Track B) |
| 3.7 | Research + Planning. Track A (MLOps decomposition) CONVERGED 2026-04-28 (R4 research) + ratified 2026-04-29 (recommendation + sibling scope + codebase survey). Track B (libsparrow_engine-vs-PyTorch latency gap) NOT STARTED. | In progress |
| 4 | Docker data management — sparrow-engine-side primitives (`?store=true`, `halt_on_store_failure`, idempotent writes, drift Tier-1/2 metric hooks). The `sparrow-engine-db` sidecar + storage + annotations + query API moved to `sparrow-data` sibling per Phase 3.7 Track A (deferred). | Not started |
| 5 | MLOps (sibling-repo decomposition per Phase 3.7 Track A) — `sparrow-data` (data substrate: storage, snapshot versioning, preprocessing orchestration, inference logging) + `sparrow-ops` (model registry, drift detection Tier-3, CI/CD, monitoring, retraining orchestrator). Manifest `[provenance]` pointer fields (formerly tagged "Phase 5a"; folded into Phase 4 per `docs/master_plan.md:360`) land in Phase 4 first. Sibling construction deferred per `feedback_dev_first_release_last.md`. | Not started |
| Fine-tuning | `fine-tuning repo` — colleague's repo (separate project). Repo-and-project-agnostic Docker training environment; consumes per-project preprocessing scripts; outputs sparrow-engine-compatible ONNX with manifest provenance metadata. | Not started (deferred; cross-team-owned) |

See `02_project_timeline.md` for dates and per-phase decisions.

## Key design decisions (see `04_design_decisions.md` for full list)

- **ONNX for all models.** Vision and audio alike. TOML manifests (not YAML, since `serde_yaml` is deprecated) drive the pre/post pipeline.
- **NCHW layout mandatory.** ORT CUDA EP has open bugs with NHWC + dynamic shapes (SafeInt overflow in Conv — ORT issues #27912, #12288). Parser rejects `nhwc` manifests at load time.
- **Engine is a process singleton.** `ENGINE_EXISTS` AtomicBool; second `Engine::new()` returns `EngineAlreadyExists`. ORT's `OrtEnv` is process-global so this matches reality rather than pretending otherwise.
- **NMS lives in the ONNX graph, never in libsparrow_engine.** Model validation at load time rejects non-conforming models.
- **libsparrow_engine owns all preprocessing and postprocessing.** Eliminates divergence between Sparrow Local and Sparrow Web. Manifests make this declarative.
- **GPU is the default.** `Device::Auto` probes CUDA at engine creation and falls back to CPU. `SPARROW_ENGINE_DEVICE=auto` is the default. Conservation workloads target throughput.
- **FFI ABI evolution via `_v2` function versioning.** No reserved fields in FFI structs. Safety via `Weak` + `AtomicBool` + `Arc` pattern for model handles.

## Key gotchas (see `06_gotchas_and_constraints.md` for depth)

- **cuDNN 9.8 Conv bug on sm_89.** PyTorch/TF wheels bundle cuDNN 9.8, which has a Conv-engine bug with asymmetric padding on RTX 6000 Ada. SpeciesNet hits it; YOLO-style detectors don't. Fix: install `nvidia-cudnn-cu12>=9.10` standalone.
- **ORT CUDA EP + NHWC + dynamic shapes = crash.** Drove the NCHW mandate. SpeciesNet was originally NHWC-exported; re-exported with `tf2onnx --inputs-as-nchw`, max diff 0.0 vs original.
- **SIGPIPE handling.** Rust disables SIGPIPE by default; CLI now resets to `SIG_DFL` so `spe ... | head` exits 141 cleanly rather than printing a broken-pipe error.
- **`active_device()` is compile-time, not runtime.** Uses `ort::ep::CUDA::is_available()` which checks whether the CUDA EP was compiled in, not whether a GPU driver is present. Reliable GPU check = watch `nvidia-smi` during a real workload.
- **`fork()` safety.** `ENGINE_EXISTS` AtomicBool leaks to forked children. Python `multiprocessing` must use `spawn`, not `fork`. Documented limitation.

## Known open issues

- **MT-17** — intermittent heap corruption on `spe pipeline --device cuda` at process exit. Mitigated in W1 S1 (2026-04-22): explicit session drop under the models write-lock + `std::mem::forget(Arc<EngineInner>)` (pykeio/ort #280 pattern). Pre-mitigation 10–33% → ~5% residual over 60 post-mitigation runs. Upstream fix not planned (pykeio/ort #564 closed not_planned). Correctness unaffected. Full RCA: `docs/bugs.md`.
- **Engine singleton test order-dependency.** Resolved with `serial_test` crate tagging all 18 engine tests `#[serial]`; 173/173 libsparrow_engine tests now pass deterministically at `dev@09ee0aa` (175 at Phase 3.5 W1 S7 close; -2 dropped during later consolidation). See `commit 7fed112`.

## What this report delivers

- Full design rationale with evidence citations (file paths, line numbers, commit SHAs, issue numbers).
- Benchmark methodology and results.
- Gotchas with root causes and workarounds.
- Audit-fix methodology that caught two high-impact bugs the first-round reviewers missed.
- Phase 3.5 complete (2026-04-28; manual-test sign-off); planned Phase 4 + Phase 3.7 Track B scope, with MT-3.5-12 (bimodal libsparrow_engine latency) carried to Track B as the open reliability item. MT-17 was mitigated in Phase 3.5 W1 S1 (drop-order; ~5% residual; correctness unaffected).

**Confidence**: HIGH
- Factual accuracy: HIGH — numbers and claims verified against `docs/master_plan.md`, `CLAUDE.md`, `docs/benchmarks.md`, `project_phase3_status.md`, and git log at `dev@09ee0aa`
- Completeness: HIGH — covers what a reader needs for a 2-page orientation; defers depth to dedicated sections
- Freshness: HIGH — 2026-04-29 (R9 Phase 3.7 Track A reframing applied to Phase 3.5/4/5 status rows + 5-component architecture; R11 ch 00 L88 past-tense Phase 3.5 + MT-3.5-12 Track B framing; R12 ch 00 L80 test-count drift 175→173 with provenance softening)

## References

- `docs/master_plan.md` § Phase list
- `docs/benchmarks.md`
- `docs/review/phase3-final/round_05/inquisitor_review.md` (final Phase 3 audit-fix verdict)
- `docs/review/phase3-testing/manual_testing_logs.md` (17 MT issues + 5 decisions)
- `CLAUDE.md` § Benchmark Results
