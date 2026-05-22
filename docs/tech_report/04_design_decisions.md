# Design Decisions

Every load-bearing decision in sparrow-engine, with **why**, **what**, and **how to apply / enforce**. Grouped by the phase in which the decision was locked in.

## v3 — locked in during multi-container Docker design

### D-v3-1: ONNX for all models (vision and audio)

**Why.** One runtime, one model format, one set of gotchas to master. PyTorch at inference time pulled in ~1 GB of deps and tied the library to a specific PyTorch major version. TFLite was considered for audio — rejected because MD_AudioBirds_V1 exports cleanly to ONNX and a second runtime doubles maintenance cost.

**What.** All onboarded models ship as `.onnx` files. libsparrow_engine calls ORT and only ORT for inference. Adding a new model is a manifest + ONNX file drop, not a code change.

**How to apply.** Manifest parser (`sparrow-engine/libsparrow_engine/src/manifest.rs`) only recognizes `format = "onnx"`. Any non-ONNX value errors at load time. New models must export to ONNX before onboarding.

---

### D-v3-2: NCHW layout mandatory

**Why.** ORT CUDA EP has open bugs with NHWC + dynamic shapes — SafeInt overflow in Conv. ORT issues #27912 and #12288. SpeciesNet was originally NHWC and crashed the CUDA EP on first inference during Phase 3 manual testing (MT-10, logged in `docs/review/phase3-testing/manual_testing_logs.md`).

**What.** All ONNX models onboarded into sparrow-engine must use NCHW input. Models exported as NHWC (e.g., TensorFlow origin) must be converted before onboarding.

**How to apply.** `tf2onnx --inputs-as-nchw` plus `onnx-simplifier` on the source model. For SpeciesNet, this produced identical output (max diff = 0.0). Manifest parser rejects `layout = "nhwc"` at parse time with a message referencing this rule and the ORT issue numbers. Parser-level rejection catches the mistake before runtime SafeInt overflow. The `Layout::Nhwc` enum variant is retained (parser rejects, preprocess code still handles it) so removing it would be a breaking API change that buys nothing beyond the parser check.

---

### D-v3-3: Inference never reads from the database (write-only, async)

**Why.** DB is a write sink for audit / annotation, not a hot path. Reading from DB during inference would add unpredictable latency and couple the inference pipeline to DB availability.

**What.** When `sparrow-engine-db` arrives in Phase 4, inference writes results via `?store=true` asynchronously. No inference endpoint reads from the DB.

**How to apply.** Phase 4 scope. Enforced at sparrow-engine-server handler level — inference handlers emit DB writes on a background task, return the inference result synchronously from ORT.

**Phase 3.7 Track A update (2026-04-29).** Phase 4 retains the sparrow-engine-side primitives (`?store=true`, write-only async path); the database itself (the former in-sparrow-engine `sparrow-engine-db` sidecar) moves to the `sparrow-data` sibling repo (deferred construction) per Phase 3.7 Track A. The write-only-async invariant is unchanged. See `docs/master_plan.md` § Phase 4 + `docs/design/architecture.md`.

---

### D-v3-4: `halt_on_store_failure` safety flag

**Why.** If the DB is down or throws on writes, users should be able to choose: keep inferring and drop writes, or halt until the DB is back. Silent data loss is unacceptable in conservation workflows where every detection record matters.

**What.** A per-deployment flag that, when true, causes the inference endpoint to return 503 on DB write failure instead of returning the inference result with a log warning.

**How to apply.** Phase 4 scope. Implemented in sparrow-engine-server configuration. Defaults to `false` (returning inference results even on DB failure) because most deployments want detection continuity.

---

### D-v3-5: Idempotent writes via `UNIQUE(media_hash, model_id)`

**Why.** Re-running inference on the same image through the same model should not create duplicate DB rows. Retries, reprocessing, and multi-worker fleets all make duplicates likely without this constraint.

**What.** `sparrow-engine-db` schema enforces `UNIQUE(media_hash, model_id)`. Re-inference produces an `UPSERT` on conflict, not a duplicate row.

**How to apply.** Phase 4 scope. SQLite schema constraint; SQLx or `rusqlite` query uses `ON CONFLICT(media_hash, model_id) DO UPDATE`.

**Phase 3.7 Track A update (2026-04-29).** The idempotency contract (`UNIQUE(media_hash, model_id)`, `UPSERT` on conflict) is unchanged; the schema itself lives in the `sparrow-data` sibling repo (deferred construction) per Phase 3.7 Track A — not in an in-sparrow-engine `sparrow-engine-db` sidecar. See `docs/master_plan.md` § Phase 4 + `docs/design/architecture.md`.

---

### D-v3-6: Normalized bbox `[0, 1]` at every public API boundary

**Why.** Relative coordinates survive resizing, cropping, and thumbnailing without math on the consumer side. Sparrow Studio Local, Sparrow Studio Web, CLI, and Python all render at their own resolution; sparrow-engine returning absolute pixels would force every consumer to re-normalize.

**What.** All `BBox` fields in public types (`sparrow-engine/libsparrow_engine/src/types.rs`) are `f32` in `[0.0, 1.0]`. Detection responses over FFI, HTTP, CLI, and Python all normalize.

**How to apply.** libsparrow_engine's `postprocess` divides by image width/height before populating the `BBox`. Consumer code (visualization, export, rendering) multiplies by display dimensions at the last step.

---

### D-v3-7: Separate `pipeline_id` field (not reused `model_id`)

**Why.** Pipelines chain multiple models (detect → crop → classify). Storing a pipeline result under `model_id = "mdv6+speciesnet"` or similar collapses the per-model detail that downstream analytics want. Keeping them separate keeps each table clean.

**What.** `PipelineResult` carries both the detector's `model_id`, the classifier's `model_id`, and a separate `pipeline_id`. DB schema (Phase 4) keys by `pipeline_id` in the pipeline-results table; detections and classifications still key by their individual `model_id`.

**How to apply.** Types in `sparrow-engine/libsparrow_engine/src/types.rs` carry the separate field. CLI `--export-format megadet` populates `model_id` with the detector's ID (MD v1.5 convention). CLI `--export-format coco` carries both.

---

### D-v3-8: TOML manifests, not YAML

**Why.** `serde_yaml` is deprecated. TOML is first-class in Rust (`toml` crate), well-maintained, and the format Cargo itself uses. YAML has historically been a source of indent and quoting bugs that TOML avoids by design.

**What.** Model manifests are `manifest.toml` under `{model_dir}/{model_id}/`. Pipeline manifests same. All deserialization via `serde` with `toml`.

**How to apply.** `sparrow-engine/libsparrow_engine/src/manifest.rs`. No YAML reader anywhere in the codebase. `Cargo.toml` deps: `serde = "1"`, `toml = "0.8"`.

---

### D-v3-9: No fine-tuning in the API

**Why.** Fine-tuning has a very different cost profile (hours to days, GPU-bound, dataset-heavy) than inference (milliseconds, single model + single image). Mixing them in one library means maintaining two very different code paths and two deployment shapes.

**What.** Fine-tuning is a separate workstream, colleague-owned. sparrow-engine is inference-only. See § 13.

**How to apply.** No training loop, no optimizer, no dataset abstractions in libsparrow_engine. If a user wants to fine-tune, they export their model to ONNX and load it into sparrow-engine.

---

### D-v3-10: Container lifecycle — `init: true` + Docker restart + internal health watchdog

**Why.** supervisord inside a container is an anti-pattern — it shadows Docker's own restart logic and obscures container-level health signals. Rust's `tokio` runtime plus `init: true` from Docker handles PID 1 signal forwarding. An internal health watchdog inside sparrow-engine-server checks ORT session health without needing an external supervisor.

**What.** Phase 2+ Docker deployments use `init: true` + `restart: unless-stopped` + a `/v1/health` endpoint. No supervisord.

**How to apply.** `docker-compose.yml` snippets; `Dockerfile` uses `CMD sparrow-engine-server` directly. Health endpoint in `sparrow-engine-server/src/handlers/health.rs`.

---

### D-v3-11: Pure-Python HTTP client SDK (sparrow-engine-client)

**Why.** Python Triton gRPC client required protocol buffer compilation and a heavy dependency footprint. A pure-Python `httpx`-based SDK is ~385 LOC, 20 tests, pip-installable, and needs no codegen step.

**What.** `sparrow-engine-client` ships as a standalone pip package. Sync and async APIs. `ModelInfo`, `Detection`, `BBox`, `Classification`, `AudioSegment` dataclasses match the HTTP schema 1:1.

**How to apply.** `sparrow-engine/sparrow-engine-client/`. Dependency: `httpx>=0.27`. Sparrow Studio Web workers consume it.

## v4 — locked in during libsparrow_engine + Sparrow Studio integration design

### D-v4-1: Engine is a process singleton

**Why.** ORT `OrtEnv` is process-global. Creating two `Engine` instances would trample each other's environment setup. Pretending otherwise invites hard-to-diagnose failures in forked / threaded deployments.

**What.** `ENGINE_EXISTS: AtomicBool` at `sparrow-engine/libsparrow_engine/src/engine.rs:112`. `Engine::new()` does an atomic swap:

```rust
pub fn new(config: EngineConfig) -> Result<Self> {
    if ENGINE_EXISTS.swap(true, Ordering::SeqCst) {
        return Err(SparrowEngineError::EngineAlreadyExists);
    }
    // ... build session, resolve device ...
}
```

`Drop for Engine` resets the flag to `false` (`engine.rs:689`).

**How to apply.** Tests that exercise engine lifecycle tag with `#[serial]` from the `serial_test` crate (`commit 7fed112`) so they do not fight the singleton. Python `multiprocessing` users must use `spawn`, not `fork`, because `fork()` duplicates the atomic state in the child. Documented as a limitation.

---

### D-v4-2: Opaque model handles with Weak + AtomicBool + Arc safety

**Why.** FFI consumers hold model handles across API boundaries. If the engine drops or the model is explicitly unloaded while the consumer still holds a handle, a naive implementation would be use-after-free. Opaque pointers plus weak references plus an active flag close this.

**What.** `ModelHandle` holds `Weak<EngineInner>`, `Arc<AtomicBool>` (active), `Arc<Mutex<Session>>`, `Arc<ModelManifest>`, `Arc<Vec<String>>` (labels). See `sparrow-engine/libsparrow_engine/src/engine.rs:180-200`.

**How to apply.** Every inference call first upgrades `Weak` and checks `active`. Either fails → returns `SparrowEngineError::EngineDropped` or `SparrowEngineError::ModelUnloaded`. No raw pointer dereference past an invalidated handle.

---

### D-v4-3: libsparrow_engine owns all preprocessing and postprocessing

**Why.** The Python stack had pre/post in the library. Sparrow Studio Local reimplemented parts in C#. Sparrow Studio Web had its own glue in workers. Three partial reimplementations of the same code meant every correctness fix landed three times.

**What.** Pre/post are declarative in the manifest (`PreprocessMethod`, `Normalization`, `Layout`, `PostprocessMethod`, `InferenceStrategy`, pad_value, input_size). libsparrow_engine reads the manifest and drives the pipeline. Consumers send raw image bytes, get structured results back.

**How to apply.** `sparrow-engine/libsparrow_engine/src/preprocess.rs`, `preprocess_audio.rs`, `postprocess.rs`. Manifest TOML snippet:

```toml
[model]
id = "mdv6"
format = "onnx"
preprocess_method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"
pad_value = 114.0
inference_strategy = "single"
postprocess_method = "confidence"
```

Any consumer doing its own preprocessing is working outside the design. Sparrow Studio Local has been migrated to call libsparrow_engine for pre/post.

---

### D-v4-4: Four workspace crates, `ffi` feature mandatory on libsparrow_engine

**Why.** v4 proposed 2 core crates (libsparrow_engine + sparrow-engine-server) with `ffi` as a mandatory feature — not a separate `libsparrow_engine-ffi` crate. Two crates are cleaner than three when they always ship together. Phase 2.5 added two consumer crates (sparrow-engine-cli, sparrow-engine-python) because those really are independent concerns.

**What.** Workspace has 4 crates + 1 Python package. `libsparrow_engine` exposes FFI via `--features ffi`.

**How to apply.** `sparrow-engine/libsparrow_engine/Cargo.toml` `[features]` has `ffi = ["dep:cbindgen", "dep:csbindgen"]`. `build.rs` checks `cfg(feature = "ffi")` and emits cdylib + staticlib artifacts.

**Phase 3.7 Track A update (2026-04-29).** A 5th workspace crate `sparrow-utils` is planned for Phase 4+ to host stateless utilities currently inside libsparrow_engine (viz, stats, export, daynight, hash; ~2,725 LOC). Workspace crate (NOT a sibling) — sparrow-engine-cli + sparrow-engine-python depend on it directly; published to crates.io for consumption by the future `sparrow-data` + `sparrow-ops` sibling repos. Decision rule: `docs/design/architecture.md` "Workspace crate vs sibling repo". Rationale: `docs/design/phase3.7/codebase_separation_survey.md` + `docs/design/phase3.7/mlops_planning.md` §6.1.

---

### D-v4-5: No reserved fields in FFI structs — function versioning for ABI evolution

**Why.** Reserved fields are a C tradition for ABI evolution. They also invite bugs: consumers reading uninitialized reserved bytes, inconsistent zeroing, accidental dependence on the reserved layout. Function versioning is simpler: `sparrow_engine_detect_v2` when the signature needs to change.

**What.** FFI structs carry only meaningful fields. Evolution is via new functions (`sparrow_engine_detect`, `sparrow_engine_detect_v2`, ...).

**How to apply.** `cbindgen` configuration enforces it via review of generated `sparrow_engine.h`. Any "reserved_N" field in a `#[repr(C)]` struct is a bug.

---

### D-v4-6: `cbindgen` + `csbindgen` for bindings

**Why.** Hand-written C / C# headers drift from Rust source. `cbindgen` generates `sparrow_engine.h` from Rust at build time. `csbindgen` generates C# DllImport declarations for Sparrow Studio Local's NuGet package. Drift becomes impossible.

**What.** Both crates are `optional = true` under the `ffi` feature. `build.rs` invokes them when `ffi` is on.

**How to apply.** `sparrow-engine/libsparrow_engine/build.rs` calls `cbindgen::Builder` and `csbindgen::Builder`. Outputs land where the packaging scripts expect them.

---

### D-v4-7: NMS lives in the ONNX graph, never in libsparrow_engine

**Why.** NMS in libsparrow_engine duplicates what the model already exports. In the Triton benchmark, a redundant NMS pass after the model's in-graph NMS deleted legitimate boxes that survived the first pass — visible as a ~4% detection gap when we compared sparrow-engine to Triton. Putting NMS in the graph makes the exported model self-contained and removes the post-inference pass.

**What.** All exported ONNX models must include NMS in the graph before onboarding. libsparrow_engine does not have an NMS implementation in postprocess.

**How to apply.** Model validation at load time rejects non-conforming models (output shape must match the "has-NMS" signature for detector models). Test fixtures verify NMS is in-graph. For new model onboarding, `tools/check_nms_in_graph.py` is run before the manifest is written.

---

### D-v4-8: Audio uses ONNX, same engine as vision

**Why.** A second runtime (TFLite, etc.) would double maintenance cost. MD_AudioBirds_V1 exports cleanly to ONNX. Audio is not special.

**What.** `MD_AudioBirds_V1.onnx` is a first-party export. Audio preprocessing (mel spectrogram: `n_fft=1024`, `hop=512`, `n_mels=224`, `sr=48000`) is in libsparrow_engine. Sliding-window inference: 1.0 s windows with 0.3 s stride. Sigmoid postprocessing. Phase 5 ("Audio") was absorbed into Phase 1.

**How to apply.** `sparrow-engine/libsparrow_engine/src/preprocess_audio.rs`, `sparrow-engine/libsparrow_engine/src/detect_audio.rs`. FFI exports: `sparrow_engine_detect_audio` (file path + options), `sparrow_engine_detect_audio_streaming` (per-segment callback), `sparrow_engine_audio_result_free`.

**Audio dependencies.** `hound` (WAV decode — zero deps, Apache-2.0), `realfft` (real-valued FFT for STFT — MIT), `rubato` (resampling all ratios — MIT).

---

### D-v4-9: Golden reference outputs generated by libsparrow_engine itself

**Why.** Using PytorchWildlife / PIL to generate reference outputs means cross-library differences (PIL letterbox rounding vs Rust `image` crate) get baked into the reference. Sparrow Engine's correctness bar would drift with PIL upgrades. Generating from libsparrow_engine means the reference is self-consistent.

**What.** Reference outputs in `test_files/` are produced by libsparrow_engine in a known-good configuration (CPU, fixed ORT version, fixed model version). Float precision tolerance: `bbox ±0.005`, `confidence ±0.12` (vision) / `±0.25` (audio) — the f32/f64 precision gap.

**How to apply.** `scripts/generate_golden_outputs.sh`. Regenerate after any change to preprocessing, postprocessing, or model version.

---

### D-v4-10: Batch detection with progress callback + auto-fallback

**Why.** Large campaigns have hundreds of thousands of images. Users want progress feedback. Some models are batch=1 only (hardware or export constraints); the API should hide that and fall back to a loop transparently.

**What.** `sparrow_engine_detect_batch` FFI export. Takes multiple images, emits a progress callback, internally loops if the model's ONNX input is batch=1.

**How to apply.** `sparrow-engine/libsparrow_engine/src/ffi.rs`. Progress callback signature documented in `sparrow_engine.h`.

---

### D-v4-11: OWL-T single-output heatmap support with tile-overlap dedup

**Why.** OWL-T is a single-output heatmap detector on high-res overhead imagery. Tiling is required. Naïve tile concatenation double-counts detections in overlap regions.

**What.** `detect_tiled` handles 1-output models. Single-class, adaptive threshold. Tile overlap deduplication via greedy center-proximity suppression.

**How to apply.** `sparrow-engine/libsparrow_engine/src/detect.rs` `detect_tiled` path.

## Phase 2 — Docker API

### D-p2-1: Initial HTTP endpoint surface, axum thin shell

**Why.** sparrow-engine-server exists to expose libsparrow_engine over HTTP. Logic lives in libsparrow_engine. Keeping the shell thin means sparrow-engine-server is easy to audit and update.

**What.** Phase 2 `sparrow-engine-server` was ~1,426 Rust LOC with 5 inference routes, 6 management routes, 2 health routes, and 23 tests. The current post-Phase-4.2 surface is tracked in [Appendix: HTTP Endpoint Inventory](appendices/http_endpoints.md).

**How to apply.** Handler code in `sparrow-engine/sparrow-engine-server/src/handlers/`. Any handler growing past ~200 LOC is a flag that logic is leaking out of libsparrow_engine.

---

### D-p2-2: Docker GPU image uses `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` base

**Why.** ORT CUDA EP silently fails without cuDNN. The plain `nvidia/cuda:runtime` base does not include cuDNN. Using the cudnn-runtime variant catches this at build time rather than at first inference call.

**What.** GPU `Dockerfile` `FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`. CPU `Dockerfile` `FROM debian:bookworm-slim`.

**How to apply.** `sparrow-engine/docker/Dockerfile.gpu` and `sparrow-engine/docker/Dockerfile.cpu`. CI build checks base-image SHA to catch drift.

---

### D-p2-3: GPU is the default for inference, testing, and benchmarks

**Why.** Conservation workloads target throughput. CPU is a fallback for edge or CI. Measuring CPU as the default would under-represent real deployment performance.

**What.** `Device::Auto` probes CUDA first, falls back to CPU. `SPARROW_ENGINE_DEVICE=auto` is the default. Benchmarks run on GPU unless explicitly CPU-only.

**How to apply.** `sparrow-engine/libsparrow_engine/src/engine.rs` `resolve_device`. CI benchmark job runs on a GPU runner.

## Phase 2.5 — CLI and Python package

### D-p2.5-1: Functionality consistency between CLI and Python

**Why.** If the CLI has a feature the Python package doesn't, users switching between them hit surprise. Maintaining two different surfaces is also a maintenance cost.

**What.** The CLI and Python package expose the same 8 MVP functions (4 inference, 3 informational, 1 configuration). Phase 3 added 6 standalone utility functions — also present in both.

**How to apply.** `docs/review/phase2.5-consumer-audit/` covers the enforcement. Functionality gaps in either surface are treated as audit findings.

---

### D-p2.5-2: Auto-load models — no explicit `load_model()`

**Why.** A separate load step is a foot-gun: users forget it, get confusing errors, and the library's job is well-understood (load on first use). Engine caches sessions so there is no performance penalty.

**What.** `Engine.get_or_load_model(model_id)` lazy-loads on first inference. Subsequent calls hit the cache. No explicit `load_model` in the MVP API (exposed but optional).

**How to apply.** `sparrow-engine/libsparrow_engine/src/engine.rs` `get_or_load_model`.

---

### D-p2.5-3: Ad-hoc pipeline — no pre-defined TOML required

**Why.** Requiring users to define a pipeline TOML before running `pipeline(img, detector="mdv6", classifier="speciesnet")` is friction. Most users want the common "detect then classify" case without config ceremony.

**What.** `pipeline(img, detector="mdv6", classifier="speciesnet")` composes ad-hoc from loaded models.

**How to apply.** `sparrow-engine/libsparrow_engine/src/engine.rs` `run_pipeline_adhoc`. Pre-defined pipelines are still supported for users who want them.

---

### D-p2.5-4: Return type for all inference functions is `list[Result]`

**Why.** Single-image vs batch-of-N branching in return type makes consumer code ugly (`if isinstance(result, list)`). Always returning a list, even for single-input calls (which return a 1-element list), keeps consumer code uniform.

**What.** `detect`, `classify`, `detect_audio`, `pipeline` always return `list[DetectResult | ClassifyResult | ...]`. Single image in → 1-element list out.

**How to apply.** Type signatures enforce it.

---

### D-p2.5-5: fork() safety — documented limitation

**Why.** `ENGINE_EXISTS` AtomicBool leaks into forked children. The child inherits "engine exists" state but no actual Engine object. Any `Engine::new()` in the child fails with `EngineAlreadyExists` even though there is no engine to share. Inherent to Rust singletons + POSIX fork.

**What.** Python `multiprocessing` users must use `spawn`, not `fork`. Documented in sparrow-engine-python README and in the Python package docs.

**How to apply.** Documentation. No code change — the bug is in POSIX fork semantics, not in sparrow-engine.

## Phase 3 — Utilities + local model catalog

### D-p3-1: Six new libsparrow_engine modules, all with free functions

**Why.** Tying utilities to `Engine` methods forces users to create an Engine just to hash a file or check day/night. These operations are pure functions of their inputs. Keeping them standalone means CLI `spe hash` and `spe day-night` are ORT-free and fast.

**What.** `hash.rs`, `daynight.rs`, `stats.rs`, `export.rs`, `viz.rs`, `catalog.rs` contain free functions. Engine gets thin convenience wrappers that call the free functions.

**How to apply.** `sparrow-engine/libsparrow_engine/src/hash.rs` etc. `CLI models verify` and Python `verify_model` bypass `_get_engine()`, route to `catalog::verify_model` directly.

---

### D-p3-2: Zero new dependencies for viz (manual pixel ops)

**Why.** `imageproc` is a heavy dep and its API surface is bigger than what sparrow-engine needs (bounding boxes, filled circles, alpha compositing). Manual pixel operations with the existing `image` crate keep the dep surface small and the code inspectable.

**What.** `viz.rs` does bboxes + filled circles + alpha compositing with manual pixel loops over `image::RgbImage`. Text labels (originally Phase 3.5 S4 behind a compile-time `viz-text` Cargo feature) were lifted in Phase 3.7 (2026-04-28) to a runtime `RenderOpts.show_labels: bool` toggle. `ab_glyph` is now an unconditional libsparrow_engine dep; CLI exposes `--show-labels` (default off); Python `visualize()` accepts the matching `show_labels` kwarg.

**How to apply.** `sparrow-engine/libsparrow_engine/src/viz.rs`. Phase 3 MVP shipped with zero new viz dependencies; Phase 3.5 W3 added `ab_glyph` (optional) for text-label glyph rasterisation; Phase 3.7 lift made `ab_glyph` unconditional. The bundled DejaVu Sans font lives at `libsparrow_engine/assets/fonts/`.

---

### D-p3-3: Unified viz pipeline — one render path for all result types

**Why.** Separate render functions for detection, classification, pipeline, overhead dots, and audio segments would diverge in rendering conventions (line width, color palette, alpha levels). Normalizing to a common intermediate (`BboxAnnotation`) lets one render path handle all of them.

**What.** All result types normalize to `BboxAnnotation` before rendering. `render()` takes the annotation list plus an `RgbImage` and draws.

**How to apply.** `sparrow-engine/libsparrow_engine/src/viz.rs` `render()`. Audio confidence heatmap is a separate function (`render_audio_heatmap`, ~100 LOC) because heatmaps render at pixel granularity; annotation list doesn't fit.

---

### D-p3-4: MegaDet v1.5 JSON + COCO JSON + CSV export

**Why.** Three formats cover the common downstream tools: MegaDet v1.5 for Timelapse and Wildlife Insights, COCO for annotation tool interop, CSV for spreadsheet and ad-hoc analysis. No `ExportRecord` intermediate — direct export from result types.

**What.** `export.rs` has `to_megadet`, `to_coco`, `to_csv`. Shared bbox conversion utilities.

**How to apply.** CLI `--export-format megadet|coco|csv` on `detect` and `pipeline`. Python `sparrow-engine.export(results, format="coco")`. COCO exporter warns on `label_id` namespace collisions in pipeline exports (fix applied in Phase 3 final audit-fix R4 MI-2).

---

### D-p3-5: In-manifest checksums for model verification

**Why.** Checksum files alongside the model (e.g., `model.onnx.sha256`) are easy to drift. Keeping checksums inside the manifest makes manifest + model a self-contained unit — you either have the manifest with correct checksums or you don't.

**What.** `[model]` section gains `onnx_sha256` and `onnx_size_bytes` fields. `#[serde(default)]` for backward compatibility.

**How to apply.** `sparrow-engine/libsparrow_engine/src/manifest.rs`. `spe models verify [--write]` populates checksums. `catalog::verify_model` does tiered verification (size check first, then SHA-256).

---

### D-p3-6: `models verify` is ORT-free

**Why.** Verification should not require booting ORT and loading a session just to check a file hash. Users with broken ORT environments still need to verify model integrity.

**What.** `spe models verify` skips engine creation. Uses `catalog::verify_model` directly. CLI handler bypasses the usual `_get_engine()`.

**How to apply.** `sparrow-engine/sparrow-engine-cli/src/main.rs` `cmd_models_verify`. Python `sparrow-engine.verify_model` same pattern.

---

### D-p3-7: BT.709 luma + threshold 85 for day/night classification

**Why.** Ports Sparrow Local `DetermineTimeOfDay()` exactly. The existing Sparrow convention is the ground truth for this use case.

**What.** `daynight::image_brightness_rgb` computes BT.709 luma (`0.2126*R + 0.7152*G + 0.0722*B`), returns mean over all pixels. `daynight::day_night` threshold > 85 on [0, 255] scale → "day", else → "night".

**How to apply.** `sparrow-engine/libsparrow_engine/src/daynight.rs`.

## Meta-decision: fresh team per round in audit-fix

**Why.** An agent that just audited a file and approved its structure is not an independent reviewer of the resulting change. Fresh agents per round read the prior round's reports but form their own view. The Phase 3 final audit-fix R2 independent scan caught a HIGH finding (sparrow-engine-client SRV1 wire-compat break) that the R1 agents missed. That pattern repeated in R3 (MI-1, MI-2).

**What.** Every round in an iterative skill (audit-fix, doc-fix, code-review, research) spawns fresh agents. Inter-round continuity lives in the reports, not in agent state.

**How to apply.** Skill lifecycle. `SendMessage(type: "shutdown_request")` at end of round → `TeamDelete` → `TeamCreate` for next round → `Agent` spawn with empty context.

## Freshness anchor (2026-05-13)

**Decisions locked in after this chapter was last refreshed (2026-04-29)**: 5-component architecture + sibling-repo decomposition (Phase 3.7 Track A); 7-crate workspace + dual cdylibs as `libsparrow_engine.so` + flavor-strict `Device::Auto` (Phase 3.8 + post-MT-4.1-2); `[provenance]` + `[drift_reference]` manifest fields + `InferenceLogRecord` `SCHEMA_VERSION="1.0"` + `?store=true` + `halt_on_store_failure` semantics + `InferenceLogSink` trait (Phase 4); lazy server boot + `SPARROW_ENGINE_PRELOAD` + `/v1/catalog` + runtime pipeline alias mgmt (Phase 4.2); `fp16_parity` per-model tolerance preset (Phase 4.3); `sparrow-engine-server` boot-time argv hardening (Phase 4.4). For the canonical list, read `CLAUDE.md § Locked-In Design Decisions` and `docs/master_plan.md`.

## Confidence

**Confidence**: HIGH
- Factual accuracy: HIGH — decisions cross-referenced against `docs/design/v3/final_decisions.md`, `docs/design/v4/libsparrow_engine/consensus_design_revised.md`, `docs/design/phase3/final_design.md`, `CLAUDE.md § Locked-In Design Decisions`
- Completeness: HIGH — every decision flagged "locked in" in CLAUDE.md is covered
- Freshness: HIGH — 2026-04-29 (R11 appended Phase 3.7 Track A update appendices to D-v3-3 + D-v3-5 mirroring D-v4-4 R3 pattern; matches HEAD)

## References

- `docs/design/v3/final_decisions.md` — v3 locked-in decisions
- `docs/design/v3/round_05/definitive_design_v3_final.md` + `round_06/design_fixes.md` — v3 API design
- `docs/design/v4/libsparrow_engine/consensus_design_revised.md` — v4 consensus design
- `docs/design/phase3/final_design.md` — Phase 3 design spec
- `CLAUDE.md` § Locked-In Design Decisions
- `docs/research/v2/round_05/research_v2_final_synthesis.md` § 1.3 — design principles
- ORT issues: #27912 (Conv SafeInt overflow with NHWC), #12288 (CUDA EP NHWC dynamic shapes)
