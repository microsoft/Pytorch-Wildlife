# Changelog

All notable changes to sparrow-engine (libsparrow_engine, sparrow-engine-cli, sparrow-engine-server, sparrow-engine-python)
are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v0.1.17

<one-line summary — filled by release agent at SKILL.md Stage 10d>


## v0.1.16

<one-line summary — filled by release agent at SKILL.md Stage 10d>


### Added

- **RP-5 — Cross-repo CI image-pin auto-PR**, landed v0.1.15 (2026-05-29).
  Replaces the operator-manual `sync_sparrow_engine.sh` flow with CI:
  - `release.yml` gains `build-and-push-docker-{cpu,gpu}` jobs that push
    CPU + GPU images to `ghcr.io/microsoft/sparrow-engine-server[-gpu]:vX.Y.Z`
    on prod tag-push (skip on hyphenated tags). GitHub-hosted `ubuntu-latest`
    runner builds both flavors (RP-19 precedent: cudarc fallback-dynamic-loading
    + nvjpeg-sys pre-generated bindings + ort load-dynamic skip the build-time
    CUDA Toolkit requirement). Separate `sparrow-engine-server-buildcache:{cpu,gpu}`
    registry caches keep the runtime image tag list clean. Auth via `GITHUB_TOKEN`
    + `packages: write` — zero secrets to manage.
  - `publish-cli-release-assets` now gates GH Release publish on docker push
    success — sparrow's auto-PR (lives on `Clamps251/sparrow @ sparrow-engine-dev`)
    polls `/releases/latest` and is therefore race-free w.r.t. image availability.
  - Design + impl artifacts live in the companion repo at
    `zhmiao/sparrow-engine-dev:docs/{design,implement}/rp-5-image-pin-auto-pr/`.

### Fixed

- **Phase 4.5 audit-fix Phase F (CI + Docker + release plumbing) — Round 1**
  2026-05-28 (HEAD `3052a70`, `6c6bbaf`); Round 2 hardening 2026-05-28.
  - **B-03**: new `check-version-consistency` preflight job in `.github/workflows/release.yml`
    enforces `git tag ↔ sparrow-engine-cli/Cargo.toml [package].version ↔
    sparrow-engine-python/pyproject.toml [project].version` agreement on tag-push.
    Round 2 (F-R2-4) extended enforcement to `workflow_dispatch` for the cli ↔ py pair
    so manual release rehearsals catch drift before tag-push time. All 8 release build
    jobs gain `needs: check-version-consistency`. `scripts/package_cli_tarball.sh`
    defaults VERSION from `cargo metadata` on `sparrow-engine-cli/Cargo.toml` when the
    caller doesn't set it, anchoring the tarball name on the same SSOT.
  - **B-04**: PyPI wheels (`sparrow-engine`, `sparrow-engine-gpu`) are decided Python-API
    only — the CLI binaries (`spe`, `spe-gpu`) and `sparrow-engine-server` do NOT ship
    via `pip`. Decision rationale in `sparrow-engine-python/pyproject.toml` `[tool.maturin]`
    comment (3 alternatives investigated, rejected on wheel-size + per-platform-binary
    grounds). Round 2 (F-R2-2) extended `[project].description` with the warning so the
    routing is visible on the PyPI project page; install via brew, system installer
    (`sparrow-engine-install.{sh,ps1}`), or GitHub Release tarball.
  - **B-06**: CPU Docker image (`docker/Dockerfile.cpu`) — bumped `ARG ORT_VERSION`
    from `1.24.2` → `1.25.1` (aligns with `onnxruntime>=1.25.1,<1.26` pin in
    `pyproject.toml` and the `api-24` ORT API the `sparrow-engine-cpu` Rust binding
    requires post commit `5c86dbf`). Added
    `RUN ln -sf libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so && ldconfig`
    after the existing `RUN ldconfig` so `dlopen("libonnxruntime.so")` (bare unversioned
    name emitted by `ort/load-dynamic` when `ORT_DYLIB_PATH` is unset) resolves via
    `/usr/local/lib` filesystem search. Symlink anchored on the ldconfig-managed SONAME
    `libonnxruntime.so.1` — version-bump-resilient.
  - **B-07**: GPU Docker image (`docker/Dockerfile.gpu`) — same ORT bump `1.24.2` →
    `1.25.1` (root cause identical to B-06: `api-24` Rust binding ↔ ORT 1.24.2 runtime
    mismatch caused the GPU server to boot and list models, then silently spin on the
    first CUDA EP / Session creation call). Same defensive
    `ln -sf libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so && ldconfig` for parity
    with B-06. Round 2 verified end-to-end via CPU + GPU image rebuilds + `/healthz` +
    `/v1/health` smoke
    + GPU `POST /v1/detect` against MDv6 — see
    `docs/review/phase4.5-cleanup-audit-fix-f/round_02/docker_smoke_results.txt`.
  - **F-R2-6** (round 2): new `check-version-consistency` step asserts
    `ARG ORT_VERSION` matches across `Dockerfile.cpu` and `Dockerfile.gpu`. Cheap grep
    guard against future ORT-bump drift (root cause family for B-06/B-07).
  - **build.sh / package_cli_tarball.sh observability** (round 1 `6c6bbaf`):
    `build.sh` post-build prints `[project.scripts]` entry-point names extracted from
    the built wheel via `wheel unpack` (returns 0 on absence — pure diagnostic).
    `package_cli_tarball.sh` defaults `VERSION` from `cargo metadata` when unset
    (CI callers pass explicit VERSION; the default keeps local invocations consistent).

### Changed

- **Phase 4 (Sparrow Engine-side data primitives for sibling integration) substantively complete**
  2026-05-07; audit-fix R1-R4 CONVERGED at HEAD `9c632e1` per `audit-fix r4 (lead-direct)`
  apply (Trajectory A; binding verdict at
  `docs/review/phase4-audit-fix/round_04/inquisitor_phase2_review.md`). 4 workstreams
  W1-W4 landed: manifest `[provenance]` (3 optional `Option<String>` fields:
  `training_dataset_id`, `training_experiment_id`, `training_repo_commit`) +
  `[drift_reference]` (`BTreeMap<String, f32>`) round-trip (W1); `InferenceLogRecord` +
  `DriftMetrics` wire types + `SCHEMA_VERSION="1.0"` constant in `sparrow-engine-types` (W2 + W4);
  `?store=true` + `halt_on_store_failure` per-request query params on detect/classify/
  audio/pipeline + `InferenceLogSink` trait + `StderrJsonLinesSink` default sink +
  `compute_drift_metrics` (PSI `eps=1e-4`, nearest-rank percentile, `image_count.max(1)`
  denom) in `sparrow-engine-server` (W3). Verification gate at HEAD `9c632e1`: sparrow-engine-types 113/0
  + sparrow-engine-server lib 16/0 + sparrow-engine-server `store_flow` 6/0 + sparrow-engine-cpu 50/0 + sparrow-engine-core
  158/0 = **343 PASS, 0 FAIL**; both `--features cpu/gpu` builds clean; clippy clean;
  banned-phrase grep ZERO Phase-4-scoped matches. 11 LOW future-work observations
  P4-AF-1..11 docketed at `docs/ideas.md § "Phase 4 Audit-Fix LOW Future-Work
  Observations"` (post-R3 close commit `0ef26a3`); all deferrable, none block Phase 4
  sign-off. Phase 4.1 manual test plan READY at commit `f1428b9`; manual test execution
  NOT STARTED. See `docs/design/phase4/{schema.md, README.md}` + `docs/changelog.md`
  2026-05-07 entry + `docs/master_plan.md § Phase 4`.

- **Phase 3.8 Step 2 (audio GPU pipeline) substantively complete** 2026-05-05;
  audit-fix R1-R5 CONVERGED (`512c6d5`) with 21 named items applied;
  9/9 audio parity tests pass; clippy clean. Wave 5 JpegDecoder hoist +
  per-model FP16 audits remain held for Phase B per Step 1 wrap-up; the
  Step 2 audio audit-fix codebase-janitor docket holds 8 line-level
  SKIP items (3 code-level + 5 already closed via this doc-fix cycle).
  Architecture: GPU mel pipeline (cuFFT R2C → power → cuBLAS sgemm
  Slaney mel filterbank → power_to_db → col→row transpose) + ORT
  IoBinding bound to CUDA compute stream + `Mutex<AudioWorkspace>`
  high-water-mark device-buffer cache. Strategies: `Strategy::SingleCall`
  (production default for non-streaming detect) / `Strategy::HybridA{16}`
  (streaming default) / `Strategy::PerBatchB` (memory-constrained
  fallback). See `docs/design/phase3.8/step2/{final_design,implementation_plan}.md`.

- **MD_AudioBirds_V1 default precision flipped to FP16** (Phase 3.8 Step 2
  post-STRETCH re-audit, 2026-05-05). Production manifest
  `sparrow-engine/models/audiobirds.toml` set `[inference] precision = "fp16"`.
  Empirical (RTX 6000 Ada, DUNAS_20230925_090000 60 s real audio, 5
  fresh × 10 inner iters): sparrow-engine-gpu FP16 8.52 ms p50 (stddev 0.07 ms);
  FP32 14.71 ms p50 (stddev 0.28 ms); FP16 is 1.71× faster than FP32 +
  2.10× faster than PW (torchaudio + onnxruntime-gpu 1.25.1 reference at
  17.94 ms) + 315.5× faster than sparrow-engine-cpu (2688.2 ms). §2.2 R2 FP16
  numerical-accuracy gates (max-abs ≤ 1e-2, mean-abs ≤ 2e-3, rel ≤ 5%,
  label flips=0) all met; W1.7-anchored FP32 parity gates (mel ≤ 5e-3
  dB, logit ≤ 3.0e-3, conf ≤ 7.5e-4, label flips=0, range exact) all
  met. FP16 verdict FLIPPED (Wave 3 HOLD-on-FP32 superseded post Fix A's
  per-call ORT setup overhead collapse). See
  `docs/research/phase3.8/step2/fp16_audit.md`.

- **Phase 3.8 Step 1 (Waves 1-4) substantively complete** 2026-05-04;
  audit-fix R1-R5 CONVERGED (`0ff0483`) with 18 named code fixes;
  tests 40/0/13 pass, clippy clean. Wave 5 JpegDecoder hoist +
  per-model FP16 audits deferred per design.

- **4 of 5 image models default to FP16 quantization; DeepFaune held on FP32** (Phase 3.8 Step 1 final,
  2026-05-04). Production manifests for MegaDetector v6, HerdNet,
  OWL-T, and Amazon Camera Trap v2 set `[inference] precision = "fp16"`; DeepFaune holds on the default FP32 (Path A per `docs/ideas.md` P3.8-1; P3.8-7 audit closed 2026-05-06 ruled out nvjpeg as a recovery lever).
  **Both sparrow-engine-gpu and sparrow-engine-cpu engines load the FP16 ONNX file**; sparrow-engine-gpu
  uses ORT CUDA EP Tensor Cores for hardware FP16; sparrow-engine-cpu uses ORT CPU EP's
  software FP16 path. Detection counts on sparrow-engine-gpu may differ from FP32 by
  0-1 detection per 100 images at the 0.2 confidence threshold (DeepFaune:
  160 → 159 on one borderline image; MDv6/HerdNet/OWL-T/Amazon: drift 0).
  Cross-engine count drift between sparrow-engine-gpu FP16 and sparrow-engine-cpu FP16 can reach
  2 / 100 images on DeepFaune because the two ORT EPs round FP16 ops differently
  — Tensor Core hardware FP16 and software FP16 are not bit-equivalent. Per-model
  speedups (sparrow-engine-gpu FP16 vs FP32 medians, RTX 6000 Ada): MDv6 1.59× (21.37 →
  13.46 ms), HerdNet 1.24× (585.88 → 473.22 ms), DeepFaune 1.16× (4.16 →
  3.60 ms, measured in audit but DeepFaune holds on FP32 in production manifest), Amazon 1.04× (1.84 → 1.76 ms). **Users who need FP32 numerical
  fidelity can override per-model by setting `[inference] precision = "fp32"`
  in the manifest** (the FP32 ONNX is preserved alongside the FP16 ONNX in
  every model directory). Gate G2 parity test thresholds in
  `sparrow-engine-gpu/tests/integration_yolo.rs` re-spec'd to count drift ≤ 2, IoU min
  ≥ 0.90 (was ≤ 1, ≥ 0.91) to match the measured cross-EP FP16 quantization
  characteristic. Full bench: `docs/research/phase3.8/step1/full_bench.md`.

### Added

- **sparrow-engine-types: `ProvenanceRecord` (W1) + `DriftReference` (W4) manifest sections**
  (Phase 4, 2026-05-07). 3 optional `Option<String>` fields on `[provenance]`
  (`training_dataset_id`, `training_experiment_id`, `training_repo_commit`);
  `BTreeMap<String, f32>` per-class frequency on `[drift_reference]`. All
  optional; manifests without these sections load unchanged (`#[serde(default)]`).
  Sparrow Engine round-trips the values; never interprets them. Sibling repos (`sparrow-data`,
  `bongo-fine-tuning`) populate from their own state. `Eq` derived on
  `ProvenanceRecord` (all-Option-String); `Serialize` derived because the same
  struct embeds in `InferenceLogRecord` on the wire (single canonical type, no
  parallel definitions).

- **sparrow-engine-types: `InferenceLogRecord` wire schema + `DriftMetrics` per-request metrics**
  (Phase 4 W2 + W4, 2026-05-07). New modules `inference_log.rs` + `drift_metrics.rs`.
  `SCHEMA_VERSION = "1.0"` constant. Fields: `schema_version`, `request_id` (UUID v4
  hex), `timestamp_utc` (RFC3339 millis UTC), `media_hash` (SHA-256 lowercase hex),
  `model_id` (pipeline_id for `/v1/pipeline`, not a constituent step's model id),
  `model_version` (reserved for `sparrow-data` to populate, currently always `None`),
  `device` (never `"auto"` — `Engine::active_device` resolves Auto), `inference_ms`
  (f64 widened from engine f32; engine processing time on `/v1/detect` single +
  `/v1/classify` + `/v1/audio/detect` + `/v1/pipeline`; wall-clock end-to-end on
  `/v1/detect/batch`), `result` (full HTTP response payload), `provenance` (manifest
  snapshot — classifier-step preferred for `/v1/pipeline`, detector-step fallback),
  `drift_metrics` (Tier-1/2 per-request). Schema-version policy: additive optional
  field changes keep "1.0"; rename / type / semantic-shift bumps to "2.0" with
  coordinated `sparrow-data` ingester change. `DriftMetrics` carries `confidence_p50`
  + `confidence_p95` (nearest-rank, NaN dropped) + `detections_per_image` (count /
  `image_count.max(1)`) + optional `class_distribution_psi` (PSI vs `[drift_reference]`,
  `eps = 1e-4` smoothing on both observed + reference, union-of-keys support; returns
  `None` when both observed and reference are empty).

- **sparrow-engine-server: `InferenceLogSink` trait + `StderrJsonLinesSink` default sink**
  (Phase 4 W3, 2026-05-07). New `sparrow-engine-server/src/sink.rs` defines
  `pub trait InferenceLogSink: Send + Sync { fn emit(&self, record:
  &InferenceLogRecord) -> Result<(), SinkError>; }` (sync emit; future HTTP / network
  sinks must internally `tokio::task::spawn_blocking`). Default `StderrJsonLinesSink`
  writes one JSON line per record under stderr lock. `Arc<dyn InferenceLogSink>`
  lives on `AppState`, default-wired to the stderr sink. The trait shape forecloses
  async sinks until upgrade to `async fn emit` is committed.

- **sparrow-engine-server: `?store=true` + `halt_on_store_failure` query params on
  detect/classify/audio/pipeline** (Phase 4 W3, 2026-05-07). Per-request,
  default-false flags. Existing clients see no change. Emit happens AFTER inference
  returns successfully; never on the error path. `store=true + halt=false` → 200
  even if sink errs (warn-log via `tracing`); `store=true + halt=true` → 500
  INTERNAL_ERROR if sink errs. Idempotency is a storage-layer property
  (`sparrow-data` sibling), not enforced by the default stderr sink.
  `media_hash` SHA-256 lowercase hex over request bytes (single image, audio
  bytes, batch first image). `provenance` populated from
  `handle.manifest().provenance.clone()` at all 5 handler call sites; for
  `/v1/pipeline`, the classifier-step manifest is preferred (detector-step
  fallback via `and_then` chain).

- **sparrow-engine-server: `compute_drift_metrics` Tier-1/2 drift compute path**
  (Phase 4 W3, 2026-05-07). New `sparrow-engine-server/src/drift.rs` defines
  `pub fn compute_drift_metrics(confidences, image_count, class_labels, reference)
  -> DriftMetrics`. PSI uses `eps = 1e-4` smoothing on observed + reference
  frequencies, summed `Σ (p_i - q_i) * ln(p_i / q_i)` over the union of class
  buckets. Nearest-rank percentile (NaN-filtered, sorted ascending).
  `image_count.max(1)` denom prevents div-by-zero on `0` (treated as `1`).
  Stateless — every request computes its own snapshot; Tier-3 (cross-request
  reference + CUSUM + alarm path) lives in the eventual `sparrow-ops` sibling.

- **sparrow-engine-server: `tests/store_flow.rs` integration suite** (Phase 4 W3, 2026-05-07).
  6 tests covering the Sink contract: collecting + failing sink fakes; round-trip
  serde; schema-version stability; T-3a NIST SHA-256 vector; T-3b SHA-256
  idempotency.

- **libsparrow_engine: manifest `[inference] precision` + `[model] file_fp16` fields**
  (Phase 3.8 pre-Phase-A, 2026-05-01, commit `5b5d3fd`). New optional manifest
  fields enable opt-in FP16 inference. `[inference] precision` accepts `"fp32"`
  (default, preserves pre-3.8 behaviour) or `"fp16"`. When `precision = "fp16"`,
  the engine loads `[model] file_fp16` instead of `[model] file`. Required for
  the FP16 path; manifest validation rejects `precision = "fp16"` without a
  `file_fp16` value. Backed by ORT's `transformers.float16` converter with
  `keep_io_types=True` (preprocess + postprocess code unchanged when switching
  precision). New `Precision` enum + `model_file_fp16: Option<String>` field on
  `ModelManifest`. Helper script `sparrow-engine/tools/convert_fp16.py` produces FP16
  ONNX files. 4 new unit tests verify default, fp16 with file, fp16 without
  file (rejected), unknown precision (rejected). 211/211 libsparrow_engine tests pass.
  Empirical bench: MDv6 CUDA-EP FP16 hits 11.11 ms median (1.69× over FP32)
  with detection-count parity preserved (244 detections, IoU mean 0.9964).
  Hardware: requires Tensor Cores (Ampere sm_80+ for fast FP16; sm_75 RTX
  20-series slower; pre-Volta has no Tensor Cores). Per-model verification
  gate is required before flipping any manifest's `precision` to `"fp16"`;
  Phase 3.8 ships MDv6 verified, others (DeepFaune, HerdNet, OWL-T, SpeciesNet,
  MD_AudioBirds) deferred to Phase 3.8 B/D per design.
  See `docs/lessons.md § "Model Onboarding & Inference Engine" → "FP16 is
  essentially free for YOLO-family detection on Tensor Cores"` for the
  durable lesson + 6-axis TRT cost matrix that justifies CUDA-EP FP16 over
  TensorRT FP16. Per-model audit gate items: `docs/ideas.md § "Phase 3.8
  Follow-ups (per-model FP16 + per-model channel-order audit)" → P3.8-1
  Per-model FP16 verification audit`.

- **libsparrow_engine: manifest `[preprocessing] channel_order` field** (Phase 3.8
  pre-Phase-A, 2026-05-01, commit `52ab55b`). New optional manifest field
  selects RGB or BGR plane ordering on the preprocess output tensor.
  `channel_order = "rgb"` (default, preserves pre-3.8 behaviour for legacy
  classifiers + ImageNet-pretrained models) or `"bgr"` for YOLO-family models
  trained via Ultralytics (which uses OpenCV's BGR convention). New
  `ChannelOrder` enum + `channel_order: Option<ChannelOrder>` field on
  `ModelManifest`. Backward-compatible: pre-3.8 manifests without the field
  parse and behave identically (default RGB). Sparrow Engine decodes images to RGB
  internally; when `Bgr` is specified, channels are swapped before tensor
  construction. New unit test
  `test_channel_order_swap_rgb_vs_bgr` verifies plane assignment in
  `preprocess::build_tensor`. **MDv6 + DeepFaune test-fixture manifests
  (under `test_files/sparrow_engine_models/`) opted into `channel_order = "bgr"`**
  — fixes a pre-existing pre-3.8 bug where sparrow-engine fed RGB tensors to
  BGR-trained YOLO models. Other model manifests (HerdNet, OWL-T,
  SpeciesNet) untouched pending per-model verification. Empirical
  evidence: stage-by-stage diagnostic on a 100-image MDv6 corpus reduces
  detection count from 249 (pre-fix) to 244 (post-fix), matching PW's 243
  within filter-implementation tolerance — closes the previously-documented
  "+2.9% sparrow-engine-vs-PW correctness axis" from Phase 3.7 R5.
  See `docs/lessons.md § "Model Onboarding & Inference Engine" → "YOLO
  models trained via Ultralytics expect BGR, not RGB"` for the durable
  lesson — when on-boarding a new YOLO-family ONNX export, default to
  `channel_order = "bgr"` and verify via stage diagnostic. Per-model audit
  gate items: `docs/ideas.md § "Phase 3.8 Follow-ups (per-model FP16 +
  per-model channel-order audit)" → P3.8-2 Per-model channel-order audit`
  + `→ P3.8-3 CPU-mode channel-order implementation defensive test`.

### Changed

- **libsparrow_engine: viz text labels lifted from compile-time feature to runtime
  flag** (Phase 3.7, 2026-04-28). The `viz-text` Cargo feature is retired.
  `ab_glyph` is now an unconditional dep; DejaVu Sans is always embedded.
  Toggle text labels per call via `RenderOpts.show_labels: bool` (default
  `false`). sparrow-engine-cli exposes `--show-labels` (default off) on `detect`,
  `classify`, `pipeline`. sparrow-engine-python `visualize()` accepts the matching
  `show_labels: bool = False` kwarg. Net cost: ~+1-2 MB binary (the font
  + glyph rasteriser are now always linked) in exchange for runtime
  toggle without rebuild. Two libsparrow_engine viz tests guard both directions
  of the toggle. Rationale: rebuilding to flip a presentation knob was
  poor UX; the user explicitly asked to lift it. See Phase 3.7 master
  plan entry.

### BREAKING

- **`spe detect-audio` default output changed** (Phase 3.5, S5 / item #6).
  The default output no longer emits one row per sliding-window segment
  (~198 rows for a 60 s recording at 1.0 s window, 0.3 s stride). It now
  emits merged confidence ranges instead: consecutive windows whose
  confidence exceeds the threshold and whose gap is less than a
  configurable merge threshold (default stride + 1 ms — computed in the
  CLI; callers of `detect_audio::merge_segments` pass `gap_s` directly)
  are collapsed into a single `(start_time_s, end_time_s,
  max_confidence)` range. The default confidence threshold for merging
  is **0.9** (per `sparrow-engine/models/audiobirds.toml`; raised from 0.5 after
  manual-test §3.3 surfaced a saturation problem — a 60 s clip
  collapsed into a single 0.0–60.0 s range under 0.5 because
  MD_AudioBirds_V1 returns ~1.0 confidence on continuous bird audio).
  Scripts that parse the old per-window output MUST
  pass `--raw-segments` to opt back in to the previous format, which is
  preserved bit-for-bit. JSON and CSV schemas both change: JSON emits a
  `ranges` array (fields: `start_time_s`, `end_time_s`, `max_confidence`,
  `class`) by default; `--raw-segments` restores the old `segments` array
  (fields: `start_time_s`, `end_time_s`, `confidence`). CSV header flips
  between `start_time_s,end_time_s,max_confidence,class` and
  `start_time_s,end_time_s,confidence` the same way. `class` is `null`
  for binary audio detectors (MD_AudioBirds_V1, the Phase 1 default); it
  is reserved for future multiclass audio models.

  **Migration**: add `--raw-segments` to any script consuming the old
  output. No library-side breakage — `detect_audio::detect_audio()` and
  `AudioDetectResult::segments` remain unchanged; only the CLI default
  flips.

### Added

- **`spe detect-audio --visualize / --output-dir`** (Phase 3.6 micro).
  Mirrors the existing `--visualize` flag on `detect`/`classify`/`pipeline`.
  Writes one `{stem}_viz.png` per audio input. The backdrop is the
  **real mel spectrogram** of the input audio (computed via the same
  DSP path as inference, using the model's manifest preprocessing
  config), with the per-window confidence heatmap (inferno colormap)
  overlaid on top; in the default merged-range mode, cyan vertical
  bars mark each range's start/end and a bottom band reflects each
  range's `max_confidence`. `--raw-segments --visualize` skips the
  cyan overlay. JSON output is unchanged (additive). New library
  surface: `sparrow_engine::viz::render_range_overlay()` + `RangeOverlayOpts`
  + `sparrow_engine::viz::render_mel_spectrogram()` + a public
  `ModelHandle::audio_preprocess_config()` accessor.

- **Canonical audio manifest** at `sparrow-engine/models/audiobirds.toml`,
  mirroring the post-Phase-3.5 herdnet/owlt template pattern. Encodes
  the production-default `confidence_threshold = 0.9`.

- **CLI progress bar** (Phase 3.5, S5 / item #1-cli). `spe detect`,
  `spe classify`, `spe detect-audio`, and `spe pipeline` now render
  an `indicatif` 0.17 progress bar on stderr showing position, bar, ETA,
  and per-second throughput. The bar is suppressed automatically when
  stderr is not a TTY (piped / redirected) and can be suppressed explicitly
  via the new global `--quiet` flag. Replaces the earlier
  `eprint!("\r[{i}/{n}] {path}")` progress line, which is now removed.

- **libsparrow_engine**: `detect_audio::merge_segments(&[AudioSegment], gap_s: f32)
  -> Vec<AudioRange>` — public helper that collapses consecutive windows
  into merged ranges. `AudioRange` is a new public type in
  `libsparrow_engine::detect_audio` with fields `start_time_s`, `end_time_s`,
  `max_confidence`, and `class: Option<String>` (None for binary
  detectors).

- **libsparrow_engine: viz dispatches on `ModelType`, not bbox pixel size**
  (Phase 3.5, S3 / item #3, MT-9 fix). New `ModelSubtype` enum
  (`Standard` | `Overhead`) in `libsparrow_engine::types`, new
  `ModelType::OverheadDetector` variant, new `RenderOpts.model_type`
  field (default `Detector`). The TOML manifest gains a `[model].subtype`
  key accepting `"standard"` | `"overhead"`; missing field defaults to
  `"standard"` for backward compatibility with pre-3.5 manifests.
  `viz::render` now dispatches to the centroid-dot path iff
  `opts.model_type == ModelType::OverheadDetector`, replacing the
  pre-S3 heuristic that compared bbox pixel size against
  `2 * point_radius` (which false-negatived for overhead models on
  high-resolution images). Canonical overhead manifests added at
  `sparrow-engine/models/herdnet.toml` and `sparrow-engine/models/owlt.toml`.

- **sparrow-engine-python: per-file `progress_callback` + `tracing` →
  `logging` bridge** (Phase 3.5, S6 / item #1-py + #9).
  `sparrow-engine.detect`, `sparrow-engine.classify`, `sparrow-engine.detect_audio`, and
  `sparrow-engine.pipeline` accept an optional
  `progress_callback: Callable[[int, int, str], None]` kwarg
  invoked once per input file after its inference attempt resolves
  (success or failure), with `(index, total, filename)`. Raising
  from the callback aborts the batch and propagates the exception
  to Python. Separately, Rust-side diagnostics now emit via
  `tracing::warn!(target: "sparrow_engine::python", …)` and are bridged into
  Python's `logging` module by `pyo3-log` at module init — events
  surface on `logging.getLogger("sparrow_engine.python")` (a child of
  `"sparrow-engine"`). This removes all former `eprintln!` writes in
  `sparrow-engine-python`, which were invisible inside Jupyter kernels
  (PyO3 #2247). Consumers that want to see Rust-side warnings
  should configure the `"sparrow-engine"` logger, e.g.
  `logging.getLogger("sparrow_engine").setLevel(logging.DEBUG)`.

- **libsparrow_engine: viz text labels behind `viz-text` feature flag** (Phase 3.5,
  S4 / item #4). **Superseded by the Phase 3.7 lift in § Changed above**
  — `viz-text` Cargo feature retired 2026-04-28; `ab_glyph` is now an
  unconditional dep; DejaVu Sans always embedded; toggle is runtime
  via `RenderOpts.show_labels`. The original W3 entry below is preserved
  as the historical landing record.

  Original W3 landing: new optional Cargo feature `viz-text` adds an
  `ab_glyph` dep and embeds DejaVu Sans at compile time. When active,
  `viz::render()` draws `"{label} {conf:.2}"` above each bbox in the
  Detector/Classifier/AudioDetector/AudioClassifier paths (overhead-dot
  path is NOT labeled — S4 scope-out). Default build is unchanged:
  `ab_glyph` is not linked, the font is not embedded. Font lives at
  `sparrow-engine-core/assets/fonts/DejaVuSans.ttf` with `LICENSE` (Bitstream
  Vera permissive license) alongside.

- **libsparrow_engine: audio heatmap end-to-end test** (Phase 3.5, S9 / item #12).
  New integration test `sparrow-engine-core/tests/audio_heatmap_e2e.rs` with
  three committed WAV fixtures under
  `sparrow-engine-core/tests/fixtures/audio/` (`short_2s.wav`, `medium_10s.wav`,
  `long_30s.wav` — synthetic, ≤960 KB each). Exercises
  `viz::render_audio_heatmap` against each fixture and asserts dimensional
  correctness, monotonic confidence→heat mapping, and inferno-warm pixel
  presence in the high-confidence band. Output PNGs land under
  `test_outputs/libsparrow_engine/audio_heatmap_e2e/` for the visual-inspection
  pass protocol documented at
  `docs/review/phase3.5-manual-test/manual_test_plan.md` §8.2 (originally
  lived under `docs/review/phase3.5-audio-heatmap/`; merged 2026-04-28 to
  keep manual testing in a single doc). ORT-free (no engine, no models).

- **Head-to-head PyTorch vs ONNX benchmark** (Phase 3.5, S10 / item #11).
  New benchmark script `scripts/bench_head_to_head.py` (~240 LOC) measures
  MegaDetector v6 through two independent inference stacks on the same
  image set: libsparrow_engine (`.onnx` via PyO3) and PytorchWildlife (`.pth` via
  `torch.hub`). Records per-image mean/median/stddev, cold start, peak
  RSS, detection count, decode time. Results (MDv6, RTX 6000 Ada, 100
  images, N=3) committed to `docs/benchmarks.md` §8. Summary: libsparrow_engine
  43.86 ms/img, PW 24.76 ms/img, +2.9 % detection delta in favor of
  libsparrow_engine (no redundant NMS). Asymmetry caveat (S5 item #6 shapes only
  the sparrow-engine column; PW uses its own defaults) is reproduced across
  the script module docstring, the `ASYMMETRY_CAVEAT` runtime-printed
  constant, and the `docs/benchmarks.md` §8.1 table caption.

- **W5 clean-room local install test pipeline** (Phase 3.5, S11 /
  item #5). New developer tool at
  `sparrow-engine/scripts/clean_room_test.sh` (~280 LOC bash orchestrator) plus
  two minimal Dockerfiles (`sparrow-engine/scripts/clean_room/Dockerfile.ubuntu22`,
  `Dockerfile.ubuntu24`) and review directory
  `docs/review/phase3.5-clean-room/` (README, GPU manual checklist,
  auto-appended results log, manual test plan). Runs the 2×3 CPU
  install matrix (Ubuntu 22.04 + 24.04 × CLI + sparrow-engine Python wheel +
  pytorchwildlife-compat shim wheel) inside throwaway Docker containers
  that pre-install only `python3 python3-pip python3-venv
  ca-certificates libssl3 libstdc++6 libgcc-s1`. Smoke tests verify
  INSTALL only (`spe --version`, `spe models list`, `python -c "import sparrow_engine"`,
  `python -c "import pytorchwildlife"`); inference is intentionally out of
  scope. Caveats:
  - **Not a CI job.** No GitHub Actions workflow, no push-tests.
  - **No PyPI install testing.** PyPI publish pipeline is deferred to
    Phase 4.5+ per `docs/design/phase3.5/final_design.md` §7.1; the
    matrix runs against locally-built wheels via mounted-volume.
  - **No GH Releases asset push-tests.** Artifacts are read from the
    dev box's `sparrow-engine/target/release/` and `sparrow-engine/sparrow-engine-python/dist/`.
  - **Local-only developer tool.** Run it before a colleague handoff
    or any internal user delivery; do not run on every commit.
  - **Static-CLI ship story** ("~35MB with static ORT" per CLAUDE.md)
    depends on a static-link build pipeline not yet in place — see
    `docs/ideas.md` ticket T-2 (Phase 3.5 follow-up). Today the
    dev-built CLI is dynamic (`readelf -d` shows
    `NEEDED [libonnxruntime.so.1]`); the matrix surfaces this as a
    faithfully-reproduced dev-env leak per the Wave 5 R1 inquisitor
    HYBRID interpretation, rather than masking it with an
    in-container ORT mount.

  GPU coverage is manual via `checklist_gpu.md` (no NVIDIA Container
  Toolkit dependency in this wave).

- **docs: S12 GPU CI runner provisioning ADR** (Phase 3.5, contingent,
  user-ratified 2026-04-22, **deferred indefinitely 2026-04-23**). New
  ADR at `docs/design/phase3.5/adrs/s12_gpu_ci_provisioning.md`
  (10 sections, 430 lines) covering provider comparison (GitHub-hosted
  GPU, self-hosted on-prem, self-hosted cloud VM, third-party), OS
  choice, token-rotation cadence, supply-chain surface (Shai-Hulud class
  attacks), maintenance burden, and alternatives rejected.
  Recommendation (historical, now parked): self-hosted on-prem RTX 6000
  Ada dev box with Docker-per-job `--ephemeral` runner on Ubuntu 24.04,
  transitioning to a hybrid (GitHub-hosted for fork PRs + self-hosted
  for trusted workflows) when the repo goes public. Eight open questions
  preserved in §10 pending user input (org placement, OSS-release
  timeline, budget cap, GPU-fidelity tolerance, etc.). **Status flipped
  to "do not implement" on 2026-04-23**: user reiterated the strict
  order `development → internal delivery → public release (source only,
  no services)` and excluded CI / GPU-runner work until development is
  fully complete; the 2026-04-22 ratification is superseded. ADR
  preserved on disk as parked research only; `final_design.md §7.2` and
  `master_plan.md` Phase 3.5 W3 entry record the deferral.

### Notes

- **Wire-format addition: `"overhead_detector"`**. `ModelType::as_str()`
  returns a new value `"overhead_detector"` for the new
  `OverheadDetector` variant. This string surfaces at six external
  boundaries: FFI (`sparrow_engine_list_models`, `sparrow_engine_engine_model_info`,
  `sparrow_engine_engine_list_models_extended`), HTTP (`GET /v1/models` via
  sparrow-engine-server), Python (`PyEngine.list_models()` / `model_info()`),
  and CLI (`spe models list` / `spe models info` JSON output). The
  CLI, sparrow-engine-python, and sparrow-engine-server each maintain a duplicated `match`
  rather than calling `ModelType::as_str()` — at
  `sparrow-engine-cli/src/main.rs:497`, `sparrow-engine-python/src/lib.rs:347`, and
  `sparrow-engine-server/src/handlers/models.rs:55`. Any future string change
  MUST update all four sites (the canonical `types.rs:222` plus the
  three duplicates). The canonical `sparrow-engine/models/herdnet.toml` and
  `sparrow-engine/models/owlt.toml` already set `subtype = "overhead"`, so
  consumers that pull W2 will see the new value immediately.
  Consumer switches that treat `model_type` as a closed enum MUST add
  the `"overhead_detector"` arm. Consumers that only care about the
  vision / audio / classifier taxonomy can map
  `"overhead_detector"` to the same handling as `"detector"`. Not an
  ABI break (no struct layout or function signature change).

- FFI: no ABI changes in this release. `sparrow_engine_detect_batch` (`ffi.rs`,
  near line 970) gains a doc comment formalizing the per-image callback
  contract so follow-up work (S6 Python progress bridge) can extend the
  payload additively without breaking existing C callers.

- Semver: this release predates 0.1.0. Consumers should pin to the exact
  commit or tagged build; the `--raw-segments` opt-in is a compatibility
  bridge, not a long-term guarantee.
