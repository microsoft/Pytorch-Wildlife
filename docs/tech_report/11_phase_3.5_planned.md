# Phase 3.5 — Planned Enhancements (COMPLETE 2026-04-28)

> **Status banner**: chapter title retained for stable URL/cross-ref; chapter content is now historical record (Phase 3.5 SIGNED OFF 2026-04-28). For backlog status see `docs/ideas.md`, for completion details see `docs/master_plan.md § Phase 3.5`.

Small-to-medium enhancements scoped to land between Phase 3 (utilities + model catalog, complete) and Phase 4 (Docker data management — scope reframed by Phase 3.7 Track A; see `docs/design/phase4/README.md`). Twelve items in the backlog.

**Status.** COMPLETE 2026-04-28 (manual-test sign-off). Wave 1 complete (2026-04-22): S1 MT-17 mitigation (item #7), S2 debug+release CI matrix + eprintln guard (item #10 + CI-side of #9), S7 Sparrow summary parity (item #2 — see `docs/design/phase3.5/adrs/sparrow_parity.md`), S8 pytorchwildlife-compat shim (item #8). Wave 2 complete (2026-04-23): S3 viz subtype dispatch / MT-9 correctness fix (item #3), S5 CLI output hygiene (items #6 + #1-cli), S6 Python progress + `tracing` bridge (items #1-py + #9). Wave 3 landed 2026-04-23 (S4 viz text labels — later lifted to runtime toggle in Phase 3.7; S9 audio heatmap e2e; S10 head-to-head bench). Wave 4 audit-fix CONVERGED 2026-04-23. Wave 5 clean-room install pipeline + audit-fix CONVERGED 2026-04-24. Manual-test signed off 2026-04-28; only open item MT-3.5-12 (bimodal libsparrow_engine latency) carried to Phase 3.7 Track B. See `docs/design/phase3.5/final_design.md` for the full waves; item-level backlog in `docs/ideas.md`.

## Priority ordering (recommended)

| Order | Item | Rationale |
|-------|------|-----------|
| 1 | #7 MT-17 investigation | Reliability bug. Intermittent, so diagnosis takes time. Start early. |
| 2 | #5 Clean-room distribution testing | Highest future leverage. Would have caught MT-4 / MT-14 / MT-15. |
| 3 | #10 Test matrix: debug + release profiles | CI-signal integrity. Release-only regressions currently ship unobserved. |
| 4 | #11 Rust+ONNX vs PyTorch+PTH head-to-head | High communication value. Direct PytorchWildlife → sparrow-engine comparison for external stakeholders. |
| 5 | #12 Audio heatmap viz test | Untested surface area; small but load-bearing for audio users. |
| 6 | #3 Model-subtype viz dispatch | Correctness for overhead models on high-res images (MT-9 proper fix). |
| 7 | #6 Output verbosity cleanup | Usability — 198 audio segments for a 60 s file is unusable default. |
| 8 | #9 tracing + Python logging bridge | Developer UX for Jupyter notebook users. Replaces invisible `eprintln!` sites. |
| 9 | #1 Progress bar | UX, small scope. |
| 10 | #8 pytorchwildlife compat shim | Migration aid. Small scope. |
| 11 | #2 Sparrow Studio summary parity | Research + implementation. Depends on review access. |
| 12 | #4 Text labels on viz | Behind feature flag. Cosmetic, small. |

## Items

### #1 — Progress bar for batch processing

**Scope.** CLI (`indicatif` crate) + Python (`tqdm` or callback). Both CLI and Python.

**Why.** Currently prints `[N/total]` to stderr. Want ETA, throughput, elapsed time, visual progress.

**Estimated scope.** ~40 LOC total across CLI + Python + passing a callback into libsparrow_engine inference loop.

**Risk.** Low. `indicatif` is well-maintained. Callback pattern matches existing batch detection FFI.

### #2 — Sparrow Studio summary parity

**Scope.** Research phase first, then extend `stats.rs` if needed.

**Why.** Sparrow Studio Local does its own detection summary (`DetermineTimeOfDay`, per-station counts, species tallies). Before Phase 4 bakes the stats schema into the DB, verify sparrow-engine's `summarize_detections` doesn't miss fields Sparrow users expect. Gap analysis → `stats.rs` extension.

**Where to look.** `/home/miao/repos/PW_refactor/sparrow_studio_local/` Sparrow Studio Local codebase.

**Estimated scope.** Depends on gap. Likely ~50 LOC if gaps are minor.

**Risk.** Low. Additive extension to `ModelInfo` or a new `Stats` variant.

### #3 — Model subtype in manifest + viz rendering dispatch

**Scope.** `manifest.rs`, `types.rs`, `viz.rs`. Add `subtype` or `rendering_hint` field to manifests (e.g., `overhead`, `standard`, `segmentation`). Dispatch in `viz.rs::render()` based on subtype.

**Why.** Overhead models (HerdNet, OWL-T) currently use the standard detection render path with a pixel-size heuristic (`bbox < point_radius*2` → render as dot). Fails on high-res overhead imagery where legitimate detections have 20-pixel bboxes on a 6000×4000 image. MT-9 flagged this.

**Estimated scope.** Manifest schema extension + `ModelType` enum update + `viz.rs::render()` dispatch. ~80 LOC.

**Dependency.** Blocks #4 (text labels) and COCO namespace-strategy (from MI-2 Phase 3.5 follow-up).

**Risk.** Medium — changes the render path for existing models. Needs golden-output regeneration for overhead model tests.

### #4 — Text labels on viz bounding boxes

**Status.** Landed in W3 (2026-04-23) as a `viz-text` Cargo feature; **lifted to runtime in Phase 3.7 (2026-04-28)**. `ab_glyph` is now an unconditional libsparrow_engine dep; toggle per-call via `RenderOpts.show_labels: bool` (CLI `--show-labels`, Python `show_labels` kwarg). Default off.

**Scope.** `viz.rs` + `ab_glyph` crate, runtime-toggle. ~120 LOC. Bundled DejaVu Sans (~750 KB) under `libsparrow_engine/assets/fonts/`.

**Why.** Originally renders colored boxes only, no label or confidence text. Hard to distinguish animal / person / vehicle in mixed images.

**Format.** `"{label} {conf:.2}"` above each bbox when `show_labels=true`.

**Dependency.** Font file bundled. No longer feature-flagged at compile time after the Phase 3.7 lift.

**Risk.** Low. Default off; binary cost ~+1-2 MB always (the lift's net cost).

### #5 — End-user distribution testing (clean-room)

**Scope.** Test scripts + CI integration + Docker containers for clean-room testing. ~200 LOC + container setup.

**Why.** Current manual testing runs against the dev environment (`scripts/ort-env.sh`, `LD_LIBRARY_PATH`, alias with `$EXTRA_LIB_PATHS`, cuDNN 9.10+ in `~/.local/cudnn`). End users have none of this. Two paths to verify:

- **(a) CLI binary from GitHub Releases**: download the static-ORT binary (~35 MB), run `spe detect ...` with NO env setup. Must work on GPU and CPU.
- **(b) Python wheel from pip**: `uv pip install sparrow-engine` into a fresh venv, `python -c "import sparrow_engine; sparrow-engine.init(); sparrow-engine.detect(...)"`. Must work without any `LD_LIBRARY_PATH` or `ort-env.sh`.

Should run in clean Docker containers (Ubuntu 22.04 + 24.04) to guarantee no host-env leakage.

**Gaps this catches.**

1. Static ORT linking bugs.
2. rpath misconfiguration.
3. Missing bundled cuDNN in wheel.
4. Missing model catalog fallback.
5. Docs that assume dev env.

**Risk.** Medium-high. Requires CI infrastructure. Catches deployment bugs that are currently invisible until a user complains.

### #6 — Output verbosity cleanup

**Scope.** CLI + libsparrow_engine (`audio.rs`, `detector.rs`, `main.rs`). ~150 LOC + doc updates.

**Why.** Current `detect-audio` on a 60 s file emits ~198 overlapping 1 s-window segments at 0.3 s stride, most at `conf ≈ 1.0`. Useless as a default for humans.

**Proposed changes.**

- (a) Default confidence threshold for audio, like detect has for bboxes. Likely 0.5 or 0.8.
- (b) Auto-merge consecutive high-confidence windows into ranges (198 windows → "birds from 0.0–60.0 s conf=1.0").
- (c) `--raw-segments` flag to preserve current behavior for power users.
- (d) Broader review: audit `detect / classify / pipeline` output shapes too — are bboxes / scores being dumped in a form that respects thresholds? Is max-detections applied consistently? Should JSON have compact vs pretty modes?

**Scope.** Design pass over all 4 inference commands first, then targeted changes in postprocessing (merge logic) + CLI defaults.

**Risk.** Medium. Changes default output format — users scripting against current output need migration or opt-out.

### #7 — Investigate pipeline GPU heap corruption (MT-17)

**Scope.** libsparrow_engine engine / ORT integration investigation.

**Why.** `spe pipeline --device auto/cuda:0` on 100 images intermittently aborts with "corrupted double-linked list" (SIGABRT). ~20–33% reproduction. GPU-only, requires both detector + classifier sessions. Detect-only and classify-only are clean. CPU pipeline is clean. Not a correctness issue — inference results are written correctly when it doesn't crash.

**Suspected cause.** ORT CUDA EP / cuDNN workspace shared between two sessions, corrupted at drop time.

**Investigation steps.**

- (a) Reproduce under AddressSanitizer for a clean stack trace.
- (b) Enable ORT verbose logging to pinpoint cuDNN handle lifecycle.
- (c) Try explicit `drop(engine)` before `main` returns.
- (d) Test with cuDNN-disabled CUDA EP (`cudnn_conv_algo_search=DEFAULT` or off).
- (e) Check ORT GitHub for similar reports with 2.0.0-rc.12.

**Workaround for users.** `spe pipeline --device cpu` for pipelines until stable, or re-run after crash.

**Risk.** Investigation is time-bounded; fix scope depends on root cause. Could be anywhere from a one-line drop-order fix to a deeper ORT version change.

**Priority.** **Highest** among Phase 3.5 items. Reliability bug, hard to reproduce, worth starting early.

### #8 — `pytorchwildlife` compatibility shim with deprecation warning

**Scope.** New `pytorchwildlife-compat` wheel OR bundled with `sparrow-engine-python`. ~30 LOC + new crate OR an `__init__.py` alias.

**Why.** PytorchWildlife users transitioning to sparrow-engine shouldn't have to rewrite imports immediately. A shim makes `import pytorchwildlife` still work, emits `DeprecationWarning("pytorchwildlife has been renamed to sparrow-engine; this alias will be removed in 0.2.0")`, and re-exports sparrow-engine's public API.

**Decision points.**

- (a) **Separate wheel** (`pip install pytorchwildlife`) vs **bundled** (sparrow-engine's `__init__.py` registers a `sys.modules['pytorchwildlife']` alias on first import). Separate wheel is cleaner and opt-in; bundled is lower friction but pollutes `sys.modules`. **Recommended: separate wheel.**
- (b) **API compat level.** If old PytorchWildlife API (e.g., `PytorchWildlife.MegaDetectorV6(...)`) differs significantly from sparrow_engine's (`sparrow-engine.detect(...)`), either ship a thin adapter layer mapping old calls to new ones OR accept that the shim only prevents `ImportError` and users must still rewrite call sites. Recommended: separate `pytorchwildlife` wheel with `__init__.py` containing `warnings.warn(...)` + `from sparrow_engine import *`. `ImportError` stays solved, type hints survive, breakage surfaces only at old-API attribute access (clean `AttributeError` gives user the info they need).

**Removal planned for** 0.2.0.

**Risk.** Low.

### #9 — `tracing` + Python logging bridge for sparrow-engine-python

**Scope.** `sparrow-engine-python/src/lib.rs` — 11 `eprintln!` call sites across `detect`, `classify`, `detect_audio`, `pipeline`, `visualize`. Initialize a `tracing` subscriber at module import that routes events to Python's `logging` module (via `pyo3-log` crate or hand-rolled bridge).

**Why.** Per `~/.claude/rules/rust.md` PyO3 rule, `eprintln!` / `println!` from Rust are invisible in Jupyter (PyO3 issue #2247). The 11 sites emit per-file skip warnings and visualization-failure warnings that silently disappear for notebook users. Each site is tagged inline with `// TODO(Phase 3.5)`.

**Approach.** Replace `eprintln!` with `tracing::warn!(...)` or `tracing::info!(...)` as appropriate. Users then control verbosity via standard `logging.getLogger("sparrow-engine").setLevel(...)`.

**Estimated scope.** ~50 LOC + `pyo3-log` dep (or ~100 LOC hand-rolled).

**Risk.** Low. Replaces output surface without changing behavior.

### #10 — Test matrix covers debug + release profiles

**Scope.** `scripts/test.sh` + CI.

**Why.** `scripts/test.sh` with no args runs `cargo test` (debug only) — cfg-gated release-only tests are silently skipped (no "ignored" marker, no output). Discovered in Phase 3 final audit-fix R1 while adding a silent-skip comment requested by the inquisitor: `heatmap_inverted_alpha_returns_input_in_release` at `libsparrow_engine/src/viz.rs:621` compiles only under `not(debug_assertions)`. Without a release-mode CI pass, release-specific regressions (e.g., release early-return paths for invariants guarded by `debug_assert!`) ship unobserved.

**Approach.**

- (a) Add a `--release` branch to `scripts/test.sh` and run both profiles in CI.
- (b) Have CI run `./scripts/test.sh` and `./scripts/test.sh --release` in separate jobs.

Plus: audit for other cfg-gated tests by grepping `#[cfg(not(debug_assertions))]` / `#[cfg(debug_assertions)]` in the workspace — each match is a test that's skipped in the opposite profile.

**Estimated scope.** Trivial script + CI config change.

**Risk.** Low. Adds a signal that's currently missing.

### #11 — Head-to-head perf comparison: libsparrow_engine Rust + ONNX vs PyTorch + PTH

**Scope.** Benchmark script + docs + results table. Exercise the same model (e.g., MegaDetector v6) via two paths:

- **Path A (sparrow-engine)**: `.onnx` export of MDv6 through libsparrow_engine on CPU and GPU.
- **Path B (PytorchWildlife baseline)**: `.pth` checkpoint of MDv6 through PytorchWildlife on the same hardware.

Measure: raw inference (no HTTP), cold start, total pipeline (startup + preprocess + infer + postprocess), memory footprint, image decode time (Rust `image` vs PIL), Docker image size, GIL contention under N workers.

**Why.** Current exec-summary benchmarks compare sparrow-engine to Triton (end-to-end HTTP, showing Triton's protocol overhead). Adding a direct Rust+ONNX vs PyTorch+PTH comparison gives external stakeholders a clean "before sparrow-engine / after sparrow-engine" signal at the library level — what a user sees when they switch their existing PytorchWildlife script for a sparrow-engine one. This is the missing piece in the current tech-report benchmark story.

**Existing partial data.**

- Raw GPU inference: Rust 71.5 ms/img vs Python + ORT 89.2 ms/img (from `docs/benchmarks.md`). Both are ONNX-based, so this isn't yet the requested comparison.
- Research synthesis projected: 4.3 s (Python) → 348 ms (Rust) cold start; 3214 ms → 91.4 ms total pipeline. These are projected numbers from Subonis and Pistek benchmarks, not measured on the current sparrow-engine codebase.

**Estimated scope.** ~150 LOC for the benchmark script (side-by-side invocation), plus the `PytorchWildlife` environment setup documented. Results table + analysis added to `07_benchmarks.md` + `10_lessons_learned.md`.

**Risk.** Low. Just measurement. Result goes into the tech report.

### #12 — Audio heatmap visualization test

**Scope.** End-to-end test of `sparrow-engine.visualize` for audio confidence heatmap. Currently untested on real audio inference runs.

**Why.** Phase 3's `viz::render_audio_heatmap` was implemented and unit-tested but not exercised against a real audio pipeline with real spectrogram windows. Sparrow Studio Local users may have relied on the audio heatmap visualization; need to confirm it renders correctly across sample rates, file lengths, and confidence distributions.

**Test plan.**

- 3 sample WAV files: short (<5 s), medium (30 s), long (>5 min).
- Run `spe detect-audio ... --visualize --output-dir ...`.
- Visual inspection of output for color-map correctness (inferno, CVD-safe), time-axis alignment, confidence-to-alpha mapping.
- Check output file format and size against spec.

**Estimated scope.** ~50 LOC of test fixtures + visual inspection checklist. Could be part of Phase 3.5 manual testing plan if that format is preferred over automated test.

**Risk.** Low. If issues found, they route to `viz.rs` `render_audio_heatmap` as bug fixes.

## Item #3 + MI-2 follow-up: COCO namespace strategy

Tangentially related to Phase 3.5 item #3 (model subtype). The MI-2 fix in Phase 3 final audit-fix R4 documents that a full namespace strategy for COCO pipeline exports (separating detector and classifier label spaces) is a Phase 3.5 item. The gating design work is item #3 (model subtype dispatch) because the namespace strategy depends on knowing which model produced each detection / classification.

## When Phase 3.5 started

Phase 3.5 opened 2026-04-21 after the triggers below fired (retained as a retrospective record and template for comparable future-phase planning):

- Tech report drafting complete (this document) → clear decision-making baseline.
- MT-17 user reports accumulate → forcing function for item #7.
- Sparrow Studio Web Phase 4 work starts → item #2 (Sparrow summary parity) was sequenced to land before Phase 4 schema freezes (landed Phase 3.5 W1 — 2026-04-22).
- External user requests for pytorchwildlife migration → item #8.

Wave 1 completed 2026-04-22 (S1/S2/S7/S8 + W1 audit-fix CONVERGED R2); Wave 2 completed 2026-04-23 (S3/S5/S6 + W2 audit-fix CONVERGED R5); Wave 3 completed 2026-04-23 (S4 text labels, S9 audio heatmap viz e2e, S10 head-to-head perf comparison); Wave 4 audit-fix CONVERGED 2026-04-23 (3 rounds, round_06..08); Wave 5 clean-room install + audit-fix CONVERGED 2026-04-24 (4 rounds, round_01..04). Manual-test signed off 2026-04-28; MT-3.5-12 (bimodal libsparrow_engine latency) is the only open item, queued to Phase 3.7 Track B. S12 GPU CI runner deferred indefinitely.

## Confidence

**Confidence**: HIGH on scope; HIGH on freshness (all five waves complete; manual-test signed off 2026-04-28).
- Factual accuracy: HIGH — items sourced from `docs/ideas.md`, scope estimates from author
- Completeness: HIGH for currently-known items; new items may emerge from Phase 4 work or user reports
- Freshness: HIGH — 2026-04-29 (all five waves complete; manual-test signed off 2026-04-28; MT-3.5-12 carried to Phase 3.7 Track B; R15 ch 11 L225 retrospective triggering condition past-tense — item #2 landed Phase 3.5 W1 2026-04-22 [NEW-R13-1 deferred from R14, applied per Option A "annotate, don't bump" convention])

## References

- `docs/ideas.md` — full item table with LOC estimates
- `docs/review/phase3-testing/manual_testing_logs.md` — MT-9 (viz overhead), MT-17 (pipeline GPU)
- `docs/review/phase3-final/round_04/inquisitor_review.md` — MI-2 COCO namespace strategy deferral
