# Gotchas and Constraints

Issues that shaped or still shape the design. Each entry includes the root cause, the symptom observed, the workaround, and the escalation / enforcement point.

## ORT CUDA EP + NHWC + dynamic shapes = SafeInt overflow

**Symptom.** SpeciesNet (originally exported NHWC from TensorFlow) crashes the ORT CUDA EP on first inference with a SafeInt overflow deep inside Conv. Logged as MT-10 during Phase 3 manual testing (`docs/review/phase3-testing/manual_testing_logs.md`).

**Root cause.** ORT CUDA EP has two open issues with NHWC + dynamic shapes — #27912 (Conv SafeInt overflow) and #12288 (CUDA EP NHWC dynamic shapes). Both have been open long enough that the sparrow-engine team treats NHWC as unsupported territory on CUDA.

**Workaround.** Convert to NCHW before onboarding. For TensorFlow-origin models: `tf2onnx --inputs-as-nchw` + `onnx-simplifier`. For SpeciesNet the conversion produced identical output (max diff = 0.0).

**Enforcement.** Manifest parser rejects `layout = "nhwc"` at parse time with a message referencing this rule and the ORT issue numbers (`sparrow-engine/libsparrow_engine/src/manifest.rs`). `Layout::Nhwc` enum variant is retained so removing it is not a breaking API change, but the parser path prevents it from being used.

**Related design decision.** D-v3-2 (NCHW mandatory) — see `04_design_decisions.md`.

## cuDNN 9.8 Conv-engine bug on sm_89

**Symptom.** SpeciesNet on RTX 6000 Ada (Ada architecture, sm_89) produces wrong outputs with the default cuDNN that ships in PyTorch and TensorFlow wheels. YOLO-style detectors don't hit this bug. Logged as MT-15.

**Root cause.** cuDNN 9.8 has a Conv-engine bug with asymmetric padding on sm_89. NVIDIA fixed the bug in cuDNN 9.10.

**Workaround.** Install standalone `nvidia-cudnn-cu12>=9.10` outside the PyTorch / TensorFlow envs. Sparrow Engine's dev env uses `~/.local/cudnn/9.10.X` via `LD_LIBRARY_PATH`.

**Enforcement.** `scripts/ort-env.sh` prefers cuDNN ≥ 9.10 over 9.8 — appends to `LD_LIBRARY_PATH` in the right order. For Docker: GPU image `FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` bundles a 9.x cuDNN — confirm version at image build time.

**Report the bug in deployment docs.** Any user running on a non-standard system needs to know about this. Deployment notes in `sparrow-engine/README.md` and Phase 4 deployment guides will call it out.

## Rust disables SIGPIPE by default

**Symptom.** `spe detect *.jpg | head` exits 1 with "fatal runtime error: writing to a broken pipe". `head` closes the pipe once it has its 10 lines; sparrow-engine keeps trying to write and Rust panics on the broken pipe.

**Root cause.** Rust sets SIGPIPE to `SIG_IGN` by default; writes to a broken pipe return `EPIPE` as an `io::Error`, which panics through Rust's unwrap / ? chain.

**Workaround.** CLI `main()` sets SIGPIPE back to `SIG_DFL` — standard Unix behavior. `spe ... | head` then exits 141 cleanly (128 + SIGPIPE).

**Enforcement.** Source: `commit 8c60050`. Inline extern signal handler restoration in `sparrow-engine-cli/src/main.rs`. No dependency change; zero-LOC cost.

## `active_device()` is compile-time, not runtime

**Symptom.** `spe device` reports `cuda:0` even when the GPU is not actually available (driver missing, no physical GPU, CUDA runtime version mismatch).

**Root cause.** `active_device()` uses `ort::ep::CUDA::is_available()` which checks whether the CUDA execution provider was compiled into the ORT library, not whether a GPU driver is present. Logged as MD-4 during Phase 3 manual testing.

**Workaround.** Reliable GPU check = run a real workload and watch `nvidia-smi`. Sparrow Engine's GPU verify alias (in Miao's dev env) runs a real inference and checks nvidia-smi utilization. Don't trust `spe device` output alone.

**Enforcement.** Documented inline in `engine.rs` `resolve_device`:

```rust
/// **Compile-time check only**: uses `ort::ep::CUDA::is_available()` which
/// checks whether the CUDA execution provider was compiled into the ORT
/// library. This does NOT verify that a physical GPU exists, that CUDA
/// drivers are installed, or that the CUDA runtime version is compatible.
```

Source: `sparrow-engine/libsparrow_engine/src/engine.rs:254-262`.

## `fork()` + `ENGINE_EXISTS` AtomicBool

**Symptom.** Python `multiprocessing.Pool` with `fork` start method (POSIX default on Linux) fails all child workers with `EngineAlreadyExists` even though no Engine object exists in the child.

**Root cause.** `fork()` duplicates the parent's memory including the `ENGINE_EXISTS` AtomicBool. Child inherits `true` but there's no Engine on the Rust side. Any `Engine::new()` in the child fails. Inherent to POSIX fork + Rust singleton — can't be fixed from sparrow_engine-python alone.

**Workaround.** Use `multiprocessing.set_start_method("spawn")` before `Pool`. macOS default is `spawn` since Python 3.8; Linux default is `fork`. Windows default is `spawn`.

**Enforcement.** Documented as a limitation in `sparrow-engine-python/README.md` and in the Python docs. No code fix — POSIX semantics prevent it.

## Engine singleton test order-dependency

**Symptom.** `cargo test -p libsparrow_engine --lib` intermittently failed with 3–5 tests panicking with `EngineAlreadyExists` or "Second engine should fail". Nondeterministic: failures varied between runs even with `--test-threads=1`. All 18 engine tests passed in isolation.

**Root cause.** 18 `#[test]` in `engine.rs` exercise the singleton. Each resets `ENGINE_EXISTS.store(false)` at start and `drop(engine)` clears it, but test runners can still interleave under `--test-threads=1` when test binaries are involved indirectly. Race between parallel test binaries rather than within one binary.

**Workaround.** Added `serial_test = "3"` as a dev-dependency and tagged all 18 tests `#[serial]`. Now 173/173 libsparrow_engine tests pass deterministically at `dev@09ee0aa` (175 at Phase 3.5 W1 S7 close; -2 dropped during later consolidation).

**Enforcement.** Source: `commit 7fed112`. Tests verified across 5+ repeat runs post-fix.

## MT-17 — pipeline GPU teardown heap corruption (MITIGATED)

**Symptom.** `spe pipeline --device cuda` on 100 images intermittently aborts with "corrupted double-linked list" (SIGABRT) at process exit *after* all inference completes. Pre-mitigation reproduction rate 10–33% on a 100-image set. Logged as MT-17 (`docs/review/phase3-testing/manual_testing_logs.md`).

**Isolation.** GPU-only. Requires both detector and classifier sessions loaded. Detect-only and classify-only are clean. CPU pipeline is clean. Inference results are written correctly — correctness is not affected, only process-exit cleanliness.

**Root cause.** ORT's CUDA EP retains execution-provider hooks into `libonnxruntime_providers_cuda.so`. Rust's `Drop` sweep on `Engine` releases session HashMaps *after* `Drop::drop()` returns, as part of the field-drop phase. glibc `_dl_fini` can finalize the CUDA provider shared object before the session field-drops run, so session-drop reads freed memory. Upstream (pykeio/ort #564, closed `not_planned` 2026-04-05) confirms this is an ORT C-API lifetime issue that the Rust `ort` wrapper cannot fix alone. Full RCA in `docs/bugs.md`.

**Mitigation applied** (`sparrow-engine/libsparrow_engine/src/engine.rs`, `Drop for Engine`):

1. Drop sessions explicitly under the `models` write-lock inside `Drop::drop` — forces ORT `Session` drops to run while the runtime is provably still mapped, rather than at the post-`drop` field-drop sweep when `_dl_fini` may have already started.
2. `std::mem::forget(Arc::clone(&self.inner))` — leaks `EngineInner` so its `SessionBuilder` does not drop during `_dl_fini`. Symmetric with pykeio/ort's own `Environment` leak (discussion #280).

Measured on 60 post-mitigation 20-run stresses: 57/60 clean vs 18/20 baseline (10–33% → ~5% residual). Mitigation is a 2–6x reduction, not elimination — the `_dl_fini` ordering race persists.

**Regression harness.** `sparrow-engine/libsparrow_engine/tests/integration_pipeline_gpu_stress.rs` (`#[ignore]` by default, requires GPU + ORT env + `sparrow_engine_models_test` manifests). Run with:

```sh
source scripts/ort-env.sh
cargo test --release -p libsparrow_engine --test integration_pipeline_gpu_stress \
    -- --ignored --test-threads=1
```

The harness is a probe (not a CI gate) because of the residual ~5% flake. Use when bumping `ort`, CUDA, or cuDNN.

**Workaround for reliability-sensitive use.** `spe pipeline --device cpu` is fully clean. Detection-only and classification-only on GPU are unaffected.

**Status.** MITIGATED in `libsparrow_engine`. Full elimination requires upstream ORT C-API changes (tracked against pykeio/ort #564).

## libsparrow_engine dev env requires dynamic ORT linking

**Symptom.** Tests fail to build on dev Ubuntu 22.04 with `glibc 2.35` — the pre-built ORT static lib (via `ort-sys`) requires `glibc 2.38+`.

**Workaround.** Dev machines link dynamically against the pip `onnxruntime` shared library. `scripts/ort-env.sh` sources the right paths:

```bash
export ORT_LIB_LOCATION=$(python -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))")/capi
export ORT_PREFER_DYNAMIC_LINK=1
export LD_LIBRARY_PATH="${ORT_LIB_LOCATION}:${LD_LIBRARY_PATH}"
```

For the shipped CLI binary and Docker images, the static ORT lib is used — those environments have newer glibc.

**Enforcement.** `scripts/test.sh` sources `ort-env.sh` before running `cargo test`. Documented in `sparrow-engine/README.md`.

## `serde_yaml` is deprecated

**Symptom.** N/A — this is a preemptive constraint.

**Root cause.** `serde_yaml` is no longer maintained. YAML parsing bugs that surface in the deprecated version will not be fixed.

**Workaround.** Don't use YAML. Use TOML throughout. All manifests are `manifest.toml`, all config is TOML.

**Enforcement.** Grep the codebase for `serde_yaml` — zero hits. `Cargo.toml` deps list has only `toml` and `serde_json`.

## ORT EP ordering

**Symptom.** If CUDA EP is registered but no GPU is present, ORT can take a long time to fall back to CPU and may emit verbose warnings that confuse end users.

**Root cause.** ORT's EP probing is best-effort. `Device::Auto` relies on the EP list order.

**Workaround.** `Device::Auto` explicitly probes CUDA first via `ort::ep::CUDA::is_available()` before registering the EP. If not available, registers CPU only. Keeps the warning noise down.

**Enforcement.** `sparrow-engine/libsparrow_engine/src/engine.rs` `resolve_device` and session builder setup. Dev env logs show one probe attempt at Engine creation; no per-inference probing.

## Detection parity: sparrow-engine vs Triton

**Symptom.** Sparrow Engine returns ~4% more detections than Triton baseline on the same 100-image test.

**Root cause.** Triton was configured to apply a second NMS pass after the model's in-graph NMS, deleting legitimate boxes that survived the first pass. This is not a sparrow-engine bug — it is a Triton configuration quirk that sparrow-engine avoids by putting all NMS in the ONNX graph and never re-applying.

**Impact.** Users migrating from Triton to sparrow-engine will see *more* detections. In the conservation use case, more recall is usually better than fewer, but it means downstream thresholds may need retuning.

**Enforcement.** D-v4-7 (NMS in graph) makes this permanent.

## Manifest backward-compatibility via `#[serde(default)]`

**Symptom.** N/A — preemptive constraint for Phase 3's manifest schema extension.

**Root cause.** Adding `onnx_sha256`, `onnx_size_bytes`, `version`, `description` to `[model]` would break parsing of older manifests that don't have those fields.

**Workaround.** `#[serde(default)]` on the new `Option<T>` fields means older manifests continue to parse; the new fields simply default to `None`.

**Enforcement.** `sparrow-engine/libsparrow_engine/src/manifest.rs` `ModelSection` struct. Tests in `manifest.rs` verify that a manifest without the new fields still parses.

## COCO `category_id` namespace collision in pipelines

**Symptom.** When a pipeline mixes classifier-labeled rows (SpeciesNet `label_id = 7` → "deer") and detector-labeled rows (MD v6 `label_id = 1` → "animal"), a user exporting to COCO could see the detector and classifier labels collide if both label spaces are 1-indexed and overlap.

**Root cause.** COCO keys category by `category_id`. `seen_categories.entry(label_id).or_insert_with(...)` silently drops the second `(label_id, label)` pair on collision. Pre-Phase-3-audit-fix behavior: wrong category name for colliding IDs. Caught in Phase 3 final audit-fix R3 by the inquisitor's independent scan (neither reviewer caught it first).

**Workaround.** As of `commit ee01898`: three-arm `Entry::Occupied | Entry::Vacant` match. First-seen wins. `HashSet<u32>` dedup for one-shot warn output per colliding ID: `eprintln!("WARN: COCO category_id collision on {}: keeping first-seen label {}", id, first_label)`. Export still produces valid COCO JSON.

**Full fix tracked as Phase 3.5 follow-up; gating work landed.** Item #3 (model subtype in manifest + viz dispatch) is the gating design work and landed in W2 R5 (2026-04-23). Phase 3.5 closed 2026-04-28 without folding in a full namespace strategy (separating detector and classifier label spaces in the output); the namespace strategy now falls to Phase 4+ scope per user-priority signals (see `11_phase_3.5_planned.md` § Item #3 + MI-2 follow-up).

**Enforcement.** Doc comment on `to_coco` explains the invariant: detector and classifier label spaces must be disjoint when mixed in a pipeline, else first-seen wins and a warning is emitted.

## all-NaN sentinel leak in statistics

**Symptom.** When `total_detections > 0` but all detection confidences are NaN (a pathological case for a broken model), `summarize_detections` returned `confidence_min = +Inf` and `confidence_max = -Inf` — the f32 fold initial values leaked to output.

**Root cause.** Reset guard used `total_detections == 0`. When detections exist but all confidences are non-finite, the fold never got a finite value to replace the sentinels.

**Workaround.** As of `commit 6766f4d`: reset guard uses `non_nan_count == 0`. If no finite confidences were seen, min/max reset to 0.0. Regression test `all_nan_confidence_does_not_leak_sentinels` asserts the full state.

**Caught in.** Phase 3 final audit-fix R3 by the inquisitor's independent scan.

## PyO3 `println!` invisible in Jupyter

**Symptom.** Debug output from sparrow_engine-python Rust code not visible in Jupyter notebooks.

**Root cause.** PyO3 issue #2247. Jupyter redirects Python stdout but not Rust-level stdout.

**Workaround.** Route diagnostic output through Python logging or through `eprintln!` to stderr.

**Enforcement.** `~/.claude/rules/rust.md` § PyO3 Bindings explicitly bans `println!` from Rust code that runs inside PyO3 consumers.

## Confidence

**Confidence**: HIGH
- Factual accuracy: HIGH — each gotcha cites commit SHA, MT number, or issue number; root causes verified against `project_tech_report_notes.md` and `phase3-testing/manual_testing_logs.md`
- Completeness: MEDIUM — covers the gotchas sparrow-engine has hit in development and testing; undiscovered production gotchas may exist
- Freshness: HIGH — 2026-04-29 (R12 ch 06 L74 test-count drift 175→173 with provenance softening + L171 COCO namespace strategy past-tense — Phase 3.5 closed without folding in full namespace strategy → Phase 4+; matches HEAD)

## References

- `docs/review/phase3-testing/manual_testing_logs.md` — MT-1 through MT-17 issue log
- `docs/lessons.md` — durable cross-session lessons
- `~/.claude/projects/.../memory/project_tech_report_notes.md` — drafting notes (gotcha catalog)
- `commit 7fed112` — serial_test for engine tests
- `commit 8c60050` — SIGPIPE fix
- `commit ee01898` — COCO label_id collision fix (MI-2)
- `commit 6766f4d` — stats all-NaN sentinel fix (MI-1)
- ORT issues: #27912 (Conv SafeInt overflow with NHWC), #12288 (CUDA EP NHWC dynamic shapes)
- PyO3 issue #2247 (println! invisible in Jupyter)
