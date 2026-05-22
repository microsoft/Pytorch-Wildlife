# Implementation Details

Things a reader needs to know to open the code and not get surprised. Grouped by subsystem.

## Supported models

Model-agnostic by design: any ONNX model that conforms to the manifest schema can be onboarded without a libsparrow_engine code change. The current onboarded set includes the full Sparrow Studio Local production model catalog plus first-party additions for audio.

Representative set (not exhaustive — actual onboarded set matches Sparrow Studio Local's catalog + first-party audio):

| Model | Purpose | Input | Output | Postprocess |
|-------|---------|-------|--------|-------------|
| MegaDetector v6 (YOLOv10-E) | General animal/person/vehicle detection | 1280×1280 RGB | `[1, 300, 6]` — (cx, cy, w, h, conf, class) with in-graph NMS | Confidence threshold, normalize bbox |
| MegaDetector v5a | Earlier MegaDetector, Sparrow Studio Local legacy support | Model-dependent | NMS-in-graph detections | Same |
| DeepFaune (DFNE) | Species detection (European camera traps) | 640×640 RGB | NMS-in-graph detections | Same |
| HerdNet | Overhead-imagery animal counting (dual-output heatmap) | Tiled input (model's native crop) | 2 heatmap outputs | Peak detection across tiles, greedy center-proximity dedup, overhead-dot rendering |
| OWL-T | Overhead-imagery single-class detector (single-output heatmap) | Tiled input | 1 heatmap output | Adaptive threshold, tile overlap dedup (same greedy center-proximity algorithm) |
| SpeciesNet Crop | Species classification (crop post-detection) | Variable crop, resized internally | Softmax over classes | Top-K classification (default K=5) |
| AI4G classifier | Species classification, AI for Good Lab first-party | Variable crop | Softmax | Top-K classification |
| MD_AudioBirds_V1 | Bird audio classifier (sliding-window) | Mel spectrogram: n_fft=1024, hop=512, n_mels=224, sr=48000 | Sigmoid per class per window | Per-window sigmoid, threshold, window merge |

## Freshness anchor (2026-05-13)

The "Representative set" table above is anchored at 2026-04-29; the Phase 4.1 manual-test matrix (`docs/master_plan.md § Phase 4.1`) cites the live onboarded set as MegaDetector v6, DeepFaune, HerdNet, OWL-T, Amazon Camera Trap v2, MD_AudioBirds_V1 (FP16 default for all except DeepFaune-on-FP32-HOLD per P3.8-7). The MegaDetector v5a row above is historical / Sparrow-Studio-Local-legacy; current Phase-4.1 testing exercises the post-Phase-3.8 set.

Onboarding a new model: drop the ONNX file into `{model_dir}/{new_id}/` with a `manifest.toml` describing preprocess method, input size, layout (must be NCHW — § 06 gotchas), normalization, postprocess method. No libsparrow_engine code change. `spe models verify` checks the in-manifest SHA-256 against the ONNX file; the new model becomes available to `spe detect`, HTTP `/v1/detect`, and all other consumer surfaces.

Golden-output integration tests run against reference files in `test_files/` for the Sparrow Studio Local production set. Float tolerance: `bbox ±0.005`, `confidence ±0.12` (vision) / `±0.25` (audio) — the f32/f64 precision gap. Source: `docs/master_plan.md`.

## Preprocessing pipeline

Image path, end to end, for the common letterbox + NCHW case:

```
  image::open(path)
    → image::RgbImage
    ↓
  letterbox (resize + pad to input_size, preserving aspect; pad_value from manifest)
    → letterboxed RgbImage
    ↓
  normalize (manifest choice: "unit" = /255.0, "imagenet" = mean/std)
    → f32 array
    ↓
  NCHW pack (HWC → CHW)
    → ndarray::Array4<f32> shape [1, 3, H, W]
    ↓
  into ORT Session::run
```

Source: `sparrow-engine/libsparrow_engine/src/preprocess.rs`.

### Letterbox rounding

Letterbox has a subtle rounding choice: where do odd-pixel margins go? sparrow-engine rounds padding top/left rather than distributing it symmetrically. This produces a 1-detection difference vs PIL-based Python preprocessing in some edge cases (217 vs 218 in the 100-image Phase 2.5 benchmark). Golden outputs use sparrow-engine's rounding as the reference.

### Audio preprocessing

```
  hound::WavReader::open(path)
    → samples: Vec<i16> or Vec<f32> (casts)
    ↓
  rubato::FftFixedInOut::new(src_sr → 48000) if resampling needed
    → resampled Vec<f32>
    ↓
  window into 1.0 s segments with 0.3 s stride
    → Vec<Vec<f32>>, each 48000 samples
    ↓
  realfft::RealFftPlanner → STFT (n_fft=1024, hop=512)
    → Vec<Complex<f32>>
    ↓
  mel filterbank (n_mels=224, manually computed once)
    → ndarray [1, 224, T] shape for each window
    ↓
  into ORT Session::run
```

Source: `sparrow-engine/libsparrow_engine/src/preprocess_audio.rs`. Dependencies: `hound` (WAV decode), `realfft` (FFT for STFT), `rubato` (resampling).

## Postprocessing

### Detection

- Filter by confidence threshold from `DetectOpts` or manifest default.
- NMS already in-graph — never re-run.
- Denormalize bbox from output coordinate space to `[0, 1]` relative to original image (not letterboxed).
- Return `Vec<Detection>`.

### Classification

- Apply softmax if manifest says so (otherwise sigmoid or raw).
- Sort by confidence descending.
- Return top-K (default K=5, configurable).

### Tiled detection (HerdNet, OWL-T)

- Split input into model-native tiles with overlap.
- Run inference per tile.
- For each tile, find peaks in the heatmap(s).
- Merge peaks across tiles via greedy center-proximity suppression — keep the highest-confidence peak within a radius threshold.
- Return `Vec<Detection>` with overhead-dot rendering hints. Phase 3.5 S3 (item #3, MT-9 correctness fix) made this a first-class manifest subtype: `ModelSubtype::{Standard, Overhead}` in `libsparrow_engine/src/types.rs`, dispatched in `viz::render()`. Landed 2026-04-23 (W2 R5 audit-fix CONVERGED).

### Audio classification

- Sigmoid per class per window.
- Threshold (default from manifest).
- Merge consecutive high-confidence windows into contiguous segments. Phase 3.5 S5 (item #6) flipped `detect-audio` default output to merged confidence ranges (default threshold 0.5, merge gap stride+1ms); `--raw-segments` opts INTO the per-window format. Landed 2026-04-23 (W2 R5 audit-fix CONVERGED).
- Return `Vec<AudioSegment>`.

## Engine lifecycle

```
  Engine::new(EngineConfig)
    ├─ atomic swap on ENGINE_EXISTS — fail if already true
    ├─ resolve device: Device::Auto → CUDA (if compiled in) or CPU
    ├─ build ORT SessionBuilder template (configured once, cloned per model)
    └─ return Engine { inner: Arc<EngineInner>, models: RwLock<HashMap>, pipelines: Mutex<HashMap>, loading_lock: Mutex<()> }

  Engine::get_or_load_model(model_id)
    ├─ read models RwLock — hit: return ModelHandle
    ├─ miss: acquire loading_lock
    ├─   read models RwLock — double-check (prevents TOCTOU double-load)
    ├─   load manifest from {model_dir}/{model_id}/manifest.toml
    ├─   validate manifest (NCHW check, NMS check for detectors, etc.)
    ├─   create ORT session from manifest + model file
    ├─   insert into models RwLock
    └─ return ModelHandle

  Engine::run_pipeline_adhoc(img, detector="mdv6", classifier="speciesnet")
    ├─ get_or_load_model(detector)
    ├─ get_or_load_model(classifier)
    ├─ detect
    ├─ for each detection, crop
    ├─ classify each crop
    └─ return Vec<PipelineResult>

  Drop for Engine
    └─ ENGINE_EXISTS.store(false)
```

Source: `sparrow-engine/libsparrow_engine/src/engine.rs`.

### Session concurrency

`ort::Session::run` takes `&mut self`. Wrapping sessions in `std::sync::Mutex` serializes inference per model (one call at a time per model). Multiple models can run concurrently. This matches ORT's own concurrency guarantees and avoids the need for a session pool at the library level.

### Why `std::sync::Mutex` and not `tokio::sync::Mutex`

Inference is CPU-bound (GPU-bound from the caller's perspective). Holding a mutex across await points is what `tokio::sync::Mutex` exists for. Inference calls are synchronous and do not await anything. `std::sync::Mutex` is the right choice — zero runtime overhead, no cross-runtime dependency. See `~/.claude/rules/rust.md` § Async Runtime.

### `spawn_blocking` for async contexts

sparrow-engine-server is async (axum / tokio). ORT inference is blocking CPU-bound work. Handlers wrap `engine.detect(...)` in `tokio::task::spawn_blocking` so the async runtime isn't blocked. Source: `sparrow-engine/sparrow-engine-server/src/handlers/detect.rs`.

## FFI safety

### Opaque pointer lifecycle

```
  C consumer                              Rust libsparrow_engine
  ──────────                              ─────────────
  sparrow_engine_engine_new(cfg)
                          ───────>       Engine::new(...)
                                         → Box<Engine>, leak to raw pointer
  p_engine <── raw pointer ──

  sparrow_engine_engine_load_model(p_engine, "mdv6")
                          ───────>       &*p_engine → engine.load_model("mdv6")
                                         → ModelHandle::new(weak, active, session, ...)
                                         → Box<ModelHandle>, leak to raw pointer
  p_model <── raw pointer ──

  sparrow_engine_detect(p_engine, p_model, img, opts)
                          ───────>       Safety check path:
                                         1. engine_ref.upgrade() — fails if Engine dropped
                                         2. active.load() — false if unloaded
                                         3. either fail → return error code
                                         4. else → session.lock(), run(), unlock
                          <──── results ──

  sparrow_engine_model_free(p_model)
                          ───────>       Box::from_raw(p_model).drop()
                                         → decrements Arc<Session> refcount
                                         → ModelHandle freed
  sparrow_engine_engine_free(p_engine)
                          ───────>       Box::from_raw(p_engine).drop()
                                         → ENGINE_EXISTS.store(false)
```

No raw pointer dereference path exists that skips the upgrade + active check. Every FFI inference function starts with those two checks.

### Why a `Mutex` inside the `Arc` for `Session`

`ort::Session` is `!Send + !Sync` because the underlying C bindings hold raw pointers. Wrapping in `Mutex` gives us `Send + Sync` and serializes calls. We carry `Arc<Mutex<Session>>` so multiple handles can share the session for a single model (standard session caching), and the mutex ensures ORT sees only one `run()` call at a time.

The `unsafe impl Send for LoadedModel {}` and `unsafe impl Sync for LoadedModel {}` in `engine.rs:128-129` are justified in the comment adjacent: "Session is behind Mutex. All other fields are Arc-wrapped or plain data."

## Python GIL handling

PyO3 default is to hold the GIL for the entire duration of a Rust function. For inference, that would block Python threads unnecessarily — ORT's C library releases cross-GIL.

sparrow-engine-python releases the GIL during inference:

```rust
// Illustrative — pattern is what matters, not the exact signature.
fn detect(py: Python, img: &PyAny, opts: DetectOpts) -> PyResult<Vec<Detection>> {
    let input = read_image_from_pyany(img)?;
    let result = py.allow_threads(|| {
        engine.detect(&input, &opts)
    })?;
    Ok(result)
}
```

Source pattern: `~/.claude/rules/rust.md` § PyO3 Bindings. Actual call sites: `sparrow-engine/sparrow-engine-python/src/lib.rs`.

### No `println!` from Rust in sparrow-engine-python

PyO3 issue #2247: `println!` from Rust is invisible in Jupyter (Jupyter redirects Python stdout but not Rust stdout). sparrow-engine-python routes all diagnostic output through Python logging via `pyo3-log` — post-Phase 3.5 S6 (2026-04-23), zero `eprintln!` sites remain in `sparrow-engine-python/src/`; `scripts/guard_no_print.sh` (Phase 3.5 S2 CI grep guard) enforces this going forward.

## Worker types (Sparrow Studio Web)

Three worker types are shipped. All speak HTTP to `sparrow-engine-server` — none link libsparrow_engine directly.

| Worker | Language | Size | Dependency | Use case |
|--------|----------|------|------------|----------|
| `worker_rust` | Rust (reqwest) | ~230 LOC, 15 MB image | `sparrow-engine-server` via HTTP | Default. Fastest for GPU workloads (reqwest beats httpx on multipart). |
| `worker_python` | Python (httpx) | ~? LOC | `sparrow-engine-client` | Teams already running Python infrastructure |
| `worker_local` | Python (tritonclient) | — | Triton gRPC | Legacy. Kept for backward-compat during cutover only. |

Benchmark difference (GPU, 100 images, end-to-end):

| Worker | Total | Per-image |
|--------|-------|-----------|
| worker_rust + Sparrow Engine GPU | 5.8 s | 58 ms |
| worker_python + Sparrow Engine GPU | 7.3 s | 73 ms |

The 15 ms/image difference is HTTP client overhead: reqwest vs httpx on `multipart/form-data` upload. On CPU workers (~2 s/image) the difference is noise.

## Export formats

### MegaDet v1.5 JSON

Schema verified from upstream `run_detector_batch.py`:

```json
{
  "info": {"detector": "...", "detection_completion_time": "..."},
  "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
  "images": [
    {
      "file": "path/to/image.jpg",
      "detections": [
        {"category": "1", "conf": 0.93, "bbox": [0.12, 0.34, 0.05, 0.08]}
      ]
    }
  ]
}
```

Bbox: `[x_min, y_min, width, height]` normalized `[0, 1]`. Source: `sparrow-engine/libsparrow_engine/src/export.rs` `to_megadet`.

### COCO JSON

Standard COCO detection format. `category_id` convention is `>= 1` (0 is reserved in `pycocotools`). Phase 3 final audit-fix R3 MI-2 caught a pipeline-specific bug: when a pipeline mixes classifier-labeled and detector-labeled rows (SpeciesNet classifier label_id + MD v6 detector label_id, both 1-indexed), the `seen_categories.entry().or_insert_with()` silently drops the second `(label_id, label)` pair. Fix (commit `ee01898`): emit `eprintln!` warn on collision with first-seen semantics; `HashSet<u32>` for one-shot warn dedup; doc subsection on the invariant and the Phase 3.5 full-namespace-strategy follow-up.

### CSV

RFC 4180 compliant. `export::csv_escape` handles commas, double quotes, newlines. Same escape function is used by both `--format csv` (inline output) and `--export-format csv` (file output) — single source of truth for escaping.

## CLI behavior

11 commands (`detect`, `classify`, `detect-audio`, `pipeline`, `models list`, `models info`, `models verify`, `device`, `init`, `hash`, `day-night`).

### Exit code discipline

| Exit code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | Per-file error on inference commands (some files failed) |
| 1 | Empty input (no files matched the pattern) |
| 1 | Other runtime errors |
| 141 | SIGPIPE (`spe ... \| head`) — clean exit, not a "fatal error" |

SIGPIPE handling: CLI resets SIGPIPE to `SIG_DFL` in `main()` so `spe detect *.jpg | head` exits 141 cleanly rather than printing "broken pipe: fatal runtime error". Source: `commit 8c60050`.

### `--recursive`

All four inference commands accept `--recursive` for directory inputs. Symlink cycle detection via `canonicalize` + visited set — protects against infinite descent into symlink loops.

### `--visualize --output-dir`

Mutually required (clap `requires`). `--visualize` without `--output-dir` errors. Directory mirroring via longest-common-prefix of input paths. Output filename: `{stem}_viz.{ext}` preserving the input's encoded format.

### `--export-format megadet|coco|csv --export-output <path>`

Suppresses inline JSON/CSV output when export is active. Parent directory created if missing (R1 reviewer fix H4).

## Python package API (Phase 2.5 + Phase 3)

14 public functions. 8 MVP (shared with CLI), 6 Phase 3 standalone.

| MVP (shared with CLI) | Phase 3 (Python-only standalone) |
|------------------------|-----------------------------------|
| `detect` | `hash_file` |
| `classify` | `day_night` |
| `detect_audio` | `verify_model` |
| `pipeline` | `summarize` |
| `list_models` | `visualize` |
| `model_info` | `export` |
| `active_device` | |
| `init` | |

Phase 3 standalone functions route to `#[pyfunction]` in `_sparrow_engine_core`, bypassing `_get_engine()`. They work on installs that don't have ORT available.

### `visualize()` two-mode design

`spe.visualize(items)` returns `list[bytes]` (PNG/JPEG-encoded image data in memory).
`spe.visualize(items, output_dir=...)` writes files to disk as a side effect.

Same rendering, different persistence. Jupyter / HTTP / pipeline use cases want bytes in memory; CLI-style scripts want files on disk. Forcing either one would be wrong for half of consumers.

### `export()` type dispatch

`spe.export(results, format="coco")` accepts `list[(path, DetectResult | PipelineResult)]` and returns `str`. `PipelineResult` auto-converts to `DetectResult` with the classification label preferred over the detector's class name. `model_id` is required for `megadet` format (MegaDet v1.5 schema requires it).

## Confidence

**Confidence**: HIGH
- Factual accuracy: HIGH — model list from master_plan, preprocessing details from libsparrow_engine source, FFI pattern from engine.rs:180-200
- Completeness: MEDIUM — subsystem coverage is broad but not exhaustive; individual deep-dives (e.g., full HerdNet tiling algorithm) defer to source
- Freshness: HIGH — 2026-04-29 (R9 R-S1 + R-S2 substantively edited L96 + L102 — Phase 3.5 S3 model-subtype dispatch landed 2026-04-23 + S5 audio merge framing + W2 R5 audit-fix convergence; matches HEAD)

## References

- `sparrow-engine/libsparrow_engine/src/` — all module sources
- `sparrow-engine/libsparrow_engine/src/engine.rs:112-252` — Engine singleton + new
- `sparrow-engine/libsparrow_engine/src/engine.rs:174-200` — ModelHandle safety pattern
- `sparrow-engine/libsparrow_engine/src/preprocess.rs`, `preprocess_audio.rs`, `postprocess.rs`
- `sparrow-engine/libsparrow_engine/src/export.rs` — three export formats + csv_escape
- `sparrow-engine/sparrow-engine-cli/src/main.rs` — 11 CLI commands
- `sparrow-engine/sparrow-engine-python/src/lib.rs` — PyO3 bindings, GIL handling
- `docs/master_plan.md:95-101` — Phase 2.5 engine benchmark
- `docs/benchmarks.md` — Phase 2 end-to-end benchmarks
- Phase 3 final audit-fix R4 inquisitor review for MI-2 fix details
