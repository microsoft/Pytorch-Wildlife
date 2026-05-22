# Architecture

> **PARTIALLY SUPERSEDED (2026-04-29)** — the 3-consumer architecture diagrammed in this chapter (libsparrow_engine → sparrow-engine-server / sparrow-engine-cli / sparrow-engine-python; Sparrow Studio Local + Sparrow Studio Web + future Avalonia as the 3 consumer boundaries) is **superseded** by the 5-component architecture ratified in Phase 3.7 Track A (Sparrow Studio + sparrow-engine + sparrow-data + sparrow-ops + fine-tuning repo). For the current architecture, see [`docs/design/architecture.md`](../design/architecture.md). The libsparrow_engine internals + workspace shape + crate boundaries documented below remain accurate; the consumer/sibling-repo decomposition does not.
>
> **Update 2026-05-13**: the "libsparrow_engine (Rust core, rlib + optional cdylib)" top-level diagram below is also superseded by the Phase 3.8 Phase A crate-carve (2026-05-02). Workspace is now **7 crates**: `sparrow-engine-types` + `sparrow-engine-core` + `sparrow-engine-cpu` + `sparrow-engine-gpu` + `sparrow-engine-server` + `sparrow-engine-cli` + `sparrow-engine-python`. Both `sparrow-engine-cpu` and `sparrow-engine-gpu` ship cdylibs as `libsparrow_engine.so` (Phase C Wave 4b lock); see `docs/design/architecture.md` for the current state and `docs/master_plan.md § Phase 3.8` for the carve history. Chapter body retained as audit trail of the pre-Phase-3.8 shape.

System shape at the crate, module, and process-boundary level.

## Top-level diagram

```
          libsparrow_engine (Rust core, rlib + optional cdylib)
      +----------------------------------------------+
      |  Inference engine, preprocessing,            |
      |  postprocessing, TOML manifests, FFI,        |
      |  utilities (hash/daynight/viz/export/stats)  |
      +----------------------+-----------------------+
                             |
     +---------+------+------+-------+---------+----------+
     |         |             |       |         |          |
     v         v             v       v         v          v
sparrow-engine-server  sparrow-engine-cli  sparrow-engine-python  C P/Invoke  HTTP   (future)
 (axum)      (`sparrow-engine`)   (PyO3 0.25)  (Sparrow    (Sparrow  Avalonia
  15 endpoint ~35 MB      CPU/GPU     Studio      Studio   desktop
  surface     static      wheels      Local,      Web, 3   .NET app
              ORT                     C# P/Invoke workers)
                                      NuGet DLL)
```

## Workspace

**Pre-Phase-3.8-Phase-A snapshot** — at the time of writing (2026-04-21), four Rust crates + one pure-Python package; a 5th workspace crate (`sparrow-utils`) was planned for Phase 4+ per Phase 3.7 Track A (see ch 04 D-v4-4). **Post-Phase-3.8-Phase-A (2026-05-02 onward)**: 7 crates total — see `docs/design/architecture.md`. `sparrow-utils` remains a Phase-4+ idea (not landed; see `docs/master_plan.md` and `docs/ideas.md`).

| Crate / package | Purpose | Produces |
|-----------------|---------|----------|
| `libsparrow_engine` | Core inference library | `rlib` always; `cdylib` + `staticlib` when `ffi` feature is enabled |
| `sparrow-engine-server` | HTTP thin shell over libsparrow_engine | binary for Docker images |
| `sparrow-engine-cli` | Rust CLI binary | `spe` binary, ~35 MB with static ORT |
| `sparrow-engine-python` | PyO3 0.25 Rust bindings | `_sparrow_engine_core` Python extension module, consumed by the `sparrow-engine` Python package |
| `sparrow-engine-client` | Python HTTP SDK (pure Python, not a Rust crate) | `sparrow_engine_client` pip package for Sparrow Studio Web workers |

Workspace manifest: `sparrow-engine/Cargo.toml`. Individual manifests: `sparrow-engine/libsparrow_engine/Cargo.toml`, `sparrow-engine/sparrow-engine-server/Cargo.toml`, `sparrow-engine/sparrow-engine-cli/Cargo.toml`, `sparrow-engine/sparrow-engine-python/Cargo.toml`. sparrow-engine-client is at `sparrow-engine/sparrow-engine-client/` with its own `pyproject.toml`.

**Dependency DAG** (strict, no cycles):

```
libsparrow_engine  →  (leaf, no workspace deps)
sparrow-engine-server  →  libsparrow_engine
sparrow-engine-cli     →  libsparrow_engine
sparrow-engine-python  →  libsparrow_engine
sparrow-engine-client  →  httpx  (pure-Python, no libsparrow_engine dependency; HTTP protocol is the interface)
```

All three Rust consumer crates link libsparrow_engine directly as a path dependency. `sparrow-engine-client` is deliberately decoupled — it speaks sparrow-engine-server's HTTP API, so upgrading `sparrow-engine-server` only requires upgrading `sparrow-engine-client` if the HTTP schema changes.

## libsparrow_engine module structure

```
sparrow-engine/libsparrow_engine/src/
  lib.rs                 — module surface + re-exports (Device, Engine, EngineConfig,
                           ModelHandle, SparrowEngineError, Result, and all types::*)
  engine.rs              — ORT singleton, session management, model loading
  manifest.rs            — TOML manifest parsing and validation
  types.rs               — public structs: Detection, BBox, Classification,
                           AudioSegment, Result variants, ModelInfo, ModelType
  error.rs               — SparrowEngineError enum, Result<T> alias

  preprocess.rs          — Image decode → letterbox/resize → normalize → NCHW pack
  preprocess_audio.rs    — WAV decode → resample → mel spectrogram → sliding windows
  postprocess.rs         — Confidence filtering, softmax, heatmap peak finding

  detect.rs              — High-level detection API
  classify.rs            — High-level classification API
  detect_audio.rs        — High-level audio classification API
  pipeline.rs            — Chained detect → crop → classify

  hash.rs                — Phase 3: SHA-256 file hashing
  daynight.rs            — Phase 3: BT.709 luma brightness heuristic
  viz.rs                 — Phase 3: unified visualization pipeline
  stats.rs               — Phase 3: batch detection statistics
  export.rs              — Phase 3: MegaDet v1.5 JSON + COCO JSON + CSV
  catalog.rs             — Phase 3: model integrity verification

  ffi.rs                 — C FFI boundary (behind `ffi` feature flag)
```

The `types.rs` surface is what every consumer sees. Detection, BBox, Classification, AudioSegment, Result. Everything else is implementation detail.

## Consumer model

Six consumer surfaces, all hitting libsparrow_engine through exactly one of three boundaries.

### Boundary 1: Rust direct link

Crates that link libsparrow_engine as a path dependency.

- **sparrow-engine-server** — axum HTTP service, 15 endpoint/method surface after Phase 4.2 (`GET /v1/catalog` and `POST /v1/pipelines` extend the Phase 2 list). Thin shell over libsparrow_engine. Inference handlers call libsparrow_engine; response serialization in `serde` JSON. Container lifecycle: `init: true` + Docker restart policy + internal health watchdog. No supervisord.
- **sparrow-engine-cli** — clap-based CLI, 11 commands. Outputs JSON (default), CSV, or MegaDet v1.5 JSON. Ships as a single static binary via GitHub Releases. Static ORT is linked in.

### Boundary 2: FFI (C ABI)

When the `ffi` feature flag is enabled, libsparrow_engine also produces `cdylib` (`libsparrow_engine.so` / `sparrow-engine.dll`) and `staticlib`. The C header is generated by `cbindgen` from Rust source at build time and is named `sparrow_engine.h`. A matching C# binding header is generated by `csbindgen` and lands in the Sparrow Studio Local NuGet package.

**FFI structs** have no reserved fields. ABI evolution is handled by function versioning: a `sparrow_engine_detect` that needs a new argument becomes `sparrow_engine_detect_v2`.

**Safety pattern for model handles.** Detection APIs return opaque pointers. The struct behind the pointer holds:

```rust
// Illustrative shape of ModelHandle — exported across FFI boundary as opaque pointer.
pub struct ModelHandle {
    pub(crate) engine_ref: Weak<EngineInner>,       // fails to upgrade if engine dropped
    pub(crate) active: Arc<AtomicBool>,             // set to false on unload; checked before inference
    pub(crate) session: Arc<Mutex<Session>>,        // ORT session; Mutex because ort::Session is &mut on run()
    pub(crate) manifest: Arc<ModelManifest>,
    pub(crate) labels: Arc<Vec<String>>,
}
```

Source: `sparrow-engine/libsparrow_engine/src/engine.rs:174-190`.

Use-after-free and use-after-unload are closed by checking `Weak::upgrade()` and `active.load()` before every inference call. If either fails, the FFI function returns an error code instead of dereferencing freed memory.

**9 Phase 3 FFI exports** added on top of the Phase 1 / Phase 2 set:

| Category | Functions |
|----------|-----------|
| Standalone (no Engine needed) | `sparrow_engine_hash_file`, `sparrow_engine_day_night`, `sparrow_engine_image_brightness`, `sparrow_engine_verify_model`, `sparrow_engine_hash_result_free`, `sparrow_engine_verify_result_free` |
| Engine wrappers | `sparrow_engine_engine_verify_model`, `sparrow_engine_engine_model_info`, `sparrow_engine_engine_list_models_extended` |

Consumers: Sparrow Studio Local (C# P/Invoke) and any future native desktop client (e.g., planned Avalonia .NET app for macOS/Linux).

### Boundary 3: HTTP

sparrow-engine-server exposes a 15 endpoint/method HTTP surface. JSON request and response. `multipart/form-data` for image upload on inference endpoints.

Consumers:

- **sparrow-engine-client** — pure-Python HTTP SDK (~385 LOC, 20 tests, later 21 after SRV1 regression test). Depends on `httpx`. Used by Sparrow Studio Web workers. Sync and async APIs.
- **worker_rust** — Rust worker, ~230 LOC, 15 MB Docker image. Uses `reqwest`. Default Sparrow Studio Web worker.
- **worker_python** — Python worker. Uses `sparrow-engine-client`. For teams already running Python infrastructure.
- **worker_local** — legacy Triton gRPC worker. Not a sparrow-engine consumer; kept for backward-compat during cutover.

### PyO3 Rust-Python in-process binding

Outside the three boundaries above, `sparrow-engine-python` uses PyO3 to link libsparrow_engine directly into the Python process. Not HTTP, not P/Invoke. Just Rust-in-Python.

- The Rust extension module is named `_sparrow_engine_core`. It exports functions and classes consumed by `sparrow-engine/__init__.py`.
- Default feature is `extension-module` — PyO3 does not link libpython, because Python will load the extension at import time. For `cargo test`, tests build with `--no-default-features` so libpython is linked.
- GIL released during ORT inference via `py.allow_threads` per the `~/.claude/rules/rust.md` PyO3 rule.
- `sparrow-engine-python` exposes the same 8 MVP functions as `sparrow-engine-cli` (functionality consistency rule; see § 04). Phase 3 adds 6 standalone Phase 3 functions that do not require an Engine (`hash_file`, `day_night`, `verify_model`, `summarize`, `visualize`, `export`).

## Data flow

Single inference call, start to finish:

```
  Consumer (CLI / HTTP / Python / C#)
              │
              ▼
  Engine.{detect|classify|detect_audio|run_pipeline_adhoc}(model_id, input, opts)
              │
              ▼                              ┌── Phase 3 standalone calls ──┐
  Engine → get_or_load_model(model_id)       │   hash_file                  │
              │                              │   day_night                  │
              ▼                              │   verify_model               │
  (lazy load: first call resolves            │   visualize                  │
    {model_dir}/{id}/manifest.toml,          │   summarize                  │
    loads ORT session, caches)               │   export                     │
              │                              └──────────────────────────────┘
              ▼
  preprocess::image::decode_and_letterbox
    (or preprocess_audio for audio)
              │
              ▼
  ORT Session::run (NCHW tensor → outputs)
              │
              ▼
  postprocess (confidence threshold,
    NMS already in-graph — never re-applied,
    softmax for classifiers,
    tile merge for HerdNet/OWL-T)
              │
              ▼
  Result: list[Detection] | list[Classification]
    | list[AudioSegment] | list[PipelineResult]
              │
              ▼
  Consumer receives typed Result.
```

Sources: `sparrow-engine/libsparrow_engine/src/lib.rs`, `sparrow-engine/libsparrow_engine/src/engine.rs`, `sparrow-engine/libsparrow_engine/src/detect.rs`.

## Interface ownership

From v4 consensus design: **sparrow-engine defines the interface. Sparrow Studio adapts to it.**

Implication: when Sparrow Studio Local or Sparrow Studio Web needs a feature, the request goes through a sparrow-engine API proposal, not a Sparrow-side workaround. This keeps libsparrow_engine the single place where pre/post logic lives.

Concrete examples:

- **MT-11** (Phase 3 manual testing): Sparrow Studio CLI displayed the extended `ModelInfo` fields (`version`, `description`, `onnx_sha256`, `onnx_size_bytes`, `default`). The Python package did too. But the HTTP response struct `ModelResponse` in `sparrow-engine-server/src/handlers/models.rs` was still at the Phase 2 shape. Audit-fix R1 found this. The fix was to extend `ModelResponse`, not to have Sparrow Studio Web workers maintain a separate enrichment step.
- **Audit-fix R2** caught a downstream breakage: Sparrow Studio Web workers unpack `ModelInfo(**m)` strictly and rejected the new `default: bool` field. Fix was `sparrow-engine-client` dataclass extension + backward-compat regression test. Again, the fix stayed at the interface; Sparrow-side code did not need to change.

## Deployment matrix

| Target | Package | Runtime requirements |
|--------|---------|----------------------|
| Sparrow Studio Local (desktop — Windows today, macOS/Linux via Avalonia port in progress) | `libsparrow_engine` cdylib (`sparrow-engine.dll` / `libsparrow_engine.dylib` / `libsparrow_engine.so`) + `sparrow_engine.h` + C# bindings via NuGet | .NET 10+ on any supported OS with ORT runtime bundled |
| Sparrow Studio Web (server) | `sparrow-engine-server` Docker image, CPU (163 MB) or GPU (~4 GB) | Docker; GPU image needs cuDNN runtime base |
| Field researcher / notebook | `pip install sparrow-engine` (wheel) | Python ≥ 3.10; CPU wheel uses load-dynamic + pip `onnxruntime` |
| Field CLI / cluster | `spe` binary via GitHub Releases | None — static ORT in the binary, ~35 MB |
| Sparrow Studio Web workers | `sparrow-engine-client` pip package | Python, HTTP access to `sparrow-engine-server` |
| Future Avalonia desktop | `libsparrow_engine` cdylib + bindings | .NET 10+; cross-platform |

## What is explicitly not in libsparrow_engine

Things the design keeps out of the core library so the library stays model-agnostic and small:

- **Model download.** `tools/download_models.py` for manual download. Python package and CLI integrate download helpers outside libsparrow_engine.
- **Database writes, annotation storage, query logs.** Phase 4 scope; sparrow-engine-side primitives (`?store=true`, write-only async path) stay in sparrow-engine, but the storage itself moves to the `sparrow-data` sibling repo per Phase 3.7 Track A (deferred construction) — the former in-sparrow-engine `sparrow-engine-db` sidecar reference is superseded; see chapter banner + `docs/design/architecture.md`.
- **Fine-tuning pipeline.** Separate workstream (§ 13).
- **Model-specific glue code.** Models are self-describing via TOML manifests. Adding a new model should be a manifest plus an ONNX file, not a code change in libsparrow_engine.

## Confidence

**Confidence**: HIGH
- Factual accuracy: HIGH — crate layout verified via `ls`, module list verified via `lib.rs`, FFI struct shape quoted from `engine.rs:174-190`
- Completeness: HIGH — covers all three consumer boundaries plus the PyO3 in-process case
- Freshness: HIGH — 2026-04-29 (R9 C1 workspace + sparrow-utils framing applied; R12 ch 03 L218 `sparrow-engine-db` sidecar → `sparrow-data` sibling Phase 3.7 Track A forward-pointer; matches current HEAD)

## References

- `sparrow-engine/Cargo.toml` — workspace manifest
- `sparrow-engine/libsparrow_engine/src/lib.rs` — module surface
- `sparrow-engine/libsparrow_engine/src/engine.rs:150-190` — Engine and ModelHandle definitions
- `docs/design/v4/libsparrow_engine/consensus_design_revised.md` — v4 consensus design
- `docs/design/v4/libsparrow_engine/design_report.md` — v4 design report
- `docs/design/phase3/final_design.md` — Phase 3 module layout decisions
- `04_design_decisions.md` — Why each architectural choice was made
