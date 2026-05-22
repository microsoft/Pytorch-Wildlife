//! Phase 3.8 Phase C Wave 3 — engine flavor dispatch shim.
//!
//! Compile-time dispatch between `sparrow-engine-cpu` and
//! `sparrow-engine-gpu` via the `cpu` / `gpu` Cargo features
//! (mutually exclusive, default = `cpu`).
//! Re-exports the active engine crate's full public surface from the
//! `sparrow_engine` crate link so the rest of the CLI (`main.rs`) can
//! keep using its local `engine_dispatch::Engine` / `engine_dispatch::viz::*` /
//! `engine_dispatch::types::*` alias without per-call rewrites.
//!
//! Mutual exclusivity and presence are enforced at the binary entry
//! point (`main.rs`) via `compile_error!` so a misconfigured feature
//! set fails loud at build time.
//!
//! Pattern source: `docs/design/phase3.8/phase_c/implementation_plan.md`
//! §2.1 + Wave 3 deliverables.

// Both `sparrow-engine-cpu` and `sparrow-engine-gpu` now set
// `[lib] name = "sparrow_engine"` (cdylib `libsparrow_engine.so`
// invariant per implementation_plan.md §2.2). Cargo features above
// ensure exactly one engine crate is in the dependency graph at a time,
// so `pub use ::sparrow_engine::*` resolves to the active engine. This
// shim is intentionally cfg-free now — the mutex check is enforced in
// main.rs.
//
// The glob picks up the active engine's `Engine`, `ModelHandle`,
// `detect`, `classify`, `detect_audio`, `pipeline`, `preprocess`,
// plus the `sparrow-engine-core` / `sparrow-engine-types` re-exports
// (`viz`, `types`, `export`, `stats`, `catalog`, `hash`, `daynight`,
// `preprocess_audio`, `SparrowEngineError`, `Device`, `EngineConfig`,
// `ModelType`, `ImageInput`, `AudioInput`, `DetectOpts`,
// `DetectResult`, `ClassifyOpts`, `ClassifyResult`, `AudioDetectOpts`,
// `AudioDetectResult`, `PipelineResult`, etc.) via
// `pub use sparrow_engine_core::*; pub use sparrow_engine_types::*;`
// at the engine crate's `lib.rs`.
pub use sparrow_engine::*;
