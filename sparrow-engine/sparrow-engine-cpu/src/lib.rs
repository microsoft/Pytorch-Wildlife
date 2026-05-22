//! sparrow-engine-cpu — Core ML inference library for wildlife conservation (CPU pipeline).
//!
//! Load ONNX models via TOML manifests, send images, get structured detections
//! or classifications back. sparrow-engine-cpu owns the CPU/CUDA-EP inference pipeline
//! (preprocessing, ORT singleton + session management, postprocessing dispatch).
//!
//! # Architecture (Phase 3.8 Phase A workspace layout)
//!
//! - `sparrow-engine-types` (sibling crate) — pure-data types: `types`, `error`,
//!   `manifest`, `Device`, `EngineConfig`, `derive_model_type`,
//!   `PreprocessMeta`, `PreprocessConfig`, `AudioRange`. Re-exported here
//!   via `pub use sparrow_engine_types::*;` so existing `sparrow_engine::*` paths keep
//!   resolving.
//! - `sparrow-engine-core` (sibling crate) — device-agnostic logic: `hash`,
//!   `daynight`, `stats`, `viz`, `export`, `catalog`, `postprocess`,
//!   `preprocess_audio`. Re-exported here via `pub use sparrow_engine_core::*;`.
//! - `engine` (this crate) — ORT singleton, session management, model loading
//! - `preprocess` (this crate) — image decode, letterbox/resize, normalization
//! - `detect` / `classify` / `pipeline` / `detect_audio` (this crate) — high-level inference
//! - `ffi` (this crate) — C FFI boundary (behind `ffi` feature flag)
//!
//! The cdylib filename remains `libsparrow_engine.so` / `sparrow_engine.dll` / `libsparrow_engine.dylib`
//! per C8 closure ([lib] name = "sparrow_engine" after the rename).

pub mod classify;
pub mod detect;
pub mod detect_audio;
pub mod engine;
pub mod pipeline;
pub mod preprocess;

#[cfg(feature = "ffi")]
pub mod ffi;

// Phase 3.8 Phase A S2 closure: glob re-exports from sparrow-engine-types and
// sparrow-engine-core so existing `sparrow_engine::SparrowEngineError`, `sparrow_engine::Device`,
// `sparrow_engine::hash`, `sparrow_engine::types::*`, etc. keep working without
// per-symbol re-export shims.
pub use sparrow_engine_core::*;
pub use sparrow_engine_types::*;

// Engine-side re-exports (Engine + ModelHandle wrap ORT sessions and stay
// in sparrow-engine-cpu).
pub use engine::{Engine, ModelHandle};
