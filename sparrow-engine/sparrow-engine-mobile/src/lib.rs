//! sparrow-engine-mobile — manifest-driven mobile inference flavor (LiteRT/TFLite).
//!
//! The Raspberry Pi/mobile peer of `sparrow-engine-cpu` and `sparrow-engine-gpu`:
//! the generic, manifest-driven [`engine::Engine`] on a LiteRT CPU backend, plus
//! single-model audio detection and a config-described audio cascade ([`pipeline`])
//! that replaces the previously hardcoded orca C exports (RP-25-FU-1).
//!
//! Errors use `anyhow` internally and stringify at the FFI boundary, matching the
//! cpu/gpu flavor's string last-error surface. A typed `SparrowEngineError`
//! surface for this flavor remains future work.

pub mod cascade;
pub mod engine;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod pipeline;
pub mod preprocess;
pub mod sys;
pub mod tflite;
pub(crate) mod timing;

// Match the CPU/GPU flavor convention: consumers can continue importing shared
// types and device-agnostic helpers through the flavor crate.
pub use sparrow_engine_core::*;
pub use sparrow_engine_types::*;
