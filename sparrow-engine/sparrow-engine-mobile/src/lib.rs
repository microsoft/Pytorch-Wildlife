//! sparrow-engine-mobile — mobile LiteRT/TensorFlow Lite backend scaffold.
//!
//! This crate is the Raspberry Pi/mobile flavor peer of `sparrow-engine-cpu` and
//! `sparrow-engine-gpu`. P2.1 wires the crate, LiteRT C bindings, and a reusable
//! LiteRT session wrapper only; the full sparrow-engine inference cascade is a
//! later milestone.
//!
//! This focused crate uses `anyhow` internally for the P2 orca path. Errors
//! stringify at the FFI boundary, matching the CPU flavor's string last-error
//! surface. Typed `SparrowEngineError` migration is tracked in RP-25-FU-1.

pub mod cascade;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod sys;
pub mod tflite;

// Match the CPU/GPU flavor convention: consumers can continue importing shared
// types and device-agnostic helpers through the flavor crate.
pub use sparrow_engine_core::*;
pub use sparrow_engine_types::*;
