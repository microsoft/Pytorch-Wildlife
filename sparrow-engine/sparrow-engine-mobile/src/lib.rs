//! sparrow-engine-mobile — mobile LiteRT/TensorFlow Lite backend scaffold.
//!
//! This crate is the Raspberry Pi/mobile flavor peer of `sparrow-engine-cpu` and
//! `sparrow-engine-gpu`. P2.1 wires the crate, LiteRT C bindings, and a reusable
//! LiteRT session wrapper only; the full sparrow-engine inference cascade is a
//! later milestone.

pub mod cascade;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod sys;
pub mod tflite;

// Match the CPU/GPU flavor convention: consumers can continue importing shared
// types and device-agnostic helpers through the flavor crate.
pub use sparrow_engine_core::*;
pub use sparrow_engine_types::*;
