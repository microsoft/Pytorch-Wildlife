//! sparrow-engine-core — device-agnostic logic for the sparrow-engine workspace.
//!
//! Phase 3.8 Phase A: stateless modules. ZERO ORT/CUDA/nvjpeg deps.
//! Engine + ORT integration lives in sparrow-engine-cpu.

pub mod catalog;
pub mod daynight;
pub mod export;
pub mod hash;
pub mod pipeline_compat;
pub mod postprocess;
pub mod preprocess;
pub mod preprocess_audio;
pub mod stats;
pub mod viz;
