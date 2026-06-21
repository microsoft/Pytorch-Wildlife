//! sparrow-engine-gpu — GPU pipeline crate.
//!
//! Per `docs/design/phase3.8/final_design.md` §2/§5 + Step 1 + Step 2 +
//! Phase C wave-decomposition, this crate owns the GPU-side primitives
//! (Step 1 + Step 2) plus the `Engine` dispatch glue (Phase C Wave 1)
//! that mirrors `sparrow_engine_cpu::Engine`'s public surface so consumer crates
//! (`sparrow-engine-server`, `sparrow-engine-cli`, `sparrow-engine-python`) can swap between
//! flavors via compile-time feature dispatch.
//!
//! - [`decode`] — nvjpeg JPEG decode + CPU fallback for unparseable inputs.
//! - [`kernels`] — custom CUDA kernels for letterbox + center-crop +
//!   resize + tiled-preprocess + audio (window-frame, power, power_to_db,
//!   transpose), compiled via `cudarc::nvrtc` at runtime.
//! - [`models`] — per-model pipelines: [`models::yolo::YoloModel`],
//!   [`models::classifier::ClassifierModel`], [`models::tiled::TiledModel`],
//!   [`models::audio::AudioModel`].
//! - [`engine`] — Phase C Wave 1 dispatch [`engine::Engine`] (concrete
//!   struct mirroring `sparrow_engine_cpu::Engine`'s surface; trait deferred per
//!   `final_design §3` footnote).
//! - [`detect`] / [`classify`] / [`detect_audio`] / [`pipeline`] —
//!   top-level free functions matching `sparrow_engine_cpu`'s. Each takes a
//!   [`engine::ModelHandle`] (or [`engine::Engine`] for pipeline) and
//!   routes to the right per-model GPU pipeline.

pub mod audio;
pub mod classify;
pub mod decode;
pub mod detect;
pub mod detect_audio;
pub mod engine;
pub mod kernels;
pub mod models;
pub mod pipeline;
pub mod profile;
pub(crate) mod trt;

// Phase 3.8 Phase C Wave 4b: C FFI surface mirrors sparrow-engine-cpu's
// `pub mod ffi;` declaration. Gated on `--features ffi` so default
// workspace builds do not pull in cbindgen + csbindgen build-deps. See
// `src/ffi.rs` for the 32 `sparrow_engine_*` exports and `exports.{map,def}` for
// the linker symbol filter applied to the cdylib.
#[cfg(feature = "ffi")]
pub mod ffi;

// Phase 3.8 Phase C W1 audit-fix R2 (I-S1): glob re-export sparrow_engine_core
// + sparrow_engine_types so consumers compiled against sparrow-engine-gpu can write
// `sparrow_engine_gpu::ImageInput` / `sparrow_engine_gpu::DetectOpts` / `sparrow_engine_gpu::ModelInfo`
// / `sparrow_engine_gpu::SparrowEngineError` / `sparrow_engine_gpu::catalog::*` etc., mirroring
// `sparrow_engine_cpu::lib.rs:39-40`. Phase C waves 2-5 (consumer wiring)
// require this surface parity so `#[cfg(feature="gpu")] use sparrow_engine_gpu as sparrow_engine;`
// vs `use sparrow_engine_cpu as sparrow_engine;` resolves identically.
pub use sparrow_engine_core::*;
pub use sparrow_engine_types::*;

// Engine-side re-exports (Engine + ModelHandle wrap GPU sessions and
// stay in sparrow-engine-gpu).
pub use engine::{Engine, ModelHandle};
