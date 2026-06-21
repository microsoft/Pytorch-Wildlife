//! sparrow-engine-types — leaf data type crate for the sparrow-engine workspace.
//!
//! Phase 3.8 Phase A. Zero ORT/CUDA/nvjpeg deps.

pub mod types;
pub mod error;
pub mod manifest;
pub mod drift_metrics;
pub mod inference_log;
// device / model_type / preprocess_meta / engine_config are crate-private:
// every symbol they expose is re-exported at the crate root below, and no
// consumer accesses them via the `sparrow_engine_types::module::*` path. Keeping them
// `pub(crate)` prevents accidental new public paths (`engine_dispatch::device::Device`
// etc.) from leaking through `sparrow-engine-cpu`'s `pub use sparrow_engine_types::*;` glob.
pub(crate) mod device;
pub(crate) mod model_type;
pub(crate) mod preprocess_meta;
pub(crate) mod engine_config;

// Crate-root re-exports for ergonomic consumer access.
pub use error::{Result, SparrowEngineError};
pub use types::*;
pub use device::Device;
pub use model_type::derive_model_type;
pub use preprocess_meta::{PreprocessConfig, PreprocessMeta};
pub use engine_config::EngineConfig;
pub use manifest::{ModelManifest, PipelineManifest, ProvenanceRecord, TrtConfig, TrtPrecision};
pub use drift_metrics::{DriftMetrics, DriftReference};
pub use inference_log::{InferenceLogRecord, SCHEMA_VERSION};

// NOTE: `pub type SparrowEngine = c_void;` is INTENTIONALLY ABSENT here.
// Per v2 CRIT-1 closure (PRESERVE), the C-FFI opaque alias stays in
// sparrow-engine-cpu/src/ffi.rs ONLY. cbindgen with parse_deps = false (per C9
// closure) cannot follow cross-crate type aliases — placing the alias
// here AND in ffi.rs would not change cbindgen's output, and placing
// it ONLY here would break sparrow_engine.h byte-identity.
