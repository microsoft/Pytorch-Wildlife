//! Per-model GPU pipeline paths (Phase 3.8 Step 1 Wave 2/3/4).
//!
//! Each module owns the per-model preprocess + ORT-CUDA-EP inference +
//! postprocess for one model family:
//!
//! - [`yolo`] — YOLO E2E (MDv6, DeepFaune): nvjpeg decode → CUDA letterbox+
//!   normalize+NCHW (BGR) → ORT CUDA EP IoBinding → CPU yolo_e2e postprocess
//!   in `sparrow-engine-core`. Filled in by Wave 2.
//! - [`classifier`] — softmax classifier (SpeciesNet): nvjpeg decode → CUDA
//!   center-crop+resize+normalize+NCHW (RGB) → ORT CUDA EP IoBinding → CPU
//!   softmax in `sparrow-engine-core`. Filled in by Wave 3.
//! - [`tiled`] — tiled detection (HerdNet, OWL-T): nvjpeg decode → tile loop
//!   {CUDA preprocess per tile → ORT CUDA EP per tile} → CPU heatmap NMS +
//!   tile-overlap dedup in `sparrow-engine-core`. Filled in by Wave 4.
//!
//! # Module API contract (locked at Wave 2/3/4 spawn)
//!
//! Each model module exports a struct (`YoloModel`, `ClassifierModel`,
//! `TiledModel`) with at minimum:
//!
//! - `pub fn load(ctx: &Arc<CudaContext>, manifest: &ModelManifest, manifest_dir: &Path) -> Result<Self, SparrowEngineError>`
//! - the inference method appropriate for the model type (`detect`,
//!   `classify`, or `detect_tiled`).
//!
//! Engine integration (`engine::Engine::load_model` etc. dispatching by
//! `derive_model_type` to the per-model `load` + inference methods) lands
//! in a follow-up wave; Wave 2/3/4 leaf files are independently testable
//! via `tests/integration_<model>.rs` that calls the module functions
//! directly without going through `Engine`.

pub mod audio;
pub mod audio_raw;
pub mod classifier;
pub mod tiled;
pub mod yolo;
