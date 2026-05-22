//! Custom CUDA kernels for the GPU preprocessing pipeline.
//!
//! Compiled at runtime via `cudarc::nvrtc::Ptx`:
//!
//! - [`letterbox`] — Wave 1. Bilinear resize preserving aspect ratio + pad +
//!   `/255.0` normalize + NCHW transpose. YOLO-family preprocessing.
//!   Honours manifest `channel_order` (RGB / BGR).
//! - [`center_crop`] — Wave 1. Square center crop + bilinear resize +
//!   `/255.0` normalize + NCHW transpose. SpeciesNet (and other classifier)
//!   preprocessing. Default RGB channel order.
//! - [`tiled_preprocess`] — Wave 4. Per-tile crop + zero-pad (edge tiles) +
//!   per-channel `(px/255 - mean) / std` normalize + NCHW transpose.
//!   HerdNet (ImageNet stats) + OWL-T (Unit stats). Honours manifest
//!   `channel_order`. No resize — assumes `tile_size == input_size` per the
//!   tiled-detection manifests.
//!
//! All kernels emit FP32 output. FP16 conversion (Wave 2 MDv6 path) is a
//! cast at the ORT boundary, not in the kernel — see Wave 2's design.

pub mod center_crop;
pub mod letterbox;
pub mod resize;
pub mod tiled_preprocess;

pub use center_crop::center_crop_gpu;
pub use letterbox::letterbox_gpu;
pub use resize::resize_gpu;
pub use tiled_preprocess::{tiled_preprocess_gpu, NormalizeStats, TiledPreprocessKernel};
