//! Tiled detection path (HerdNet, OWL-T) — Phase 3.8 Step 1 Wave 4.
//!
//! Pipeline shape (final_design §6 Phase D + Step 1 plan §3 Wave 4):
//! ```text
//! ImageInput -> nvjpeg decode (CPU fallback for non-baseline JPEGs)
//!            -> tile loop {
//!                 GPU tiled_preprocess (crop + zero-pad + (px/255-mean)/std + NCHW)
//!                 -> DtoH host roundtrip (Value-from-GPU not in ort 2.0.0-rc.12 surface)
//!                 -> ORT CUDA EP run (implicit re-upload, same as Wave 2 yolo.rs)
//!               }
//!            -> CPU per-tile peak finding (mirrors sparrow-engine-core::postprocess)
//!            -> CPU tile-overlap dedup (greedy center-proximity suppression)
//! ```
//!
//! Per-model differences:
//! - **HerdNet** (`sparrow-engine/models/herdnet.toml`): dual-output heatmap
//!   (location + species). `tile_size = input_size = [512, 512]`,
//!   `tile_overlap = 0`, `normalization = imagenet`, `adaptive = false`.
//! - **OWL-T** (`sparrow-engine/models/owlt.toml`): single-output heatmap, adaptive
//!   per-tile threshold. `tile_size = input_size = [512, 512]`,
//!   `tile_overlap = 160`, `normalization = unit`, `adaptive = true`.
//!
//! # GPU-pipeline coverage post-amend (2026-05-03)
//!
//! The Wave 4 amend (lead override 2026-05-03) lands a dedicated
//! `kernels/tiled_preprocess.cu` GPU kernel that handles per-tile
//! crop + zero-pad (edge tiles) + per-channel `(px/255 - mean) / std`
//! normalize + NCHW transpose for both Unit (OWL-T) and ImageNet (HerdNet)
//! stats. The previous CPU-preprocess path is gone:
//!
//! - **Decode** is on GPU via [`crate::decode::decode_jpeg`] (nvjpeg fast
//!   path, with image-crate CPU fallback for non-baseline / EXIF JPEGs).
//! - **Preprocess** is on GPU via [`crate::kernels::tiled_preprocess_gpu`].
//! - **Inference** is on GPU via ORT CUDA EP per tile.
//! - **Heatmap NMS + dedup** stay on CPU per `final_design §6 Phase D`.
//!
//! # IoBinding follow-up (carries the same caveat as Wave 2 yolo.rs)
//!
//! The GPU preprocess output (FP32 NCHW `CudaSlice`) is `clone_dtoh`'d to a
//! host `Array4<f32>` for `Session::run` because `ort 2.0.0-rc.12` does not
//! expose a clean public `Value` constructor backed by GPU memory; the only
//! safe path requires unsafe FFI into ort-sys which Wave 4 explicitly
//! defers (same wall coder-w2 hit on yolo.rs — see that file's
//! `run_inference_fp32` doc-comment block). True IoBinding wiring stays
//! a shared follow-up across yolo.rs + tiled.rs + classifier.rs.
//!
//! Owned by coder-w4 in /implement Wave 4 spawn (initial commit `2f22ff8` +
//! Gate 4 framing fix `317fceb` + this GPU-preprocess amend).

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};
use ndarray::ArrayViewD;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use sparrow_engine_core::postprocess::{
    apply_max_detections, is_local_maximum, label_for_id, owl_adaptive_threshold,
    resolve_confidence_threshold, validate_heatmap_maps,
};
use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{
    self, ChannelOrder, InferenceStrategy, Layout, ModelManifest, Normalization, PostprocessMethod,
    Precision, PreprocessMethod,
};
use sparrow_engine_types::{BBox, DetectOpts, DetectResult, Detection, ImageInput};

use crate::decode::GpuImage;
use crate::kernels::tiled_preprocess::{
    tiled_preprocess_gpu, NormalizeStats, TiledPreprocessKernel,
};
use crate::trt::ep::{manifest_cache_material, CudaEpConfig, GpuIdentity, TrtEpBuilder};

// ---------------------------------------------------------------------------
// Cached tile-loop parameters (locked at load time, validated against manifest)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// JpegDecoder — local stateful nvjpeg cache (Wave 5 hoist target)
// ---------------------------------------------------------------------------
//
// Wave 1's `decode::decode_jpeg` calls `nvjpegCreateSimple` +
// `nvjpegJpegStateCreate` per call, costing ~787 ms each (measured by
// coder-w3 on the classifier path). Tiled detection invokes decode once
// per `detect_tiled` call (one image, many tiles), so a per-call setup
// dominated post-amend latency.
//
// The fix lives here in [`JpegDecoder`] rather than in `decode.rs`
// because Wave 1 + Wave 4 + Wave 5 ship in parallel and `decode.rs` is
// owned by Wave 1; restructuring around `nvjpegCreateSimple` +
// `nvjpegJpegStateCreate` re-use is a Wave 5 task. When that lands,
// hoist this `JpegDecoder` into `decode.rs` proper and delete this
// duplicate; coder-w3's classifier.rs has the same struct for the same
// reason and will be folded together.
//
// SHARED-WAIT consolidation: this is a near-byte-for-byte port of
// `coder-w3-2531012-3937/sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs::JpegDecoder`,
// re-checked with the team-lead's heads-up before duplication. Any
// behavior fix should land in both copies until Wave 5 consolidates.

/// Stateful JPEG decoder backed by nvjpeg, with a CPU fallback.
///
/// Owns one `nvjpegHandle_t` + `nvjpegJpegState_t` pair, reused across
/// calls. Drop-time RAII guards release them via `nvjpegJpegStateDestroy`
/// then `nvjpegDestroy`. The decoder is tied to a CUDA context (kept alive
/// via the `_ctx` Arc) so the handle is usable for `nvjpegDecode` calls on
/// streams created from that context.
///
/// Falls back to `image` crate CPU decode for inputs nvjpeg cannot handle
/// (progressive / non-baseline JPEG, EXIF orientation requiring rotation,
/// non-JPEG bytes such as PNG). CPU-decoded buffers are uploaded to the
/// active stream via `clone_htod`.
pub struct JpegDecoder {
    handle: nvjpeg_sys::nvjpegHandle_t,
    state: nvjpeg_sys::nvjpegJpegState_t,
    /// Keep the CUDA context alive as long as the decoder lives.
    _ctx: Arc<CudaContext>,
}

// SAFETY: nvjpeg handle + state are bound to the CUDA primary context at
// creation time (held alive via `self._ctx`). `Send` is sound BECAUSE moving
// the decoder across threads keeps the same primary context — CUDA primary
// contexts are per-process per-device, not per-thread. `Sync` is sound because
// `decode_to_gpu` requires `&mut self`, so concurrent `&JpegDecoder` access
// cannot reach the FFI. (TiledModel additionally wraps this decoder in a
// `Mutex` to serialize the per-call decode against the per-call session run.)
//
// HOWEVER: callers MUST pass a stream derived from the same `Arc<CudaContext>`
// held in `self._ctx`. Cross-context use (different ordinal) is undefined
// behavior. The decoder does not currently assert this at runtime; cudarc does
// not expose a context-ordinal accessor on `&CudaStream` cheaply, so the
// constraint is documented + enforced by code review rather than dynamic
// assertion. The Wave 5 hoist (`sparrow-engine-gpu/src/decode.rs::JpegDecoder`) is the
// natural site to add a runtime check via `cuStreamGetCtx` if needed.
unsafe impl Send for JpegDecoder {}
unsafe impl Sync for JpegDecoder {}

impl std::fmt::Debug for JpegDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JpegDecoder").finish_non_exhaustive()
    }
}

impl JpegDecoder {
    /// Create a stateful nvjpeg decoder bound to the given CUDA context.
    /// Allocates the nvjpeg handle + state once; reused per `decode_to_gpu`
    /// call.
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        use nvjpeg_sys as nvj;

        // Phase E (2026-05-25): consult the dlopen loader BEFORE the
        // existing nvjpegCreateSimple call. If libnvjpeg.so.12 is missing /
        // wrong major / has missing symbols, surface the rich NvjpegInitError
        // (remediation text) via SparrowEngineError::NvjpegUnavailable instead
        // of letting the thin wrapper below flatten to status=1.
        if let Err(err) = nvj::nvjpeg() {
            return Err(SparrowEngineError::NvjpegUnavailable(err.to_string()));
        }

        // SAFETY: FFI; status checked.
        unsafe {
            let mut handle: nvj::nvjpegHandle_t = std::ptr::null_mut();
            let s = nvj::nvjpegCreateSimple(&mut handle);
            if s != nvj::nvjpegStatus_t_NVJPEG_STATUS_SUCCESS as nvj::nvjpegStatus_t {
                return Err(SparrowEngineError::Ort(format!(
                    "nvjpegCreateSimple failed: status={s}"
                )));
            }

            let mut state: nvj::nvjpegJpegState_t = std::ptr::null_mut();
            let s = nvj::nvjpegJpegStateCreate(handle, &mut state);
            if s != nvj::nvjpegStatus_t_NVJPEG_STATUS_SUCCESS as nvj::nvjpegStatus_t {
                let _ = nvj::nvjpegDestroy(handle);
                return Err(SparrowEngineError::Ort(format!(
                    "nvjpegJpegStateCreate failed: status={s}"
                )));
            }

            Ok(JpegDecoder {
                handle,
                state,
                _ctx: ctx.clone(),
            })
        }
    }

    /// Decode JPEG bytes to a GPU-resident HWC RGB u8 buffer using the
    /// cached nvjpeg handle + state. Falls back to CPU decode for inputs
    /// nvjpeg cannot handle.
    ///
    /// The `SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1` env var skips the nvjpeg fast
    /// path entirely and routes every image through the `image` crate
    /// (CPU decode + HtoD copy). Used to isolate "is residual cross-engine
    /// drift coming from JPEG decode (nvjpeg vs image-crate IDCT) or from
    /// the GPU preprocess kernel?". Mirrors `models/yolo.rs::decode_to_gpu`
    /// (Phase 3.8 Step 1 audit-fix R1 B1) so the diagnostic A/B knob works
    /// identically across all production model paths.
    fn decode_to_gpu(&mut self, stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<GpuImage> {
        if std::env::var("SPARROW_ENGINE_GPU_FORCE_CPU_DECODE").as_deref() == Ok("1") {
            return decode_via_cpu_fallback(stream, bytes);
        }
        if let Ok(img) = self.decode_via_nvjpeg(stream, bytes) {
            return Ok(img);
        }
        decode_via_cpu_fallback(stream, bytes)
    }

    fn decode_via_nvjpeg(&mut self, stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<GpuImage> {
        use cudarc::driver::DevicePtrMut;
        use nvjpeg_sys as nvj;
        use std::os::raw::{c_int, c_uchar};

        if has_nontrivial_exif_orientation(bytes) {
            return Err(SparrowEngineError::ImageDecode(
                "EXIF orientation requires CPU fallback".into(),
            ));
        }

        // SAFETY: FFI calls, return values checked.
        unsafe {
            let mut n_components: c_int = 0;
            let mut subsampling: nvj::nvjpegChromaSubsampling_t = 0;
            let mut widths = [0i32; nvj::NVJPEG_MAX_COMPONENT as usize];
            let mut heights = [0i32; nvj::NVJPEG_MAX_COMPONENT as usize];
            let s = nvj::nvjpegGetImageInfo(
                self.handle,
                bytes.as_ptr() as *const c_uchar,
                bytes.len(),
                &mut n_components,
                &mut subsampling,
                widths.as_mut_ptr(),
                heights.as_mut_ptr(),
            );
            if s != nvj::nvjpegStatus_t_NVJPEG_STATUS_SUCCESS as nvj::nvjpegStatus_t {
                return Err(SparrowEngineError::ImageDecode(format!(
                    "nvjpegGetImageInfo failed: status={s}"
                )));
            }
            let w = widths[0] as u32;
            let h = heights[0] as u32;
            if w == 0 || h == 0 {
                return Err(SparrowEngineError::ImageDecode(
                    "nvjpeg reported zero width or height".into(),
                ));
            }

            let total = w as usize * h as usize * 3;
            let mut out: cudarc::driver::CudaSlice<u8> = stream
                .alloc_zeros::<u8>(total)
                .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros: {e}")))?;

            let cu_stream = stream.cu_stream() as nvj::cudaStream_t;
            let s = {
                let (dev_handle, _sync) = out.device_ptr_mut(stream);
                let dev_ptr = dev_handle as *mut c_uchar;
                let mut ni: nvj::nvjpegImage_t = std::mem::zeroed();
                ni.channel[0] = dev_ptr;
                ni.pitch[0] = (w as usize) * 3;

                nvj::nvjpegDecode(
                    self.handle,
                    self.state,
                    bytes.as_ptr() as *const c_uchar,
                    bytes.len(),
                    nvj::nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGBI as nvj::nvjpegOutputFormat_t,
                    &mut ni,
                    cu_stream,
                )
            };
            if s != nvj::nvjpegStatus_t_NVJPEG_STATUS_SUCCESS as nvj::nvjpegStatus_t {
                return Err(SparrowEngineError::ImageDecode(format!(
                    "nvjpegDecode failed: status={s} (likely non-baseline / progressive)"
                )));
            }
            stream
                .synchronize()
                .map_err(|e| SparrowEngineError::Ort(format!("cudarc stream.synchronize: {e}")))?;
            Ok(GpuImage {
                data: out,
                width: w,
                height: h,
            })
        }
    }
}

impl Drop for JpegDecoder {
    fn drop(&mut self) {
        // SAFETY: state and handle were allocated by nvjpeg in `new()`.
        unsafe {
            if !self.state.is_null() {
                let _ = nvjpeg_sys::nvjpegJpegStateDestroy(self.state);
            }
            if !self.handle.is_null() {
                let _ = nvjpeg_sys::nvjpegDestroy(self.handle);
            }
        }
    }
}

/// CPU-decode fallback. Mirrors `crate::decode::decode_via_cpu_fallback`
/// but lives here so we don't need to expose that internal-private
/// function from `decode.rs`.
fn decode_via_cpu_fallback(stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<GpuImage> {
    use image::ImageReader;
    let dyn_img = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?
        .decode()
        .map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?;
    let rgb = dyn_img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w == 0 || h == 0 {
        return Err(SparrowEngineError::ImageDecode(
            "decoded image has zero width or height".into(),
        ));
    }
    let buf = rgb.into_raw();
    let dev = stream
        .clone_htod(buf.as_slice())
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc clone_htod: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc stream.synchronize: {e}")))?;
    Ok(GpuImage {
        data: dev,
        width: w,
        height: h,
    })
}

/// Cheap EXIF orientation pre-check. Mirrors `decode::has_nontrivial_exif_orientation`.
fn has_nontrivial_exif_orientation(bytes: &[u8]) -> bool {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return false;
    }
    let mut i = 2usize;
    while i + 4 <= bytes.len() {
        if bytes[i] != 0xFF {
            return false;
        }
        let marker = bytes[i + 1];
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        if i + 4 > bytes.len() {
            return false;
        }
        let seg_len = ((bytes[i + 2] as usize) << 8) | (bytes[i + 3] as usize);
        if seg_len < 2 || i + 2 + seg_len > bytes.len() {
            return false;
        }
        if marker == 0xE1 && seg_len >= 8 && &bytes[i + 4..i + 10] == b"Exif\0\0" {
            let tiff = i + 10;
            if bytes.len() < tiff + 8 {
                return false;
            }
            let le = &bytes[tiff..tiff + 2] == b"II";
            let read16 = |p: usize| -> u16 {
                if le {
                    u16::from_le_bytes([bytes[p], bytes[p + 1]])
                } else {
                    u16::from_be_bytes([bytes[p], bytes[p + 1]])
                }
            };
            let read32 = |p: usize| -> u32 {
                if le {
                    u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]])
                } else {
                    u32::from_be_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]])
                }
            };
            let ifd0_off = read32(tiff + 4) as usize;
            let ifd0 = tiff + ifd0_off;
            if bytes.len() < ifd0 + 2 {
                return false;
            }
            let n_entries = read16(ifd0) as usize;
            let entries_start = ifd0 + 2;
            for k in 0..n_entries {
                let entry = entries_start + k * 12;
                if bytes.len() < entry + 4 {
                    return false;
                }
                let tag = read16(entry);
                if tag == 0x0112 {
                    if bytes.len() < entry + 10 {
                        return false;
                    }
                    let val = read16(entry + 8);
                    return val != 1 && val != 0;
                }
            }
            return false;
        }
        i += 2 + seg_len;
    }
    false
}

#[derive(Debug, Clone, Copy)]
struct TileParams {
    tile_w: u32,
    tile_h: u32,
    tile_overlap: u32,
    peak_threshold: f32,
    adaptive: bool,
    point_to_box_half_size: u32,
    /// Kept on the params snapshot for diagnostics + potential future
    /// callers; the inner loop reads the pre-derived `stats` field instead.
    #[allow(dead_code)]
    normalization: Normalization,
    channel_order: ChannelOrder,
}

// ---------------------------------------------------------------------------
// TiledModel
// ---------------------------------------------------------------------------

/// One loaded tiled-detection model on GPU.
///
/// Cheap to clone via `Arc`; owns:
/// - the ORT CUDA-EP session (`Mutex` for `&mut self` `session.run`),
/// - the manifest (for dimensions / normalization / channel order),
/// - the labels (for `label_for_id`),
/// - locked-in tile parameters (`TileParams`) extracted from the manifest at
///   load time so the per-tile inner loop never re-reads the manifest.
///
/// Direct-function tests construct `TiledModel::load(...)` without going
/// through `sparrow_engine_gpu::Engine`. The Engine integration is a Wave 5 follow-up
/// (final_design §3 footnote: `SparrowEngineApi` trait deferred to Phase B).
pub struct TiledModel {
    session: Arc<Mutex<Session>>,
    manifest: Arc<ModelManifest>,
    labels: Arc<Vec<String>>,
    params: TileParams,
    /// NVRTC-compiled GPU preprocess kernel (`tiled_preprocess.cu`).
    /// Owned by the model so the kernel module is loaded once at `load`-time
    /// rather than per-call.
    preprocess: TiledPreprocessKernel,
    /// Cached normalization stats matching `params.normalization`. Populated
    /// at load time so the per-tile inner loop never re-derives them.
    stats: NormalizeStats,
    /// Stateful nvjpeg decoder (cached handle + state). Wrapped in a `Mutex`
    /// so `detect_tiled` keeps an `&self` receiver while still mutating the
    /// nvjpeg internals. The `Mutex` is uncontended on single-threaded
    /// workloads (~10s of ns per acquire/release), so the only cost is the
    /// type-system bookkeeping. Wave 5 hoists this into
    /// `sparrow-engine-gpu/src/decode.rs` proper alongside coder-w3's identical
    /// classifier-side `JpegDecoder` cache; a `&mut decoder` parameter at
    /// the engine call site is the cross-Wave consolidation target.
    decoder: Mutex<JpegDecoder>,
    model_id: String,
    /// Device ordinal captured at load time. Used to (a) pin the ORT CUDA
    /// EP to the same GPU as the kernels, and (b) validate per-call `ctx`
    /// matches the session's pinned EP device. Mirrors the field on
    /// `YoloModel` and `ClassifierModel` (Phase 3.8 Step 1 audit-fix R1
    /// B2: replaces a hardcoded `with_device_id(0)` that ignored
    /// multi-GPU configurations).
    device_id: i32,
}

// SAFETY: All non-Send/Sync ORT types (`Session`) are wrapped behind
// `std::sync::Mutex`. `JpegDecoder` is wrapped in a separate `Mutex`. Other
// fields (`Arc<ModelManifest>`, `Arc<Vec<String>>`, `TileParams` POD,
// `TiledPreprocessKernel` NVRTC module, `NormalizeStats` POD, `String`,
// `i32`) are all thread-safe. The session lock serializes the only mutable
// ORT access. Mirrors the SAFETY rationale on `YoloModel` and
// `ClassifierModel` (Phase 3.8 Step 1 audit-fix R2 A5 / N-NEW-2
// harmonization). Also mirrors `sparrow_engine_cpu::engine::ModelHandle`.
unsafe impl Send for TiledModel {}
unsafe impl Sync for TiledModel {}

impl TiledModel {
    /// Construct a tiled-detection model from a parsed `ModelManifest`.
    ///
    /// `ctx` is the CUDA context. Compiles the
    /// [`crate::kernels::tiled_preprocess`] kernel against this context once
    /// at load time so the per-tile inner loop pays zero NVRTC cost.
    ///
    /// `manifest` must declare:
    /// - `inference.strategy = "tiled"` with `tile_size` + `tile_overlap`,
    /// - `postprocessing.method = "heatmap_peaks"` with peak_threshold,
    /// - `preprocessing.method = "resize"` (letterbox is yolo-family).
    ///
    /// Returns `SparrowEngineError::InvalidManifest` if the manifest does not match
    /// the expected tiled-heatmap shape.
    pub fn load(
        ctx: &Arc<CudaContext>,
        manifest: &ModelManifest,
        manifest_dir: &Path,
    ) -> Result<Self> {
        // --- Validate manifest shape -------------------------------------------------
        let (tile_w, tile_h, tile_overlap) = match manifest.inference_strategy {
            InferenceStrategy::Tiled {
                tile_size,
                tile_overlap,
            } => (tile_size[0], tile_size[1], tile_overlap),
            other => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "TiledModel requires inference.strategy = \"tiled\", got {other:?}",
                )));
            }
        };

        // Reject `tile_overlap >= tile_w || tile_overlap >= tile_h`. The
        // detect_tiled inner loop computes
        // `stride = tile_w.saturating_sub(tile_overlap).max(1)`; when
        // overlap >= dim the stride collapses to 1 and the tile loop runs
        // O(W*H) ORT.run calls instead of O((W/stride)*(H/stride)). For a
        // 6000x4000 image with tile_w=512 + tile_overlap=512 that would be
        // ~24M tile inferences — effectively a livelock. Production
        // manifests don't trigger this (HerdNet overlap=0, OWL-T
        // overlap=160), but a manifest typo would. Catching it at load
        // time is cheap and prevents a foot-gun (Phase 3.8 Step 1
        // audit-fix R1 B5).
        if tile_overlap >= tile_w || tile_overlap >= tile_h {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "TiledModel: tile_overlap = {tile_overlap} must be strictly less than \
                 tile_size = [{tile_w}, {tile_h}] (overlap >= dim collapses stride to 1 \
                 and runs O(W*H) inferences per image)"
            )));
        }

        let (peak_threshold, adaptive, point_to_box_half_size) = match &manifest.postprocess_method
        {
            PostprocessMethod::HeatmapPeaks {
                peak_threshold,
                adaptive,
                point_to_box_half_size,
            } => (*peak_threshold, *adaptive, *point_to_box_half_size),
            other => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "TiledModel requires postprocessing.method = \"heatmap_peaks\", got {other:?}",
                )));
            }
        };

        // The Wave 4 MVP only handles `method = resize` (the only mode used by
        // HerdNet / OWL-T). Letterbox is yolo-family (Wave 2). MelSpectrogram
        // is audio (Step 2). Reject the others with a clear error so a
        // mis-pointed manifest fails at load time, not deep inside the inner
        // loop.
        if !matches!(manifest.preprocess_method, PreprocessMethod::Resize) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "TiledModel only supports preprocessing.method = \"resize\" in Wave 4 MVP, \
                 got {:?}",
                manifest.preprocess_method,
            )));
        }

        let normalization = manifest.normalization.ok_or_else(|| {
            SparrowEngineError::InvalidManifest(
                "tiled image manifest missing 'normalization' field".to_string(),
            )
        })?;

        // The manifest validation already ensures input_size and tile_size
        // agree for tiled models, but be defensive at the leaf module too.
        let input_size = manifest.input_size.ok_or_else(|| {
            SparrowEngineError::InvalidManifest(
                "tiled image manifest missing 'input_size' field".to_string(),
            )
        })?;
        if input_size != [tile_w, tile_h] {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "tiled manifest mismatch: input_size = {input_size:?} but tile_size = [{tile_w}, {tile_h}] — \
                 they must agree (single-tile inference, no resize)",
            )));
        }
        if !matches!(manifest.layout, Some(Layout::Nchw)) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "tiled MVP requires layout = nchw, got {:?}",
                manifest.layout,
            )));
        }
        let channel_order = manifest.channel_order.unwrap_or_default(); // Default = Rgb.

        // --- Resolve ONNX file path -------------------------------------------------
        let onnx_path = match manifest.precision {
            Precision::Fp32 => manifest_dir.join(&manifest.model_file),
            Precision::Int8 => manifest_dir.join(&manifest.model_file),
            Precision::Fp16 => {
                manifest_dir.join(manifest.model_file_fp16.as_ref().ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(
                        "manifest precision = fp16 requires file_fp16".to_string(),
                    )
                })?)
            }
        };

        // --- Build ORT session (TRT→CUDA→CPU EP policy, crate::trt::ep) ---------------
        // Pin ORT CUDA EP to the same device ordinal as `ctx`. Previously
        // hardcoded to device 0, which silently mis-pinned multi-GPU
        // configurations (Phase 3.8 Step 1 audit-fix R1 B2). `error_on_failure`
        // surfaces CUDA EP registration failures immediately rather than
        // silently degrading to the CPU fallback EP — sparrow-engine-gpu's GPU-first
        // intent is that CUDA EP MUST work; the CPU EP is kept only as a
        // last-resort runtime fallback.
        let device_id: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        let builder = Session::builder()
            .map_err(|e| SparrowEngineError::Ort(format!("ort SessionBuilder: {e}")))?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| SparrowEngineError::Ort(format!("ort with_optimization_level: {e}")))?;
        let gpu = GpuIdentity::from_context(ctx)?;
        let manifest_cache_material = manifest_cache_material(manifest);
        let providers = TrtEpBuilder::new(
            &manifest.id,
            manifest.trt.as_ref(),
            &gpu,
            CudaEpConfig::new(device_id),
            &onnx_path,
            &manifest_cache_material,
        )
        .execution_providers()?;
        let mut builder = builder
            .with_execution_providers(providers)
            .map_err(|e| SparrowEngineError::Ort(format!("ort with_execution_providers: {e}")))?;
        let session = builder.commit_from_file(&onnx_path).map_err(|e| {
            SparrowEngineError::Ort(format!("ort commit_from_file({onnx_path:?}): {e}"))
        })?;

        // Phase 3.8 Step 1 audit-fix R2 B10 (M-NEW-5): the FP32 binding code
        // assumes Float32 I/O. Reject FP16 ONNX converted without
        // `keep_io_types=True` early.
        validate_input_dtype_fp32(&session, &manifest.id)?;
        // Phase 3.8 Step 1 audit-fix R2 B8 (M-NEW-1): catch wrong-shape ONNX
        // at load time. Tiled heatmap output is rank-4 [N, C, H, W].
        validate_output_shape_tiled(&session, &manifest.id)?;

        // --- Load labels ------------------------------------------------------------
        let labels = match (&manifest.label_file, &manifest.label_format) {
            (Some(file), Some(fmt)) => {
                let label_path = manifest_dir.join(file);
                manifest::load_labels(&label_path, fmt)?
            }
            _ => Vec::new(),
        };

        // --- Compile the GPU preprocess kernel against this context ---------------
        let preprocess = TiledPreprocessKernel::new(ctx)?;

        // --- Cache the nvjpeg handle + state so per-call decode skips
        //     ~787 ms `nvjpegCreateSimple` + `nvjpegJpegStateCreate` setup
        //     (lead heads-up 2026-05-03; same fix coder-w3 applied to
        //     classifier.rs). Wave 5 hoists this cache into decode.rs.
        let decoder = Mutex::new(JpegDecoder::new(ctx)?);

        // --- Pre-derive normalization stats so the inner loop is allocation-free --
        let stats = match normalization {
            Normalization::Unit => NormalizeStats::UNIT,
            Normalization::Imagenet => NormalizeStats::IMAGENET,
            Normalization::None => {
                // The sparrow-engine-cpu MVP doesn't use Normalization::None for tiled
                // models either; reject up-front rather than emitting silently
                // wrong tensors. Add a kernel branch + stats variant if a
                // future model requires it.
                return Err(SparrowEngineError::InvalidManifest(
                    "tiled GPU preprocess does not support normalization = none yet".to_string(),
                ));
            }
        };

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            manifest: Arc::new(manifest.clone()),
            labels: Arc::new(labels),
            params: TileParams {
                tile_w,
                tile_h,
                tile_overlap,
                peak_threshold,
                adaptive,
                point_to_box_half_size,
                normalization,
                channel_order,
            },
            preprocess,
            stats,
            decoder,
            model_id: manifest.id.clone(),
            device_id,
        })
    }

    /// Device ordinal captured at load time (matches `ctx.ordinal() as i32`
    /// at the time of [`TiledModel::load`]).
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Convenience: parse a manifest file then load.
    pub fn load_from_path(ctx: &Arc<CudaContext>, manifest_path: &Path) -> Result<Self> {
        let manifest = manifest::load_manifest(manifest_path)?;
        // Flavor-strict: reject non-ONNX (the shared loader now accepts tflite for
        // the mobile flavor). Mirrors gpu/cpu Engine::load_model + AudioModel::load.
        if manifest.format != "onnx" {
            return Err(SparrowEngineError::UnsupportedFormat {
                format: manifest.format.clone(),
            });
        }
        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        Self::load(ctx, &manifest, manifest_dir)
    }

    /// Run tiled detection on a single image.
    ///
    /// Per-tile path (closes the GPU-pipeline gap left by the initial Wave 4
    /// MVP):
    ///
    /// 1. GPU decode via [`crate::decode::decode_jpeg`] (nvjpeg fast path,
    ///    image-crate CPU fallback for non-baseline / EXIF JPEGs and for the
    ///    `ImageInput::FilePath` / `ImageInput::Raw` inputs handled here).
    /// 2. GPU `tiled_preprocess_gpu` (crop + zero-pad + (px/255-mean)/std +
    ///    NCHW transpose + RGB↔BGR plane swap), one launch per tile.
    /// 3. DtoH host roundtrip — needed because `ort 2.0.0-rc.12` does not
    ///    expose a public `Value` constructor backed by GPU memory; same wall
    ///    coder-w2 hit on `models/yolo.rs::run_inference_fp32`. ORT then
    ///    re-uploads the host tensor to GPU and runs CUDA EP.
    /// 4. CPU per-tile peak finding via the sparrow-engine-core helpers.
    /// 5. CPU tile-overlap dedup (greedy center-proximity).
    pub fn detect_tiled(
        &self,
        ctx: &Arc<CudaContext>,
        image: &ImageInput,
        opts: &DetectOpts,
    ) -> Result<DetectResult> {
        let start = Instant::now();

        // Validate the per-call ctx matches the session's pinned device.
        // Cheap guard against caller-side bugs (mixing engines on different
        // GPU ordinals). Mirrors `YoloModel::detect` and
        // `ClassifierModel::classify`; closes the multi-GPU misuse hole left
        // by the previous hardcoded `with_device_id(0)` (Phase 3.8 Step 1
        // audit-fix R1 B2).
        let ctx_ordinal: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        if ctx_ordinal != self.device_id {
            return Err(SparrowEngineError::Ort(format!(
                "TiledModel::detect_tiled: ctx device {} != session device {}",
                ctx_ordinal, self.device_id
            )));
        }

        // 1. GPU decode via the cached JpegDecoder (nvjpeg handle + state
        //    re-used across calls). CPU fallback for non-baseline / EXIF /
        //    non-JPEG inputs handled inside `image_input_to_gpu`.
        let stream: Arc<CudaStream> = ctx.default_stream();
        let gpu_image = {
            let mut decoder = self
                .decoder
                .lock()
                .map_err(|_| SparrowEngineError::Ort("JpegDecoder mutex poisoned".into()))?;
            image_input_to_gpu(&stream, &mut decoder, image)?
        };
        let img_w = gpu_image.width;
        let img_h = gpu_image.height;

        let p = self.params;
        let stride_x = p.tile_w.saturating_sub(p.tile_overlap).max(1);
        let stride_y = p.tile_h.saturating_sub(p.tile_overlap).max(1);

        let threshold = resolve_confidence_threshold(opts.confidence_threshold, p.peak_threshold)?;
        let half = p.point_to_box_half_size as f32;
        let img_wf = img_w as f32;
        let img_hf = img_h as f32;

        let tile_tensor_len = checked_tensor_len_3hw(p.tile_h, p.tile_w)?;
        let mut host_buf: Vec<f32> = vec![0.0f32; tile_tensor_len];

        let mut all_detections: Vec<Detection> = Vec::new();

        // 2. Tile loop. Per-tile preprocess on GPU, per-tile inference on GPU.
        let mut y = 0u32;
        while y < img_h {
            let mut x = 0u32;
            while x < img_w {
                let crop_w = p.tile_w.min(img_w - x);
                let crop_h = p.tile_h.min(img_h - y);

                // 2a. GPU preprocess: crop + zero-pad + normalize + NCHW.
                //     Edge tiles (crop_w < tile_w || crop_h < tile_h) are
                //     handled inside the kernel via zero-pad — byte-equivalent
                //     to sparrow-engine-cpu's `RgbImage::new(...) + copy_from(...)`
                //     fallback. Result is FP32 NCHW on device.
                let dst_gpu = tiled_preprocess_gpu(
                    &stream,
                    &self.preprocess,
                    &gpu_image,
                    x,
                    y,
                    crop_w,
                    crop_h,
                    p.tile_w,
                    p.tile_h,
                    self.stats,
                    p.channel_order,
                )?;

                // 2b. DtoH roundtrip into the reused host buffer. cudarc
                //     orders kernel + memcpy on the same stream, so we only
                //     need one synchronize() after the memcpy to guarantee
                //     the host buffer is populated before ORT reads it.
                stream
                    .memcpy_dtoh(&dst_gpu, host_buf.as_mut_slice())
                    .map_err(|e| {
                        SparrowEngineError::Ort(format!("cudarc memcpy_dtoh (tile): {e}"))
                    })?;
                stream.synchronize().map_err(|e| {
                    SparrowEngineError::Ort(format!("cudarc synchronize (after dtoh): {e}"))
                })?;

                // 2c. ORT CUDA EP run. The session's CUDA EP re-uploads the
                //     host tensor to GPU and runs on device. True IoBinding
                //     wiring is the same follow-up across yolo.rs +
                //     classifier.rs + this file.
                let arr = ndarray::ArrayView4::<f32>::from_shape(
                    (1, 3, p.tile_h as usize, p.tile_w as usize),
                    host_buf.as_slice(),
                )
                .map_err(|e| SparrowEngineError::Ort(format!("tile tensor reshape: {e}")))?;
                let input_value = TensorRef::from_array_view(arr)
                    .map_err(|e| SparrowEngineError::Ort(format!("ort TensorRef: {e}")))?;
                let mut guard = self.session.lock().map_err(|_| {
                    SparrowEngineError::Ort("TiledModel session lock poisoned".into())
                })?;
                let outputs = guard
                    .run(ort::inputs![input_value])
                    .map_err(|e| SparrowEngineError::Ort(format!("ort session.run: {e}")))?;

                // 2d. Extract heatmap outputs:
                //     - Dual-output (HerdNet): outputs[0] = loc_map, outputs[1] = cls_map.
                //     - Single-output (OWL-T): outputs[0] = heatmap (single class).
                // Phase 3.8 Step 1 doc-fix R1 F-T10: explicit defensive
                // `is_empty` guard before the position-indexed access.
                // `SessionOutputs` is a BTreeMap wrapper (no `.first()` slice
                // method); indexing by ordinal `[0]` via `Index<usize>` would
                // panic on empty. Convert to a structured `SparrowEngineError::Ort`.
                if outputs.len() == 0 {
                    return Err(SparrowEngineError::Ort(
                        "session.run returned no outputs (tiled inference)".into(),
                    ));
                }
                let loc_view: ArrayViewD<'_, f32> = outputs[0]
                    .try_extract_array::<f32>()
                    .map_err(|e| SparrowEngineError::Ort(format!("ort extract loc: {e}")))?;
                let has_cls = outputs.len() > 1;
                let cls_4d = if has_cls {
                    let cls_view: ArrayViewD<'_, f32> = outputs[1]
                        .try_extract_array::<f32>()
                        .map_err(|e| SparrowEngineError::Ort(format!("ort extract cls: {e}")))?;
                    Some(
                        cls_view
                            .into_dimensionality::<ndarray::Ix4>()
                            .map_err(|e| SparrowEngineError::Ort(format!("ort cls dim: {e}")))?
                            .to_owned(),
                    )
                } else {
                    None
                };
                let loc_4d = loc_view
                    .into_dimensionality::<ndarray::Ix4>()
                    .map_err(|e| SparrowEngineError::Ort(format!("ort loc dim: {e}")))?
                    .to_owned();

                drop(outputs);
                drop(guard);

                // 2e. Per-tile peak finding. Mirrors sparrow_engine_cpu detect_tiled
                //     exactly (same plateau tie-breaking, adaptive threshold,
                //     dual-vs-single-output classification).
                let tile_dets = peaks_for_tile(
                    &loc_4d,
                    cls_4d.as_ref(),
                    &self.labels,
                    &p,
                    threshold,
                    x,
                    y,
                    img_wf,
                    img_hf,
                    half,
                )?;
                all_detections.extend(tile_dets);

                x += stride_x;
            }
            y += stride_y;
        }

        // 3. Sort descending by confidence (deterministic dedup target ordering).
        all_detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4. Tile-overlap dedup: collapse near-duplicates near tile boundaries.
        //    Greedy center-proximity suppression with radius 2 * half_size,
        //    same as sparrow-engine-cpu detect_tiled.
        if p.tile_overlap > 0 {
            deduplicate_tiled(&mut all_detections, img_wf, img_hf, half * 2.0);
        }

        apply_max_detections(&mut all_detections, opts.max_detections);

        let elapsed = start.elapsed();
        Ok(DetectResult {
            detections: all_detections,
            image_width: img_w,
            image_height: img_h,
            processing_time_ms: elapsed.as_secs_f32() * 1000.0,
        })
    }

    /// Model ID from the manifest (e.g. `"herdnet-general-2022"`).
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Borrow the manifest for diagnostics / pipeline orchestration.
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }
}

/// Validate that the session's first input is `Float32`. FP16 ONNX must be
/// converted with `onnxruntime.transformers.float16.keep_io_types=True` so
/// the I/O dtypes remain Float32 (the FP16 Cast nodes are internal to the
/// graph). True-FP16 I/O would crash the FP32 binding code at `session.run`
/// with a typed-tensor mismatch; reject at load time instead.
///
/// Phase 3.8 Step 1 audit-fix R2 B10 (M-NEW-5).
fn validate_input_dtype_fp32(session: &ort::session::Session, model_id: &str) -> Result<()> {
    use ort::value::{TensorElementType, ValueType};
    match session.inputs().first().map(|o| o.dtype()) {
        Some(ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        }) => Ok(()),
        Some(other) => Err(SparrowEngineError::InvalidManifest(format!(
            "model '{model_id}' input dtype must be Float32 (FP16 ONNX must be converted with \
             keep_io_types=True so I/O remains FP32), got {other:?}"
        ))),
        None => Err(SparrowEngineError::InvalidManifest(format!(
            "model '{model_id}' has no inputs"
        ))),
    }
}

/// Validate the tiled-heatmap output shape at load time.
///
/// Tiled inference (HerdNet, OWL-T) emits a heatmap as the first output:
/// rank-4 `[batch, channels, H, W]`. Dynamic dims (`-1`) are accepted.
/// Some tiled models (HerdNet) emit a second output (per-tile classification
/// logits); we only validate the first output's rank — the second output's
/// shape is checked by `peaks_for_tile` at runtime.
///
/// Phase 3.8 Step 1 audit-fix R2 B8 (M-NEW-1). Mirrors `yolo.rs::validate_output_shape`.
fn validate_output_shape_tiled(session: &ort::session::Session, model_id: &str) -> Result<()> {
    // Phase 3.8 Step 1 doc-fix R1 F-T10: collapse the `is_empty` early-return +
    // position-indexing into a single safe `?` chain. Mirrors the F-C9 pattern
    // in classifier.rs::validate_output_shape_softmax.
    let first_output =
        session
            .outputs()
            .first()
            .ok_or_else(|| SparrowEngineError::OutputShapeMismatch {
                id: model_id.to_string(),
                shape: "no outputs".to_string(),
                method: "heatmap_peaks".to_string(),
            })?;
    let dims: Vec<i64> = match first_output.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => vec![],
    };
    if dims.len() != 4 {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: format!("{dims:?} (expected rank-4 [N, C, H, W])"),
            method: "heatmap_peaks".to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-tile peak finding (mirrors sparrow_engine_cpu detect_tiled inner loop)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn peaks_for_tile(
    loc_4d: &ndarray::Array4<f32>,
    cls_4d: Option<&ndarray::Array4<f32>>,
    labels: &[String],
    params: &TileParams,
    threshold: f32,
    tile_x: u32,
    tile_y: u32,
    img_wf: f32,
    img_hf: f32,
    half: f32,
) -> Result<Vec<Detection>> {
    if let Some(cls) = cls_4d {
        validate_heatmap_maps(&loc_4d.view(), Some(&cls.view()), "tiled detector")?;
    } else {
        validate_heatmap_maps(&loc_4d.view(), None, "tiled detector")?;
    }
    let loc_h = loc_4d.shape()[2];
    let loc_w = loc_4d.shape()[3];
    let scale_x = params.tile_w as f32 / loc_w as f32;
    let scale_y = params.tile_h as f32 / loc_h as f32;
    let loc_view4 = loc_4d.view();

    // For OWL-T-style single-output models with adaptive thresholding:
    //   threshold = max(peak_threshold, tile_max * peak_threshold, 0.1)
    // For sigmoid-bounded heatmaps ([0,1]) this never exceeds peak_threshold.
    let has_cls = cls_4d.is_some();
    let effective_base_threshold = if !has_cls && params.adaptive {
        let tile_max = loc_4d.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        owl_adaptive_threshold(threshold, tile_max)
    } else {
        threshold
    };

    let mut detections = Vec::new();
    for py in 0..loc_h {
        for px in 0..loc_w {
            let val = loc_4d[[0, 0, py, px]];
            if val < effective_base_threshold {
                continue;
            }
            // 8-connected local maximum with plateau tie-breaking.
            if !is_local_maximum(&loc_view4, py, px, loc_h, loc_w) {
                continue;
            }

            let (class_id, confidence) = if let Some(cls) = cls_4d {
                let cls_h = cls.shape()[2];
                let cls_w = cls.shape()[3];
                let num_classes = cls.shape()[1];

                let cy = if cls_h == loc_h {
                    py
                } else {
                    (py * cls_h / loc_h).min(cls_h - 1)
                };
                let cx = if cls_w == loc_w {
                    px
                } else {
                    (px * cls_w / loc_w).min(cls_w - 1)
                };

                let mut best_id = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for c in 0..num_classes {
                    let v = cls[[0, c, cy, cx]];
                    if v > best_val {
                        best_val = v;
                        best_id = c;
                    }
                }
                (best_id, val * best_val)
            } else {
                (0usize, val)
            };

            if confidence < threshold {
                continue;
            }

            // Map heatmap coord → full-image pixel coord.
            let full_px = tile_x as f32 + px as f32 * scale_x;
            let full_py = tile_y as f32 + py as f32 * scale_y;

            let x_min = ((full_px - half) / img_wf).clamp(0.0, 1.0);
            let y_min = ((full_py - half) / img_hf).clamp(0.0, 1.0);
            let x_max = ((full_px + half) / img_wf).clamp(0.0, 1.0);
            let y_max = ((full_py + half) / img_hf).clamp(0.0, 1.0);

            let label = label_for_id(labels, class_id as u32);

            detections.push(Detection {
                bbox: BBox {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                },
                label,
                label_id: class_id as u32,
                confidence,
            });
        }
    }
    Ok(detections)
}

// ---------------------------------------------------------------------------
// Tile-overlap deduplication (greedy center-proximity, copied from sparrow_engine_cpu)
// ---------------------------------------------------------------------------

/// Suppress lower-confidence duplicates whose bbox centers lie within
/// `radius_pixels` of a higher-confidence detection.
///
/// Expects `detections` sorted by confidence descending. O(N^2) — fine for
/// camera-trap workloads where N is in the hundreds.
fn deduplicate_tiled(detections: &mut Vec<Detection>, img_w: f32, img_h: f32, radius_pixels: f32) {
    if detections.len() <= 1 {
        return;
    }
    let r_sq = radius_pixels * radius_pixels;
    let mut keep = vec![true; detections.len()];
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        let ci_x = (detections[i].bbox.x_min + detections[i].bbox.x_max) * 0.5 * img_w;
        let ci_y = (detections[i].bbox.y_min + detections[i].bbox.y_max) * 0.5 * img_h;
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            let cj_x = (detections[j].bbox.x_min + detections[j].bbox.x_max) * 0.5 * img_w;
            let cj_y = (detections[j].bbox.y_min + detections[j].bbox.y_max) * 0.5 * img_h;
            let dist_sq = (ci_x - cj_x).powi(2) + (ci_y - cj_y).powi(2);
            if dist_sq < r_sq {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

// ---------------------------------------------------------------------------
// ImageInput → GPU upload helper
// ---------------------------------------------------------------------------

/// Upload an [`ImageInput`] to a GPU-resident HWC u8 RGB buffer using the
/// cached [`JpegDecoder`].
///
/// - `Encoded(bytes)` → `decoder.decode_to_gpu` (nvjpeg fast path with the
///   cached handle/state; image-crate CPU fallback for non-baseline / EXIF /
///   non-JPEG formats).
/// - `FilePath(path)` → read file then `decoder.decode_to_gpu`.
/// - `Raw { data, width, height, stride, format }` → re-pack into HWC RGB
///   then `clone_htod`. Mirrors sparrow_engine_cpu::preprocess::decode_to_rgb's
///   per-format byte-shuffling (RGB / BGR / RGBA / BGRA).
fn image_input_to_gpu(
    stream: &Arc<CudaStream>,
    decoder: &mut JpegDecoder,
    input: &ImageInput,
) -> Result<GpuImage> {
    match input {
        ImageInput::Encoded(bytes) => decoder.decode_to_gpu(stream, bytes),
        ImageInput::FilePath(path) => {
            if !path.exists() {
                return Err(SparrowEngineError::ImageFileNotFound(path.clone()));
            }
            let bytes = std::fs::read(path)
                .map_err(|e| SparrowEngineError::ImageDecode(format!("read {path:?}: {e}")))?;
            decoder.decode_to_gpu(stream, &bytes)
        }
        // Phase 3.8 Step 1 audit-fix R2 B9 (M-NEW-2): the inline RGB/BGR/RGBA/BGRA
        // shuffler was hoisted to `crate::decode::raw_to_gpu` and shared with
        // `YoloModel::detect` and `ClassifierModel::classify`.
        ImageInput::Raw {
            data,
            width,
            height,
            stride,
            format,
        } => crate::decode::raw_to_gpu(stream, data, *width, *height, *stride, *format),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
//
// These are deterministic CPU-only checks that don't require a GPU or an ORT
// session. The GPU preprocess kernel itself is exercised end-to-end by the
// integration tests in `tests/integration_tiled.rs`; per-pixel kernel parity
// vs a CPU reference is a Wave-5 follow-up bench (the old CPU `build_nchw_tensor`
// + `normalize_pixel` helpers + their unit tests were removed when we cut
// over the per-tile path to GPU; reinstating them as a cross-EP parity
// reference belongs in a separate kernel-parity test file alongside
// `tests/kernels_parity.rs`).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::LabelFormat;
    use sparrow_engine_types::ModelSubtype;

    /// Construct a minimal valid tiled manifest. Tests override individual
    /// fields to probe specific validation paths.
    fn dummy_tiled_manifest(tile_size: [u32; 2], tile_overlap: u32) -> ModelManifest {
        ModelManifest {
            id: "tiled-test".into(),
            format: "onnx".into(),
            model_file: "test.onnx".into(),
            preprocess_method: PreprocessMethod::Resize,
            input_size: Some(tile_size),
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Imagenet),
            pad_value: None,
            channel_order: Some(ChannelOrder::Rgb),
            precision: Precision::Fp32,
            model_file_fp16: None,
            inference_strategy: InferenceStrategy::Tiled {
                tile_size,
                tile_overlap,
            },
            trt: None,
            postprocess_method: PostprocessMethod::HeatmapPeaks {
                peak_threshold: 0.2,
                adaptive: false,
                point_to_box_half_size: 8,
            },
            confidence_threshold: None,
            label_file: Some("labels.txt".into()),
            label_format: Some(LabelFormat::NameIndexCsv),
            default: false,
            subtype: ModelSubtype::Standard,
            onnx_sha256: None,
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
        }
    }

    fn cuda_or_skip(test_name: &str) -> Option<Arc<CudaContext>> {
        if std::env::var("SPARROW_ENGINE_GPU_TESTS").as_deref() == Ok("0") {
            eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping {test_name}");
            return None;
        }
        match CudaContext::new(0) {
            Ok(c) => Some(c),
            Err(_) => {
                eprintln!("CUDA unavailable → skipping {test_name}");
                None
            }
        }
    }

    /// Phase 3.8 Step 1 audit-fix R1 B5 regression test.
    ///
    /// `tile_overlap = tile_w` triggers `stride = saturating_sub(w,w).max(1) = 1`
    /// in the inner loop; for a 6000×4000 image with tile_w=512 + overlap=512
    /// that yields ~24M tile inferences. Production manifests don't trigger
    /// this (HerdNet overlap=0, OWL-T overlap=160), but a manifest typo
    /// could lock up the engine. Validation must fire at load time.
    #[test]
    fn load_rejects_tile_overlap_equal_to_tile_size() {
        let ctx = match cuda_or_skip("load_rejects_tile_overlap_equal_to_tile_size") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        let m = dummy_tiled_manifest([512, 512], 512);
        match TiledModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg))
                if msg.contains("tile_overlap") && msg.contains("must be strictly less than") =>
            {
                // expected
            }
            Err(other) => {
                panic!("expected InvalidManifest(tile_overlap...) rejection, got Err({other:?})")
            }
            Ok(_) => panic!("expected InvalidManifest rejection, got Ok(_)"),
        }
    }

    /// Phase 3.8 Step 1 audit-fix R1 B5 regression test (over-condition).
    ///
    /// `tile_overlap > tile_w` is also rejected — same root cause as the
    /// equality case (stride collapses to 1).
    #[test]
    fn load_rejects_tile_overlap_exceeding_tile_size() {
        let ctx = match cuda_or_skip("load_rejects_tile_overlap_exceeding_tile_size") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        let m = dummy_tiled_manifest([512, 512], 768);
        match TiledModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg)) if msg.contains("tile_overlap") => {
                // expected
            }
            Err(other) => {
                panic!("expected InvalidManifest(tile_overlap...) rejection, got Err({other:?})")
            }
            Ok(_) => panic!("expected InvalidManifest rejection, got Ok(_)"),
        }
    }

    fn make_det(center_x_px: f32, center_y_px: f32, confidence: f32) -> Detection {
        let img_w = 6000.0_f32;
        let img_h = 4000.0_f32;
        let half = 10.0_f32;
        Detection {
            bbox: BBox {
                x_min: (center_x_px - half) / img_w,
                y_min: (center_y_px - half) / img_h,
                x_max: (center_x_px + half) / img_w,
                y_max: (center_y_px + half) / img_h,
            },
            label: "buffalo".to_string(),
            label_id: 0,
            confidence,
        }
    }

    #[test]
    fn dedup_collapses_exact_duplicates() {
        let mut dets = vec![make_det(100.0, 200.0, 0.9), make_det(100.0, 200.0, 0.7)];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert_eq!(dets.len(), 1);
        assert!((dets[0].confidence - 0.9).abs() < 1e-6, "higher conf kept");
    }

    #[test]
    fn dedup_preserves_distinct_detections() {
        let mut dets = vec![make_det(100.0, 200.0, 0.9), make_det(600.0, 200.0, 0.7)];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert_eq!(dets.len(), 2, "distinct centers (500px apart) survive");
    }

    #[test]
    fn dedup_collapses_near_duplicates_within_radius() {
        let mut dets = vec![make_det(100.0, 200.0, 0.8), make_det(105.0, 200.0, 0.6)];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert_eq!(dets.len(), 1);
        assert!((dets[0].confidence - 0.8).abs() < 1e-6);
    }
}
