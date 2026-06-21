//! Softmax classifier path (SpeciesNet). Phase 3.8 Step 1 Wave 3.
//!
//! Pipeline shape (Wave 3 amend, after lead direction "option A"):
//! ```text
//! ImageInput -> nvjpeg decode -> CUDA preprocess (dispatched on manifest)
//!            -> ORT CUDA EP (TensorRefMut::from_raw, GPU-resident input)
//!            -> CPU softmax (sparrow-engine-core)
//! ```
//!
//! ## Preprocess dispatch
//!
//! `classify` reads `manifest.preprocess_method` and routes to the
//! matching GPU kernel. SpeciesNet's manifest is method = "resize", so it
//! lands in [`crate::kernels::resize::resize_gpu`] — a multi-tap
//! convolutional bilinear bit-tight against `fast_image_resize::Resizer`
//! with `ResizeAlg::Convolution(FilterType::Bilinear)`, the algorithm
//! `sparrow-engine-cpu/src/preprocess.rs::resize_simd` uses. Result: top-1 parity
//! 10/10 across the SpeciesNet test subset (Wave 3 amend).
//!
//! Forward-compat slot: if a future `sparrow-engine-types` schema adds a
//! `PreprocessMethod::CenterCropResize` variant (or similar), this
//! function will route it to [`crate::kernels::center_crop::center_crop_gpu`]
//! — the `center_crop` kernel parameter is held in the signature for
//! exactly this case. Until then the parameter is unused in the active
//! path.
//!
//! ## Wave 3 IoBinding strategy
//!
//! Wave 3 binds the GPU input pointer directly via
//! [`ort::value::TensorRefMut::from_raw`] with a CUDA `MemoryInfo`, then
//! calls `Session::run`. ORT's run path consumes the device pointer
//! without additional H→D copy when source memory_info matches the EP's
//! device. Per-call buffer freshness (each `center_crop_gpu` allocates a
//! new `CudaSlice<f32>`) means the IoBinding "stable pointer" optimization
//! does NOT apply here; Wave 5 sweep can revisit.
//!
//! ## Channel order
//!
//! Honors manifest `channel_order` (default RGB; SpeciesNet manifest has no
//! field → defaults via `sparrow_engine_types::manifest::load_manifest`). Per Hard
//! Constraints in the Wave 3 directive: do NOT flip without empirical
//! evidence.
//!
//! ## Stateful JPEG decoder
//!
//! Wave 3 needs a stateful nvjpeg decoder per the directive's
//! `decoder: &mut /* nvjpeg decoder state */` slot. Wave 1's
//! `crate::decode::decode_jpeg` creates + destroys nvjpeg handle/state
//! on every call; empirical measurement shows this costs **~787 ms per
//! image** on RTX 6000 Ada — pure ORT inference (no decode) on the same
//! model is 4 ms median (see `wave_3_bench.md`). The per-call nvjpeg
//! setup dominates by 200×.
//!
//! The fix lives here in [`JpegDecoder`] rather than in `decode.rs`
//! (Wave 1 territory). It is a duplicate of `decode.rs`'s nvjpeg path
//! restructured around `nvjpegCreateSimple` + `nvjpegJpegStateCreate`
//! at construction time, then reused per call. Wave 5 sweep can hoist
//! `JpegDecoder` into `decode.rs` proper and delete the duplicate;
//! Wave 2 (yolo) and Wave 4 (tiled) will hit the same perf cliff and
//! can either copy this struct or wait for the Wave 5 hoist.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ndarray::{ArrayView2, ArrayViewD};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Shape, TensorRef, TensorRefMut};
use sparrow_engine_core::postprocess;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{
    self, ChannelOrder, Layout, ModelManifest, Normalization, PostprocessMethod, Precision,
    PreprocessMethod,
};
use sparrow_engine_types::{ClassifyOpts, ClassifyResult, ImageInput};

use crate::decode::GpuImage;
use crate::kernels::center_crop::CenterCropKernel;
use crate::kernels::resize::{resize_gpu, ResizeKernel};
use crate::kernels::tiled_preprocess::NormalizeStats;
use crate::trt::ep::{manifest_cache_material, CudaEpConfig, GpuIdentity, TrtEpBuilder};

// ===========================================================================
// JpegDecoder — stateful nvjpeg wrapper.
//
// Owns one nvjpeg handle + state pair, reused across calls. Drop-time RAII
// guards release them via `nvjpegJpegStateDestroy` + `nvjpegDestroy`.
// Tied to a CUDA context (tracked via the `ctx` Arc) so the handle is
// usable for `nvjpegDecode` calls on streams created from that context.
// ===========================================================================

/// Stateful JPEG decoder backed by nvjpeg, with a CPU fallback.
///
/// Construct once per pipeline; pass `&mut self` to
/// [`ClassifierModel::classify`]. Internally caches the nvjpeg handle and
/// JpegState so each decode reuses both — the per-call cost drops from
/// ~787 ms (Wave 1's `decode_jpeg`) to a few ms.
///
/// Falls back to `image` crate CPU decode for inputs nvjpeg cannot handle
/// (progressive / non-baseline JPEG, EXIF orientation requiring rotation,
/// non-JPEG bytes such as PNG). CPU-decoded buffers are uploaded to the
/// active stream via `clone_htod`.
pub struct JpegDecoder {
    handle: nvjpeg_sys::nvjpegHandle_t,
    state: nvjpeg_sys::nvjpegJpegState_t,
    /// Keep the CUDA context alive as long as the decoder lives. Not used
    /// directly inside `decode_to_gpu` (the caller passes a stream), but
    /// guarantees nvjpegDestroy in `Drop` runs in a context-bound state.
    _ctx: Arc<CudaContext>,
}

// SAFETY: nvjpeg handle + state are bound to the CUDA primary context at
// creation time (held alive via `self._ctx`). `Send` is sound BECAUSE moving
// the decoder across threads keeps the same primary context — CUDA primary
// contexts are per-process per-device, not per-thread. `Sync` is sound because
// `decode_to_gpu` requires `&mut self`, so concurrent `&JpegDecoder` access
// cannot reach the FFI.
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
    /// Allocates the handle + state once; reused per `decode_to_gpu` call.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
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
    pub fn decode_to_gpu(&mut self, stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<GpuImage> {
        if std::env::var("SPARROW_ENGINE_GPU_FORCE_CPU_DECODE").as_deref() == Ok("1") {
            return decode_via_cpu_fallback(stream, bytes);
        }
        match self.decode_via_nvjpeg(stream, bytes) {
            Ok(img) => return Ok(img),
            Err(e) => {
                // Phase 3.8 Step 1 doc-fix R1 F-C6: log nvjpeg failures so the
                // CPU-fallback frequency is observable in production.
                tracing::warn!(
                    target: "sparrow_engine_gpu::decode",
                    error = %e,
                    "nvjpeg decode failed; falling back to CPU decode (image-crate)"
                );
            }
        }
        decode_via_cpu_fallback(stream, bytes)
    }

    fn decode_via_nvjpeg(&mut self, stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<GpuImage> {
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
            let mut out: CudaSlice<u8> = stream
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
        // SAFETY: state and handle were allocated by nvjpeg in `new()`;
        // drop them in reverse order.
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

/// CPU-decode fallback. Decodes via the `image` crate (handles all
/// formats, honours EXIF), then `cudaMemcpy` the HWC RGB buffer to GPU.
/// Mirrors `crate::decode::decode_via_cpu_fallback`.
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

/// EXIF orientation pre-check on raw JPEG bytes. Returns `true` only when
/// the orientation tag is present AND non-trivial (anything other than 1).
/// Mirrors `crate::decode::has_nontrivial_exif_orientation` (private).
fn has_nontrivial_exif_orientation(bytes: &[u8]) -> bool {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return false; // not JPEG
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

/// SpeciesNet (and other softmax classifier) GPU inference path.
///
/// Owns the ORT `Session` for one classifier model + per-session metadata
/// (input/output names, CUDA `MemoryInfo` for tensor binding, manifest +
/// labels). One `ClassifierModel` per loaded classifier; multi-model
/// orchestration lives in `crate::engine::Engine` (Phase B follow-up).
#[derive(Debug)]
pub struct ClassifierModel {
    /// ORT session. Mutex because `Session::run` is `&mut self` and the
    /// model is shared across the engine's reader threads.
    session: Mutex<Session>,
    /// Manifest snapshot (cloned at load time; immutable thereafter).
    manifest: Arc<ModelManifest>,
    /// Class labels indexed by `label_id`.
    labels: Arc<Vec<String>>,
    /// First input outlet name (single-input for SpeciesNet; cached so we
    /// don't re-borrow the session on every classify call).
    input_name: String,
    /// First output outlet name.
    output_name: String,
    /// CUDA `MemoryInfo` for binding device pointers as ORT inputs.
    /// Pre-built at load time; cloned per-call (`MemoryInfo: Clone` via
    /// `MemoryInfo::new` round-trip).
    cuda_mem_info: MemoryInfo,
    /// Device ordinal captured at load time. Used to validate that
    /// per-call `ctx` matches the session's EP device.
    device_id: i32,
}

// SAFETY: All non-Send/Sync ORT types (Session) are wrapped behind
// std::sync::Mutex. MemoryInfo is Clone and inherently device-scoped POD-
// like. Manifest, labels, names are Arc / String / i32. The session lock
// serializes the only mutable access.
unsafe impl Send for ClassifierModel {}
unsafe impl Sync for ClassifierModel {}

fn read_image_file(path: &Path) -> Result<Vec<u8>> {
    if !path.exists() {
        return Err(SparrowEngineError::ImageFileNotFound(path.to_path_buf()));
    }
    std::fs::read(path).map_err(SparrowEngineError::from)
}

impl ClassifierModel {
    /// Load a SpeciesNet-style classifier on the given CUDA context.
    ///
    /// Steps:
    /// 1. Validate manifest is a vision softmax classifier.
    /// 2. Resolve ONNX file path (FP32 default; FP16 opt-in via manifest).
    /// 3. Load labels (`sparrow_engine_types::manifest::load_labels`).
    /// 4. Build ORT session with CUDA EP pinned to `ctx.ordinal()`.
    /// 5. Cache input/output names and CUDA `MemoryInfo`.
    ///
    /// # Errors
    /// - `SparrowEngineError::InvalidManifest` if the manifest is not a softmax vision
    ///   classifier (audio rejected; non-softmax postprocess rejected;
    ///   `precision = fp16` without `model_file_fp16` rejected). Matches the
    ///   error shape used by `YoloModel::load` and `TiledModel::load`
    ///   (Phase 3.8 Step 1 audit-fix R2 B6 / S-NEW-3).
    /// - `SparrowEngineError::Ort` for ORT session-creation / commit failures only
    ///   (the variant is reserved for ORT runtime errors).
    /// - `SparrowEngineError::ManifestNotFound` / `LabelFileNotFound` propagated
    ///   from `load_labels`.
    pub fn load(
        ctx: &Arc<CudaContext>,
        manifest: &ModelManifest,
        manifest_dir: &Path,
    ) -> Result<Self> {
        // 1. Validate model type.
        if matches!(
            manifest.preprocess_method,
            PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. }
        ) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "ClassifierModel::load: manifest '{}' is an audio model (preprocess = {}), expected vision classifier",
                manifest.id,
                manifest.preprocess_method.as_str(),
            )));
        }
        if !matches!(manifest.postprocess_method, PostprocessMethod::Softmax) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "ClassifierModel::load: manifest '{}' has postprocess = {}, expected softmax",
                manifest.id,
                manifest.postprocess_method.as_str(),
            )));
        }

        // Phase 3.8 Step 1 audit-fix R3 M8: hoist remaining classify()-time
        // manifest checks into load() for fail-fast semantics. Each was using
        // `SparrowEngineError::Ort` instead of `InvalidManifest` (B6 parity gap).
        // Sites pre-hoist:
        //   - classify() input_size                  → SparrowEngineError::Ort
        //   - classify() Letterbox preprocess        → SparrowEngineError::Ort
        //   - classify() Normalization::None         → SparrowEngineError::Ort
        //   - classify() Layout::Nhwc                → SparrowEngineError::Ort
        // Hoisted here with the variant matched to B6 (load-time manifest
        // defects use `InvalidManifest`).

        // `input_size` is required: classifier preprocess (Resize) needs target dims.
        if manifest.input_size.is_none() {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "ClassifierModel::load: manifest '{}' missing input_size",
                manifest.id
            )));
        }

        // `Letterbox` is YOLO-family preprocess; classifier engine cannot run it.
        if matches!(manifest.preprocess_method, PreprocessMethod::Letterbox) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "ClassifierModel::load: manifest '{}' specifies preprocess method 'letterbox', \
                 which is YOLO-family preprocess (not a classifier preprocess). \
                 Use a detector engine, or change the manifest method.",
                manifest.id
            )));
        }

        // `Normalization::None` is rejected: classifier preprocess always normalizes.
        if matches!(manifest.normalization, Some(Normalization::None)) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "ClassifierModel::load: manifest '{}' specifies normalization = 'none'; \
                 classifier resize_gpu has no raw-passthrough variant. Add a \
                 NormalizeStats::RAW and a kernel branch if a future model needs it.",
                manifest.id
            )));
        }

        // `Layout::Nhwc` is unsupported: Wave 1 kernels emit NCHW only, and ORT
        // CUDA EP has known bugs with NHWC + dynamic shapes (issues #27912, #12288).
        if matches!(manifest.layout, Some(Layout::Nhwc)) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "ClassifierModel::load: manifest '{}' specifies NHWC layout but \
                 sparrow-engine-gpu kernels emit NCHW only (CUDA EP NHWC + dynamic shapes \
                 is bug-prone — ORT issues #27912, #12288)",
                manifest.id
            )));
        }

        // 2. Resolve ONNX file path. Phase 3.8 FP16 opt-in honored here.
        let onnx_path = match manifest.precision {
            Precision::Fp32 => manifest_dir.join(&manifest.model_file),
            Precision::Int8 => manifest_dir.join(&manifest.model_file),
            Precision::Fp16 => manifest_dir.join(manifest.model_file_fp16.as_ref().ok_or_else(
                || {
                    SparrowEngineError::InvalidManifest(format!(
                        "ClassifierModel::load: manifest '{}' precision=fp16 but model_file_fp16 missing",
                        manifest.id
                    ))
                },
            )?),
        };

        // 3. Load labels.
        let labels = match (&manifest.label_file, &manifest.label_format) {
            (Some(file), Some(fmt)) => {
                let label_path = manifest_dir.join(file);
                manifest::load_labels(&label_path, fmt)?
            }
            _ => Vec::new(),
        };

        // 4. Build ORT session pinned to the same CUDA device as `ctx`.
        // GPU-first design: `error_on_failure()` on the CUDA EP catches silent
        // CUDA-registration failures (driver/cuDNN init drama) — that's the
        // real GPU-first guard. The CPU EP is the per-op last-resort fallback
        // for ops the CUDA EP doesn't implement (rare but real for novel
        // graphs); it is NOT a silent full-CPU degradation path. With
        // `error_on_failure()` set, a CUDA-EP registration failure surfaces
        // immediately; the CPU EP only runs ops that the CUDA EP could not.
        // Harmonized with `yolo.rs::build_session` and `tiled.rs` per Phase
        // 3.8 Step 1 audit-fix R2 B7 (S-NEW-6).
        let device_id: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;

        // Per ort 2.0.0-rc.12 source (`ep/mod.rs:336-369`), EP registration
        // failures are silent by default — `error_on_failure()` opts into
        // hard failure for the CUDA EP. The CPU EP is not gated this way:
        // its job is per-op fallback, not full-engine fallback.
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
        let session = Session::builder()
            .map_err(|e| SparrowEngineError::Ort(format!("ort Session::builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| SparrowEngineError::Ort(format!("with_optimization_level: {e}")))?
            .with_execution_providers(providers)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("with_execution_providers(TRT, CUDA, CPU): {e}"))
            })?
            .commit_from_file(&onnx_path)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("commit_from_file({onnx_path:?}): {e}"))
            })?;

        // Phase 3.8 Step 1 audit-fix R2 B10 (M-NEW-5): the FP32 binding code
        // below assumes Float32 I/O. Reject FP16 ONNX converted without
        // `keep_io_types=True` early.
        validate_input_dtype_fp32(&session, &manifest.id)?;
        // Phase 3.8 Step 1 audit-fix R2 B8 (M-NEW-1): catch wrong-shape ONNX
        // at load time instead of garbage softmax output at first inference.
        validate_output_shape_softmax(&session, &manifest.id, labels.len())?;

        // 5. Cache input/output names + CUDA MemoryInfo template.
        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| {
                SparrowEngineError::Ort(format!("session for '{}' has no inputs", manifest.id))
            })?
            .name()
            .to_owned();
        let output_name = session
            .outputs()
            .first()
            .ok_or_else(|| {
                SparrowEngineError::Ort(format!("session for '{}' has no outputs", manifest.id))
            })?
            .name()
            .to_owned();

        let cuda_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| SparrowEngineError::Ort(format!("MemoryInfo::new(CUDA): {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            manifest: Arc::new(manifest.clone()),
            labels: Arc::new(labels),
            input_name,
            output_name,
            cuda_mem_info,
            device_id,
        })
    }

    /// Run classification on a single image.
    ///
    /// Pipeline: nvjpeg decode → CUDA preprocess (dispatched from manifest)
    /// → ORT CUDA EP (zero-copy via `TensorRefMut::from_raw`) → CPU softmax.
    ///
    /// Preprocess dispatch on `manifest.preprocess_method`:
    /// - `Resize` → [`resize_gpu`] (plain bilinear, no aspect preservation,
    ///   matches `sparrow-engine-cpu`'s `resize_direct`). SpeciesNet (`unit`) and
    ///   Amazon Camera Trap v2 (`imagenet`) both use this path; the
    ///   `manifest.normalization` field selects the per-channel mean/std
    ///   passed to the kernel via `NormalizeStats`.
    /// - `Letterbox` → rejected: letterbox is YOLO-family preprocess, not
    ///   classifier preprocess. Misconfiguration.
    /// - `MelSpectrogram` → already rejected at `load` time.
    ///
    /// Normalization dispatch on `manifest.normalization`:
    /// - `Unit` → `NormalizeStats::UNIT` (mean=0, std=1; bit-exact identity).
    /// - `Imagenet` → `NormalizeStats::IMAGENET` (torchvision standard stats).
    /// - `None` → rejected; classifier preprocess always normalizes.
    ///
    /// The `center_crop` kernel parameter is held for forward-compat with
    /// a future `PreprocessMethod::CenterCropResize` variant in
    /// `sparrow-engine-types`. Until that variant lands, the parameter is unused
    /// in the active code path; declaring it keeps the call signature
    /// stable across the eventual enum extension and matches the Wave 3
    /// directive's contract.
    ///
    /// # Errors
    /// - `SparrowEngineError::ImageDecode` if both nvjpeg and CPU fallback decode fail.
    /// - `SparrowEngineError::Ort` for kernel launch / ORT errors / device mismatch.
    /// - `SparrowEngineError::InvalidManifest` for unsupported preprocess method
    ///   (`Letterbox`, `MelSpectrogram`), unsupported normalization (`None`),
    ///   or unsupported layout (`Nhwc`). These are defense-in-depth — `load()`
    ///   already rejects them at manifest validation time (Phase 3.8 Step 1
    ///   audit-fix R3 M8); reachable only if the manifest mutates post-load.
    /// - I/O errors when `image` is `ImageInput::FilePath`.
    pub fn classify(
        &self,
        ctx: &Arc<CudaContext>,
        center_crop: &CenterCropKernel,
        resize: &ResizeKernel,
        decoder: &mut JpegDecoder,
        image: &ImageInput,
        opts: &ClassifyOpts,
    ) -> Result<ClassifyResult> {
        let _ = center_crop; // see docstring — held for forward-compat
        let start = Instant::now();

        // Validate the per-call ctx matches the session's pinned device.
        // Cheap guard to catch caller-side bugs (mixing engines).
        let ctx_ordinal: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        if ctx_ordinal != self.device_id {
            return Err(SparrowEngineError::Ort(format!(
                "ClassifierModel::classify: ctx device {} != session device {}",
                ctx_ordinal, self.device_id
            )));
        }

        let stream = ctx.default_stream();

        // 1. Resolve `ImageInput` to a GPU-resident HWC RGB u8 buffer.
        //    - Encoded JPEG bytes → cached nvjpeg fast path (`decoder.decode_to_gpu`).
        //    - FilePath → read bytes from disk → cached nvjpeg fast path.
        //    - Raw → direct H→D copy via `crate::decode::raw_to_gpu`.
        //
        // Phase 3.8 Step 1 audit-fix R2 B9 (M-NEW-2): Raw was previously
        // rejected with `SparrowEngineError::Ort("not yet implemented")`. The
        // canonical helper lives in `crate::decode::raw_to_gpu` and is
        // shared with `YoloModel::detect` and `TiledModel::detect_tiled`.
        let gpu_img = match image {
            ImageInput::Encoded(b) => decoder.decode_to_gpu(&stream, b)?,
            ImageInput::FilePath(p) => {
                let bytes = read_image_file(p)?;
                decoder.decode_to_gpu(&stream, &bytes)?
            }
            ImageInput::Raw {
                data,
                width,
                height,
                stride,
                format,
            } => crate::decode::raw_to_gpu(&stream, data, *width, *height, *stride, *format)?,
        };
        let original_w = gpu_img.width;
        let original_h = gpu_img.height;

        // 2. Resolve target size + channel order from manifest.
        let input_size = self.manifest.input_size.ok_or_else(|| {
            // Defense-in-depth: load() rejects this at manifest validation
            // time (Phase 3.8 Step 1 audit-fix R3 M8). Reachable only if the
            // manifest mutates post-load (impossible via the public API).
            SparrowEngineError::InvalidManifest(format!(
                "manifest '{}' missing input_size",
                self.manifest.id
            ))
        })?;
        let target_w = input_size[0];
        let target_h = input_size[1];
        // Bongo's manifest defaults `channel_order` to RGB when absent.
        let channel_order = self.manifest.channel_order.unwrap_or(ChannelOrder::Rgb);
        // Resolve normalization stats from the manifest. SpeciesNet:
        // `unit` → identity. Amazon CTV2: `imagenet` → torchvision-standard
        // mean/std. `none` is rejected up-front: classifier preprocess always
        // produces a normalized tensor; if a future model needs raw 0..255
        // input, extend `NormalizeStats` (e.g., `RAW` with mean=0, std=1/255).
        let stats = match self.manifest.normalization.unwrap_or(Normalization::Unit) {
            Normalization::Unit => NormalizeStats::UNIT,
            Normalization::Imagenet => NormalizeStats::IMAGENET,
            Normalization::None => {
                // Defense-in-depth: load() rejects this at manifest validation
                // time (Phase 3.8 Step 1 audit-fix R3 M8). Reachable only if the
                // manifest mutates post-load (impossible via the public API).
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "ClassifierModel::classify: manifest '{}' specifies normalization = 'none'; \
                     classifier resize_gpu has no raw-passthrough variant. Add a NormalizeStats::RAW \
                     and a kernel branch if a future model needs it.",
                    self.manifest.id
                )));
            }
        };

        // 3. GPU preprocess dispatched on manifest method.
        // SpeciesNet's manifest is `Resize` + `unit`, so it lands in
        // `resize_gpu` with `NormalizeStats::UNIT` (mean=[0,0,0], std=[1,1,1])
        // — bit-exact identity vs the pre-Amazon `/255`-only kernel.
        // Amazon CTV2 takes the same path with `NormalizeStats::IMAGENET`.
        let dev_tensor: CudaSlice<f32> = match self.manifest.preprocess_method {
            PreprocessMethod::Resize => resize_gpu(
                &stream,
                resize,
                &gpu_img,
                target_w,
                target_h,
                channel_order,
                stats,
            )?,
            PreprocessMethod::Letterbox => {
                // Defense-in-depth: load() rejects this at manifest validation
                // time (Phase 3.8 Step 1 audit-fix R3 M8). Reachable only if the
                // manifest mutates post-load (impossible via the public API).
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "ClassifierModel::classify: manifest '{}' specifies preprocess method 'letterbox', which is YOLO-family preprocess (not a classifier preprocess). Use a detector engine, or change the manifest method.",
                    self.manifest.id
                )));
            }
            PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. } => {
                // Defense-in-depth: load() rejects audio manifests at validation
                // time (B6 / S-NEW-3). Aligned to `InvalidManifest` for variant
                // consistency with the other classify() defense-in-depth arms
                // (Phase 3.8 Step 1 audit-fix R3 M8 cleanup).
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "ClassifierModel::classify: manifest '{}' has audio preprocess ({}) — rejected at load",
                    self.manifest.id,
                    self.manifest.preprocess_method.as_str(),
                )));
            }
        };

        // Synchronize so the tensor is fully written before ORT consumes it.
        stream
            .synchronize()
            .map_err(|e| SparrowEngineError::Ort(format!("stream.synchronize before run: {e}")))?;

        // 4. Bind tensor as a CUDA-resident ORT input + run inference.
        // Validate the kernel-output layout against manifest layout.
        let layout = self.manifest.layout.unwrap_or(Layout::Nchw);
        let shape: Shape = match layout {
            Layout::Nchw => Shape::from([1i64, 3, target_h as i64, target_w as i64]),
            Layout::Nhwc => {
                // Defense-in-depth: load() rejects this at manifest validation
                // time (Phase 3.8 Step 1 audit-fix R3 M8). Reachable only if the
                // manifest mutates post-load (impossible via the public API).
                // Wave 1 kernel emits NCHW; an NHWC manifest would require a
                // separate kernel branch (out of Step 1 scope; NHWC was ruled
                // out for ORT CUDA EP in v3 design decisions).
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "ClassifierModel::classify: manifest '{}' specifies NHWC layout but sparrow-engine-gpu kernels emit NCHW only (CUDA EP NHWC + dynamic shapes is bug-prone — ORT issues #27912, #12288)",
                    self.manifest.id
                )));
            }
        };

        // Two input-binding paths, controlled at run-time:
        // - Default: zero-copy CUDA-resident binding via TensorRefMut::from_raw
        //   with a CUDA `MemoryInfo`. ORT CUDA EP consumes the device pointer
        //   directly when memory_info matches the EP's device.
        // - `SPARROW_ENGINE_GPU_CLASSIFIER_HOST_ROUNDTRIP=1`: copy the device tensor
        //   back to host (`stream.clone_dtoh`), build an ndarray, and bind
        //   via TensorRef::from_array_view. This is the sparrow-engine-cpu equivalent
        //   path and the diagnostic comparison point — useful for isolating
        //   whether per-call slowdown sits in CUDA-EP-friendly binding or in
        //   host-bound preprocess. Default off; selected when present.
        let host_roundtrip =
            std::env::var("SPARROW_ENGINE_GPU_CLASSIFIER_HOST_ROUNDTRIP").as_deref() == Ok("1");

        let classifications = if host_roundtrip {
            let host_buf: Vec<f32> = stream.clone_dtoh(&dev_tensor).map_err(|e| {
                SparrowEngineError::Ort(format!("stream.clone_dtoh (host roundtrip): {e}"))
            })?;
            stream.synchronize().map_err(|e| {
                SparrowEngineError::Ort(format!("stream.synchronize after dtoh: {e}"))
            })?;
            let arr: ndarray::Array4<f32> = ndarray::Array4::from_shape_vec(
                (1, 3, target_h as usize, target_w as usize),
                host_buf,
            )
            .map_err(|e| SparrowEngineError::Ort(format!("Array4::from_shape_vec: {e}")))?;
            let input_value = TensorRef::from_array_view(&arr)
                .map_err(|e| SparrowEngineError::Ort(format!("TensorRef::from_array_view: {e}")))?;
            let mut guard = self.session.lock().map_err(|_| {
                SparrowEngineError::Ort("ClassifierModel session lock poisoned".into())
            })?;
            let outputs = guard
                .run(ort::inputs![&self.input_name => input_value])
                .map_err(|e| {
                    SparrowEngineError::Ort(format!("Session::run (host roundtrip): {e}"))
                })?;
            extract_softmax_top_k(&outputs, &self.output_name, &self.labels, opts)?
        } else {
            // Zero-copy CUDA path.
            let (dev_ptr_u64, _sync) = dev_tensor.device_ptr(&stream);
            let mem_info = self.cuda_mem_info.clone();

            // SAFETY: `dev_ptr_u64` is the device pointer to `dev_tensor`,
            // which is owned by this scope and live for the lifetime of
            // the inference call. The pointer is valid for `1 * 3 *
            // target_h * target_w` f32 values (verified by the
            // `resize_gpu` post-condition: it allocates exactly
            // `3*tgt_h*tgt_w` f32s — see `kernels/resize.rs`). `mem_info`
            // describes the device the pointer lives on (CUDA device
            // `self.device_id`, AllocatorType::Device).
            let input_tensor = unsafe {
                TensorRefMut::<f32>::from_raw(
                    mem_info,
                    dev_ptr_u64 as usize as *mut std::ffi::c_void,
                    shape,
                )
            }
            .map_err(|e| SparrowEngineError::Ort(format!("TensorRefMut::from_raw: {e}")))?;

            let mut guard = self.session.lock().map_err(|_| {
                SparrowEngineError::Ort("ClassifierModel session lock poisoned".into())
            })?;
            let outputs = guard
                .run(ort::inputs![&self.input_name => input_tensor])
                .map_err(|e| SparrowEngineError::Ort(format!("Session::run: {e}")))?;
            extract_softmax_top_k(&outputs, &self.output_name, &self.labels, opts)?
        };
        // dev_tensor drops at end of scope (after the block).

        let elapsed = start.elapsed();
        Ok(ClassifyResult {
            classifications,
            image_width: original_w,
            image_height: original_h,
            processing_time_ms: elapsed.as_secs_f32() * 1000.0,
        })
    }

    /// Manifest snapshot for diagnostics / engine integration.
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Manifest ID accessor. Mirrors `YoloModel::model_id` and
    /// `TiledModel::model_id`; aligns with the system-wide
    /// `sparrow_engine_cpu::engine::ModelHandle::model_id()` convention.
    pub fn model_id(&self) -> &str {
        &self.manifest.id
    }
}

/// Extract logits from a SessionOutputs and apply softmax. Shared between
/// the zero-copy CUDA-binding path and the host-roundtrip diagnostic path
/// in [`ClassifierModel::classify`].
fn extract_softmax_top_k(
    outputs: &ort::session::SessionOutputs<'_>,
    output_name: &str,
    labels: &[String],
    opts: &ClassifyOpts,
) -> Result<Vec<sparrow_engine_types::Classification>> {
    let output = outputs.get(output_name).ok_or_else(|| {
        SparrowEngineError::Ort(format!("classifier output '{output_name}' not found"))
    })?;
    let output_view: ArrayViewD<'_, f32> = output
        .try_extract_array::<f32>()
        .map_err(|e| SparrowEngineError::Ort(format!("try_extract_array: {e}")))?;
    let ndim = output_view.ndim();
    let view_2d: ArrayView2<f32> = if ndim == 2 {
        output_view
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| SparrowEngineError::Ort(format!("into_dimensionality 2D: {e}")))?
    } else if ndim == 1 {
        let len = output_view.len();
        output_view
            .into_shape_with_order((1, len))
            .map_err(|e| SparrowEngineError::Ort(format!("into_shape_with_order 1->2: {e}")))?
    } else {
        return Err(SparrowEngineError::Ort(format!(
            "Unexpected classifier output rank {ndim}; expected 1 or 2"
        )));
    };
    postprocess::try_softmax(&view_2d, labels, opts)
}

impl ClassifierModel {
    /// Class labels (cloned `Arc`) for callers that need to map label_id ↔ name.
    pub fn labels(&self) -> &Arc<Vec<String>> {
        &self.labels
    }
}

/// Validate that the session's first input is `Float32`. FP16 ONNX must be
/// converted with `onnxruntime.transformers.float16.keep_io_types=True` so
/// the I/O dtypes remain Float32 (the FP16 Cast nodes are internal to the
/// graph). True-FP16 I/O would crash the FP32 binding code at `session.run`
/// with a typed-tensor mismatch; reject at load time instead.
///
/// Phase 3.8 Step 1 audit-fix R2 B10 (M-NEW-5).
fn validate_input_dtype_fp32(session: &Session, model_id: &str) -> Result<()> {
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

/// Validate the softmax classifier output shape at load time.
///
/// Accepts:
/// - rank-1 `[N]` (raw logits, no batch dim)
/// - rank-2 `[1, N]` (with explicit batch=1)
/// - rank-2 `[-1, N]` (dynamic batch dim)
///
/// `num_classes` must equal `N` if `num_classes > 0` (i.e., when labels are
/// loaded from the manifest). When `num_classes == 0` (no label file), only
/// the rank check fires; the dimension match is skipped.
///
/// Phase 3.8 Step 1 audit-fix R2 B8 (M-NEW-1). Mirrors `yolo.rs::validate_output_shape`.
fn validate_output_shape_softmax(
    session: &Session,
    model_id: &str,
    num_classes: usize,
) -> Result<()> {
    // Phase 3.8 Step 1 doc-fix R1 F-C9: collapse the `is_empty` early-return +
    // position-indexing into a single safe `?` chain. Eliminates the implicit
    // panic risk on `outputs[0]` in source even though the gate above made it
    // unreachable today.
    let first_output =
        session
            .outputs()
            .first()
            .ok_or_else(|| SparrowEngineError::OutputShapeMismatch {
                id: model_id.to_string(),
                shape: "no outputs".to_string(),
                method: "softmax".to_string(),
            })?;
    let dims: Vec<i64> = match first_output.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => vec![],
    };
    let last_dim = match dims.len() {
        1 => dims[0],
        2 if dims[0] == 1 || dims[0] == -1 => dims[1],
        _ => {
            return Err(SparrowEngineError::OutputShapeMismatch {
                id: model_id.to_string(),
                shape: format!("{dims:?} (expected rank-1 [N] or rank-2 [1, N] / [-1, N])"),
                method: "softmax".to_string(),
            });
        }
    };
    // Dynamic last dim (-1) accepted; concrete dim must match num_classes if
    // labels are loaded.
    if last_dim != -1 && num_classes != 0 && last_dim as usize != num_classes {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: format!("{dims:?} (expected last dim = {num_classes})"),
            method: "softmax".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{InferenceStrategy, LabelFormat, Normalization};
    use sparrow_engine_types::ModelSubtype;

    fn dummy_manifest(method: PostprocessMethod) -> ModelManifest {
        ModelManifest {
            id: "test".into(),
            format: "onnx".into(),
            model_file: "test.onnx".into(),
            preprocess_method: PreprocessMethod::Resize,
            input_size: Some([480, 480]),
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Unit),
            pad_value: None,
            channel_order: None,
            precision: Precision::Fp32,
            model_file_fp16: None,
            inference_strategy: InferenceStrategy::Single,
            trt: None,
            postprocess_method: method,
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

    #[test]
    fn read_image_file_missing_path_returns_image_file_not_found() {
        let missing = Path::new("sparrow-engine-gpu-test-missing-classifier-image.jpg");
        assert!(
            !missing.exists(),
            "test sentinel path unexpectedly exists: {}",
            missing.display()
        );
        match super::read_image_file(missing) {
            Err(SparrowEngineError::ImageFileNotFound(path)) => assert_eq!(path, missing),
            Err(other) => panic!("expected ImageFileNotFound, got {other:?}"),
            Ok(_) => panic!("expected missing classifier image to fail"),
        }
    }

    #[test]
    fn load_rejects_audio_manifest() {
        // The validation short-circuits before any CUDA work, but
        // ClassifierModel::load still takes &Arc<CudaContext> in its
        // signature, so we need a real (or skipped) context.
        let mut m = dummy_manifest(PostprocessMethod::Softmax);
        m.preprocess_method = PreprocessMethod::MelSpectrogram {
            sample_rate: 16000,
            n_fft: 512,
            hop_length: 160,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8000.0,
            top_db: 80.0,
            window: "hann".into(),
            mel_scale: "slaney".into(),
            filter_norm: "slaney".into(),
            fill_highfreq: false,
        };

        let ctx = match cuda_or_skip("load_rejects_audio_manifest") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        match ClassifierModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg)) if msg.contains("audio model") => {}
            Err(other) => panic!("expected audio rejection, got Err({other:?})"),
            Ok(_) => panic!("expected audio rejection, got Ok(_)"),
        }
    }

    #[test]
    fn load_rejects_non_softmax_postprocess() {
        let m = dummy_manifest(PostprocessMethod::YoloE2e);
        let ctx = match cuda_or_skip("load_rejects_non_softmax_postprocess") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        match ClassifierModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg)) if msg.contains("expected softmax") => {}
            Err(other) => panic!("expected non-softmax rejection, got Err({other:?})"),
            Ok(_) => panic!("expected non-softmax rejection, got Ok(_)"),
        }
    }

    /// Phase 3.8 Step 1 audit-fix R2 B6 regression test (S-NEW-3).
    ///
    /// `precision = fp16` without an accompanying `model_file_fp16` was
    /// previously rejected with `SparrowEngineError::Ort` (semantically wrong — this
    /// is a manifest-validation failure, not an ORT runtime error). The
    /// audit-fix reroutes the rejection to `SparrowEngineError::InvalidManifest`,
    /// matching `YoloModel::load` and `TiledModel::load`.
    #[test]
    fn load_rejects_fp16_without_model_file_fp16() {
        let mut m = dummy_manifest(PostprocessMethod::Softmax);
        m.precision = Precision::Fp16;
        m.model_file_fp16 = None;
        let ctx = match cuda_or_skip("load_rejects_fp16_without_model_file_fp16") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        match ClassifierModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg))
                if msg.contains("model_file_fp16") && msg.contains("fp16") => {}
            Err(other) => panic!(
                "expected InvalidManifest(... model_file_fp16 missing ...), got Err({other:?})"
            ),
            Ok(_) => panic!("expected InvalidManifest rejection, got Ok(_)"),
        }
    }

    // -------------------------------------------------------------------
    // Phase 3.8 Step 1 audit-fix R3 M8 regression tests
    //
    // Mirror the existing `load_rejects_*` patterns. These cover the 4
    // manifest-defect sites hoisted from `classify()` runtime path into
    // `load()` for fail-fast semantics:
    //   - Letterbox preprocess (was line 711-715 of classify pre-hoist)
    //   - missing input_size  (was line 667-672 of classify pre-hoist)
    //   - Normalization::None (was line 686-692 of classify pre-hoist)
    //   - NHWC layout         (was line 741-744 of classify pre-hoist)
    // All four pre-hoist sites returned `SparrowEngineError::Ort`; they now return
    // `SparrowEngineError::InvalidManifest` at load time, matching B6 + B10.
    // -------------------------------------------------------------------

    #[test]
    fn load_rejects_letterbox_preprocess_method() {
        let mut m = dummy_manifest(PostprocessMethod::Softmax);
        m.preprocess_method = PreprocessMethod::Letterbox;
        let ctx = match cuda_or_skip("load_rejects_letterbox_preprocess_method") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        match ClassifierModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg)) if msg.contains("letterbox") => {}
            Err(other) => panic!("expected letterbox rejection, got Err({other:?})"),
            Ok(_) => panic!("expected letterbox rejection, got Ok(_)"),
        }
    }

    #[test]
    fn load_rejects_missing_input_size() {
        let mut m = dummy_manifest(PostprocessMethod::Softmax);
        m.input_size = None;
        let ctx = match cuda_or_skip("load_rejects_missing_input_size") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        match ClassifierModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg)) if msg.contains("input_size") => {}
            Err(other) => panic!("expected missing-input_size rejection, got Err({other:?})"),
            Ok(_) => panic!("expected missing-input_size rejection, got Ok(_)"),
        }
    }

    #[test]
    fn load_rejects_normalization_none() {
        let mut m = dummy_manifest(PostprocessMethod::Softmax);
        m.normalization = Some(Normalization::None);
        let ctx = match cuda_or_skip("load_rejects_normalization_none") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        match ClassifierModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg)) if msg.contains("normalization") => {}
            Err(other) => panic!("expected normalization-none rejection, got Err({other:?})"),
            Ok(_) => panic!("expected normalization-none rejection, got Ok(_)"),
        }
    }

    #[test]
    fn load_rejects_nhwc_layout() {
        let mut m = dummy_manifest(PostprocessMethod::Softmax);
        m.layout = Some(Layout::Nhwc);
        let ctx = match cuda_or_skip("load_rejects_nhwc_layout") {
            Some(c) => c,
            None => return,
        };
        let manifest_dir = Path::new("/tmp");
        match ClassifierModel::load(&ctx, &m, manifest_dir) {
            Err(SparrowEngineError::InvalidManifest(msg)) if msg.contains("NHWC") => {}
            Err(other) => panic!("expected NHWC-layout rejection, got Err({other:?})"),
            Ok(_) => panic!("expected NHWC-layout rejection, got Ok(_)"),
        }
    }
}
