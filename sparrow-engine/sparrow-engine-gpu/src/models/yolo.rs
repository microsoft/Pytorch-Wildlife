//! YOLO E2E inference path (MDv6, DeepFaune). Phase 3.8 Step 1.
//!
//! Pipeline shape (`docs/design/phase3.8/final_design.md §4`):
//! ```text
//! ImageInput -> nvjpeg decode -> CUDA letterbox+normalize+NCHW (BGR)
//!            -> ORT CUDA EP (GPU-resident IoBinding) -> CPU yolo_e2e
//!            postprocess (sparrow-engine-core)
//! ```
//!
//! # Public API
//!
//! - [`YoloModel::load`]: parse manifest, build ORT session against the
//!   FP32 or FP16 ONNX file, validate output shape, cache the stateful
//!   nvjpeg decoder + per-load env-var snapshots.
//! - [`YoloModel::detect`]: single-image GPU-resident pipeline.
//! - [`YoloModel::detect_batch_pipelined`]: bench-only experimental
//!   batched path that overlaps decode-of-N+1 with ORT.run-of-N. Opt-in
//!   from the bench harness via `SPARROW_ENGINE_GPU_YOLO_BATCH_PIPELINE=1`.
//!
//! # Inference path
//!
//! GPU-resident IoBinding via [`ort::value::TensorRefMut::from_raw`] with
//! a CUDA `MemoryInfo`: the letterbox kernel's GPU output buffer is bound
//! directly as the ORT input, skipping ORT's implicit Host→Device upload
//! that the legacy host-roundtrip path paid (~24 ms / image at MDv6's
//! `1 × 3 × 1280 × 1280` FP32 input shape). Output is a small
//! `~7.2 KB` `[1, 300, 6]` tensor; we DtoH-copy it to drive
//! `sparrow_engine_core::postprocess::yolo_e2e` on CPU. FP16 ONNX (converted via
//! `onnxruntime.transformers.float16` with `keep_io_types=True`) accepts
//! FP32 input and emits FP32 output — the Cast lives inside the graph,
//! so a single Rust inference path handles both precisions.
//!
//! cuDNN algo selection defaults to ORT's HEURISTIC search (Path 2 follow-up
//! Lever B per `docs/research/phase3.8/step1/mdv6_perf_investigation.md`).
//! The `SPARROW_ENGINE_GPU_YOLO_CONV_SEARCH` env var below selects EXHAUSTIVE / HEURISTIC
//! / DEFAULT for diagnostic A/B.
//!
//! NO `cuda_graph` — refuted in Phase 3.7 R5
//! (`docs/research/phase3.7/track_b/experiments/results.md § R5`): stale-by-1
//! correctness bug in `ort 2.0.0-rc.12` cuda_graph capture + no measurable
//! speedup once IoBinding alone removes the implicit upload.
//!
//! # Environment overrides
//!
//! Three of the four env vars below are read at `YoloModel::load()` time and
//! cached on the struct (no per-call `std::env::var` syscalls in the hot path).
//!
//! | env var                          | accepted | effect |
//! |----------------------------------|----------|--------|
//! | `SPARROW_ENGINE_GPU_FORCE_CPU_DECODE`     | `1`      | Skip nvjpeg; CPU-decode every image. Diagnostic A/B for nvjpeg-vs-image-crate parity. |
//! | `SPARROW_ENGINE_GPU_YOLO_HOST_ROUNDTRIP`  | `1`      | Use the legacy host-roundtrip ORT path (DtoH input, let ORT re-upload). Diagnostic A/B vs the GPU-resident default; production cost is ~24 ms/image of pure roundtrip. |
//! | `SPARROW_ENGINE_GPU_YOLO_CONV_SEARCH`     | `default`/`heuristic`/`exhaustive` (case-insensitive) | Override cuDNN convolution algo selection. Default = `heuristic` (Path 2 Lever B). |
//! | `SPARROW_ENGINE_GPU_YOLO_BATCH_PIPELINE`  | `1` (bench harness only) | Route the bench harness through `detect_batch_pipelined` instead of per-image `detect`. Not read inside `models/yolo.rs`. |
//!
//! **Exception**: `SPARROW_ENGINE_GPU_FORCE_CPU_DECODE` is re-read on every call inside
//! `JpegDecoder::decode_to_gpu` so the runtime A/B toggle works without process
//! restart (Wave 2 + Wave 3 parity-debug workflow). The cost is a single
//! `getenv` call per inference (~50 ns); negligible vs the nvjpeg / letterbox /
//! ORT timings. The other three env vars above are cached at load.

use std::path::Path;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ndarray::{Array2, ArrayView2};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{Shape, TensorRef, TensorRefMut};
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{
    self, ChannelOrder, Layout, ModelManifest, Normalization, PostprocessMethod, Precision,
    PreprocessMethod,
};
use sparrow_engine_types::{DetectOpts, DetectResult, Detection, ImageInput, PreprocessMeta};

use crate::decode::GpuImage;
use crate::kernels::letterbox::{letterbox_gpu, LetterboxKernel, LetterboxMeta};
use crate::trt::ep::{manifest_cache_material, CudaEpConfig, GpuIdentity, TrtEpBuilder};

// ===========================================================================
// JpegDecoder — stateful nvjpeg wrapper.
//
// Owns one nvjpeg handle + state pair, reused across calls. Drop-time RAII
// releases via `nvjpegJpegStateDestroy` + `nvjpegDestroy`. Bound to a CUDA
// context (kept alive via the `_ctx` Arc) so the handle is valid for
// `nvjpegDecode` calls on streams created from that context.
//
// This mirrors `models/classifier.rs::JpegDecoder` (coder-w3, Wave 3). A
// shared lift into `decode.rs` is a Wave 5 task; for now each model file
// caches its own decoder. Per-call cost drops from ~785 ms (Wave 1's
// `decode::decode_jpeg`) to a few ms.
// ===========================================================================

/// Stateful JPEG decoder backed by nvjpeg, with a CPU fallback.
pub struct JpegDecoder {
    handle: nvjpeg_sys::nvjpegHandle_t,
    state: nvjpeg_sys::nvjpegJpegState_t,
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
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        use nvjpeg_sys as nvj;
        // Phase E (2026-05-25): consult the dlopen loader BEFORE the
        // existing nvjpegCreateSimple call. If libnvjpeg.so.12 is missing /
        // wrong major / has missing symbols, surface the rich NvjpegInitError
        // (remediation text) via SparrowEngineError::NvjpegUnavailable instead
        // of letting the thin wrapper below flatten to status=1 (which would
        // bubble up as a misleading "ONNX Runtime error: nvjpegCreateSimple
        // failed: status=1").
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

    fn decode_to_gpu(&mut self, stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<GpuImage> {
        // Wave-2-amend experiment toggle: setting `SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1`
        // skips the nvjpeg fast path entirely and routes every image through
        // the `image` crate (CPU decode + HtoD copy). Used to isolate "is the
        // remaining DeepFaune parity drift from JPEG-decode divergence
        // (nvjpeg vs image crate IDCT) or from the GPU letterbox kernel?".
        // Default behaviour unchanged: nvjpeg fast path with CPU fallback.
        if std::env::var("SPARROW_ENGINE_GPU_FORCE_CPU_DECODE").as_deref() == Ok("1") {
            return decode_via_cpu_fallback(stream, bytes);
        }
        match self.decode_via_nvjpeg(stream, bytes) {
            Ok(img) => return Ok(img),
            Err(e) => {
                // Phase 3.8 Step 1 doc-fix R1 F-C6: log nvjpeg failures so the
                // CPU-fallback frequency is observable in production. Mirrors
                // `models/classifier.rs::JpegDecoder::decode_to_gpu`.
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

/// CPU-decode fallback (mirrors `crate::decode::decode_via_cpu_fallback`,
/// which is private to that module).
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

/// Cheap EXIF orientation pre-check (mirrors `crate::decode`'s private
/// `has_nontrivial_exif_orientation`).
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

/// A loaded YOLO E2E model on GPU (MDv6 / DeepFaune family).
///
/// Holds an ORT session bound to the CUDA EP, the parsed manifest, the
/// label table, and a stateful nvjpeg decoder. `detect` runs the GPU
/// pipeline end-to-end.
pub struct YoloModel {
    /// ORT session for this model. Wrapped behind `Mutex` because
    /// `Session::run` requires `&mut self`. Cheap to lock — single
    /// inference call holds it for the duration of one image.
    session: Arc<Mutex<Session>>,
    /// Cached nvjpeg decoder (handle + state). Behind a `Mutex` because
    /// `JpegDecoder::decode_to_gpu` requires `&mut self` while `detect`
    /// is `&self` (matches the Wave 2 brief's external surface).
    decoder: Mutex<JpegDecoder>,
    /// Parsed manifest.
    manifest: Arc<ModelManifest>,
    /// Ordered label names. Index = label_id.
    labels: Arc<Vec<String>>,
    /// Resolved input dimensions from the manifest. Cached to avoid
    /// re-parsing on every `detect` call.
    input_w: u32,
    input_h: u32,
    /// Resolved channel order (RGB / BGR). MDv6 + DeepFaune ship BGR.
    channel_order: ChannelOrder,
    /// Pad value in the closed range 0 to 1 (after `/255` normalization).
    pad_value: f32,
    /// Default confidence threshold from manifest. Overrides default if
    /// `DetectOpts::confidence_threshold` is None.
    default_threshold: f32,
    /// CUDA `MemoryInfo` for binding device pointers as ORT inputs.
    /// Pre-built at load time; cloned per-call. Mirrors `ClassifierModel`'s
    /// pattern (Wave 3 commit 64646b8).
    cuda_mem_info: MemoryInfo,
    /// Device ordinal captured at load time. Validates per-call ctx
    /// matches the session's pinned EP device.
    device_id: i32,
    /// Cached value of `SPARROW_ENGINE_GPU_YOLO_HOST_ROUNDTRIP=1` env var,
    /// resolved once at `load()` time. Selects the legacy host-roundtrip
    /// inference path for diagnostic A/B vs the GPU-resident default.
    /// Process-lifetime: changing the env var after `load()` has no
    /// effect (single-process bench is the supported usage).
    use_host_roundtrip: bool,
}

// SAFETY: All non-Send/Sync ORT types (`Session`) are wrapped behind
// `std::sync::Mutex`. `JpegDecoder` is wrapped in a separate `Mutex`.
// Manifest + labels are `Arc<POD>` / `String` / `i32` (thread-safe POD).
// The session lock serializes the only mutable ORT access. Mirrors the
// SAFETY rationale on `ClassifierModel` and `TiledModel` (Phase 3.8 Step 1
// audit-fix R2 A5 / N-NEW-2 harmonization).
unsafe impl Send for YoloModel {}
unsafe impl Sync for YoloModel {}

// ============================================================================
// Pipelined batch decode-ahead state (Lever A of Path 2 perf follow-up).
// ============================================================================

/// A pre-staged preprocess result: GPU-resident NCHW input tensor + the
/// letterbox geometry + original-image dims + the stream that owns the
/// pending GPU work. Consumed by `detect_consume_and_prepare_next`.
struct PreparedPreprocess {
    input_tensor_f32: CudaSlice<f32>,
    lb_meta: LetterboxMeta,
    original_w: u32,
    original_h: u32,
    decode_stream: Arc<CudaStream>,
}

/// Caller-side handle for the pipelined detect_batch path. Holds the
/// "next image" preprocess slot that's filled while the previous image's
/// ORT.run is in flight.
///
/// Crate-internal: only constructed inside [`YoloModel::detect_batch_pipelined`]
/// and threaded through private helpers. Not part of the stable public surface.
pub(crate) struct YoloDecodeAhead {
    pending: Option<PreparedPreprocess>,
}

impl YoloModel {
    /// Load a YOLO model from a parsed manifest.
    ///
    /// Builds an ORT session with CUDA EP + HEURISTIC cuDNN algo selection
    /// (Path 2 Lever B; overridable via `SPARROW_ENGINE_GPU_YOLO_CONV_SEARCH`),
    /// validates the output shape (rank-3 with last dim = 6 or dynamic),
    /// loads label file, and caches the manifest + dimensions.
    ///
    /// `manifest_dir` is used to resolve the (relative) `model_file` /
    /// `model_file_fp16` / `label_file` paths.
    ///
    /// Errors:
    /// - `InvalidManifest` if preprocess/postprocess methods aren't
    ///   `Letterbox` / `YoloE2e`.
    /// - `OutputShapeMismatch` if the ONNX output shape isn't compatible
    ///   with `yolo_e2e` postprocess.
    /// - `Ort` on ORT session creation failures.
    pub fn load(
        ctx: &Arc<CudaContext>,
        manifest: &ModelManifest,
        manifest_dir: &Path,
    ) -> Result<Self> {
        // Validate this manifest matches the YOLO E2E shape we implement.
        if !matches!(manifest.preprocess_method, PreprocessMethod::Letterbox) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "YoloModel requires preprocess_method = letterbox, got {:?}",
                manifest.preprocess_method
            )));
        }
        if !matches!(
            manifest.postprocess_method,
            PostprocessMethod::YoloE2e | PostprocessMethod::MegadetV5a { .. }
        ) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "YoloModel requires postprocess_method = yolo_e2e or megadet_v5a, got {:?}",
                manifest.postprocess_method
            )));
        }

        let input_size = manifest.input_size.ok_or_else(|| {
            SparrowEngineError::InvalidManifest("YoloModel requires manifest.input_size".into())
        })?;
        let layout = manifest.layout.unwrap_or(Layout::Nchw);
        if !matches!(layout, Layout::Nchw) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "YoloModel requires NCHW layout, got {layout:?}"
            )));
        }
        let normalization = manifest.normalization.unwrap_or(Normalization::Unit);
        if !matches!(normalization, Normalization::Unit) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "YoloModel requires unit normalization, got {normalization:?}"
            )));
        }

        let channel_order = manifest.channel_order.unwrap_or(ChannelOrder::Rgb);
        // `manifest.pad_value` is already in [0,1] post-`/255` units (per the
        // manifest spec — sparrow-engine-cpu's `preprocess_config_from_manifest` reads
        // it without further normalization). Default 0.447 ≈ 114/255, the
        // PW/Ultralytics letterbox gray. DO NOT divide by 255 again — the
        // earlier /255 here was a real bug that made pad regions ~0
        // (near-black) instead of gray, producing detection-count drift +2/100
        // on MDv6 1280×960 inputs (which need 160 px of vertical padding).
        let pad_value = manifest.pad_value.unwrap_or(114.0 / 255.0);

        // Resolve ONNX file path.
        let onnx_path = match manifest.precision {
            Precision::Fp32 => manifest_dir.join(&manifest.model_file),
            Precision::Int8 => manifest_dir.join(&manifest.model_file),
            Precision::Fp16 => {
                manifest_dir.join(manifest.model_file_fp16.as_ref().ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(
                        "precision = fp16 requires model_file_fp16 in manifest".into(),
                    )
                })?)
            }
        };

        // Load labels.
        let labels = match (&manifest.label_file, &manifest.label_format) {
            (Some(file), Some(fmt)) => {
                let label_path = manifest_dir.join(file);
                manifest::load_labels(&label_path, fmt)?
            }
            _ => Vec::new(),
        };

        // CUDA MemoryInfo + device_id captured up-front so `build_session`
        // can pin the ORT CUDA EP to the same device ordinal as `ctx`.
        // Previously `build_session` used `CUDA::default()` with no
        // `with_device_id` pin, which silently resolved to device 0 in
        // ORT's EP factory and could mis-pin multi-GPU setups even though
        // the per-call ordinal guard at the top of `detect()` would catch
        // a later mismatch (Phase 3.8 Step 1 audit-fix R1 B2 MODIFY).
        let device_id: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;

        // Build ORT session via the TRT→CUDA→CPU EP policy
        // (crate::trt::ep::TrtEpBuilder): TRT only when the manifest opts in,
        // else CUDA EP first, CPU per-op fallback.
        // EXHAUSTIVE cuDNN algo selection is ORT's default for the CUDA
        // EP — matches sparrow-engine-cpu.
        let gpu = GpuIdentity::from_context(ctx)?;
        let session = build_session(ctx, &gpu, manifest, &onnx_path, device_id)?;
        validate_output_shape(
            &session,
            &manifest.id,
            &manifest.postprocess_method,
            labels.len(),
        )?;
        // Phase 3.8 Step 1 audit-fix R2 B10 (M-NEW-5): the YOLO binding code
        // assumes FP32 input + FP32 output. FP16 ONNX must be converted with
        // `onnxruntime.transformers.float16.keep_io_types=True` so the
        // session.inputs()[0]/outputs()[0] dtypes stay Float32 (Cast nodes
        // are internal). True-FP16 I/O would crash at session.run with a
        // typed-tensor mismatch; reject at load time instead.
        validate_input_dtype_fp32(&session, &manifest.id)?;
        validate_output_dtype_fp32(&session, &manifest.id)?;

        // Construct the cached nvjpeg decoder once; reused by every
        // `detect()` call. The Wave 1 `decode::decode_jpeg` path created /
        // destroyed `nvjpegHandle` + `nvjpegJpegState` per call, costing
        // ~785 ms / call. Caching drops that to a few ms.
        let decoder = JpegDecoder::new(ctx)?;

        // CUDA MemoryInfo for GPU-resident input binding. Mirrors
        // `ClassifierModel::load`. Used by `run_inference_profiled` to
        // construct ORT TensorRefMut against the device pointer directly,
        // skipping the host roundtrip the legacy path paid. `device_id`
        // is the same ordinal already passed into `build_session`.
        let cuda_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| SparrowEngineError::Ort(format!("MemoryInfo::new(CUDA): {e}")))?;

        // Cache the diagnostic A/B env var at load() — set-once semantics,
        // matching `crate::profile::enabled()` and sparrow-engine-cpu's diagnostic
        // env-var pattern. Removes a `std::env::var` syscall from each
        // `detect()` hot path.
        let use_host_roundtrip =
            std::env::var("SPARROW_ENGINE_GPU_YOLO_HOST_ROUNDTRIP").as_deref() == Ok("1");

        Ok(YoloModel {
            session: Arc::new(Mutex::new(session)),
            decoder: Mutex::new(decoder),
            manifest: Arc::new(manifest.clone()),
            labels: Arc::new(labels),
            input_w: input_size[0],
            input_h: input_size[1],
            channel_order,
            pad_value,
            default_threshold: manifest.confidence_threshold.unwrap_or(0.2),
            cuda_mem_info,
            device_id,
            use_host_roundtrip,
        })
    }

    /// Run YOLO E2E detection on a single image, GPU end-to-end.
    ///
    /// Pipeline:
    /// 1. nvjpeg decode → GPU HWC RGB u8 buffer (CPU fallback for non-baseline).
    /// 2. CUDA letterbox kernel → GPU NCHW FP32 buffer (channel order from manifest).
    /// 3. ORT inference. FP16 ONNX (converted via `onnxruntime.transformers.float16`
    ///    with `keep_io_types=True`) accepts FP32 input and emits FP32 output;
    ///    the FP16 graph performs the Cast internally. Single inference path
    ///    handles both precisions.
    /// 4. DtoH copy of small `[1, 300, 6]` output (~7.2 KB).
    /// 5. CPU yolo_e2e postprocess (sparrow-engine-core).
    ///
    /// `letterbox` is a borrowed engine-level kernel handle. The stateful
    /// nvjpeg decoder is owned internally by `YoloModel` (`self.decoder`,
    /// behind a `Mutex`) — set up at `load()` time and reused per call.
    pub fn detect(
        &self,
        ctx: &Arc<CudaContext>,
        letterbox: &LetterboxKernel,
        image: &ImageInput,
        opts: &DetectOpts,
    ) -> Result<DetectResult> {
        let start = std::time::Instant::now();
        let prof_on = crate::profile::enabled();

        // Validate the per-call ctx matches the session's pinned device.
        // Cheap guard against caller-side bugs (mixing engines on different
        // GPU ordinals). Mirrors `ClassifierModel::classify`.
        let ctx_ordinal: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        if ctx_ordinal != self.device_id {
            return Err(SparrowEngineError::Ort(format!(
                "YoloModel::detect: ctx device {} != session device {}",
                ctx_ordinal, self.device_id
            )));
        }

        // 1. Acquire the CUDA stream (one per call; single-image path).
        // Phase 3.8 Step 1 audit-fix R2 A1 (S-NEW-1): use the per-context
        // default stream to harmonize with `ClassifierModel::classify` and
        // `TiledModel::detect_tiled`. The pipelined batch path
        // (`detect_batch_pipelined`) still allocates one fresh stream per
        // batch, which is the right pattern for batch.
        let stream = ctx.default_stream();
        let t_stream = start.elapsed();

        // 2. Resolve `ImageInput` to a GPU-resident HWC RGB u8 buffer.
        //    - Encoded JPEG bytes / FilePath → cached nvjpeg fast path.
        //    - Raw → direct H→D copy via `crate::decode::raw_to_gpu`
        //      (Phase 3.8 Step 1 audit-fix R2 B9 / M-NEW-2; replaces the
        //      previous PNG-encode round-trip + RGB-only restriction).
        let (decoded, t_bytes): (GpuImage, std::time::Duration) = match image {
            ImageInput::Raw {
                data,
                width,
                height,
                stride,
                format,
            } => {
                let t = start.elapsed();
                let img =
                    crate::decode::raw_to_gpu(&stream, data, *width, *height, *stride, *format)?;
                (img, t)
            }
            ImageInput::Encoded(_) | ImageInput::FilePath(_) => {
                let bytes = image_input_to_bytes(image)?;
                let t = start.elapsed();
                let mut dec = self.decoder.lock().map_err(|_| {
                    SparrowEngineError::Ort("YoloModel decoder lock poisoned".into())
                })?;
                let img = dec.decode_to_gpu(&stream, &bytes)?;
                (img, t)
            }
        };
        let original_w = decoded.width;
        let original_h = decoded.height;
        let t_decode = start.elapsed();

        // 3. CUDA letterbox + normalize + NCHW.
        let (input_tensor_f32, lb_meta): (CudaSlice<f32>, LetterboxMeta) = letterbox_gpu(
            &stream,
            letterbox,
            &decoded,
            self.input_w,
            self.input_h,
            self.pad_value,
            self.channel_order,
        )?;

        // Synchronize so the GPU buffer is ready before binding.
        stream
            .synchronize()
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc stream.synchronize: {e}")))?;
        let t_letterbox = start.elapsed();

        // 4. ORT inference. Both FP32 and FP16 ONNX (the latter converted via
        // `onnxruntime.transformers.float16` with `keep_io_types=True`)
        // accept FP32 input and emit FP32 output; the FP16 graph performs
        // the Cast internally. Single inference path.
        let (raw_output, t_dtoh_in_ms, t_run_ms, t_extract_ms) =
            self.run_inference_profiled(&stream, input_tensor_f32)?;
        let t_infer = start.elapsed();
        tracing::trace!(
            target: "sparrow_engine_gpu::yolo",
            t_bytes_us = t_bytes.as_micros() as u64,
            t_decode_us = (t_decode - t_bytes).as_micros() as u64,
            t_letterbox_us = (t_letterbox - t_decode).as_micros() as u64,
            t_infer_us = (t_infer - t_letterbox).as_micros() as u64,
            "detect stages"
        );

        // 5. Postprocess on CPU. yolo_e2e expects [N, 6] view.
        let view: ArrayView2<f32> = raw_output.view();
        let pp_meta = preprocess_meta_from_letterbox(&lb_meta, original_w, original_h);

        let t_pre_pp = start.elapsed();
        let detections: Vec<Detection> = match self.manifest.postprocess_method {
            PostprocessMethod::YoloE2e => sparrow_engine_core::postprocess::try_yolo_e2e(
                &view,
                &self.labels,
                opts,
                &pp_meta,
                self.default_threshold,
            )?,
            PostprocessMethod::MegadetV5a { iou_threshold } => {
                sparrow_engine_core::postprocess::try_megadet_v5a(
                    &view,
                    &self.labels,
                    opts,
                    &pp_meta,
                    self.default_threshold,
                    iou_threshold,
                )?
            }
            _ => unreachable!("YoloModel::load rejects other postprocess methods"),
        };
        let t_after_pp = start.elapsed();

        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

        if prof_on {
            let mut rec = std::collections::HashMap::new();
            rec.insert("bytes", t_bytes.as_secs_f64() * 1000.0);
            // Phase 3.8 Step 1 audit-fix R4 B12: saturate to zero. R2 A1
            // hoisted stream creation (line 621) before image_input_to_bytes
            // (lines 637/644), so t_stream < t_bytes always holds — the
            // pre-R2 ordering invariant assumed by `t_stream - t_bytes` is
            // invalid post-R2 and would panic with "overflow when subtracting
            // durations". The 'stream_create' record key is preserved for
            // downstream profile-aggregation compatibility.
            rec.insert(
                "stream_create",
                t_stream.saturating_sub(t_bytes).as_secs_f64() * 1000.0,
            );
            rec.insert("decode", (t_decode - t_stream).as_secs_f64() * 1000.0);
            rec.insert("letterbox", (t_letterbox - t_decode).as_secs_f64() * 1000.0);
            rec.insert("dtoh_in", t_dtoh_in_ms);
            rec.insert("infer", t_run_ms);
            rec.insert("extract", t_extract_ms);
            rec.insert(
                "postprocess",
                (t_after_pp - t_pre_pp).as_secs_f64() * 1000.0,
            );
            rec.insert("total", elapsed_ms as f64);
            rec.insert("orig_w", original_w as f64);
            rec.insert("orig_h", original_h as f64);
            crate::profile::push(rec);
        }

        Ok(DetectResult {
            detections,
            image_width: original_w,
            image_height: original_h,
            processing_time_ms: elapsed_ms,
        })
    }

    /// Run YOLO E2E detection on an already-prepared (decoded + letterboxed)
    /// preprocess slot, then immediately kick off decode + letterbox of the
    /// NEXT image on the prepare-side stream. Used by `detect_batch_pipelined`
    /// to overlap nvjpeg + letterbox of image N+1 with ORT.run of image N.
    ///
    /// Caller invariant: `slot.pending` must be `Some(...)`. The slot is
    /// filled by `YoloModel::prepare_into` (a private helper called from the
    /// first iteration of `detect_batch_pipelined` before the first call to
    /// this function).
    ///
    /// `next_bytes`: optional bytes of image N+1 to prepare; if `None`,
    /// nothing is enqueued (used for the final image in a batch).
    fn detect_consume_and_prepare_next(
        &self,
        ctx: &Arc<CudaContext>,
        letterbox: &LetterboxKernel,
        slot: &mut YoloDecodeAhead,
        next_bytes: Option<&[u8]>,
        opts: &DetectOpts,
    ) -> Result<DetectResult> {
        let start = std::time::Instant::now();
        let prof_on = crate::profile::enabled();

        // Validate ctx ordinal.
        let ctx_ordinal: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        if ctx_ordinal != self.device_id {
            return Err(SparrowEngineError::Ort(format!(
                "YoloModel::detect_consume_and_prepare_next: ctx device {} != session device {}",
                ctx_ordinal, self.device_id
            )));
        }

        // Step 1: take the pre-staged preprocess slot.
        let pending = slot.pending.take().ok_or_else(|| {
            SparrowEngineError::Ort(
                "detect_consume_and_prepare_next: slot.pending is None — call prepare() first"
                    .into(),
            )
        })?;
        let PreparedPreprocess {
            input_tensor_f32,
            lb_meta,
            original_w,
            original_h,
            decode_stream,
        } = pending;

        // Step 2: ensure the decode_stream's preprocess work is done before
        // ORT consumes the device pointer. Cheap in the steady state because
        // the previous detect call's ORT.run was running concurrent with
        // this stream's nvjpeg + letterbox; on the first call this is a
        // full wait.
        decode_stream
            .synchronize()
            .map_err(|e| SparrowEngineError::Ort(format!("decode_stream.synchronize: {e}")))?;
        let t_pre_ort = start.elapsed();

        // Step 3: kick off nvjpeg + letterbox of image N+1 on decode_stream
        // BEFORE blocking on ORT.run for image N. nvjpeg is async on the
        // stream; letterbox launches on the same stream. The host returns
        // immediately and the GPU work runs concurrent with ORT below.
        if let Some(next) = next_bytes {
            let next_decoded = {
                let mut dec = self.decoder.lock().map_err(|_| {
                    SparrowEngineError::Ort("YoloModel decoder lock poisoned".into())
                })?;
                dec.decode_to_gpu(&decode_stream, next)?
            };
            let next_w = next_decoded.width;
            let next_h = next_decoded.height;
            let (next_in, next_lb) = letterbox_gpu(
                &decode_stream,
                letterbox,
                &next_decoded,
                self.input_w,
                self.input_h,
                self.pad_value,
                self.channel_order,
            )?;
            slot.pending = Some(PreparedPreprocess {
                input_tensor_f32: next_in,
                lb_meta: next_lb,
                original_w: next_w,
                original_h: next_h,
                decode_stream: decode_stream.clone(),
            });
            // Note: the next call's `decode_stream.synchronize()` will wait on
            // these ops; no host sync here.
        }

        // Step 4: run ORT on the current image's prepared tensor. ORT uses
        // its own internal CUDA EP stream; the post-letterbox decode_stream
        // sync above ensures the input tensor is visible to it.
        // Pass decode_stream as the "input stream" since the device pointer
        // was last written by ops on that stream.
        let (raw_output, t_dtoh_in_ms, t_run_ms, t_extract_ms) =
            self.run_inference_profiled(&decode_stream, input_tensor_f32)?;

        // Step 5: postprocess.
        let view: ArrayView2<f32> = raw_output.view();
        let pp_meta = preprocess_meta_from_letterbox(&lb_meta, original_w, original_h);

        let t_pre_pp = start.elapsed();
        let detections: Vec<Detection> = match self.manifest.postprocess_method {
            PostprocessMethod::YoloE2e => sparrow_engine_core::postprocess::try_yolo_e2e(
                &view,
                &self.labels,
                opts,
                &pp_meta,
                self.default_threshold,
            )?,
            PostprocessMethod::MegadetV5a { iou_threshold } => {
                sparrow_engine_core::postprocess::try_megadet_v5a(
                    &view,
                    &self.labels,
                    opts,
                    &pp_meta,
                    self.default_threshold,
                    iou_threshold,
                )?
            }
            _ => unreachable!("YoloModel::load rejects other postprocess methods"),
        };
        let t_after_pp = start.elapsed();

        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

        if prof_on {
            let mut rec = std::collections::HashMap::new();
            rec.insert("bytes", 0.0); // bytes already prepared by caller
            rec.insert("stream_create", 0.0); // reused
            rec.insert("decode_wait", t_pre_ort.as_secs_f64() * 1000.0);
            rec.insert("letterbox", 0.0); // already in decode_wait above
            rec.insert("dtoh_in", t_dtoh_in_ms);
            rec.insert("infer", t_run_ms);
            rec.insert("extract", t_extract_ms);
            rec.insert(
                "postprocess",
                (t_after_pp - t_pre_pp).as_secs_f64() * 1000.0,
            );
            rec.insert("total", elapsed_ms as f64);
            rec.insert("orig_w", original_w as f64);
            rec.insert("orig_h", original_h as f64);
            // Pipelined record marker: distinguishes from the non-pipelined
            // `detect()` path. Aggregator scripts can split on this key.
            rec.insert("pipelined", 1.0);
            crate::profile::push(rec);
        }

        Ok(DetectResult {
            detections,
            image_width: original_w,
            image_height: original_h,
            processing_time_ms: elapsed_ms,
        })
    }

    /// Run a batch of images with nvjpeg + letterbox of image N+1 overlapped
    /// against ORT.run of image N. Lever A of Path 2 perf follow-up.
    ///
    /// Returns one [`DetectResult`] per input image, in the same order. The
    /// per-image timing window starts when this method begins working on
    /// that image and ends when it returns its `DetectResult` — the same
    /// per-call boundary as [`Self::detect`] but with concurrent decode.
    ///
    /// Note: the FIRST image in the batch pays full decode + letterbox +
    /// ORT cost (no overlap possible since there's no preceding ORT.run).
    /// Steady-state per-image cost: max(decode + letterbox, ORT.run).
    ///
    /// `bytes_iter`: iterator of `&[u8]` JPEG bytes. The caller is expected
    /// to pre-load file contents (mirrors [`Self::detect`] best practice).
    ///
    /// **Bench-only experimental API.** Opt-in via the bench harness's
    /// `SPARROW_ENGINE_GPU_YOLO_BATCH_PIPELINE=1` env var (the harness picks between
    /// this method and [`Self::detect`]). Not part of the stable public
    /// surface; semantics may change without notice.
    pub fn detect_batch_pipelined<'a, I>(
        &self,
        ctx: &Arc<CudaContext>,
        letterbox: &LetterboxKernel,
        bytes_iter: I,
        opts: &DetectOpts,
    ) -> Result<Vec<DetectResult>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let inputs: Vec<&[u8]> = bytes_iter.into_iter().collect();
        let mut results = Vec::with_capacity(inputs.len());
        if inputs.is_empty() {
            return Ok(results);
        }

        // Validate ctx ordinal once.
        let ctx_ordinal: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        if ctx_ordinal != self.device_id {
            return Err(SparrowEngineError::Ort(format!(
                "YoloModel::detect_batch_pipelined: ctx device {} != session device {}",
                ctx_ordinal, self.device_id
            )));
        }

        // Allocate a single decode_stream for the batch. nvjpeg + letterbox
        // run on this stream; ORT runs on its own internal EP stream.
        let decode_stream = ctx
            .new_stream()
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc new_stream: {e}")))?;

        let mut slot = YoloDecodeAhead { pending: None };

        // Prime: prepare image[0] before the loop.
        self.prepare_into(&decode_stream, letterbox, inputs[0], &mut slot)?;

        for i in 0..inputs.len() {
            let next_bytes: Option<&[u8]> = if i + 1 < inputs.len() {
                Some(inputs[i + 1])
            } else {
                None
            };
            let r =
                self.detect_consume_and_prepare_next(ctx, letterbox, &mut slot, next_bytes, opts)?;
            results.push(r);
        }

        Ok(results)
    }

    /// Internal helper: kick off nvjpeg + letterbox of `bytes` on `stream`
    /// and store the resulting preprocess tensor in `slot.pending`.
    /// Does NOT block on host. Used to prime the pipeline.
    fn prepare_into(
        &self,
        stream: &Arc<CudaStream>,
        letterbox: &LetterboxKernel,
        bytes: &[u8],
        slot: &mut YoloDecodeAhead,
    ) -> Result<()> {
        let decoded = {
            let mut dec = self
                .decoder
                .lock()
                .map_err(|_| SparrowEngineError::Ort("YoloModel decoder lock poisoned".into()))?;
            dec.decode_to_gpu(stream, bytes)?
        };
        let original_w = decoded.width;
        let original_h = decoded.height;
        let (in_tensor, lb_meta) = letterbox_gpu(
            stream,
            letterbox,
            &decoded,
            self.input_w,
            self.input_h,
            self.pad_value,
            self.channel_order,
        )?;
        slot.pending = Some(PreparedPreprocess {
            input_tensor_f32: in_tensor,
            lb_meta,
            original_w,
            original_h,
            decode_stream: stream.clone(),
        });
        Ok(())
    }

    /// Manifest ID accessor (used by the engine + tests). Mirrors
    /// `TiledModel::model_id` and `ClassifierModel::model_id`; aligns with
    /// the system-wide `sparrow_engine_cpu::engine::ModelHandle::model_id()`
    /// convention.
    pub fn model_id(&self) -> &str {
        &self.manifest.id
    }

    /// Manifest accessor (read-only).
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Device ordinal captured at load time (matches `ctx.ordinal() as i32`
    /// at the time of [`YoloModel::load`]). Phase 3.8 Step 1 audit-fix R1
    /// B2 MODIFY makes this the same value passed into ORT's CUDA EP via
    /// `with_device_id`; the per-call guard at the top of `detect()` then
    /// rejects context-ordinal mismatches.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

// ---------------------------------------------------------------------------
// Private ORT inference helpers
// ---------------------------------------------------------------------------

impl YoloModel {
    /// Run a single ORT inference call with timing probes for the three
    /// internal sub-stages: DtoH (always 0 on the GPU-resident path),
    /// ORT.run (actual inference), output extract+to_owned. Returns
    /// `(output, dtoh_ms, infer_ms, extract_ms)`.
    ///
    /// **Default path: GPU-resident** input binding via
    /// `ort::value::TensorRefMut::from_raw` with CUDA `MemoryInfo`. ORT CUDA
    /// EP consumes the device pointer directly when memory_info matches
    /// the EP's device. Mirrors `ClassifierModel::classify` (Wave 3 commit
    /// `64646b8`) which has shipped in production for SpeciesNet (FP32) +
    /// Amazon CTV2 (FP16).
    ///
    /// **Fallback** behind env-var `SPARROW_ENGINE_GPU_YOLO_HOST_ROUNDTRIP=1`:
    /// legacy path that DtoH-copies preprocess output to host and lets ORT
    /// re-upload via its CUDA EP allocator. Kept for diagnostic A/B
    /// comparison; DO NOT use in production (~24 ms/image of pure
    /// roundtrip on 1 × 3 × 1280 × 1280 f32 input).
    fn run_inference_profiled(
        &self,
        stream: &Arc<CudaStream>,
        input_gpu: CudaSlice<f32>,
    ) -> Result<(Array2<f32>, f64, f64, f64)> {
        let total_in = (3 * self.input_h * self.input_w) as usize;
        debug_assert_eq!(input_gpu.len(), total_in);

        if self.use_host_roundtrip {
            return self.run_inference_host_roundtrip(stream, input_gpu);
        }

        let mut guard = self
            .session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("YoloModel session lock poisoned".into()))?;
        let session: &mut Session = &mut guard;

        // GPU-resident path: zero-copy bind. dtoh is 0 by construction.
        let dtoh_ms = 0.0_f64;

        let shape = Shape::from([1_i64, 3, self.input_h as i64, self.input_w as i64]);
        let (dev_ptr_u64, _sync) = input_gpu.device_ptr(stream);
        let mem_info = self.cuda_mem_info.clone();

        // SAFETY: `dev_ptr_u64` is the device pointer to `input_gpu`, owned
        // by this function scope and live for the lifetime of the inference
        // call. The pointer is valid for `1 * 3 * input_h * input_w` f32
        // values (allocated by `letterbox_gpu`'s `alloc_zeros`). `mem_info`
        // describes the device the pointer lives on (CUDA device
        // `self.device_id`, `AllocatorType::Device`). The `CudaSlice` is
        // dropped after ORT.run returns; the post-letterbox sync in
        // `detect()` ensures the kernel write is visible to ORT's EP.
        let input_tensor = unsafe {
            TensorRefMut::<f32>::from_raw(
                mem_info,
                dev_ptr_u64 as usize as *mut std::ffi::c_void,
                shape,
            )
        }
        .map_err(|e| SparrowEngineError::Ort(format!("TensorRefMut::from_raw: {e}")))?;

        let t1 = std::time::Instant::now();
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| SparrowEngineError::Ort(format!("ort Session::run: {e}")))?;
        let t3 = std::time::Instant::now();

        let output = outputs.values().next().ok_or_else(|| {
            SparrowEngineError::Ort("yolo_e2e session returned no outputs".to_string())
        })?;
        let view = output
            .try_extract_array::<f32>()
            .map_err(|e| SparrowEngineError::Ort(format!("ort try_extract_array: {e}")))?;
        let shape_out = view.shape().to_vec();
        let view_2d = if shape_out.len() == 3 && shape_out[0] == 1 {
            view.index_axis(ndarray::Axis(0), 0)
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| SparrowEngineError::Ort(format!("squeeze [1,N,6]: {e}")))?
        } else if shape_out.len() == 2 {
            view.into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| SparrowEngineError::Ort(format!("downcast [N,6]: {e}")))?
        } else {
            return Err(SparrowEngineError::Ort(format!(
                "yolo_e2e output shape unexpected: {shape_out:?}"
            )));
        };
        let owned = view_2d.to_owned();
        drop(outputs);
        let t4 = std::time::Instant::now();

        let infer_ms = (t3 - t1).as_secs_f64() * 1000.0;
        let extract_ms = (t4 - t3).as_secs_f64() * 1000.0;
        Ok((owned, dtoh_ms, infer_ms, extract_ms))
    }

    /// Legacy host-roundtrip inference path. Selected via
    /// `SPARROW_ENGINE_GPU_YOLO_HOST_ROUNDTRIP=1`. Kept for diagnostic A/B
    /// comparison vs the GPU-resident default.
    fn run_inference_host_roundtrip(
        &self,
        stream: &Arc<CudaStream>,
        input_gpu: CudaSlice<f32>,
    ) -> Result<(Array2<f32>, f64, f64, f64)> {
        let mut guard = self
            .session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("YoloModel session lock poisoned".into()))?;
        let session: &mut Session = &mut guard;

        let t0 = std::time::Instant::now();
        let host_in = stream
            .clone_dtoh(&input_gpu)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc memcpy_dtov (input): {e}")))?;
        stream.synchronize().map_err(|e| {
            SparrowEngineError::Ort(format!("cudarc stream.synchronize (input): {e}"))
        })?;
        let t1 = std::time::Instant::now();

        let arr = ndarray::Array4::<f32>::from_shape_vec(
            (1, 3, self.input_h as usize, self.input_w as usize),
            host_in,
        )
        .map_err(|e| SparrowEngineError::Ort(format!("input tensor reshape: {e}")))?;
        let input_value = TensorRef::from_array_view(&arr).map_err(|e| {
            SparrowEngineError::Ort(format!("ort::TensorRef::from_array_view: {e}"))
        })?;
        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(|e| SparrowEngineError::Ort(format!("ort Session::run: {e}")))?;
        let t3 = std::time::Instant::now();

        let output = outputs.values().next().ok_or_else(|| {
            SparrowEngineError::Ort("yolo_e2e session returned no outputs".to_string())
        })?;
        let view = output
            .try_extract_array::<f32>()
            .map_err(|e| SparrowEngineError::Ort(format!("ort try_extract_array: {e}")))?;
        let shape = view.shape().to_vec();
        let view_2d = if shape.len() == 3 && shape[0] == 1 {
            view.index_axis(ndarray::Axis(0), 0)
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| SparrowEngineError::Ort(format!("squeeze [1,N,6]: {e}")))?
        } else if shape.len() == 2 {
            view.into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| SparrowEngineError::Ort(format!("downcast [N,6]: {e}")))?
        } else {
            return Err(SparrowEngineError::Ort(format!(
                "yolo_e2e output shape unexpected: {shape:?}"
            )));
        };
        let owned = view_2d.to_owned();
        drop(outputs);
        let t4 = std::time::Instant::now();

        let dtoh_ms = (t1 - t0).as_secs_f64() * 1000.0;
        let infer_ms = (t3 - t1).as_secs_f64() * 1000.0;
        let extract_ms = (t4 - t3).as_secs_f64() * 1000.0;
        Ok((owned, dtoh_ms, infer_ms, extract_ms))
    }
}

// ---------------------------------------------------------------------------
// ORT session construction
// ---------------------------------------------------------------------------

/// Build an ORT session for this model via the TRT→CUDA→CPU EP policy
/// (`crate::trt::ep::TrtEpBuilder`): TensorRT when the manifest opts in,
/// otherwise the CUDA EP, with a CPU per-op fallback.
///
/// `device_id` is the ordinal of the CUDA device the session must pin to;
/// passed in from `YoloModel::load` (which captured it from
/// `ctx.ordinal()`). Phase 3.8 Step 1 audit-fix R1 B2 MODIFY: previously
/// the function used `CUDA::default()` with no `with_device_id`, which
/// resolved to device 0 in ORT's EP factory regardless of the caller's
/// actual context — the per-call ordinal guard at the top of `detect()`
/// would catch a later mismatch, but the session itself was mis-pinned.
fn build_session(
    _ctx: &Arc<CudaContext>,
    gpu: &GpuIdentity,
    manifest: &ModelManifest,
    onnx_path: &Path,
    device_id: i32,
) -> Result<Session> {
    use ort::ep::cuda::ConvAlgorithmSearch;
    use ort::session::builder::GraphOptimizationLevel;

    let builder = Session::builder().map_err(|e| SparrowEngineError::Ort(e.to_string()))?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| SparrowEngineError::Ort(e.to_string()))?;
    // YOLO-specific tuning: pin ORT's intra/inter-op thread pools to 1 so
    // that the per-call CPU work (postprocess copy + bbox decode) does not
    // contend with `detect_batch_pipelined`'s decode-ahead worker thread
    // for CPU cores. classifier + tiled paths do not use a pipelined batch
    // worker, so they default to ORT's auto-thread-pool sizing. See
    // `docs/design/phase3.7/perf_research.md` (Phase 3.7 R2 convergence) +
    // `docs/research/phase3.8/step1/mdv6_perf_investigation.md`.
    let builder = builder
        .with_intra_threads(1)
        .map_err(|e| SparrowEngineError::Ort(e.to_string()))?;
    let builder = builder
        .with_inter_threads(1)
        .map_err(|e| SparrowEngineError::Ort(e.to_string()))?;
    // Lever B (Path 2 follow-up): cuDNN HEURISTIC algo selection (default
    // for YOLO sessions) instead of EXHAUSTIVE. EXHAUSTIVE picks an algo
    // by running every candidate at session-creation time; on RTX 6000 Ada
    // the selection is non-deterministic across fresh processes and adds
    // 0.2-0.4 ms of per-image variance after Phase 3.7 R2's bimodality
    // analysis (`docs/research/phase3.7/track_b/results.md`). HEURISTIC
    // picks deterministically from cuDNN's lookup table. Trade: median
    // may shift by ~0 to +0.5 ms (case-by-case); cross-run stddev tightens.
    //
    // Override at run-time with SPARROW_ENGINE_GPU_YOLO_CONV_SEARCH=exhaustive |
    // heuristic | default for A/B testing (case-insensitive).
    let conv_search = match std::env::var("SPARROW_ENGINE_GPU_YOLO_CONV_SEARCH")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("exhaustive") => ConvAlgorithmSearch::Exhaustive,
        Some("default") => ConvAlgorithmSearch::Default,
        _ => ConvAlgorithmSearch::Heuristic,
    };
    let cuda = CudaEpConfig::new(device_id).with_conv_algorithm_search(conv_search);
    let manifest_cache_material = manifest_cache_material(manifest);
    let providers = TrtEpBuilder::new(
        &manifest.id,
        manifest.trt.as_ref(),
        gpu,
        cuda,
        onnx_path,
        &manifest_cache_material,
    )
    .execution_providers()?;
    let mut builder = builder
        .with_execution_providers(providers)
        .map_err(|e| SparrowEngineError::Ort(e.to_string()))?;
    builder
        .commit_from_file(onnx_path)
        .map_err(|e| SparrowEngineError::Ort(format!("ort session commit_from_file: {e}")))
}

/// Validate the YOLO output dimensions declared on the first session output.
/// Supported shapes:
/// - `YoloE2e`: `[N, 6]` or `[1, N, 6]`
/// - `MegadetV5a`: `[N, 5+C]` or `[1, N, 5+C]`
///
/// Dynamic `-1` is allowed for `N`, the optional batch axis, and the last
/// dimension. Any static batch axis other than `1` is rejected because the
/// runtime extraction path only accepts `[N, C]` or `[1, N, C]`.
fn validate_output_dims(
    dims: &[i64],
    model_id: &str,
    method: &PostprocessMethod,
    num_labels: usize,
) -> Result<()> {
    let method_str = method.as_str().to_string();
    if dims.len() != 2 && dims.len() != 3 {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: format!("{dims:?}"),
            method: method_str,
        });
    }
    let last = dims[dims.len() - 1];
    let static_last_ok = match method {
        PostprocessMethod::YoloE2e => last == 6 || last == -1,
        PostprocessMethod::MegadetV5a { .. } => {
            let _ = num_labels;
            // Runtime MegaDet postprocess accepts any `[N, 5+C]` with `C > 0`
            // and tolerates `labels.len() != num_classes` via `unknown_<id>`
            // fallback, so load-time shape validation must not key the static
            // last dimension to `num_labels`.
            last == -1 || last > 5
        }
        _ => unreachable!("validated at load: only YoloE2e | MegadetV5a reach here"),
    };
    if !static_last_ok {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: format!("{dims:?}"),
            method: method_str.clone(),
        });
    }
    match dims {
        [n, _] if *n == -1 || *n > 0 => Ok(()),
        [batch, n, _] if (*batch == 1 || *batch == -1) && (*n == -1 || *n > 0) => Ok(()),
        _ => Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: format!("{dims:?}"),
            method: method_str,
        }),
    }
}

/// Validate the YOLO E2E output shape on session inputs.
fn validate_output_shape(
    session: &Session,
    model_id: &str,
    method: &PostprocessMethod,
    num_labels: usize,
) -> Result<()> {
    let outputs = session.outputs();
    let first_output = outputs
        .first()
        .ok_or_else(|| SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: "no outputs".to_string(),
            method: method.as_str().to_string(),
        })?;
    let dims: Vec<i64> = match first_output.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => vec![],
    };
    validate_output_dims(&dims, model_id, method, num_labels)
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

/// Validate that the session's first output is `Float32`. The YOLO binding
/// extracts `outputs()[0]` as `f32`, so load-time rejection is cheaper than
/// surfacing a typed-tensor mismatch on first inference.
fn validate_output_dtype_fp32(session: &Session, model_id: &str) -> Result<()> {
    use ort::value::{TensorElementType, ValueType};
    match session.outputs().first().map(|o| o.dtype()) {
        Some(ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        }) => Ok(()),
        Some(other) => Err(SparrowEngineError::InvalidManifest(format!(
            "model '{model_id}' output dtype must be Float32 (FP16 ONNX must be converted with \
             keep_io_types=True so I/O remains FP32), got {other:?}"
        ))),
        None => Err(SparrowEngineError::InvalidManifest(format!(
            "model '{model_id}' has no outputs"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an `ImageInput::Encoded` or `ImageInput::FilePath` payload to a
/// JPEG/PNG-encoded byte buffer for nvjpeg / CPU fallback decode.
///
/// `ImageInput::Raw` is NOT handled here — the Raw path goes directly through
/// `crate::decode::raw_to_gpu` (Phase 3.8 Step 1 audit-fix R2 B9 / M-NEW-2),
/// which avoids the wasteful PNG-encode-just-to-decode round-trip the
/// previous `raw_to_png` helper performed. Callers must dispatch on the
/// `ImageInput` variant before invoking this function.
fn image_input_to_bytes(image: &ImageInput) -> Result<Vec<u8>> {
    match image {
        ImageInput::Encoded(b) => Ok(b.clone()),
        ImageInput::FilePath(p) => {
            if !p.exists() {
                return Err(SparrowEngineError::ImageFileNotFound(p.clone()));
            }
            std::fs::read(p).map_err(SparrowEngineError::Io)
        }
        ImageInput::Raw { .. } => Err(SparrowEngineError::ImageDecode(
            "image_input_to_bytes: ImageInput::Raw must be routed through \
             crate::decode::raw_to_gpu directly, not encoded as bytes"
                .into(),
        )),
    }
}

/// Construct a [`PreprocessMeta`] from a [`LetterboxMeta`] for postprocess
/// coordinate undo. yolo_e2e's `denormalize_and_normalize` expects the meta
/// fields populated with original image dimensions + letterbox geometry.
fn preprocess_meta_from_letterbox(lb: &LetterboxMeta, orig_w: u32, orig_h: u32) -> PreprocessMeta {
    PreprocessMeta {
        scale: lb.scale,
        pad_x: lb.pad_x,
        pad_y: lb.pad_y,
        original_width: orig_w,
        original_height: orig_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn megadet_v5a_method() -> PostprocessMethod {
        PostprocessMethod::MegadetV5a {
            iou_threshold: 0.45,
        }
    }

    /// Phase 3.8 Step 1 audit-fix R2 B9 (M-NEW-2) regression test.
    ///
    /// `image_input_to_bytes` no longer accepts `ImageInput::Raw` — that path
    /// is routed through `crate::decode::raw_to_gpu` directly. Calling the
    /// helper with `Raw` is now an explicit error to make the routing
    /// invariant easy to spot in code review and to catch any regression
    /// where a future caller forgets to dispatch on the variant first.
    #[test]
    fn image_input_to_bytes_rejects_raw() {
        let raw = ImageInput::Raw {
            data: vec![0u8; 12],
            width: 2,
            height: 2,
            stride: 6,
            format: sparrow_engine_types::PixelFormat::Rgb,
        };
        let res = image_input_to_bytes(&raw);
        match res {
            Err(SparrowEngineError::ImageDecode(msg)) if msg.contains("raw_to_gpu") => {}
            Err(other) => panic!("expected ImageDecode(... raw_to_gpu ...), got Err({other:?})"),
            Ok(_) => panic!("expected error rejection of Raw, got Ok(_)"),
        }
    }

    #[test]
    fn preprocess_meta_carries_letterbox_geometry() {
        let lb = LetterboxMeta {
            scale: 0.5,
            pad_x: 10.0,
            pad_y: 20.0,
            original_width: 2560,
            original_height: 1920,
        };
        let pp = preprocess_meta_from_letterbox(&lb, 2560, 1920);
        assert_eq!(pp.scale, 0.5);
        assert_eq!(pp.pad_x, 10.0);
        assert_eq!(pp.pad_y, 20.0);
        assert_eq!(pp.original_width, 2560);
        assert_eq!(pp.original_height, 1920);
    }

    #[test]
    fn validate_output_dims_accepts_supported_shapes() {
        validate_output_dims(&[8400, 6], "mdv6", &PostprocessMethod::YoloE2e, 0)
            .expect("rank-2 yolo output should be accepted");
        validate_output_dims(&[8400, 8], "mdv5a-unlabeled", &megadet_v5a_method(), 0)
            .expect("megadet output should accept unlabeled manifests");
        validate_output_dims(
            &[1, 8400, 8],
            "mdv5a-short-labels",
            &megadet_v5a_method(),
            2,
        )
        .expect("megadet output should accept short label lists");
        validate_output_dims(&[1, 8400, 8], "mdv5a", &megadet_v5a_method(), 3)
            .expect("batch-1 megadet output should still be accepted");
        validate_output_dims(&[-1, -1, -1], "dynamic", &megadet_v5a_method(), 3)
            .expect("dynamic output dims should be accepted");
    }

    #[test]
    fn validate_output_dims_rejects_unsupported_batch_axis() {
        let err = validate_output_dims(&[2, 8400, 6], "bad-batch", &PostprocessMethod::YoloE2e, 0)
            .expect_err("batch>1 must be rejected at load time");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }

    #[test]
    fn validate_output_dims_rejects_nonpositive_candidate_count() {
        let err = validate_output_dims(&[0, 6], "bad-count", &PostprocessMethod::YoloE2e, 0)
            .expect_err("N=0 must be rejected at load time");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }

    #[test]
    fn validate_output_dims_rejects_megadet_static_last_dim_below_six() {
        let err = validate_output_dims(&[8400, 5], "bad-last-5", &megadet_v5a_method(), 0)
            .expect_err("megadet outputs with last_dim == 5 must be rejected");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));

        let err = validate_output_dims(&[8400, 4], "bad-last-4", &megadet_v5a_method(), 2)
            .expect_err("megadet outputs with last_dim <= 5 must be rejected");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }
}
