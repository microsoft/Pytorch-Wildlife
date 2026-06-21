//! W1.5 — ORT CUDA EP IoBinding wrapper for the audio classifier. **Tier A.**
//!
//! Wave 0 measured `audio.ort` at **800.7 ms / 60 s clip** (median, 5
//! fresh-process runs) — the second-largest stage after `mel_gemm`.
//! Step 1 image MDv6 saw **2.81×** from GPU-resident IoBinding (36.7 → 13.5 ms;
//! `docs/research/phase3.8/step1/full_bench.md`). For audio classifier
//! the expected gain is similar (~800 → ~280 ms, ~520 ms saved).
//!
//! # IoBinding shape
//!
//! `MD_AudioBirds_V1.onnx` accepts `[batch_size, 1, 224, time_steps]` `f32`
//! and emits `[batch_size, 1]` `f32`. Both `batch_size` and `time_steps`
//! are dynamic dim_params (verified by lead's ONNX inspection in
//! `docs/design/phase3.8/step2/round_02/cameratraps_audit.md` §F0.10).
//!
//! For Wave 1 we hold `time_steps = 90` (1 s segment at 48 kHz, hop 512,
//! n_fft 2048 → 90 mel frames per segment). Wave 2 may switch to whole-clip
//! mel + chunk-of-16 ORT (per arch-prag's ratified hybrid in
//! `arch-perf_proposal_r2.md §R2.7`); the IoBinding wrapper here is
//! shape-agnostic so future layouts plug in unchanged.
//!
//! # Strategy
//!
//! Following the Step 1 pattern in `crate::models::classifier::ClassifierModel`
//! and `crate::models::yolo::YoloModel`:
//!
//! 1. Build the `ort::Session` with CUDA EP + CPU EP fallback.
//! 2. Cache an `ort::memory::MemoryInfo` for `AllocationDevice::CUDA` once
//!    at load.
//! 3. Per-call: bind the GPU-resident mel input via
//!    `ort::value::TensorRefMut::from_raw(mem_info, dev_ptr, shape)`.
//! 4. Run with `Session::run` — ORT consumes the device pointer directly,
//!    skipping the implicit H→D copy the legacy host-roundtrip path paid.
//! 5. `try_extract_array` on the output, then DtoH the small `[batch, 1]`
//!    logits (~64 B at batch=16) for CPU sigmoid + threshold + callback.
//!
//! Wave 1's parity gate sets max-abs Δ = 0.0 (bind-once vs bind-per-call
//! must produce bit-exact logits on the same input mel tensor).

use std::path::Path;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Shape, TensorRef, TensorRefMut};
use sparrow_engine_types::error::{Result, SparrowEngineError};

use crate::trt::ep::{CudaEpConfig, GpuIdentity, TrtEpBuilder};

/// CUDA-bound audio classifier with cached IoBinding metadata.
pub struct AudioOrtSession {
    /// ORT session. Wrapped in `Mutex` because `Session::run` is `&mut self`.
    session: Mutex<Session>,
    /// Cached input outlet name (`MD_AudioBirds_V1` has a single input).
    input_name: String,
    /// Cached output outlet name.
    output_name: String,
    /// CUDA `MemoryInfo` template for binding device pointers as ORT inputs.
    cuda_mem_info: MemoryInfo,
    /// Captured CUDA device ordinal.
    device_id: i32,
    /// Owned CUDA stream the session was bound to via
    /// `ort::ep::CUDA::with_compute_stream`. Held here so the stream
    /// outlives the session per the `with_compute_stream` safety
    /// contract — Drop order is field-declaration order, so `session`
    /// drops before `stream`.
    #[allow(dead_code)] // held purely for lifetime; not accessed.
    stream: Arc<CudaStream>,
}

// SAFETY: ORT `Session` is wrapped in a `Mutex`, mirroring the pattern in
// `crate::models::yolo::YoloModel` + `crate::models::classifier::ClassifierModel`.
// `MemoryInfo` is Clone + POD-like.
unsafe impl Send for AudioOrtSession {}
unsafe impl Sync for AudioOrtSession {}

fn checked_audio_chunk_end(
    mel_len: usize,
    offset_elements: usize,
    batch: usize,
    n_mels: usize,
    time_steps: usize,
) -> Result<usize> {
    let chunk_len = batch
        .checked_mul(n_mels)
        .and_then(|v| v.checked_mul(time_steps))
        .ok_or_else(|| {
            SparrowEngineError::Ort(format!(
                "audio chunk element count overflow: batch={batch}, n_mels={n_mels}, time_steps={time_steps}"
            ))
        })?;
    let end = offset_elements.checked_add(chunk_len).ok_or_else(|| {
        SparrowEngineError::Ort(format!(
            "audio chunk offset overflow: offset={offset_elements}, chunk_len={chunk_len}"
        ))
    })?;
    if end > mel_len {
        return Err(SparrowEngineError::Ort(format!(
            "audio chunk offset OOB: offset={offset_elements}, batch={batch}, n_mels={n_mels}, time_steps={time_steps}, mel_d.len()={mel_len}"
        )));
    }
    Ok(end)
}

impl AudioOrtSession {
    /// Build the audio classifier session pinned to `ctx`'s device.
    ///
    /// `onnx_path` is the path to `MD_AudioBirds_V1.onnx`. The session is
    /// constructed with CUDA EP first (with `error_on_failure()` so silent
    /// CUDA-registration failures surface immediately), CPU EP as the
    /// per-op fallback (NOT a silent full-engine downgrade — same policy
    /// as Step 1).
    ///
    /// Phase 3.8 Step 2 perf-fix Fix D: the CUDA EP is bound to `stream`
    /// via `with_compute_stream`, so ORT's internal CUDA work is
    /// scheduled on the same stream as the cudarc mel-pipeline kernels
    /// in `crate::models::audio::AudioModel`. Stream ordering naturally
    /// serializes mel → ORT without the CPU-blocking
    /// `cudaStreamSynchronize` that the pre-fix path paid (~2-5 ms per
    /// detect). The stream pointer must outlive the session — the
    /// `Arc<CudaStream>` is held here as a field so dropping the session
    /// doesn't invalidate the pointer.
    pub fn load(
        ctx: &Arc<cudarc::driver::CudaContext>,
        stream: &Arc<CudaStream>,
        onnx_path: &Path,
    ) -> Result<Self> {
        let device_id: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;

        // SAFETY: `stream.cu_stream()` returns a `sys::CUstream` (raw CUDA
        // stream pointer). ORT's `with_compute_stream` is documented to
        // accept a `cudaStream_t`, which is the same opaque pointer type
        // (`struct CUstream_st *`). The stream's lifetime is anchored
        // by the `Arc<CudaStream>` we store in `Self::stream`; field-
        // declaration order ensures `session` drops before `stream`, so
        // ORT cannot dereference the stream after it's freed.
        let cu_stream_ptr = stream.cu_stream() as *mut ();

        let gpu = GpuIdentity::from_context(ctx)?;
        let model_id = onnx_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("audio-ort");
        let providers = TrtEpBuilder::new(
            model_id,
            None,
            &gpu,
            CudaEpConfig::new(device_id).with_compute_stream(cu_stream_ptr),
            onnx_path,
            "audio-ort-io-no-manifest",
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
            .commit_from_file(onnx_path)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("commit_from_file({onnx_path:?}): {e}"))
            })?;

        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| SparrowEngineError::Ort("audio session has no inputs".into()))?
            .name()
            .to_owned();
        let output_name = session
            .outputs()
            .first()
            .ok_or_else(|| SparrowEngineError::Ort("audio session has no outputs".into()))?
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
            input_name,
            output_name,
            cuda_mem_info,
            device_id,
            stream: stream.clone(),
        })
    }

    /// Run the audio classifier with **GPU-resident IoBinding**.
    ///
    /// - `mel_d`: `[batch * n_mels * time_steps]` row-major `f32` on GPU.
    ///   Layout is NCHW with `C=1`, i.e. logical shape `[batch, 1, n_mels,
    ///   time_steps]`. The pointer must remain valid for the duration of
    ///   the `Session::run` call (kept alive via `&CudaSlice<f32>` borrow).
    /// - `stream`: CUDA stream the buffer was produced on.
    /// - `batch`, `n_mels`, `time_steps`: shape dimensions.
    ///
    /// Returns the flat logits `Vec<f32>` of length `batch` (output shape
    /// `[batch, 1]`). Caller applies sigmoid + threshold on CPU.
    ///
    /// Equivalent to
    /// `run_iobinding_at_offset(stream, mel_d, 0, batch, n_mels, time_steps)`
    /// — when `offset_elements = 0` the chunk-pointer arithmetic is a
    /// no-op (`dev_ptr_u64 + 0 = dev_ptr_u64`), so the bind shape +
    /// session-run path is byte-identical. S8 collapse (R2 audit-fix
    /// 2026-05-05) removed the duplicated body.
    pub fn run_iobinding(
        &self,
        stream: &Arc<CudaStream>,
        mel_d: &CudaSlice<f32>,
        batch: usize,
        n_mels: usize,
        time_steps: usize,
    ) -> Result<Vec<f32>> {
        self.run_iobinding_at_offset(stream, mel_d, 0, batch, n_mels, time_steps)
    }

    /// Run with GPU-resident IoBinding **at an element offset within
    /// `mel_d`**.
    ///
    /// Phase 3.8 Step 2 Wave 2 (e2e orchestrator): the whole-clip
    /// Strategy A pipeline produces a single ORT-ready mel buffer of
    /// `[n_segments * n_mels * time_steps]` elements; the ORT loop
    /// iterates that buffer in chunks of `T` segments, binding sub-
    /// slices into ORT without copying. `offset_elements` is the
    /// starting f32 offset (0, T*n_mels*time_steps, 2*T*n_mels*time_steps,
    /// ...) and `batch` is the chunk size in segments.
    ///
    /// SAFETY: caller MUST guarantee that
    /// `(offset_elements + batch * n_mels * time_steps) * 4 bytes` is in
    /// bounds of the underlying `mel_d` allocation. This is enforced by
    /// `AudioModel::run_strategy_a` in `crate::models::audio` — outside
    /// callers should construct the offset against the same `n_segments
    /// * n_mels * time_steps` they used to allocate `mel_d`.
    pub fn run_iobinding_at_offset(
        &self,
        stream: &Arc<CudaStream>,
        mel_d: &CudaSlice<f32>,
        offset_elements: usize,
        batch: usize,
        n_mels: usize,
        time_steps: usize,
    ) -> Result<Vec<f32>> {
        checked_audio_chunk_end(mel_d.len(), offset_elements, batch, n_mels, time_steps)?;
        let shape: Shape = Shape::from([batch as i64, 1, n_mels as i64, time_steps as i64]);
        let (dev_ptr_u64, _sync) = mel_d.device_ptr(stream);
        // f32 pointer arithmetic: byte_offset = offset_elements * 4.
        let chunk_ptr_u64 =
            dev_ptr_u64 + (offset_elements as u64) * (std::mem::size_of::<f32>() as u64);
        let mem_info = self.cuda_mem_info.clone();

        // SAFETY: caller's contract above. We construct the device ptr
        // at the slab start; ORT reads exactly `batch * n_mels *
        // time_steps` f32 values from it.
        let input_tensor = unsafe {
            TensorRefMut::<f32>::from_raw(
                mem_info,
                chunk_ptr_u64 as usize as *mut std::ffi::c_void,
                shape,
            )
        }
        .map_err(|e| {
            SparrowEngineError::Ort(format!("TensorRefMut::from_raw (audio chunk): {e}"))
        })?;

        let mut guard = self
            .session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("AudioOrtSession session lock poisoned".into()))?;
        let outputs = guard
            .run(ort::inputs![&self.input_name => input_tensor])
            .map_err(|e| {
                SparrowEngineError::Ort(format!("Session::run (audio iobinding chunk): {e}"))
            })?;

        let output = outputs.get(self.output_name.as_str()).ok_or_else(|| {
            SparrowEngineError::Ort(format!("audio output '{}' not found", self.output_name))
        })?;
        let view = output
            .try_extract_array::<f32>()
            .map_err(|e| SparrowEngineError::Ort(format!("try_extract_array (chunk): {e}")))?;
        let logits: Vec<f32> = view.iter().copied().collect();
        Ok(logits)
    }

    /// Run the audio classifier with the legacy host-roundtrip path
    /// (DtoH the mel tensor first, build an `ndarray`, bind via
    /// `TensorRef::from_array_view`). Used by the parity test as the
    /// "bind-per-call" reference: max-abs Δ vs `run_iobinding` MUST be
    /// `0.0` (G0e gate).
    pub fn run_host_roundtrip(
        &self,
        stream: &Arc<CudaStream>,
        mel_d: &CudaSlice<f32>,
        batch: usize,
        n_mels: usize,
        time_steps: usize,
    ) -> Result<Vec<f32>> {
        let host: Vec<f32> = stream
            .clone_dtoh(mel_d)
            .map_err(|e| SparrowEngineError::Ort(format!("clone_dtoh (host roundtrip): {e}")))?;
        stream
            .synchronize()
            .map_err(|e| SparrowEngineError::Ort(format!("stream.synchronize after dtoh: {e}")))?;
        let arr: ndarray::Array4<f32> =
            ndarray::Array4::from_shape_vec((batch, 1, n_mels, time_steps), host)
                .map_err(|e| SparrowEngineError::Ort(format!("Array4::from_shape_vec: {e}")))?;
        let input_value = TensorRef::from_array_view(&arr)
            .map_err(|e| SparrowEngineError::Ort(format!("TensorRef::from_array_view: {e}")))?;
        let mut guard = self
            .session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("AudioOrtSession session lock poisoned".into()))?;
        let outputs = guard
            .run(ort::inputs![&self.input_name => input_value])
            .map_err(|e| {
                SparrowEngineError::Ort(format!("Session::run (audio host roundtrip): {e}"))
            })?;
        let output = outputs.get(self.output_name.as_str()).ok_or_else(|| {
            SparrowEngineError::Ort(format!("audio output '{}' not found", self.output_name))
        })?;
        let view = output
            .try_extract_array::<f32>()
            .map_err(|e| SparrowEngineError::Ort(format!("try_extract_array: {e}")))?;
        let logits: Vec<f32> = view.iter().copied().collect();
        Ok(logits)
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl std::fmt::Debug for AudioOrtSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioOrtSession")
            .field("input_name", &self.input_name)
            .field("output_name", &self.output_name)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_audio_chunk_end_accepts_exact_end() {
        let end = checked_audio_chunk_end(100, 40, 3, 5, 4).expect("in bounds");
        assert_eq!(end, 100);
    }

    #[test]
    fn checked_audio_chunk_end_rejects_oob() {
        let err =
            checked_audio_chunk_end(99, 40, 3, 5, 4).expect_err("one element past end must fail");
        assert!(
            err.to_string().contains("OOB"),
            "error should mention OOB, got {err}"
        );
    }

    #[test]
    fn checked_audio_chunk_end_rejects_overflow() {
        let err = checked_audio_chunk_end(usize::MAX, usize::MAX, 2, 2, 2)
            .expect_err("overflow must fail");
        assert!(
            err.to_string().contains("overflow"),
            "error should mention overflow, got {err}"
        );
    }
}
