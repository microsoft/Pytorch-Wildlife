//! Phase 3.8 Step 2 Wave 2 — End-to-end GPU audio model orchestrator.
//!
//! Wires the Wave 1 primitives (`crate::audio::*`) into a complete
//! `AudioInput` → mel-spectrogram → ORT → `AudioDetectResult` pipeline
//! for `MD_AudioBirds_V1`. Two strategies are exposed for D1
//! head-to-head benchmarking:
//!
//! - [`Strategy::HybridA`] (whole-clip GPU mel + chunk-of-T ORT). Single
//!   H2D upload of the resampled waveform; single window-frame /
//!   cuFFT R2C / power / cuBLAS sgemm / col→row transpose / power_to_db
//!   pass over the whole clip; ORT IoBinding loop iterates the resulting
//!   GPU mel buffer in slices of `T` segments.
//! - [`Strategy::PerBatchB`] (per-batch host-framing). Loops over batches
//!   of `T` segments. CPU pre-computes 16-segment offsets, H2Ds the slice,
//!   runs the same GPU pipeline at batch granularity, and immediately
//!   binds + runs ORT before iterating to the next batch.
//!
//! See `docs/design/phase3.8/step2/round_02/arch-perf_proposal_r2.md
//! §R2.7` (D1 hybrid case) and `arch-par_proposal_r2.md §9` (per-batch
//! case) for the design rationale.
//!
//! # Public API
//!
//! - [`AudioModel::load`]: parse manifest, build `AudioOrtSession` against
//!   `MD_AudioBirds_V1.onnx`, upload mel filterbank + Hann window once,
//!   compile the four CUDA kernels (window-frame / power / power_to_db /
//!   transpose), construct cuBLAS handle.
//! - [`AudioModel::detect`]: end-to-end inference against an
//!   [`AudioInput`].
//! - [`AudioModel::detect_streaming`]: per-segment callback variant
//!   (matches `sparrow_engine_cpu::detect_audio::detect_audio_streaming`).
//!
//! # Singleton + ORT environment
//!
//! `AudioModel::load` builds its OWN `AudioOrtSession`; it does NOT
//! depend on `sparrow_engine_gpu::Engine`. This matches the Wave 2/3/4 image
//! pattern (`YoloModel` / `ClassifierModel` / `TiledModel` are all
//! standalone-loadable). The Phase C consumer wiring (`sparrow-engine-cli` /
//! `sparrow-engine-python` / `sparrow-engine-server`) re-uses these standalone constructors
//! behind a feature flag.
//!
//! # Layout cheat-sheet
//!
//! Inside the GPU pipeline, mel data lives in two distinct buffers:
//!
//! - `mel_col_d`: column-major `[n_mels, total_frames]` (the cuBLAS sgemm
//!   output). Per-segment slab is byte-contiguous, internal layout is
//!   col-major.
//! - `mel_row_d`: per-segment row-major `[n_mels, frames_per_seg]`,
//!   bound directly into ORT IoBinding. The col→row
//!   [`crate::audio::transpose`] kernel populates this from `mel_col_d`.
//!
//! # Status (Wave 2)
//!
//! Wave 2 commit 1 lands the scaffold and `load`. Commits 2 + 3 fill in
//! Strategy A and Strategy B. Commit 4 wires the FP32-parity test on
//! the real DUNAS corpus. Commit 5 wires the bench harness.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::cufft::sys as cufft_sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use sparrow_engine_core::preprocess_audio::{self, AudioPreprocessConfig, AudioSamples};
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{
    self, InferenceStrategy, ModelManifest, PostprocessMethod, Precision, PreprocessMethod,
};
use sparrow_engine_types::{
    AudioClass, AudioDetectOpts, AudioDetectResult, AudioInput, AudioSegment,
};

use crate::audio::cufft_plan::{alloc_complex_output, BatchedR2cPlan};
use crate::audio::hann::{upload_hann_window, upload_mel_filterbank, UploadedMelFilterbank};
use crate::audio::mel_gemm::MelGemm;
use crate::audio::ort_io::AudioOrtSession;
use crate::audio::power_kernel::{power_gpu, PowerKernel};
use crate::audio::power_to_db::{power_to_db_gpu, PowerToDbKernel};
use crate::audio::transpose::{transpose_per_segment_gpu, TransposeKernel};
use crate::audio::window_frame::{window_frame_gpu, WindowFrameKernel};

const DEFAULT_AUDIO_CLASSIFIER_TOP_K: usize = 5;

// ---------------------------------------------------------------------------
// Strategy + opts
// ---------------------------------------------------------------------------

/// D1 sliding-window strategy (Wave 2 head-to-head).
#[derive(Debug, Clone, Copy)]
pub enum Strategy {
    /// Whole-clip GPU mel + ORT chunk-of-T (`arch-perf §R2.7`).
    ///
    /// One H2D, one batched cuFFT call sized for ALL frames in the clip,
    /// one cuBLAS sgemm. ORT loop iterates the GPU mel buffer in slices
    /// of `ort_chunk_segments`.
    HybridA { ort_chunk_segments: usize },
    /// Per-batch host-framing (`arch-par §9`).
    ///
    /// Loop over batches of `batch_segments`. CPU pre-computes per-batch
    /// segment offsets, H2D a slice of the waveform, run the same GPU
    /// pipeline at batch granularity, immediately ORT.
    PerBatchB { batch_segments: usize },
    /// Whole-clip GPU mel + **one** ORT call covering all segments.
    ///
    /// Production default for non-streaming `AudioModel::detect`. Avoids
    /// the per-`Session::run` setup overhead surfaced by the
    /// post-Wave-4 perf triage (`docs/research/phase3.8/step2/
    /// perf_triage_report.md`): on a 60 s DUNAS clip with `n_segments=
    /// 198`, switching from `HybridA{16}` (13 chunks) or `HybridA{197}`
    /// (2 chunks, off-by-one) to a single ORT call collapses ~62 ms of
    /// chunk-loop overhead. Equivalent to `HybridA{n_segments}` but
    /// avoids the off-by-one / overflow hazards of forcing the chunk
    /// value to a saturated `usize`.
    SingleCall,
}

impl Strategy {
    /// Effective batch size on the ORT side. Used by the bench harness
    /// to label cells. `SingleCall` reports `0` as a sentinel — the
    /// effective chunk is `n_segments`, which depends on the input clip
    /// duration; the runtime resolves it inside `detect_inner`.
    pub fn ort_chunk(&self) -> usize {
        match self {
            Strategy::HybridA { ort_chunk_segments } => *ort_chunk_segments,
            Strategy::PerBatchB { batch_segments } => *batch_segments,
            Strategy::SingleCall => 0,
        }
    }

    /// Short label for log output / bench cells.
    pub fn short_label(&self) -> String {
        match self {
            Strategy::HybridA { ort_chunk_segments } => format!("A_T{ort_chunk_segments}"),
            Strategy::PerBatchB { batch_segments } => format!("B_T{batch_segments}"),
            Strategy::SingleCall => "A_single".to_string(),
        }
    }
}

/// Full options for [`AudioModel::detect`].
#[derive(Debug, Clone)]
pub struct GpuAudioDetectOpts {
    /// Standard sliding-window opts (threshold + segment duration/stride
    /// overrides). Same shape as `sparrow_engine_types::AudioDetectOpts`.
    pub base: AudioDetectOpts,
    /// D1 strategy (whole-clip vs per-batch).
    pub strategy: Strategy,
}

impl GpuAudioDetectOpts {
    /// Default strategy for **non-streaming** [`AudioModel::detect`]:
    /// a single ORT call covering all segments in the clip.
    ///
    /// Produces the lowest latency for full-clip detect (no
    /// per-`Session::run` setup overhead amortized across multiple
    /// chunks). On 60 s DUNAS clips this saves ~62 ms versus
    /// `HybridA{16}` per `docs/research/phase3.8/step2/
    /// perf_triage_report.md`.
    pub fn default_strategy() -> Strategy {
        Strategy::SingleCall
    }

    /// Default strategy for **streaming**
    /// [`AudioModel::detect_streaming`]: 16-segment chunks (matches
    /// `sparrow-engine-cpu/src/detect_audio.rs::DEFAULT_BATCH_SIZE`).
    ///
    /// Per the Wave 2 D2 decision (Variant B), the streaming variant
    /// keeps the chunk-of-16 cadence so per-batch callbacks fire as
    /// segments become available. Single-call would defer all callbacks
    /// to the end of the clip, defeating the streaming contract.
    pub fn default_strategy_streaming() -> Strategy {
        Strategy::HybridA {
            ort_chunk_segments: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Workspace — Fix C: high-water-mark device-buffer cache reused across detect
// calls, eliminating the per-call `cudaMallocAsync` + zero-fill cost
// (~5 ms on a 60 s clip per `perf_triage_report.md` § "Step 4 — Full path
// recommendation" item #3). Buffers grow on demand and never shrink, so
// successive same-size calls pay zero allocation cost.
// ---------------------------------------------------------------------------

/// Single high-water-mark device buffer.
///
/// `capacity` tracks the allocated element count; `ensure(needed)` lazily
/// (re)allocates when the request exceeds the current capacity, replacing
/// the prior allocation (cudarc's `Drop` returns the memory to the
/// async-alloc pool / driver). The buffer contents are NOT zeroed on
/// reuse — every consumer in the audio pipeline overwrites the buffer
/// before the next stage reads it (window-frame writes the full
/// `windowed_d`; cuFFT writes the full `complex_d`; power kernel writes
/// the full `power_d`; cuBLAS GEMM writes the full `mel_col_d`;
/// transpose writes the full `mel_row_d` per segment). Trading
/// `alloc_zeros` for `alloc_zeros` (first time only) preserves the
/// safety contract of the original code path.
struct WorkspaceBuf<T: DeviceRepr + ValidAsZeroBits> {
    buf: Option<CudaSlice<T>>,
    capacity: usize,
}

impl<T: DeviceRepr + ValidAsZeroBits> WorkspaceBuf<T> {
    const fn new() -> Self {
        Self {
            buf: None,
            capacity: 0,
        }
    }

    /// Ensure the buffer has at least `needed` elements. Lazy-grow only —
    /// never shrinks. No-op if `needed == 0`.
    fn ensure(&mut self, stream: &Arc<CudaStream>, needed: usize, label: &str) -> Result<()> {
        if needed == 0 {
            return Ok(());
        }
        if self.capacity < needed {
            // Grow: drop the old allocation by overwriting the Option and
            // alloc a fresh buffer sized for `needed`. We call
            // `alloc_zeros` (matching the previous per-call behaviour) so
            // first-time content is deterministic; subsequent calls reuse
            // without re-zeroing.
            self.buf = Some(stream.alloc_zeros::<T>(needed).map_err(|e| {
                SparrowEngineError::Ort(format!("AudioWorkspace alloc_zeros ({label}): {e}"))
            })?);
            self.capacity = needed;
        }
        Ok(())
    }

    /// Borrow the buffer mutably. Panics if `ensure` has not been called
    /// (caller bug — every code path under `AudioModel` ensures before use).
    fn get_mut(&mut self) -> &mut CudaSlice<T> {
        self.buf
            .as_mut()
            .expect("WorkspaceBuf::get_mut called before ensure()")
    }
}

/// Cached device buffers for the audio mel pipeline.
///
/// Held inside `AudioModel` behind a `Mutex` so `detect` can mutate
/// through a `&self` receiver. All five buffers grow independently to
/// the high-water-mark across all calls on this engine instance.
struct AudioWorkspace {
    /// `[total_frames * n_fft]` f32 — output of `window_frame_gpu`,
    /// input of cuFFT R2C.
    windowed: WorkspaceBuf<f32>,
    /// `[total_frames * n_freqs]` complex f32 — output of cuFFT R2C,
    /// input of `power_gpu`.
    complex: WorkspaceBuf<cufft_sys::float2>,
    /// `[total_frames * n_freqs]` f32 — output of `power_gpu`, input of
    /// the cuBLAS sgemm.
    power: WorkspaceBuf<f32>,
    /// `[n_mels * total_frames]` f32 col-major — output of cuBLAS sgemm,
    /// input of the per-segment transpose.
    mel_col: WorkspaceBuf<f32>,
    /// `[n_segments * n_mels * frames_per_seg]` f32 row-major — output
    /// of the transpose + in-place `power_to_db`, bound directly into
    /// ORT IoBinding.
    mel_row: WorkspaceBuf<f32>,
}

impl AudioWorkspace {
    const fn new() -> Self {
        Self {
            windowed: WorkspaceBuf::new(),
            complex: WorkspaceBuf::new(),
            power: WorkspaceBuf::new(),
            mel_col: WorkspaceBuf::new(),
            mel_row: WorkspaceBuf::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// AudioModel
// ---------------------------------------------------------------------------

/// CUDA-resident audio detection pipeline for `MD_AudioBirds_V1`.
pub struct AudioModel {
    #[allow(dead_code)] // ctx kept alive for kernel + session lifetimes.
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    ort_session: AudioOrtSession,
    config: AudioPreprocessConfig,

    // GPU constants — uploaded once at load.
    hann_d: CudaSlice<f32>,
    filterbank_d: UploadedMelFilterbank,

    // Compiled kernels.
    window_frame_kernel: WindowFrameKernel,
    power_kernel: PowerKernel,
    power_to_db_kernel: PowerToDbKernel,
    transpose_kernel: TransposeKernel,

    // cuBLAS handle (constructed from same stream).
    mel_gemm: MelGemm,

    // Cached cuFFT plans, keyed by `total_frames` so back-to-back calls
    // with the same audio length pay the plan-creation cost ONCE.
    plans: Mutex<HashMap<usize, Arc<BatchedR2cPlan>>>,

    /// Fix C: per-call mel-pipeline device buffers, cached across calls.
    /// Lazy-grown high-water-mark — same audio duration on the second
    /// detect call pays zero allocation cost.
    workspace: Mutex<AudioWorkspace>,

    // Manifest-derived defaults.
    postprocess: AudioPostprocess,
    num_classes: usize,
    labels: Vec<String>,
    threshold: f32,
    segment_duration_s: f32,
    stride_s: f32,
    sample_rate: u32,
}

// SAFETY: every field is itself `Send + Sync` (CudaContext + CudaStream
// are `Send + Sync`; `AudioOrtSession` declares `Send + Sync`; CudaSlice
// is `Send + Sync`; kernels hold cudarc CudaFunction which is Send+Sync;
// MelGemm declares Debug-only and uses `Send`+`Sync` internally;
// HashMap<_, Arc<_>> behind Mutex). Mirror Wave 1 audio + Step 1 image
// model patterns.
unsafe impl Send for AudioModel {}
unsafe impl Sync for AudioModel {}

impl AudioModel {
    /// Build the audio model from a manifest path.
    ///
    /// - Parses + validates the manifest.
    /// - Resolves the ONNX path (`manifest_dir/<model_file>`), respecting
    ///   the FP32 / FP16 split (audio model is always FP32 per Wave 1
    ///   primitive bench).
    /// - Builds the `AudioOrtSession` (CUDA EP first, CPU fallback).
    /// - Uploads Hann window + mel filterbank.
    /// - Compiles the 4 NVRTC kernels.
    pub fn load(ctx: &Arc<CudaContext>, manifest_path: &Path) -> Result<Self> {
        let manifest = manifest::load_manifest(manifest_path)?;
        // Flavor-strict: the gpu flavor runs ONNX via ORT. The shared loader now
        // also accepts `tflite` (mobile flavor); reject it here with a clear error
        // rather than failing later with an opaque ORT parse error. Mirrors
        // gpu/cpu Engine::load_model.
        if manifest.format != "onnx" {
            return Err(SparrowEngineError::UnsupportedFormat {
                format: manifest.format.clone(),
            });
        }
        let manifest_dir = manifest_path
            .parent()
            .ok_or_else(|| {
                SparrowEngineError::Ort(format!(
                    "manifest path has no parent dir: {manifest_path:?}"
                ))
            })?
            .to_path_buf();
        Self::load_from_manifest(ctx, &manifest, &manifest_dir)
    }

    /// Lower-level loader: takes a parsed manifest + manifest dir.
    pub fn load_from_manifest(
        ctx: &Arc<CudaContext>,
        manifest: &ModelManifest,
        manifest_dir: &Path,
    ) -> Result<Self> {
        // 1. Validate that this manifest describes an audio (mel) model.
        let (sample_rate, segment_duration_s, stride_s, threshold) =
            extract_audio_params(manifest)?;
        let (postprocess, labels) = resolve_audio_postprocess(manifest, manifest_dir)?;
        let num_classes = postprocess.num_classes();
        let config =
            AudioPreprocessConfig::from_manifest(&manifest.preprocess_method).ok_or_else(|| {
                SparrowEngineError::NotAnAudioModel {
                    id: manifest.id.clone(),
                    method: "non-mel".to_string(),
                }
            })?;

        // 2. Resolve ONNX path. Phase 3.8 Step 2 Wave 3 wired the FP16
        // path: when `[inference] precision = "fp16"` the manifest must
        // also carry `[model] file_fp16 = "..."` (validated at parse
        // time). FP16 ONNX is generated via `sparrow-engine/tools/convert_fp16.py`
        // (audio CNN: 91 nodes, plain Conv/MaxPool/Gemm graph; FP16
        // converter `keep_io_types=True` so input/output stay FP32).
        // Mirrors the YOLO/Classifier/Tiled patterns in this crate.
        let onnx_path = match manifest.precision {
            Precision::Fp32 => manifest_dir.join(&manifest.model_file),
            Precision::Int8 => manifest_dir.join(&manifest.model_file),
            Precision::Fp16 => {
                manifest_dir.join(manifest.model_file_fp16.as_ref().ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(
                        "AudioModel: precision = fp16 requires [model] file_fp16 in manifest"
                            .into(),
                    )
                })?)
            }
        };
        if !onnx_path.exists() {
            return Err(SparrowEngineError::Ort(format!(
                "audio ONNX file does not exist: {onnx_path:?}"
            )));
        }

        // 3. Build a dedicated non-default CUDA stream for the audio
        // pipeline. Phase 3.8 Step 2 perf-fix Fix D: ORT's CUDA EP is
        // bound to this stream (`with_compute_stream`), so all mel-
        // pipeline kernels (window-frame / cuFFT / power / cuBLAS GEMM /
        // transpose / power_to_db) and the ORT inference share one
        // stream — kernel ordering is enforced by the stream itself,
        // eliminating the pre-ORT `cudaStreamSynchronize` that the
        // legacy default-stream path paid (~2-5 ms per detect on 60 s
        // clips, see `perf_triage_report.md` § "Step 4 — Full path
        // recommendation" item #4).
        //
        // Dedicated (non-NULL) stream is REQUIRED here: ORT's
        // `with_compute_stream(NULL)` would be interpreted as "no user
        // stream set" by the underlying ONNX Runtime CUDA EP, which
        // would default ORT back to its own internal stream and
        // re-introduce the cross-stream sync.
        let stream = ctx
            .new_stream()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.new_stream (audio): {e}")))?;

        // 4. Build ORT session (CUDA EP first, CPU fallback) bound to
        // this stream.
        let ort_session = AudioOrtSession::load(ctx, &stream, &onnx_path)?;

        // 5. GPU constants — uploaded on the audio model's stream so
        // they become visible to subsequent kernels via cudarc's
        // event-tracked sync.
        let hann_d = upload_hann_window(&stream, config.n_fft as usize)?;
        let filterbank_d = upload_mel_filterbank(&stream, &config)?;

        // 6. Kernels.
        let window_frame_kernel = WindowFrameKernel::new(ctx)?;
        let power_kernel = PowerKernel::new(ctx)?;
        let power_to_db_kernel = PowerToDbKernel::new(ctx)?;
        let transpose_kernel = TransposeKernel::new(ctx)?;

        // 7. cuBLAS handle (constructed against the same dedicated
        // stream so cuBLAS sgemm dispatches alongside the kernels +
        // ORT).
        let mel_gemm = MelGemm::new(stream.clone(), filterbank_d.n_mels, filterbank_d.n_freqs)?;

        Ok(AudioModel {
            ctx: ctx.clone(),
            stream,
            ort_session,
            config,
            hann_d,
            filterbank_d,
            window_frame_kernel,
            power_kernel,
            power_to_db_kernel,
            transpose_kernel,
            mel_gemm,
            plans: Mutex::new(HashMap::new()),
            workspace: Mutex::new(AudioWorkspace::new()),
            postprocess,
            num_classes,
            labels,
            threshold,
            segment_duration_s,
            stride_s,
            sample_rate,
        })
    }

    /// Borrow the four mel-pipeline kernels packaged in the
    /// [`PipelineKernels`] view.
    ///
    /// S7 extract (R2 audit-fix 2026-05-05): `Strategy::HybridA` /
    /// `SingleCall` / `PerBatchB` and the diagnostic
    /// `compute_mel_per_segment` all need this same 5-field struct
    /// literal. Localising the construction to one method means a
    /// future kernel addition (Phase B/C) is a one-line touch
    /// instead of a 3-call-site sweep.
    fn pipeline_kernels(&self) -> PipelineKernels<'_> {
        PipelineKernels {
            window_frame_kernel: &self.window_frame_kernel,
            power_kernel: &self.power_kernel,
            power_to_db_kernel: &self.power_to_db_kernel,
            transpose_kernel: &self.transpose_kernel,
        }
    }

    /// Resolve the per-call sliding-window parameters (manifest defaults
    /// overrideable via `opts`).
    fn resolve_window(&self, opts: &AudioDetectOpts) -> Result<WindowParams> {
        let segment_duration_s = opts.segment_duration_s.unwrap_or(self.segment_duration_s);
        let stride_s = opts.stride_s.unwrap_or(self.stride_s);
        let threshold = opts.confidence_threshold.unwrap_or(self.threshold);
        let (segment_samples, stride_samples) = preprocess_audio::validate_audio_window_params(
            segment_duration_s,
            stride_s,
            threshold,
            self.sample_rate,
            self.config.n_fft,
        )?;
        Ok(WindowParams {
            segment_samples,
            stride_samples,
            threshold,
            sample_rate: self.sample_rate,
            n_fft: self.config.n_fft as usize,
            hop_length: self.config.hop_length as usize,
            n_mels: self.filterbank_d.n_mels,
            n_freqs: self.filterbank_d.n_freqs,
            top_db: self.config.top_db,
        })
    }

    /// Stub (Wave 2 commit 1): full implementation lands in commits 2 + 3.
    pub fn detect(
        &self,
        audio: &AudioInput,
        opts: &GpuAudioDetectOpts,
    ) -> Result<AudioDetectResult> {
        self.detect_inner(audio, opts, &mut None)
    }

    /// Streaming variant — invokes `on_segment` for each above-threshold
    /// detection in production order.
    ///
    /// **Callback cadence diverges from `sparrow-engine-cpu`** (Phase 3.8 Step 2
    /// Wave 2 architectural choice). Both crates fire callbacks
    /// post-threshold in chronological order, but the timing differs:
    ///
    /// | Crate | Callback fires |
    /// | --- | --- |
    /// | `sparrow-engine-cpu::detect_audio_streaming` | per-batch inside the chunk loop |
    /// | `sparrow-engine-gpu::AudioModel::detect_streaming` | once at detect-end (after the full chunk loop completes) |
    ///
    /// The GPU path defers callbacks because `Strategy::HybridA` /
    /// `Strategy::SingleCall` hold the [`AudioWorkspace`] mutex for the
    /// duration of the chunk loop ([`run_strategy_a`] acquires at the H2D
    /// stage and drops BEFORE postprocess, see `audio.rs:638..766`). Two
    /// constraints force the per-detect-end fire site:
    ///
    /// 1. **Reentrant deadlock**: invoking `on_segment` while the
    ///    workspace mutex is held would deadlock if the callback re-calls
    ///    `detect()` on the same [`AudioModel`].
    /// 2. **Workspace cache invariant** (Fix C): dropping + re-acquiring
    ///    the mutex mid-detect would defeat the high-water-mark cache
    ///    that amortizes alloc cost across detect calls.
    ///
    /// Phase C consumer wiring (server / CLI / Python streaming
    /// adapters) MUST account for this divergence — clients that need
    /// per-batch cadence either (a) consume `sparrow-engine-cpu` directly,
    /// (b) issue smaller-segment detect calls, or (c) post Phase 3.8
    /// Step 2 wait for an explicit per-chunk callback API on
    /// `AudioModel`. `sparrow-engine-cpu`'s per-batch cadence remains the
    /// reference contract for `sparrow_engine_detect_audio_streaming` FFI users
    /// until that API lands.
    pub fn detect_streaming(
        &self,
        audio: &AudioInput,
        opts: &GpuAudioDetectOpts,
        mut on_segment: impl FnMut(&AudioSegment),
    ) -> Result<AudioDetectResult> {
        self.detect_inner(
            audio,
            opts,
            &mut Some(&mut on_segment as &mut dyn FnMut(&AudioSegment)),
        )
    }

    /// Inner dispatch: load + segment + run the chosen strategy.
    fn detect_inner(
        &self,
        audio: &AudioInput,
        opts: &GpuAudioDetectOpts,
        on_segment: &mut Option<&mut dyn FnMut(&AudioSegment)>,
    ) -> Result<AudioDetectResult> {
        let start = Instant::now();
        let win = self.resolve_window(&opts.base)?;

        // Decode + resample on CPU (matches sparrow-engine-cpu; Wave 2 brief keeps
        // decode + resample CPU per the Step 2 scope).
        let samples = preprocess_audio::load_audio(audio, &self.config)?;
        let total_samples = samples.data.len();
        let duration_s = samples.duration_s;

        // Pre-compute segment offsets (matches sparrow-engine-cpu termination logic).
        let segment_offsets = preprocess_audio::compute_segment_offsets(
            total_samples,
            win.segment_samples,
            win.stride_samples,
        );
        let n_segments = segment_offsets.len();
        // Shared audio option validation guarantees `segment_samples >= n_fft`,
        // so the subtraction below cannot underflow.
        let frames_per_seg = ((win.segment_samples - win.n_fft) / win.hop_length) + 1;

        let segments = match opts.strategy {
            Strategy::HybridA { ort_chunk_segments } => self.run_strategy_a(
                &samples,
                &segment_offsets,
                &win,
                frames_per_seg,
                ort_chunk_segments,
                on_segment,
            )?,
            Strategy::PerBatchB { batch_segments } => self.run_strategy_b(
                &samples,
                &segment_offsets,
                &win,
                frames_per_seg,
                batch_segments,
                on_segment,
            )?,
            // Strategy A with chunk = n_segments → exactly one
            // `Session::run` call. The empty-clip case is handled by
            // `run_strategy_a`'s early-return at `n_segments == 0` (see
            // `run_strategy_a` body), so passing the raw `n_segments`
            // here is safe even when it's 0.
            Strategy::SingleCall => self.run_strategy_a(
                &samples,
                &segment_offsets,
                &win,
                frames_per_seg,
                n_segments,
                on_segment,
            )?,
        };

        let elapsed = start.elapsed();
        Ok(AudioDetectResult {
            segments,
            duration_s,
            sample_rate: win.sample_rate,
            processing_time_ms: elapsed.as_secs_f32() * 1000.0,
        })
    }

    /// Strategy A — whole-clip GPU mel + ORT chunk-of-T.
    ///
    /// Phase 3.8 Step 2 Wave 2 commit 2 (arch-perf §R2.7 hybrid).
    ///
    /// Pipeline (one launch each, sized for the whole 60 s clip's
    /// `total_frames = n_segments * frames_per_seg`):
    /// 1. H2D the resampled waveform (one copy) + per-frame absolute
    ///    sample offsets.
    /// 2. window_frame kernel over all frames at once.
    /// 3. cuFFT R2C batched (cached plan keyed by `total_frames`).
    /// 4. power kernel.
    /// 5. cuBLAS sgemm.
    /// 6. col→row transpose into per-segment row-major buffer.
    /// 7. power_to_db (per-segment in-place).
    /// 8. ORT loop over chunks of `ort_chunk` segments.
    ///
    /// Tracing emits per-stage `audio.gpu.<stage>` events for the bench
    /// harness.
    fn run_strategy_a(
        &self,
        samples: &AudioSamples,
        segment_offsets: &[usize],
        win: &WindowParams,
        frames_per_seg: usize,
        ort_chunk: usize,
        on_segment: &mut Option<&mut dyn FnMut(&AudioSegment)>,
    ) -> Result<Vec<AudioSegment>> {
        // Empty-clip short-circuit MUST come before the `ort_chunk == 0`
        // guard. `Strategy::SingleCall` now passes `ort_chunk = n_segments`
        // (without the prior `n_segments.max(1)` defensive clamp) and on
        // empty input that would otherwise turn the ort_chunk guard into
        // a spurious error path. Reordering preserves the historical
        // `Ok(Vec::new())` behavior for empty clips.
        let n_segments = segment_offsets.len();
        if n_segments == 0 {
            return Ok(Vec::new());
        }
        if ort_chunk == 0 {
            return Err(SparrowEngineError::Ort(
                "Strategy::HybridA ort_chunk_segments must be > 0".to_string(),
            ));
        }
        let total_frames = n_segments * frames_per_seg;
        if total_frames == 0 {
            return Ok(Vec::new());
        }

        // ---- Stage 0: H2D once ----
        let t_h2d = Instant::now();
        let samples_d = self
            .stream
            .clone_htod(&samples.data)
            .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (samples): {e}")))?;
        let frame_starts = build_frame_starts(segment_offsets, frames_per_seg, win.hop_length)?;
        let frame_starts_d = self
            .stream
            .clone_htod(&frame_starts)
            .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (frame_starts): {e}")))?;
        tracing::info!(
            stage = "audio.gpu.h2d",
            duration_ns = t_h2d.elapsed().as_nanos() as u64,
            samples = samples.data.len(),
        );

        // ---- Stage 1-7: GPU mel pipeline ----
        // Fix C: borrow the cached workspace for the duration of this
        // detect call. Buffers grow to the high-water-mark; same-size
        // back-to-back calls pay zero allocation cost. Held across the
        // ORT loop too (no `drop()` between mel and ORT) so the cached
        // capacity persists for the next call.
        let t_mel = Instant::now();
        let mut ws = self
            .workspace
            .lock()
            .map_err(|_| SparrowEngineError::Ort("AudioWorkspace lock poisoned".to_string()))?;
        ws.windowed
            .ensure(&self.stream, total_frames * win.n_fft, "windowed")?;
        ws.complex
            .ensure(&self.stream, total_frames * win.n_freqs, "complex")?;
        ws.power
            .ensure(&self.stream, total_frames * win.n_freqs, "power")?;
        ws.mel_col
            .ensure(&self.stream, win.n_mels * total_frames, "mel_col")?;
        ws.mel_row.ensure(
            &self.stream,
            n_segments * win.n_mels * frames_per_seg,
            "mel_row",
        )?;

        // Disjoint-field borrow split — the borrow checker accepts five
        // simultaneous mutable refs to disjoint struct fields via
        // destructuring. Each `get_mut` pulls out the inner CudaSlice.
        let AudioWorkspace {
            windowed,
            complex,
            power,
            mel_col,
            mel_row,
        } = &mut *ws;
        let windowed_d = windowed.get_mut();
        let complex_d = complex.get_mut();
        let power_d = power.get_mut();
        let mel_col_d = mel_col.get_mut();
        let mel_row_d = mel_row.get_mut();

        let plan = self.get_or_build_plan(total_frames)?;
        let kernels = self.pipeline_kernels();
        run_mel_pipeline_into(
            &self.stream,
            &kernels,
            &plan,
            &self.mel_gemm,
            &samples_d,
            &frame_starts_d,
            &self.hann_d,
            &self.filterbank_d.data,
            windowed_d,
            complex_d,
            power_d,
            mel_col_d,
            mel_row_d,
            win.n_fft,
            win.n_mels,
            win.n_freqs,
            n_segments,
            frames_per_seg,
            total_frames,
            samples.data.len(),
            win.top_db,
        )?;
        // Fix D: removed `synchronize_stream(&self.stream)?` — the ORT
        // session's CUDA EP is bound to `self.stream` via
        // `ort::ep::CUDA::with_compute_stream` (`AudioOrtSession::load`),
        // so the chunk-loop ORT calls are stream-ordered after the mel
        // pipeline kernels without a CPU-blocking sync. Saves ~2-5 ms
        // per detect on 60 s clips per `perf_triage_report.md` § "Step 4
        // — Full path recommendation" item #4.
        tracing::info!(
            stage = "audio.gpu.mel",
            duration_ns = t_mel.elapsed().as_nanos() as u64,
            n_segments = n_segments,
        );

        // ---- Stage 8: ORT chunk-of-T ----
        // Note: the cached mel-pipeline buffers (`windowed`, `complex`,
        // `power`, `mel_col`) stay allocated across the ORT loop now —
        // peak VRAM during a 60 s clip detect remains at the analytic
        // ~411 MiB documented in `wave2_e2e_bench.md` § "VRAM peak"; the
        // only observable change is that VRAM is reserved BETWEEN detect
        // calls instead of being returned. Steady-state peak unchanged.
        let t_ort = Instant::now();
        let mut all_logits: Vec<f32> = Vec::with_capacity(n_segments * self.num_classes);
        // The reported `chunk_elements` is the per-call buffer
        // footprint, capped at `n_segments` so callers can pass
        // `ort_chunk = usize::MAX`-style sentinels (e.g. via
        // `Strategy::SingleCall`) without overflow.
        let chunk_elements = ort_chunk.min(n_segments) * win.n_mels * frames_per_seg;
        let mut chunk_idx = 0usize;
        let mut chunk_seq = 0usize;
        while chunk_idx < n_segments {
            let chunk_batch = (n_segments - chunk_idx).min(ort_chunk);
            let offset_elements = chunk_idx * win.n_mels * frames_per_seg;
            let t_chunk = Instant::now();
            let logits = self.ort_session.run_iobinding_at_offset(
                &self.stream,
                mel_row_d,
                offset_elements,
                chunk_batch,
                win.n_mels,
                frames_per_seg,
            )?;
            tracing::info!(
                stage = "audio.gpu.ort.chunk",
                duration_ns = t_chunk.elapsed().as_nanos() as u64,
                chunk_seq = chunk_seq,
                chunk_batch = chunk_batch,
            );
            let expected_logits = chunk_batch * self.num_classes;
            if logits.len() != expected_logits {
                return Err(SparrowEngineError::Ort(format!(
                    "Audio model returned {} logits for chunk of {chunk_batch} x {} classes; expected exactly {expected_logits}",
                    logits.len(),
                    self.num_classes,
                )));
            }
            if !logits.iter().all(|logit| logit.is_finite()) {
                return Err(SparrowEngineError::Ort(
                    "Audio model returned non-finite logits".to_string(),
                ));
            }
            all_logits.extend(logits);
            chunk_idx += chunk_batch;
            chunk_seq += 1;
        }
        tracing::info!(
            stage = "audio.gpu.ort",
            duration_ns = t_ort.elapsed().as_nanos() as u64,
            chunk_elements = chunk_elements,
            n_segments = n_segments,
        );
        // Release the workspace lock before postprocess so the next
        // detect call doesn't block on the (host-only) sigmoid + collect
        // stage.
        drop(ws);

        // ---- Stage 9: postprocess (sigmoid + threshold + collect/callback) ----
        let t_post = Instant::now();
        let segments = collect_segments_for_postprocess(
            &all_logits,
            segment_offsets,
            samples.data.len(),
            win,
            &self.postprocess,
            &self.labels,
            on_segment,
        );
        tracing::info!(
            stage = "audio.gpu.post",
            duration_ns = t_post.elapsed().as_nanos() as u64,
            n_segments = n_segments,
        );
        Ok(segments)
    }

    /// Strategy B — per-batch host-framing.
    ///
    /// Phase 3.8 Step 2 Wave 2 commit 3 (arch-par §9 R2 stance).
    ///
    /// Loop over batches of `batch_segments`. For each batch:
    /// 1. CPU pre-computes the batch's `batch_segments` segment offsets +
    ///    derives per-frame absolute sample offsets relative to the
    ///    sliced waveform.
    /// 2. H2D the slice of waveform that covers all those offsets +
    ///    H2D the per-frame offsets vector (relative to the slice).
    /// 3. Run the same window-frame / cuFFT / power / cuBLAS sgemm /
    ///    transpose / power_to_db chain at batch granularity. cuFFT
    ///    plan is cached per `(n_fft, total_frames=batch*frames_per_seg)`
    ///    so all but the first batch reuse the plan.
    /// 4. ORT IoBinding bind+run+DtoH (single call for the batch).
    /// 5. Postprocess the batch's logits.
    ///
    /// The trade-off vs Strategy A: more launches (one per batch ×
    /// 6 stages = ~6 × num_batches kernel launches), but per-batch
    /// peak GPU memory is much smaller (~64 MB at batch=16 vs ~410 MB
    /// for the whole-clip path).
    fn run_strategy_b(
        &self,
        samples: &AudioSamples,
        segment_offsets: &[usize],
        win: &WindowParams,
        frames_per_seg: usize,
        batch_segments: usize,
        on_segment: &mut Option<&mut dyn FnMut(&AudioSegment)>,
    ) -> Result<Vec<AudioSegment>> {
        if batch_segments == 0 {
            return Err(SparrowEngineError::Ort(
                "Strategy::PerBatchB batch_segments must be > 0".to_string(),
            ));
        }
        let n_segments = segment_offsets.len();
        if n_segments == 0 {
            return Ok(Vec::new());
        }

        // Pre-allocate per-batch buffers (sized for the FULL batch). The
        // last batch may be smaller; cuFFT plan keyed by total_frames
        // means a partial last batch builds a fresh plan one-shot. To
        // keep the bench fair we always run full-batch except the last.
        // Empirically (Wave 1 W1.5) batch=16 maxes out the audio model;
        // smaller residual batches don't change the conclusion.
        let total_frames_full = batch_segments * frames_per_seg;
        let mut windowed_d =
            alloc_audio_buf(&self.stream, total_frames_full * win.n_fft, "windowed")?;
        let mut complex_d = alloc_complex_output(&self.stream, total_frames_full, win.n_freqs)?;
        let mut power_d = alloc_audio_buf(&self.stream, total_frames_full * win.n_freqs, "power")?;
        let mut mel_col_d =
            alloc_audio_buf(&self.stream, win.n_mels * total_frames_full, "mel_col")?;
        let mut mel_row_d = alloc_audio_buf(
            &self.stream,
            batch_segments * win.n_mels * frames_per_seg,
            "mel_row",
        )?;

        // The last batch may have fewer segments. We still allocate
        // dedicated smaller buffers for that batch (so the cuFFT plan
        // can match its total_frames). Keeping the partial-batch buffers
        // separate avoids replanning cuFFT mid-call.
        let mut last_batch_buffers: Option<LastBatchBuffers> = None;

        let kernels = self.pipeline_kernels();

        let mut all_logits: Vec<f32> = Vec::with_capacity(n_segments * self.num_classes);
        let total_samples = samples.data.len();
        let total_audio_samples = total_samples;

        for batch_start in (0..n_segments).step_by(batch_segments) {
            let batch_end = (batch_start + batch_segments).min(n_segments);
            let batch_n = batch_end - batch_start;
            let total_frames_batch = batch_n * frames_per_seg;
            let batch_offsets = &segment_offsets[batch_start..batch_end];

            // Compute per-frame absolute offsets for THIS batch.
            let frame_starts = build_frame_starts(batch_offsets, frames_per_seg, win.hop_length)?;

            // H2D: copy the WHOLE waveform once would amortize across
            // batches but the brief mandates "One H2D per batch (16-segment
            // waveform slice)" for Strategy B (arch-par §9). To match,
            // we copy only the sample range this batch reads, with
            // padded tail.
            let t_h2d = Instant::now();
            let first_frame_start = batch_offsets[0];
            // Last frame in this batch reads up to
            // `last_seg + (frames_per_seg-1)*hop_length + n_fft`. Pad to
            // segment boundary if any sample is OOB to mirror sparrow-engine-cpu
            // tail-handling.
            let last_seg = *batch_offsets.last().unwrap();
            let last_read = last_seg + (frames_per_seg - 1) * win.hop_length + win.n_fft;
            let slice_end_in_audio = last_read.min(total_audio_samples);
            let slice_start = first_frame_start;
            let slice_audio_len = slice_end_in_audio.saturating_sub(slice_start);
            let mut slice_buf = Vec::<f32>::with_capacity(last_read - slice_start);
            slice_buf.extend_from_slice(&samples.data[slice_start..slice_end_in_audio]);
            // Zero-pad if last batch is partial-tail (matches CPU
            // `padded.resize(segment_samples, 0.0)` semantics).
            let needed = last_read - slice_start;
            if slice_buf.len() < needed {
                slice_buf.resize(needed, 0.0);
            }

            let samples_d = self
                .stream
                .clone_htod(&slice_buf)
                .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (samples slice): {e}")))?;

            // Frame starts are absolute against the FULL audio. Subtract
            // slice_start to make them relative to samples_d.
            let frame_starts_relative: Vec<i32> = frame_starts
                .iter()
                .map(|&f| f - slice_start as i32)
                .collect();
            let frame_starts_d = self
                .stream
                .clone_htod(&frame_starts_relative)
                .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (frame_starts): {e}")))?;
            tracing::info!(
                stage = "audio.gpu.h2d",
                duration_ns = t_h2d.elapsed().as_nanos() as u64,
                batch_n = batch_n,
                slice_len = slice_buf.len(),
            );

            // Pick the right buffers + cuFFT plan for THIS batch.
            let plan = self.get_or_build_plan(total_frames_batch)?;
            let (windowed_ref, complex_ref, power_ref, mel_col_ref, mel_row_ref);
            if batch_n == batch_segments {
                windowed_ref = &mut windowed_d;
                complex_ref = &mut complex_d;
                power_ref = &mut power_d;
                mel_col_ref = &mut mel_col_d;
                mel_row_ref = &mut mel_row_d;
            } else {
                // Partial-tail batch: allocate one-shot buffers sized
                // for its total_frames.
                if last_batch_buffers.is_none() {
                    last_batch_buffers = Some(LastBatchBuffers {
                        windowed: alloc_audio_buf(
                            &self.stream,
                            total_frames_batch * win.n_fft,
                            "windowed",
                        )?,
                        complex: alloc_complex_output(
                            &self.stream,
                            total_frames_batch,
                            win.n_freqs,
                        )?,
                        power: alloc_audio_buf(
                            &self.stream,
                            total_frames_batch * win.n_freqs,
                            "power",
                        )?,
                        mel_col: alloc_audio_buf(
                            &self.stream,
                            win.n_mels * total_frames_batch,
                            "mel_col",
                        )?,
                        mel_row: alloc_audio_buf(
                            &self.stream,
                            batch_n * win.n_mels * frames_per_seg,
                            "mel_row",
                        )?,
                    });
                }
                let lb = last_batch_buffers.as_mut().unwrap();
                windowed_ref = &mut lb.windowed;
                complex_ref = &mut lb.complex;
                power_ref = &mut lb.power;
                mel_col_ref = &mut lb.mel_col;
                mel_row_ref = &mut lb.mel_row;
            }

            let t_mel = Instant::now();
            run_mel_pipeline_into(
                &self.stream,
                &kernels,
                &plan,
                &self.mel_gemm,
                &samples_d,
                &frame_starts_d,
                &self.hann_d,
                &self.filterbank_d.data,
                windowed_ref,
                complex_ref,
                power_ref,
                mel_col_ref,
                mel_row_ref,
                win.n_fft,
                win.n_mels,
                win.n_freqs,
                batch_n,
                frames_per_seg,
                total_frames_batch,
                slice_audio_len,
                win.top_db,
            )?;
            // Fix D: dropped pre-ORT `synchronize_stream` — ORT CUDA EP
            // shares `self.stream` (`with_compute_stream`), so the next
            // `run_iobinding` call is stream-ordered after this batch's
            // mel kernels with no CPU-blocking sync.
            tracing::info!(
                stage = "audio.gpu.mel",
                duration_ns = t_mel.elapsed().as_nanos() as u64,
                batch_n = batch_n,
            );

            let t_ort = Instant::now();
            let logits = self.ort_session.run_iobinding(
                &self.stream,
                mel_row_ref,
                batch_n,
                win.n_mels,
                frames_per_seg,
            )?;
            tracing::info!(
                stage = "audio.gpu.ort",
                duration_ns = t_ort.elapsed().as_nanos() as u64,
                batch_n = batch_n,
            );
            let expected_logits = batch_n * self.num_classes;
            if logits.len() != expected_logits {
                return Err(SparrowEngineError::Ort(format!(
                    "Audio model returned {} logits for batch of {batch_n} x {} classes; expected exactly {expected_logits}",
                    logits.len(),
                    self.num_classes,
                )));
            }
            if !logits.iter().all(|logit| logit.is_finite()) {
                return Err(SparrowEngineError::Ort(
                    "Audio model returned non-finite logits".to_string(),
                ));
            }
            all_logits.extend(logits);
        }

        let t_post = Instant::now();
        let segments = collect_segments_for_postprocess(
            &all_logits,
            segment_offsets,
            samples.data.len(),
            win,
            &self.postprocess,
            &self.labels,
            on_segment,
        );
        tracing::info!(
            stage = "audio.gpu.post",
            duration_ns = t_post.elapsed().as_nanos() as u64,
            n_segments = n_segments,
        );
        Ok(segments)
    }

    // -----------------------------------------------------------------
    // Plumbing helpers (used by Wave 2 strategies).
    // -----------------------------------------------------------------

    /// Look up (or build) a cuFFT plan for the requested batch size.
    pub(crate) fn get_or_build_plan(&self, total_frames: usize) -> Result<Arc<BatchedR2cPlan>> {
        let mut plans = self.plans.lock().map_err(|_| {
            SparrowEngineError::Ort("AudioModel plan-cache lock poisoned".to_string())
        })?;
        if let Some(existing) = plans.get(&total_frames) {
            return Ok(existing.clone());
        }
        let plan = BatchedR2cPlan::new(
            self.stream.clone(),
            self.config.n_fft as usize,
            total_frames,
        )?;
        let plan_arc = Arc::new(plan);
        plans.insert(total_frames, plan_arc.clone());
        Ok(plan_arc)
    }

    /// Diagnostic accessor — useful for the bench harness.
    pub fn config(&self) -> &AudioPreprocessConfig {
        &self.config
    }

    /// Diagnostic accessor.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Diagnostic accessor.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Diagnostic accessor: per-segment row-major mel buffer (post
    /// power_to_db). Used by the FP32 parity test in
    /// `tests/audio_e2e_parity.rs` to compare against sparrow-engine-cpu's
    /// `mel_spectrogram`. Returns `Vec<Vec<f32>>` of length n_segments,
    /// each inner buffer of `n_mels * frames_per_seg` f32 row-major.
    ///
    /// Always uses Strategy A's whole-clip path so both branches share
    /// the same buffer layout.
    pub fn compute_mel_per_segment(
        &self,
        audio: &AudioInput,
        opts: &AudioDetectOpts,
    ) -> Result<MelDebugSnapshot> {
        let win = self.resolve_window(opts)?;
        let samples = preprocess_audio::load_audio(audio, &self.config)?;
        let total_samples = samples.data.len();
        let segment_offsets = preprocess_audio::compute_segment_offsets(
            total_samples,
            win.segment_samples,
            win.stride_samples,
        );
        let n_segments = segment_offsets.len();
        // Shared audio option validation guarantees `segment_samples >= n_fft`,
        // so the subtraction below cannot underflow.
        let frames_per_seg = ((win.segment_samples - win.n_fft) / win.hop_length) + 1;
        if n_segments == 0 {
            return Ok(MelDebugSnapshot {
                segments: Vec::new(),
                segment_offsets: Vec::new(),
                frames_per_seg,
                n_mels: win.n_mels,
                sample_rate: win.sample_rate,
            });
        }
        let total_frames = n_segments * frames_per_seg;

        // H2D + run mel pipeline.
        let samples_d = self
            .stream
            .clone_htod(&samples.data)
            .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (samples): {e}")))?;
        let frame_starts = build_frame_starts(&segment_offsets, frames_per_seg, win.hop_length)?;
        let frame_starts_d = self
            .stream
            .clone_htod(&frame_starts)
            .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (frame_starts): {e}")))?;

        let mut windowed_d = alloc_audio_buf(&self.stream, total_frames * win.n_fft, "windowed")?;
        let mut complex_d = alloc_complex_output(&self.stream, total_frames, win.n_freqs)?;
        let mut power_d = alloc_audio_buf(&self.stream, total_frames * win.n_freqs, "power")?;
        let mut mel_col_d = alloc_audio_buf(&self.stream, win.n_mels * total_frames, "mel_col")?;
        let mut mel_row_d = alloc_audio_buf(
            &self.stream,
            n_segments * win.n_mels * frames_per_seg,
            "mel_row",
        )?;

        let plan = self.get_or_build_plan(total_frames)?;
        let kernels = self.pipeline_kernels();
        run_mel_pipeline_into(
            &self.stream,
            &kernels,
            &plan,
            &self.mel_gemm,
            &samples_d,
            &frame_starts_d,
            &self.hann_d,
            &self.filterbank_d.data,
            &mut windowed_d,
            &mut complex_d,
            &mut power_d,
            &mut mel_col_d,
            &mut mel_row_d,
            win.n_fft,
            win.n_mels,
            win.n_freqs,
            n_segments,
            frames_per_seg,
            total_frames,
            samples.data.len(),
            win.top_db,
        )?;
        synchronize_stream(&self.stream)?;

        let mel_row_host: Vec<f32> = self
            .stream
            .clone_dtoh(&mel_row_d)
            .map_err(|e| SparrowEngineError::Ort(format!("clone_dtoh (mel_row): {e}")))?;

        // Slice per segment.
        let seg_size = win.n_mels * frames_per_seg;
        let mut per_segment = Vec::with_capacity(n_segments);
        for s in 0..n_segments {
            let slab = mel_row_host[s * seg_size..(s + 1) * seg_size].to_vec();
            per_segment.push(slab);
        }
        Ok(MelDebugSnapshot {
            segments: per_segment,
            segment_offsets,
            frames_per_seg,
            n_mels: win.n_mels,
            sample_rate: win.sample_rate,
        })
    }

    /// Diagnostic: run ORT IoBinding on a host-supplied mel buffer.
    /// Returns the raw logits (`batch` long). Used by the parity test
    /// to isolate "ORT response on the CPU mel" vs "ORT response on the
    /// GPU mel".
    pub fn run_ort_logits_on_host_mel(
        &self,
        mel_row_host: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>> {
        // Determine frames_per_seg from buffer length + n_mels.
        let n_mels = self.filterbank_d.n_mels;
        if mel_row_host.is_empty() || batch == 0 {
            return Ok(Vec::new());
        }
        let per_seg = mel_row_host.len() / batch;
        if per_seg * batch != mel_row_host.len() {
            return Err(SparrowEngineError::Ort(format!(
                "run_ort_logits_on_host_mel: buffer len {} is not divisible by batch {batch}",
                mel_row_host.len()
            )));
        }
        if !per_seg.is_multiple_of(n_mels) {
            return Err(SparrowEngineError::Ort(format!(
                "run_ort_logits_on_host_mel: per-seg len {per_seg} not divisible by n_mels {n_mels}"
            )));
        }
        let frames_per_seg = per_seg / n_mels;
        let mel_d = self
            .stream
            .clone_htod(mel_row_host)
            .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (host mel): {e}")))?;
        synchronize_stream(&self.stream)?;
        let logits =
            self.ort_session
                .run_iobinding(&self.stream, &mel_d, batch, n_mels, frames_per_seg)?;
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// MelDebugSnapshot — diagnostic return type for parity testing.
// ---------------------------------------------------------------------------

/// Per-segment GPU mel snapshot (post power_to_db). One inner buffer per
/// segment, sized `n_mels * frames_per_seg` and laid out row-major
/// `[n_mels, frames_per_seg]`.
#[derive(Debug, Clone)]
pub struct MelDebugSnapshot {
    pub segments: Vec<Vec<f32>>,
    pub segment_offsets: Vec<usize>,
    pub frames_per_seg: usize,
    pub n_mels: usize,
    pub sample_rate: u32,
}

/// Strategy B helper: dedicated buffers for the partial-tail batch
/// (sized for `< batch_segments * frames_per_seg`). Allocated lazily so
/// non-partial-tail clips never pay the alloc cost.
struct LastBatchBuffers {
    windowed: CudaSlice<f32>,
    complex: CudaSlice<cudarc::cufft::sys::float2>,
    power: CudaSlice<f32>,
    mel_col: CudaSlice<f32>,
    mel_row: CudaSlice<f32>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowParams {
    pub segment_samples: usize,
    pub stride_samples: usize,
    pub threshold: f32,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub n_freqs: usize,
    pub top_db: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioPostprocess {
    Detector,
    Classifier { num_classes: usize, top_k: usize },
}

impl AudioPostprocess {
    fn num_classes(self) -> usize {
        match self {
            AudioPostprocess::Detector => 1,
            AudioPostprocess::Classifier { num_classes, .. } => num_classes,
        }
    }
}

fn extract_audio_params(manifest: &ModelManifest) -> Result<(u32, f32, f32, f32)> {
    let sample_rate = match &manifest.preprocess_method {
        PreprocessMethod::MelSpectrogram { sample_rate, .. } => *sample_rate,
        PreprocessMethod::RawAudio { .. } => {
            // Phase D round 2 B-08: RawAudio routes through
            // `RawAudioModel` (parallel struct) — `AudioModel` is
            // mel-only. The engine dispatcher in
            // `engine.rs::load_from_manifest` selects the right
            // variant before reaching this helper, so a RawAudio
            // landing here is an engine-side wiring bug, not a
            // user-facing error.
            return Err(SparrowEngineError::Ort(format!(
                "internal: AudioModel::extract_audio_params received RawAudio manifest '{}'; \
                 should have been dispatched to RawAudioModel by engine.rs::load_from_manifest",
                manifest.id
            )));
        }
        other => {
            return Err(SparrowEngineError::NotAnAudioModel {
                id: manifest.id.clone(),
                method: other.as_str().to_string(),
            });
        }
    };
    let (segment_duration_s, stride_s) = match manifest.inference_strategy {
        InferenceStrategy::SlidingWindow {
            segment_duration_s,
            segment_stride_s,
        } => (segment_duration_s, segment_stride_s),
        _ => (1.0, 0.3),
    };
    let threshold = match &manifest.postprocess_method {
        PostprocessMethod::Sigmoid {
            confidence_threshold,
        } => *confidence_threshold,
        _ => manifest.confidence_threshold.unwrap_or(0.5),
    };
    Ok((sample_rate, segment_duration_s, stride_s, threshold))
}

fn resolve_audio_postprocess(
    manifest: &ModelManifest,
    manifest_dir: &Path,
) -> Result<(AudioPostprocess, Vec<String>)> {
    let labels = match (&manifest.label_file, &manifest.label_format) {
        (Some(file), Some(format)) => manifest::load_labels(&manifest_dir.join(file), format)?,
        _ => Vec::new(),
    };
    let label_count = (!labels.is_empty()).then_some(labels.len());
    let postprocess =
        resolve_audio_postprocess_from_parts(&manifest.postprocess_method, label_count)?;
    Ok((postprocess, labels))
}

pub(crate) fn resolve_audio_postprocess_from_parts(
    method: &PostprocessMethod,
    label_count: Option<usize>,
) -> Result<AudioPostprocess> {
    match method {
        PostprocessMethod::Sigmoid { .. } => Ok(AudioPostprocess::Detector),
        PostprocessMethod::Softmax => {
            let num_classes = label_count.ok_or_else(|| {
                SparrowEngineError::InvalidManifest(
                    "mel-input softmax audio classifiers require a labels file so the GPU flavor can resolve class count at load time".to_string(),
                )
            })?;
            if num_classes == 0 {
                return Err(SparrowEngineError::InvalidManifest(
                    "mel-input softmax audio classifier labels file is empty".to_string(),
                ));
            }
            Ok(AudioPostprocess::Classifier {
                num_classes,
                top_k: DEFAULT_AUDIO_CLASSIFIER_TOP_K.min(num_classes).max(1),
            })
        }
        other => Err(SparrowEngineError::InvalidManifest(format!(
            "AudioModel supports only sigmoid or softmax postprocess, got {:?}",
            other
        ))),
    }
}

/// Sigmoid activation (matches `sparrow-engine-cpu/src/detect_audio.rs::sigmoid`).
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Convert per-segment logits into above-threshold [`AudioSegment`]s,
/// invoking the optional callback for each.
///
/// Mirrors `sparrow-engine-cpu/src/detect_audio.rs:301-321` end-time computation:
/// each segment's `end_time_s = (offset + segment_samples).min(total_samples) / sr`.
pub(crate) fn collect_segments(
    logits: &[f32],
    segment_offsets: &[usize],
    total_samples: usize,
    win: &WindowParams,
    on_segment: &mut Option<&mut dyn FnMut(&AudioSegment)>,
) -> Vec<AudioSegment> {
    let mut segments: Vec<AudioSegment> = Vec::new();
    for (i, &seg_offset) in segment_offsets.iter().enumerate() {
        let logit = logits[i];
        debug_assert!(
            logit.is_finite(),
            "audio logits must be finite before collect_segments"
        );
        if !logit.is_finite() {
            continue;
        }
        let confidence = sigmoid(logit);
        if confidence >= win.threshold {
            let (start_s, end_s) = preprocess_audio::segment_time_range(
                seg_offset,
                win.segment_samples,
                total_samples,
                win.sample_rate,
            );
            let seg = AudioSegment {
                start_time_s: start_s,
                end_time_s: end_s,
                confidence,
                classes: vec![AudioClass {
                    class_idx: 0,
                    label: None,
                    probability: confidence,
                }],
            };
            if let Some(cb) = on_segment.as_deref_mut() {
                cb(&seg);
            }
            segments.push(seg);
        }
    }
    segments
}

fn collect_segments_for_postprocess(
    logits: &[f32],
    segment_offsets: &[usize],
    total_samples: usize,
    win: &WindowParams,
    postprocess: &AudioPostprocess,
    labels: &[String],
    on_segment: &mut Option<&mut dyn FnMut(&AudioSegment)>,
) -> Vec<AudioSegment> {
    match *postprocess {
        AudioPostprocess::Detector => {
            collect_segments(logits, segment_offsets, total_samples, win, on_segment)
        }
        AudioPostprocess::Classifier { num_classes, top_k } => collect_classifier_segments(
            logits,
            segment_offsets,
            total_samples,
            win,
            num_classes,
            top_k,
            labels,
            on_segment,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_classifier_segments(
    logits: &[f32],
    segment_offsets: &[usize],
    total_samples: usize,
    win: &WindowParams,
    num_classes: usize,
    top_k: usize,
    labels: &[String],
    on_segment: &mut Option<&mut dyn FnMut(&AudioSegment)>,
) -> Vec<AudioSegment> {
    let mut segments = Vec::with_capacity(segment_offsets.len());
    for (i, &seg_offset) in segment_offsets.iter().enumerate() {
        let start = i * num_classes;
        let end = start + num_classes;
        let Some(window_logits) = logits.get(start..end) else {
            break;
        };
        if !window_logits.iter().all(|logit| logit.is_finite()) {
            continue;
        }
        let probs = softmax(window_logits);
        let classes: Vec<AudioClass> = top_k_indices(&probs, top_k)
            .into_iter()
            .map(|(idx, p)| AudioClass {
                class_idx: idx as u32,
                label: labels.get(idx).cloned(),
                probability: p,
            })
            .collect();
        let confidence = classes.first().map(|c| c.probability).unwrap_or(0.0);
        let (start_time_s, end_time_s) = preprocess_audio::segment_time_range(
            seg_offset,
            win.segment_samples,
            total_samples,
            win.sample_rate,
        );
        let seg = AudioSegment {
            start_time_s,
            end_time_s,
            confidence,
            classes,
        };
        if let Some(cb) = on_segment.as_deref_mut() {
            cb(&seg);
        }
        segments.push(seg);
    }
    segments
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

fn top_k_indices(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    indexed.truncate(k);
    indexed
}

/// Helper for Wave 2 strategies: synchronize the audio model's stream.
pub(crate) fn synchronize_stream(stream: &Arc<CudaStream>) -> Result<()> {
    stream
        .synchronize()
        .map_err(|e| SparrowEngineError::Ort(format!("AudioModel stream.synchronize: {e}")))
}

/// Helper: allocate a zero-init f32 device buffer of `total_elements`,
/// formatting the failure with `label` for diagnostic context.
///
/// S6 collapse (R2 audit-fix 2026-05-05): replaces the four near-identical
/// `alloc_windowed/alloc_power/alloc_mel_col/alloc_mel_row` helpers with
/// one parameterised function. Callers compute `total_elements` from the
/// dimension product they need (`total_frames * n_fft`,
/// `total_frames * n_freqs`, `n_mels * total_frames`, `n_segments *
/// n_mels * frames_per_seg`). `alloc_complex_output` (different element
/// type — `cufft_sys::float2`) intentionally remains a separate helper.
pub(crate) fn alloc_audio_buf(
    stream: &Arc<CudaStream>,
    total_elements: usize,
    label: &str,
) -> Result<CudaSlice<f32>> {
    stream
        .alloc_zeros::<f32>(total_elements)
        .map_err(|e| SparrowEngineError::Ort(format!("AudioModel alloc_zeros ({label}): {e}")))
}

pub(crate) struct PipelineKernels<'a> {
    pub window_frame_kernel: &'a WindowFrameKernel,
    pub power_kernel: &'a PowerKernel,
    pub power_to_db_kernel: &'a PowerToDbKernel,
    pub transpose_kernel: &'a TransposeKernel,
}

/// One-shot helper: run window-frame → cuFFT → power → cuBLAS sgemm →
/// transpose → power_to_db on the input buffer, leaving the result in
/// `mel_row_d` (per-segment row-major).
///
/// Caller arrives with `windowed_d` (a `[total_frames * n_fft]` buffer
/// to be populated) and `power_d` (size `[total_frames * n_freqs]`),
/// `mel_col_d` (size `[n_mels * total_frames]`) and `mel_row_d` (size
/// `[n_segments * n_mels * frames_per_seg]`); all four are caller-
/// allocated so consumers can reuse them across batches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_mel_pipeline_into(
    stream: &Arc<CudaStream>,
    kernels: &PipelineKernels<'_>,
    plan: &BatchedR2cPlan,
    mel_gemm: &MelGemm,
    samples_d: &CudaSlice<f32>,
    frame_starts_d: &CudaSlice<i32>,
    hann_d: &CudaSlice<f32>,
    filterbank_d: &CudaSlice<f32>,
    windowed_d: &mut CudaSlice<f32>,
    complex_d: &mut CudaSlice<cudarc::cufft::sys::float2>,
    power_d: &mut CudaSlice<f32>,
    mel_col_d: &mut CudaSlice<f32>,
    mel_row_d: &mut CudaSlice<f32>,
    n_fft: usize,
    n_mels: usize,
    n_freqs: usize,
    n_segments: usize,
    frames_per_seg: usize,
    total_frames: usize,
    total_samples: usize,
    top_db: f32,
) -> Result<()> {
    // Stage 1: window-frame kernel.
    window_frame_gpu(
        stream,
        kernels.window_frame_kernel,
        samples_d,
        frame_starts_d,
        hann_d,
        windowed_d,
        n_fft,
        total_frames,
        total_samples,
    )?;
    // Stage 2: cuFFT R2C.
    plan.exec(windowed_d, complex_d)?;
    // Stage 3: power kernel.
    power_gpu(
        stream,
        kernels.power_kernel,
        complex_d,
        power_d,
        total_frames,
        n_freqs,
    )?;
    // Stage 4: cuBLAS sgemm (filterbank @ power → col-major mel).
    mel_gemm.run(filterbank_d, power_d, mel_col_d, total_frames)?;
    // Stage 5: per-segment col→row transpose.
    transpose_per_segment_gpu(
        stream,
        kernels.transpose_kernel,
        mel_col_d,
        mel_row_d,
        n_segments,
        n_mels,
        frames_per_seg,
    )?;
    // Stage 6: power_to_db (in place on mel_row_d; layout-agnostic kernel).
    power_to_db_gpu(
        stream,
        kernels.power_to_db_kernel,
        mel_row_d,
        n_segments,
        n_mels,
        frames_per_seg,
        top_db,
    )?;
    Ok(())
}

/// Build the `[total_frames]` host vector of absolute sample offsets,
/// one entry per frame (`frame_start = seg_offset + frame_idx *
/// hop_length`).
///
/// Returns `Err(SparrowEngineError::Ort(_))` if any frame start exceeds
/// `i32::MAX` — `window_frame_kernel` indexes by `i32` (matching the
/// `window_frame_gpu` precondition at `audio/window_frame.rs:100..104`)
/// so silently saturating at `i32::MAX` (the prior behavior) would
/// produce wrong frames at the upper boundary. With 48 kHz mono the
/// boundary is ~12.4 hours; well past production durations, but the
/// fail-loud `Err` matches the GPU kernel contract and the brief
/// invariant 7 ("no silent fallback").
pub(crate) fn build_frame_starts(
    segment_offsets: &[usize],
    frames_per_seg: usize,
    hop_length: usize,
) -> Result<Vec<i32>> {
    let mut starts = Vec::with_capacity(segment_offsets.len() * frames_per_seg);
    for &seg in segment_offsets {
        for f in 0..frames_per_seg {
            let abs = seg + f * hop_length;
            starts.push(i32::try_from(abs).map_err(|_| {
                SparrowEngineError::Ort(format!(
                    "frame start {abs} exceeds i32::MAX (kernel boundary, ~12.4 h @ 48 kHz)"
                ))
            })?);
        }
    }
    Ok(starts)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_offsets_match_cpu_loop() {
        // Cross-check the shared core helper against the documented termination
        // contract used by both CPU + GPU audio sliding-window code.
        let offsets = preprocess_audio::compute_segment_offsets(48000, 48000, 14400);
        assert_eq!(
            offsets.len(),
            1,
            "1 s of audio with 1 s segment + 0.3 s stride should yield 1 segment"
        );
        assert_eq!(offsets, vec![0]);

        // 2 s clip with same segment/stride: produces 5 segments at offsets
        // 0, 14400, 28800, 43200, 57600 (the last is partial-tail; CPU
        // breaks AFTER pushing it).
        // 2s @ 48k = 96000 samples.
        let offsets = preprocess_audio::compute_segment_offsets(96000, 48000, 14400);
        assert_eq!(offsets.len(), 5);
        assert_eq!(offsets, vec![0, 14400, 28800, 43200, 57600]);

        // Empty audio yields no segments.
        let offsets = preprocess_audio::compute_segment_offsets(0, 48000, 14400);
        assert!(offsets.is_empty());
    }

    #[test]
    fn build_frame_starts_layout() {
        let offsets = vec![0usize, 14400];
        let starts = build_frame_starts(&offsets, 4, 512).expect("build_frame_starts");
        // Segment 0: 0, 512, 1024, 1536. Segment 1: 14400, 14912, 15424, 15936.
        let expected = vec![0i32, 512, 1024, 1536, 14400, 14912, 15424, 15936];
        assert_eq!(starts, expected);
    }

    /// Phase 3.8 Step 2 audit-fix R2 (R1-F6 / 2026-05-05): the prior
    /// `min(i32::MAX as usize) as i32` saturation silently truncated
    /// frame starts past the kernel boundary. The fix surfaces an
    /// `Err(SparrowEngineError::Ort)` instead — locks the new contract.
    #[test]
    fn build_frame_starts_errors_past_i32_max() {
        let huge = i32::MAX as usize + 1;
        let offsets = vec![huge];
        let err = build_frame_starts(&offsets, 1, 1)
            .expect_err("build_frame_starts should refuse offsets past i32::MAX");
        let msg = format!("{err}");
        assert!(
            msg.contains("exceeds i32::MAX"),
            "unexpected error message: {msg}"
        );
    }

    /// Phase 3.8 Step 2 perf-fix (post-Wave-4 triage):
    /// Lock the per-strategy default contracts:
    ///
    /// - `default_strategy()` (non-streaming) MUST be `SingleCall` —
    ///   single-chunk ORT call, ~62 ms savings on 60 s DUNAS clips.
    /// - `default_strategy_streaming()` MUST stay `HybridA{16}` to
    ///   preserve per-batch callback cadence (Variant B / Wave 2 D2).
    ///
    /// Reverting either default would silently re-introduce the
    /// per-`Session::run` setup-overhead regression diagnosed in
    /// `docs/research/phase3.8/step2/perf_triage_report.md`.
    #[test]
    fn default_strategies_split_streaming_vs_non_streaming() {
        match GpuAudioDetectOpts::default_strategy() {
            Strategy::SingleCall => {}
            other => panic!(
                "default_strategy() must be SingleCall (non-streaming perf default), got {other:?}"
            ),
        }
        match GpuAudioDetectOpts::default_strategy_streaming() {
            Strategy::HybridA { ort_chunk_segments: 16 } => {}
            other => panic!(
                "default_strategy_streaming() must be HybridA{{16}} for per-batch cadence, got {other:?}"
            ),
        }
    }

    #[test]
    fn postprocess_mode_sigmoid_detector_uses_single_class() {
        let mode = resolve_audio_postprocess_from_parts(
            &PostprocessMethod::Sigmoid {
                confidence_threshold: 0.7,
            },
            None,
        )
        .expect("sigmoid postprocess should resolve");
        assert_eq!(mode, AudioPostprocess::Detector);
        assert_eq!(mode.num_classes(), 1);
    }

    #[test]
    fn postprocess_mode_softmax_classifier_uses_label_count_and_top_k() {
        let mode = resolve_audio_postprocess_from_parts(&PostprocessMethod::Softmax, Some(3))
            .expect("softmax postprocess should resolve with labels");
        assert_eq!(
            mode,
            AudioPostprocess::Classifier {
                num_classes: 3,
                top_k: 3,
            }
        );
        assert_eq!(mode.num_classes(), 3);
    }

    #[test]
    fn postprocess_mode_softmax_rejects_missing_labels() {
        let err = resolve_audio_postprocess_from_parts(&PostprocessMethod::Softmax, None)
            .expect_err("softmax without labels must be rejected");
        assert!(format!("{err}").contains("labels file"));
    }

    #[test]
    fn strategy_short_label_for_single_call() {
        assert_eq!(Strategy::SingleCall.short_label(), "A_single");
        assert_eq!(Strategy::SingleCall.ort_chunk(), 0);
    }
}
