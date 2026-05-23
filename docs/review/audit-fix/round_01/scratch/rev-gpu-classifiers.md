# Reviewer scratch — GPU/classifier/audio paths

## Findings

No behavioral findings in the owned files:

- `sparrow-engine-gpu/src/classify.rs:23-39` and `sparrow-engine-cpu/src/classify.rs:23-39` reject both `MelSpectrogram` and `RawAudio` before image preprocessing or softmax dispatch.
- `sparrow-engine-gpu/src/detect.rs:32-49` and `sparrow-engine-cpu/src/detect.rs:27-44` reject both audio preprocess methods before vision detection dispatch.
- `sparrow-engine-cpu/src/preprocess.rs:73-99` rejects audio preprocess methods if they reach image `preprocess()`.
- `sparrow-engine-gpu/src/ffi.rs:581-653` preserves V2 top-K class arrays with stable owned arenas; `sparrow_engine_detect_audio_v2` routes through `detect_audio` and converts the same `AudioDetectResult` as CPU.

## Test Gaps

- `sparrow-engine-{cpu,gpu}/src/classify.rs` and `detect.rs`: add RawAudio/Perch-style manifest validation tests, not only MelSpectrogram/Yolo fixtures, so the image APIs keep rejecting audio before preprocessing.
- `sparrow-engine-gpu/src/ffi.rs:581-653`: existing unit coverage checks V2 conversion. Add an ABI parity/symbol test that includes `sparrow_engine_detect_audio_v2` and `sparrow_engine_audio_result_v2_free` on both CPU and GPU cdylibs, because the file-level comment still states the historical symbol-count gate.

## Cross-Scope Findings

### CS-1 — RawAudio segment-duration override is silently ignored

- Evidence: `sparrow-engine-cpu/src/detect_audio.rs:164-165` resolves `segment_duration_s` from `AudioDetectOpts`, but the RawAudio branch then hard-codes `segment_samples = window_samples` at `sparrow-engine-cpu/src/detect_audio.rs:217-223`; only `segment_stride_s` is used. Public opts document segment duration as overrideable at `sparrow-engine-types/src/types.rs:221-227`.
- Proposed fix: for `PreprocessMethod::RawAudio`, reject `opts.segment_duration_s` unless it equals `window_samples / sample_rate` within rounding tolerance, or remove/disable that override for RawAudio callers.
- Rationale: Perch 2 has a fixed graph input window. Silently ignoring a caller-supplied duration makes requested 2s/10s windows still run as the manifest's fixed window, which is observable in `start_time_s/end_time_s` and top-K output cadence.
- Confidence: HIGH — verified in source paths above.

### CS-2 — RawAudio top-K postprocess only checks flattened element count

- Evidence: `sparrow-engine-cpu/src/detect_audio.rs:608-624` accepts any logits output shape whose flattened length equals `batch_len * num_classes`, then slices by `i * num_classes..(i + 1) * num_classes` at `sparrow-engine-cpu/src/detect_audio.rs:640-643`. The setup probe validates only the single-window shape `[1, num_classes]` at `sparrow-engine-cpu/src/detect_audio.rs:296-306`.
- Proposed fix: in the batched RawAudio loop, require `output_view.ndim() == 2`, `shape[0] == batch_len`, and `shape[1] == num_classes` before flattening.
- Rationale: A rank-3 or transposed output with the same element count would be grouped into wrong per-window class slices, producing incorrect top-K classes without an error.
- Confidence: MEDIUM-HIGH — the current check is source-verified; exploitability depends on an ONNX output shape with matching flattened length.

### CS-3 — GPU V2 FFI exists, but GPU RawAudio/Perch 2 path is explicitly unavailable

- Evidence: `sparrow-engine-gpu/src/ffi.rs:1339-1364` exports `sparrow_engine_detect_audio_v2`, but it calls `crate::detect_audio::detect_audio`; GPU audio validation rejects `PreprocessMethod::RawAudio` at `sparrow-engine-gpu/src/detect_audio.rs:40-50`, and GPU model loading also rejects RawAudio at `sparrow-engine-gpu/src/models/audio.rs:1334-1345`.
- Proposed fix: if GPU Perch 2 support is required, implement a RawAudio GPU path or route Perch 2 to CPU before GPU model load. If CPU-only is intentional, update FFI/API docs and installer/model-selection surfaces to state V2 top-K Perch output is CPU-only.
- Rationale: The V2 ABI is parity-safe, but GPU flavor cannot currently produce Perch 2 top-K output despite carrying the V2 entry point.
- Confidence: HIGH — verified in GPU FFI, validation, and model-load code.

STATUS: DONE
