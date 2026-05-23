# Round 1 file ownership

## Auditor owns (structural — audio pipeline backbone)
- sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs
- sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs
- sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs
- sparrow-engine/sparrow-engine-core/src/viz.rs
- sparrow-engine/sparrow-engine-core/tests/audio_heatmap_e2e.rs
- sparrow-engine/sparrow-engine-gpu/src/models/audio.rs
- sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs

## Reviewer owns (behavioral — FFI/bindings/types/validation/display)
- sparrow-engine/sparrow-engine-cli/src/main.rs
- sparrow-engine/sparrow-engine-cpu/src/classify.rs
- sparrow-engine/sparrow-engine-cpu/src/detect.rs
- sparrow-engine/sparrow-engine-cpu/src/ffi.rs
- sparrow-engine/sparrow-engine-cpu/src/preprocess.rs
- sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs
- sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs
- sparrow-engine/sparrow-engine-cpu/tests/integration_reexports.rs
- sparrow-engine/sparrow-engine-gpu/src/classify.rs
- sparrow-engine/sparrow-engine-gpu/src/detect.rs
- sparrow-engine/sparrow-engine-gpu/src/ffi.rs
- sparrow-engine/sparrow-engine-python/src/lib.rs
- sparrow-engine/sparrow-engine-server/src/response.rs
- sparrow-engine/sparrow-engine-types/src/manifest.rs
- sparrow-engine/sparrow-engine-types/src/model_type.rs
- sparrow-engine/sparrow-engine-types/src/types.rs

## Rationale
- Auditor takes the audio-pipeline backbone files (large diffs, new pipeline code) — structural concerns dominate
- Reviewer takes 16 files; uses task() delegation (max 5 sub-agents) since general-purpose has task tool
- Reviewer takes FFI surfaces (V2 ABI = security-/correctness-critical), parsers/validation (manifest+model_type+types), Python/server/CLI bindings (consumer-facing surface)
