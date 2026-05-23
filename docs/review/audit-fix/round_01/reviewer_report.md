# Reviewer Report — Round 1

Commit: `53b65bf43fbbfcc157d59552e98b390e1ae3d1dc`

## Changes Applied

<a name="ITEM-REV-001"></a>
- ITEM-REV-001 | sparrow-engine/sparrow-engine-types/src/manifest.rs:741 | RawAudio window/sample consistency previously ran before inference parsing and skipped when segment_duration_s was absent → audio manifests now must use sliding_window, timing fields must be finite and positive, and RawAudio window_samples is checked against the validated sliding-window duration. Prevents single/tiled RawAudio manifests and non-finite strides from reaching runtime.

<a name="ITEM-REV-002"></a>
- ITEM-REV-002 | sparrow-engine/sparrow-engine-types/src/manifest.rs:889 | Unsupported audio preprocess/postprocess combinations previously parsed successfully → manifest load now accepts only MelSpectrogram+Sigmoid and RawAudio+Softmax. model_type.rs direct legacy mappings were kept to avoid breaking existing direct-helper callers; the manifest parser is now the enforcement boundary.

<a name="ITEM-REV-003"></a>
- ITEM-REV-003 | sparrow-engine/sparrow-engine-cpu/src/ffi.rs:553; sparrow-engine/sparrow-engine-gpu/src/ffi.rs:562 | Empty V1/V2 audio results previously exposed Vec::as_ptr() dangling sentinels when len=0 → CPU and GPU V1/V2 builders now return data=null for empty segment arrays while preserving stable pointers for non-empty arrays.

<a name="ITEM-REV-004"></a>
- ITEM-REV-004 | sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs:28 | FFI symbol smoke test previously sampled only legacy symbols and nm check allowed extras → test now pins detect_audio_v2/audio_result_v2_free and requires actual exported sparrow_engine_* symbols to equal exports.def with the 34-symbol count.

<a name="ITEM-REV-005"></a>
- ITEM-REV-005 | sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs:232 | Perch 2 integration previously exercised only Rust API output → added ignored ffi-feature V2 ABI test that loads Perch 2, calls sparrow_engine_detect_audio_v2, checks two segments, classes_len=5, top-1 confidence parity, labels, and frees the V2 result.

<a name="ITEM-REV-006"></a>
- ITEM-REV-006 | sparrow-engine/sparrow-engine-cpu/tests/integration_reexports.rs:90 | Re-export test previously omitted AudioClass → now constructs sparrow_engine::AudioClass at crate root.

<a name="ITEM-REV-007"></a>
- ITEM-REV-007 | sparrow-engine/sparrow-engine-python/src/lib.rs:1639 | PyO3 conversion tests previously covered only one labeled class → added multi-class order, None-label, empty-classes, and confidence/top-1 parity coverage.

<a name="ITEM-REV-008"></a>
- ITEM-REV-008 | sparrow-engine/sparrow-engine-server/src/response.rs:265 | Server response tests previously covered only labeled multi-class entries → added mixed labeled/unlabeled multi-class serialization coverage.

<a name="ITEM-REV-009"></a>
- ITEM-REV-009 | sparrow-engine/sparrow-engine-cli/src/main.rs:1513 | --visualize post-filter previously applied a 0.5 fallback to thresholdless softmax audio output → helper now post-filters only when a manifest threshold exists, so Perch 2 visualization does not change printed output cardinality.

## Tests Added/Updated

- Added/updated manifest parser tests for RawAudio sliding_window, window_samples tolerance/mismatch, non-finite timing, and unsupported audio combinations.
- Added CPU/GPU FFI unit tests for V1/V2 empty segment arrays, V2 zero-class segments, and nul-byte label sanitization.
- Added FFI symbol equality and Perch 2 V2 ABI integration coverage.
- Added Rust re-export, PyO3 conversion, server response, and CLI threshold-policy tests.

Validation run after edits:
- `git diff --check` — PASS.
- `./scripts/test.sh -p sparrow-engine-types manifest::tests -- --test-threads=1` — PASS (66 passed).
- `./scripts/test.sh -p sparrow-engine-types phase_a_r1_model_type_tests -- --test-threads=1` — PASS (11 passed).
- `./scripts/test.sh -p sparrow-engine-cpu --features ffi ffi::tests -- --test-threads=1` — PASS (3 passed).
- `./scripts/test.sh -p sparrow-engine-cpu --features ffi --test integration_ffi_symbols -- --test-threads=1` — PASS (2 passed).
- `./scripts/test.sh -p sparrow-engine-cpu --features ffi --test integration_reexports -- --test-threads=1` — PASS (5 passed).
- `./scripts/test.sh -p sparrow-engine-cpu --features ffi --test integration_perch2 -- --test-threads=1` — PASS (1 passed, 2 ignored).
- `./scripts/test.sh -p sparrow-engine-cli audio -- --test-threads=1` — PASS (13 passed, 2 ignored).
- `./scripts/test.sh -p sparrow-engine-server response::tests -- --test-threads=1` — PASS (4 passed).
- `./scripts/test.sh -p sparrow-engine-python --lib --no-default-features --features cpu convert_audio_segment -- --test-threads=1` — PASS (3 passed).
- `./scripts/test.sh -p sparrow-engine-gpu --features ffi ffi::tests -- --test-threads=1` — PASS (3 passed).
- `./scripts/test.sh -p sparrow-engine-cpu model_type_from_preprocess_postprocess -- --test-threads=1` — PASS (1 passed).

Note: an initial Python test command without `--features cpu` failed because the package requires exactly one of cpu/gpu when default features are disabled; reran with `--features cpu` and passed.

## Cross-Scope Findings

- sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs: RawAudio still resolves AudioDetectOpts.segment_duration_s but uses manifest window_samples as the actual segment length; round 2 should reject incompatible overrides or define RawAudio override semantics.
- sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs: RawAudio top-K batching still validates only flattened output length; round 2 should require rank-2 [batch_len, num_classes] before slicing logits.
- sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py: AudioClass is not re-exported at the Python top level; file was appended to the ledger for round 2.
- sparrow-engine/sparrow-engine-server/src/handlers/audio.rs: store=true drift labels still use model_id instead of Perch 2 top-1 classes; file was appended to the ledger for round 2.
- sparrow-engine-gpu RawAudio/Perch 2 remains unsupported even though V2 FFI exists; docs/model-selection surfaces should avoid implying GPU Perch 2 support until implemented.

## Skipped

- No source edits made outside reviewer-owned files. Accidental cargo fmt changes to unowned files were reverted before committing.
- REJECTED items: none.

STATUS: DONE COMMIT=53b65bf43fbbfcc157d59552e98b390e1ae3d1dc
