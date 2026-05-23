# Reviewer Plan — Round 1

Scope: behavioral/boundary review for Perch 2 RawAudio + audio V2 top-K surfaces. Source edits are planned only for reviewer-owned files; cross-scope findings are deferred to round 2.

## Planned Fixes

<a name="ITEM-REV-001"></a>
`ITEM-REV-001 | sparrow-engine/sparrow-engine-types/src/manifest.rs:644 | Move RawAudio window/duration validation after inference parsing, require audio manifests to use sliding_window, reject non-finite segment_duration_s/segment_stride_s, and always compare window_samples against the validated sliding-window duration. | RawAudio manifests can currently omit sliding-window fields via strategy=single/tiled and bypass the window_samples consistency check; +inf timing fields also pass manifest parsing and reach runtime.`

Acceptance details:
- Add manifest tests for valid RawAudio sliding_window, missing/non-sliding strategy rejection, window_samples mismatch, ±1 sample tolerance, and non-finite duration/stride rejection.
- Keep existing mel audio manifests valid only when they use explicit sliding_window parameters.

<a name="ITEM-REV-002"></a>
`ITEM-REV-002 | sparrow-engine/sparrow-engine-types/src/manifest.rs:824; sparrow-engine/sparrow-engine-types/src/model_type.rs:24 | Enforce the implemented audio preprocess/postprocess matrix at manifest load and narrow derive_model_type audio arms/tests to implemented combinations: MelSpectrogram+Sigmoid and RawAudio+Softmax. | RawAudio+Sigmoid, MelSpectrogram+Softmax, and audio+vision postprocessors are accepted/advertised even though runtime dispatch either rejects or processes them with the wrong audio path.`

Acceptance details:
- Reject unsupported audio preprocess/postprocess combinations with `InvalidManifest` before model-type derivation is used.
- Update model_type tests so unsupported combos are not advertised as AudioDetector/AudioClassifier by the public helper.

<a name="ITEM-REV-003"></a>
`ITEM-REV-003 | sparrow-engine/sparrow-engine-cpu/src/ffi.rs:528; sparrow-engine/sparrow-engine-gpu/src/ffi.rs:537 | Normalize CPU/GPU audio FFI top-level result pointers so V1 and V2 return data = null when len = 0, and add unit coverage for empty results, zero-class segments, and nul-byte label sanitization. | The V2 builders currently replace the initialized null pointer with Vec::as_ptr() even for empty vectors, exposing a dangling-but-len-0 sentinel while nested empty class arrays use null.`

Acceptance details:
- Apply the same null-on-empty convention in both CPU and GPU audio result builders.
- Preserve existing free semantics and non-empty pointer ownership.

<a name="ITEM-REV-004"></a>
`ITEM-REV-004 | sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs:27 | Add the V2 audio functions to the compile-time FFI link-smoke set and make the Linux nm check fail on extra exported sparrow_engine_* symbols as well as missing ones. | The current symbol test samples five legacy functions and only checks exports.def ⊆ actual, so accidental V2 removal from the Rust FFI surface or accidental extra exports can pass.`

Acceptance details:
- Pin `sparrow_engine_detect_audio_v2` and `sparrow_engine_audio_result_v2_free` in the link-smoke test.
- Assert actual exported `sparrow_engine_*` names equal exports.def and retain the 34-symbol count.

<a name="ITEM-REV-005"></a>
`ITEM-REV-005 | sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs:96 | Add an ignored ffi-feature Perch 2 integration path that calls sparrow_engine_detect_audio_v2, validates two 5s segments with classes_len = 5/top-1 confidence parity, then frees via sparrow_engine_audio_result_v2_free. | The current Perch 2 integration test proves the Rust API emits top-K classes but does not exercise the public V2 C ABI layout, labels, ownership, or free path.`

Acceptance details:
- Skip gracefully when the Perch 2 bundle is absent, matching the existing ignored test.
- Gate FFI-specific assertions behind `#[cfg(feature = "ffi")]`.

<a name="ITEM-REV-006"></a>
`ITEM-REV-006 | sparrow-engine/sparrow-engine-cpu/tests/integration_reexports.rs:79 | Add root re-export coverage for sparrow_engine::AudioClass alongside AudioSegment/AudioDetectResult. | AudioClass is the new public type behind AudioSegment.classes, but the crate-root re-export test does not currently prove downstream Rust consumers can name it.`

<a name="ITEM-REV-007"></a>
`ITEM-REV-007 | sparrow-engine/sparrow-engine-python/src/lib.rs:1608 | Extend PyO3 conversion tests for multi-class order/probabilities, None labels, empty classes, and confidence == classes[0].probability when non-empty. | The binding code appears correct, but the current unit test covers only a single labeled AudioClass and misses Perch 2 top-K and legacy empty-class edge cases.`

<a name="ITEM-REV-008"></a>
`ITEM-REV-008 | sparrow-engine/sparrow-engine-server/src/response.rs:236 | Add server response serialization coverage for multi-class AudioSegment.classes entries whose labels are None. | response.rs already preserves optional classes additively, but tests only cover Some(label); class_idx/probability must remain serialized when labels are absent.`

<a name="ITEM-REV-009"></a>
`ITEM-REV-009 | sparrow-engine/sparrow-engine-cli/src/main.rs:1558 | Change detect-audio --visualize post-filtering so thresholdless RawAudio/Softmax classifier output is not filtered by the CLI-only 0.5 fallback or by a CLI threshold that the non-visualized RawAudio runtime ignores. | Visualization currently lowers inference threshold to 0 and then post-filters printed JSON/CSV/merged output; for Perch 2 this can drop low-probability windows only when --visualize is present, changing machine-readable results as a side effect of visualization.`

Acceptance details:
- Keep the existing post-filter behavior for sigmoid audio detectors with a manifest confidence threshold.
- Add a CLI unit regression around the helper/policy so visualization does not change RawAudio/Softmax output cardinality.

## Cross-Scope Findings — Deferred to Round 2

- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs`: RawAudio resolves `AudioDetectOpts.segment_duration_s` but then hard-codes `segment_samples = window_samples`; either reject incompatible overrides or define RawAudio-specific override behavior.
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs`: RawAudio top-K batching validates only flattened output length; require rank-2 `[batch_len, num_classes]` before slicing logits by window.
- `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py`: top-level package re-exports `AudioSegment`/`AudioResult` but not `AudioClass`, so Python users receive AudioClass instances without a `sparrow_engine.AudioClass` name for annotations/isinstance checks. Appended this file to the ledger for round 2.
- `sparrow-engine/sparrow-engine-server/src/handlers/audio.rs`: `store=true` drift labels use `model_id` for every audio segment, ignoring Perch 2 top-1 classes now present in the response. Appended this file to the ledger for round 2.
- `sparrow-engine-gpu/src/detect_audio.rs` / `sparrow-engine-gpu/src/models/audio.rs`: GPU exports the V2 FFI symbol but RawAudio/Perch 2 remains explicitly unsupported; docs/model-selection surfaces should avoid implying GPU Perch 2 support until implemented.

## Skipped / No Planned Fix

- No source changes planned for CPU/GPU `classify.rs`, `detect.rs`, or CPU `preprocess.rs`; reviewed dispatch guards already reject audio preprocess methods from image APIs.
- No behavioral source change planned for `types.rs`; the observed no-label binary-detector `classes` mismatch is produced in `detect_audio.rs`, which is outside this reviewer ownership for source edits.

STATUS: PLAN-READY
