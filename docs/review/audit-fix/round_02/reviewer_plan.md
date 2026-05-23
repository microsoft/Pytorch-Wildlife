# Reviewer plan — round 2

Scope: behavioral/boundary review for FFI, bindings, types, validation, and display surfaces. Sub-agent notes were written under `round_02/scratch/` and source files remain unmodified in Step 1.

## Items

<a name="ITEM-REV-R2-001"></a>
`ITEM-REV-R2-001 | sparrow-engine/sparrow-engine-types/src/model_type.rs:24 | Align public derive_model_type with the manifest-enforced audio matrix: only MelSpectrogram+Sigmoid derives AudioDetector and RawAudio+Softmax derives AudioClassifier; unsupported audio preprocess/postprocess pairs derive generic Detector/Classifier and never promote to OverheadDetector. Update model_type tests for Mel+Softmax, RawAudio+Sigmoid, and audio+Overhead fallback semantics. | Round 1 left derive_model_type lenient even though it is pub and crate-root re-exported; direct callers can still observe AudioDetector/AudioClassifier for combinations manifest loading rejects.`

Notes / risk:
- `sparrow-engine/sparrow-engine-cpu/src/engine.rs` has an unowned internal unit test that still asserts MelSpectrogram+Softmax => AudioClassifier. If this item is approved, that cross-scope test will need round-3 ownership or an explicit ownership expansion before full-suite validation.

<a name="ITEM-REV-R2-002"></a>
`ITEM-REV-R2-002 | sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py:14 | Re-export AudioClass from the native module and include "AudioClass" in __all__. | Python users receive AudioClass instances through AudioSegment.classes but cannot name sparrow_engine.AudioClass for isinstance checks or annotations.`

Acceptance details:
- `AudioClass` is imported alongside `AudioSegment` and `AudioResult`.
- `"AudioClass" in sparrow_engine.__all__` holds in an import smoke check.

<a name="ITEM-REV-R2-003"></a>
`ITEM-REV-R2-003 | sparrow-engine/sparrow-engine-server/src/handlers/audio.rs:103 | Compute store=true drift labels from each segment's top-1 class label when present, falling back to model_id when classes are absent or the top-1 label is None. Add handler-local unit coverage for multiclass top-1, empty classes, and unlabeled top-1 fallbacks. | Current drift labels use model_id for every segment, losing Perch 2 class identity in stored inference logs and drift metrics.`

Acceptance details:
- Do not substitute lower-ranked class labels when the top-1 label is None.
- Preserve model_id fallback for binary detector responses where response.classes is omitted.

<a name="ITEM-REV-R2-004"></a>
`ITEM-REV-R2-004 | sparrow-engine/sparrow-engine-cpu/src/ffi.rs:343; sparrow-engine/sparrow-engine-gpu/src/ffi.rs:352 | Apply the audio FFI null-on-empty pointer convention to non-audio FFI arrays: detection data, classify top_results, and pipeline data should be null when their corresponding len is 0 in both CPU and GPU crates. Add unit tests for empty detection/classification/pipeline conversions. | Round 1 fixed dangling Vec::as_ptr() sentinels for audio V1/V2 empty results, but analogous non-audio FFI arrays still expose non-null dangling pointers when len=0.`

## No planned source edits

- No changes planned for `sparrow-engine/sparrow-engine-cli/src/main.rs`; sub-agent review found no regression in the round-1 visualize threshold policy.
- No changes planned for CPU/GPU `classify.rs`, `detect.rs`, or CPU `preprocess.rs`; no new behavioral issue was found in these owned files.
- No changes planned for `sparrow-engine/sparrow-engine-types/src/manifest.rs` or `types.rs`; manifest enforcement and shared audio type shape already match the round-1 fix surface.
- No changes planned for `sparrow-engine/sparrow-engine-python/src/lib.rs` or `sparrow-engine-server/src/response.rs`; the native AudioClass class and response serialization were already covered in round 1.

## Cross-Scope Findings — Deferred to Round 3

- `sparrow-engine/sparrow-engine-cpu/src/engine.rs`: internal `model_type_from_preprocess_postprocess` test still asserts MelSpectrogram+Softmax derives AudioClassifier. This conflicts with ITEM-REV-R2-001's stricter public helper semantics and is outside reviewer ownership for source edits.
- `sparrow-engine/sparrow-engine-server/src/discover.rs`: audio-classifier fixture comments still describe MelSpectrogram+Softmax as the AudioClassifier path, but manifest validation now rejects that combination; this is outside reviewer ownership and appears test-comment/fixture-shape related rather than a round-2 source edit.

STATUS: PLAN-READY
