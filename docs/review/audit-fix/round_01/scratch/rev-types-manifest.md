# Reviewer scratch: types/manifest/model_type

## Findings

### REV-TYPES-001 — Audio preprocess/postprocess compatibility is not enforced, and advertised model types exceed implemented dispatch
- **File:line:** `sparrow-engine/sparrow-engine-types/src/manifest.rs:508-521`, `manifest.rs:824-874`, `sparrow-engine/sparrow-engine-types/src/model_type.rs:24-38`; cross-check `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:638-662`.
- **Observed evidence:** `raw_audio` parsing only requires `sample_rate` and `window_samples`; postprocessing then accepts any method. `derive_model_type` maps `RawAudio + Sigmoid` to `AudioDetector` and `RawAudio + Softmax` to `AudioClassifier`. The CPU raw-audio inference loop always applies softmax/top-K and builds classifier-style `AudioClass` entries, regardless of manifest postprocess.
- **Behavior risk:** A `raw_audio + sigmoid` or `raw_audio + vision postprocess` manifest can load or be advertised with a plausible `ModelType`, but runtime dispatch treats raw audio as a softmax classifier. The same validation gap exists for `MelSpectrogram + Softmax`: `model_type.rs` advertises `AudioClassifier`, while the mel loop is sigmoid-detector shaped.
- **Proposed fix:** Add a manifest compatibility matrix after postprocess parsing: current implemented behavior should allow `MelSpectrogram + Sigmoid` and `RawAudio + Softmax`; reject audio preprocess plus vision postprocessors, and reject `RawAudio + Sigmoid` / `MelSpectrogram + Softmax` unless those paths are actually implemented. Update `derive_model_type` tests to match the accepted matrix.
- **Rationale:** Invalid manifests should fail at load/parse time instead of producing misleading model-type/default-model dispatch and incorrect postprocessing.
- **Confidence:** HIGH.

### REV-TYPES-002 — `raw_audio` manifests can bypass sliding-window duration/stride requirements
- **File:line:** `sparrow-engine/sparrow-engine-types/src/manifest.rs:644-673`, `manifest.rs:755-803`; cross-check `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:801-819` and `detect_audio.rs:217-222`.
- **Observed evidence:** The `window_samples == segment_duration_s * sample_rate` check runs only when `raw.inference.segment_duration_s` is present. The parser still accepts `strategy = "single"` and `strategy = "tiled"`, where `segment_duration_s` / `segment_stride_s` are not required. CPU raw-audio runtime then falls back to default `(1.0, 0.3)` window params for non-sliding strategies, but uses `window_samples` as the actual segment size.
- **Behavior risk:** A malformed Perch/raw-audio manifest can omit the sliding-window fields and skip the consistency check, then run with an implicit stride unrelated to the declared raw window.
- **Proposed fix:** Require `InferenceStrategy::SlidingWindow` for all audio preprocess methods, or at minimum for `RawAudio`. Move the raw-audio `window_samples` consistency check after inference strategy parsing and run it against the validated `SlidingWindow.segment_duration_s`.
- **Rationale:** Raw-audio windows are fixed-size model inputs; duration/stride must be explicit and internally consistent.
- **Confidence:** HIGH.

### REV-TYPES-003 — Raw-audio stride validation misses `+inf`
- **File:line:** `sparrow-engine/sparrow-engine-types/src/manifest.rs:783-791`; cross-check `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:221-227` and `sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs:251-259`.
- **Observed evidence:** Manifest sliding-window validation checks `segment_duration_s <= 0.0` and `segment_stride_s <= 0.0`, but not `is_finite()`. The mel path later calls `validate_audio_window_params`, which rejects non-finite values. The raw path computes `stride_samples = (segment_stride_s * sample_rate).round() as usize` and only rejects `0`; `+inf` saturates instead of being rejected.
- **Behavior risk:** A TOML `segment_stride_s = inf` can reach raw-audio runtime and produce nonsensical offset stepping instead of a manifest error.
- **Proposed fix:** Add finite checks in manifest sliding-window parsing, or route raw-audio stride validation through the same finite/sample-count validation helper used by the mel path.
- **Rationale:** Parser behavior should be consistent across audio preprocess methods and reject non-finite timing fields before runtime.
- **Confidence:** MEDIUM-HIGH.

### REV-TYPES-004 — `AudioSegment.classes` empty-class semantics conflict with current binary detector construction
- **File:line:** `sparrow-engine/sparrow-engine-types/src/types.rs:172-187`; cross-check `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:490-498`.
- **Observed evidence:** `types.rs` says binary detectors use a one-entry `classes` vec or empty when no labels file is present. The mel/sigmoid CPU path always emits one `AudioClass { class_idx: 0, label: detector_label.clone(), probability: confidence }`; when there is no labels file, `detector_label` is `None`, but `classes` is still non-empty.
- **Behavior risk:** Consumers using FFI/Python/server surfaces can observe `classes_len == 1` with a null/None label where the shared type contract says empty means legacy no-label binary detector.
- **Proposed fix:** Either change the mel detector construction to emit an empty class vec when `detector_label.is_none()`, or revise the shared type contract and add tests documenting that no-label binary detectors still emit class 0 with `label = None`.
- **Rationale:** The public type invariant should match actual output semantics; otherwise clients cannot reliably interpret empty vs no-label classes.
- **Confidence:** HIGH.

## Test Gaps

- **RawAudio manifest matrix:** Add `sparrow-engine-types` manifest tests for a valid Perch-style `raw_audio + sliding_window + softmax` manifest, missing `sample_rate`, missing/zero `window_samples`, `window_samples` mismatch, ±1 sample tolerance, non-sliding strategy rejection, and unsupported postprocess rejection. Current manifest audio tests cover mel-only fixtures (`manifest.rs:1797-2012`).
- **Model type vs runtime dispatch:** Add table tests asserting only implemented audio preprocess/postprocess combinations derive audio model types; invalid combinations should be rejected by manifest validation before `derive_model_type` is used.
- **Timing finite checks:** Add manifest tests for `segment_stride_s = inf` and `segment_duration_s = inf` on raw audio. Mel runtime validation has coverage in `sparrow-engine-core/src/preprocess_audio.rs:1014-1021`; raw-audio manifest/runtime validation does not.
- **AudioSegment invariants:** Add constructor/output tests that verify `classes` is sorted descending, `confidence == classes[0].probability` when classes is non-empty, and the chosen no-label binary detector semantics.

## Cross-Scope Findings

- **Raw-audio runtime ignores `AudioDetectOpts.segment_duration_s`:** `types.rs:219-225` documents segment-duration override, but CPU raw audio uses manifest `window_samples` as `segment_samples` (`detect_audio.rs:217-222`) after resolving opts (`detect_audio.rs:164-165`). If raw-audio input size must remain fixed, reject or warn on incompatible `opts.segment_duration_s`; otherwise compute `segment_samples` from the override and validate it against the model input shape.
- **GPU raw-audio is explicitly unsupported:** `sparrow-engine-gpu/src/detect_audio.rs:40-50` rejects `RawAudio`. If Perch 2 is CPU-only for now, ensure docs/default-model resolution do not imply GPU `AudioClassifier` support for raw-audio manifests.

STATUS: DONE
