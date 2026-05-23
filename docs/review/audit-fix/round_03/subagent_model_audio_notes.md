# Reviewer R3 Model/Audio Notes

## Scope Checked

Focused review of the R2 model/audio fixes and immediate edge cases in:

- `sparrow-engine/sparrow-engine-types/src/model_type.rs`
- `sparrow-engine/sparrow-engine-cpu/src/engine.rs`
- `sparrow-engine/sparrow-engine-types/src/manifest.rs`
- `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py`
- `sparrow-engine/sparrow-engine-python/src/lib.rs`
- `sparrow-engine/sparrow-engine-server/src/handlers/audio.rs`
- `sparrow-engine/sparrow-engine-server/src/response.rs`

Direct-import/context spot checks only:

- `sparrow-engine/sparrow-engine-types/src/types.rs` for `AudioSegment` / `AudioClass` semantics.
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs` and `sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs` to verify what the audio producer emits for binary and RawAudio+Softmax paths.

No source edits made. No tests/builds run because this is a read-only audit pass and the instruction allowed writes only under this round-03 review directory.

## Findings

### ITEM-REV-R3-001 — Stored audio drift labels are computed after lossy response conversion

`audio_detect` converts the native `AudioDetectResult` into `AudioDetectResponse` first, then computes store/drift labels from `response.segments` (`sparrow-engine-server/src/handlers/audio.rs:101-121`). `AudioSegmentResponse::from` intentionally drops `classes` whenever the native segment has 0 or 1 classes (`sparrow-engine-server/src/response.rs:161-176`). The drift helper then falls back to `model_id` when `classes` is missing (`sparrow-engine-server/src/handlers/audio.rs:28-36`).

That means the R2 “drift labels from segment top-1” fix only works for segments with 2+ serialized classes. Any K=1 labeled audio segment is stored under the model id instead of the native top-1 label. This can happen for labeled binary detectors, one-class audio classifiers, or any future caller/model path that emits exactly one top-class entry. The direct producer confirms the binary mel path emits exactly one `AudioClass` when a label exists (`sparrow-engine-cpu/src/detect_audio.rs:344-346`, `485-493`), and `types.rs` documents K=1 for binary detectors (`sparrow-engine-types/src/types.rs:156-179`).

Impact: stored `InferenceLogRecord` drift class distributions and PSI inputs are wrong for the K=1 labeled-audio edge case. The public JSON omission can remain for backwards compatibility; the store/drift path should not depend on that lossy representation.

### ITEM-REV-R3-002 — RawAudio+Softmax load-time shape validation checks the wrong output head

`Engine::load_model` calls `validate_output_shape(&session, &manifest)` at load time (`sparrow-engine-cpu/src/engine.rs:293-295`), but `validate_output_shape` always validates `outputs[0]` (`sparrow-engine-cpu/src/engine.rs:901-916`). For `PostprocessMethod::Softmax`, that only verifies the first output has rank 1 or 2 (`sparrow-engine-cpu/src/engine.rs:999-1013`).

RawAudio+Softmax audio classifiers such as Perch 2 are multi-head models; the integration test states the intended logits head is named `label`, not necessarily output 0 (`sparrow-engine-cpu/tests/integration_perch2.rs:3-12`). The runtime audio path later resolves the `label` output by name and probes it (`sparrow-engine-cpu/src/detect_audio.rs:256-320`). Therefore load-time validation can pass by validating an unrelated first output (for example an embedding head), while the real classifier logits head is malformed or absent. With labels present, inference may fail later via the probe/label-count checks; with no labels, a multi-output model lacking a `label` head can fall back to output 0 and classify an embedding tensor.

Impact: RawAudio+Softmax manifests are not fully validated at load time, and a malformed multi-output audio classifier can fail only on first inference or silently use the wrong tensor in the no-label fallback case.

## Cross-Scope Findings

- CPU/GPU parity should be checked if ITEM-REV-R3-002 is fixed. `sparrow-engine-gpu/src/engine.rs` is outside this focused file list, but RawAudio+Softmax validation should remain flavor-consistent.
- The producer-side evidence for ITEM-REV-R3-001 lives in `sparrow-engine-cpu/src/detect_audio.rs`, outside the focused seven-file list. The owned-file bug is still in `audio.rs`/`response.rs`: storage derives labels from a lossy server response instead of the native segment data.

## Suggested Reviewer Plan Items

- `ITEM-REV-R3-001`: In `sparrow-engine-server/src/handlers/audio.rs`, compute drift labels from the native `result.segments` before converting to `AudioDetectResponse`, or keep a parallel native label vector before consuming `result`. Add a unit test covering a single labeled `AudioClass` so store labels use the top-1 label while the JSON response may still omit `classes`.
- `ITEM-REV-R3-002`: In `sparrow-engine-cpu/src/engine.rs`, make output validation model/preprocess-aware for `RawAudio + Softmax`: validate the named `label` output when multiple outputs are present; allow output 0 only for single-output classifiers; reject multi-output RawAudio+Softmax models without a `label` head. Add helper-level tests for single-output softmax, multi-output with `label`, and multi-output without `label`. Mirror/audit GPU validation separately.

STATUS: FINDINGS
