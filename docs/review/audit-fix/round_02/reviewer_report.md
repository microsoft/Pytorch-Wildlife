# Reviewer report — round 2

Commit: `20aa643db16bf628ea5ddd4a808196437d389a72`

## Changes Applied

<a name="ITEM-REV-R2-001"></a>
- ITEM-REV-R2-001 | `sparrow-engine/sparrow-engine-types/src/model_type.rs`; `sparrow-engine/sparrow-engine-cpu/src/engine.rs` | Tightened `derive_model_type` so only MelSpectrogram+Sigmoid derives `AudioDetector` and RawAudio+Softmax derives `AudioClassifier`; unsupported audio preprocess fallbacks now derive generic `Detector`/`Classifier` and do not promote to `OverheadDetector`. Per MODIFY approval, appended `sparrow-engine/sparrow-engine-cpu/src/engine.rs` to the ledger and updated its dependent in-crate test.

<a name="ITEM-REV-R2-002"></a>
- ITEM-REV-R2-002 | `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py` | Re-exported `AudioClass` from the native module and added it to `__all__` so users can reference `sparrow_engine.AudioClass` for annotations/isinstance checks.

<a name="ITEM-REV-R2-003"></a>
- ITEM-REV-R2-003 | `sparrow-engine/sparrow-engine-server/src/handlers/audio.rs` | Store=true drift labels now use `classes[0].label` by index-0 lookup, with model-id fallback when classes are omitted/empty or top-1 label is `None`. Added tests covering no re-sort, empty/missing classes, unlabeled top-1, and empty-string label preservation.

<a name="ITEM-REV-R2-004"></a>
- ITEM-REV-R2-004 | `sparrow-engine/sparrow-engine-cpu/src/ffi.rs`; `sparrow-engine/sparrow-engine-gpu/src/ffi.rs` | Detection, classification top-results, and pipeline FFI arrays now return null data pointers when len=0 in both CPU and GPU crates. Added unit coverage for each empty-array conversion in both crates.

## Tests Added/Updated

- Updated `model_type.rs` tests for strict audio matrix and no overhead promotion on unsupported audio fallbacks.
- Updated CPU engine direct-helper test for MelSpectrogram+Softmax => `Classifier`.
- Added CPU/GPU FFI empty non-audio array tests.
- Added server audio drift-label helper tests.

Validation run:
- `./scripts/test.sh -p sparrow-engine-types phase_a_r1_model_type_tests -- --test-threads=1` — PASS.
- `./scripts/test.sh -p sparrow-engine-cpu model_type_from_preprocess_postprocess -- --test-threads=1` — PASS.
- `./scripts/test.sh -p sparrow-engine-cpu --features ffi ffi::tests -- --test-threads=1` — PASS.
- `./scripts/test.sh -p sparrow-engine-gpu --features ffi ffi::tests -- --test-threads=1` — PASS.
- `./scripts/test.sh -p sparrow-engine-server handlers::audio::tests -- --test-threads=1` — PASS.
- Python `AudioClass` re-export smoke script — PASS.
- `git diff --check` — PASS.
- `rustfmt --edition 2021 --check sparrow-engine/sparrow-engine-types/src/model_type.rs sparrow-engine/sparrow-engine-cpu/src/ffi.rs sparrow-engine/sparrow-engine-gpu/src/ffi.rs` — PASS.
- `cargo fmt --check` — FAILS on unrelated workspace formatting diffs; not used as the acceptance gate for this reviewer patch.

## Cross-Scope Findings

- `sparrow-engine/sparrow-engine-server/src/discover.rs`: audio-classifier fixture comments still describe MelSpectrogram+Softmax as the AudioClassifier path even though manifest validation now rejects that combination. Source behavior did not block the targeted server tests; comment/fixture cleanup remains outside reviewer ownership.

## Skipped

- No changes made for REJECTED items; none were rejected.
- No source changes made for `sparrow-engine/sparrow-engine-cli/src/main.rs`, CPU/GPU image `classify.rs`/`detect.rs`, CPU `preprocess.rs`, `manifest.rs`, `types.rs`, Python `src/lib.rs`, or server `response.rs` because no approved round-2 item targeted them.

STATUS: DONE COMMIT=20aa643db16bf628ea5ddd4a808196437d389a72
