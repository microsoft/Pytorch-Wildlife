# model_type.rs review notes — round 2 Step 1

Status: completed read-only review.

## Required context
- Read `~/.copilot/skills/_shared/iterative-anti-drift.md`: convergence is ledger + coverage anchored; editor/reviewer plan items should append audited/fixed coverage entries and hard convergence requires a later clean round.
- Read `docs/review/audit-fix/SCOPE_LEDGER.json`: `manifest.rs`, `model_type.rs`, `types.rs`, and re-export tests are in scope; reviewer owns the target files listed in round 2 ownership.
- Read `docs/review/audit-fix/COVERAGE_LOG.jsonl`: round 1 fixed ITEM-REV-002 in `manifest.rs`/`model_type.rs`; inquisitor verified the public helper deviation.
- Read `docs/review/audit-fix/round_01/inquisitor_review.md`: previous finding says `derive_model_type` is public + crate-root re-exported and still lenient for unsupported audio pairs.
- Read `docs/review/audit-fix/round_02/file_ownership.md`: reviewer owns `sparrow-engine-types/src/{manifest.rs,model_type.rs,types.rs}` and `sparrow-engine-cpu/tests/integration_reexports.rs`.

## Evidence
- `sparrow-engine-types/src/model_type.rs` currently maps all four audio preprocess/postprocess pairs to audio model types:
  - Mel + Sigmoid -> `AudioDetector`
  - Mel + Softmax -> `AudioClassifier`
  - RawAudio + Sigmoid -> `AudioDetector`
  - RawAudio + Softmax -> `AudioClassifier`
- `sparrow-engine-types/src/manifest.rs` accepts only two implemented audio pairs after round 1:
  - Mel + Sigmoid
  - RawAudio + Softmax
  Unsupported audio pairs return `InvalidManifest("unsupported audio preprocess/postprocess combination ...")`.
- `sparrow-engine-types/src/lib.rs` re-exports `derive_model_type` at crate root; `sparrow-engine-cpu/tests/integration_reexports.rs` proves `sparrow_engine::derive_model_type` is externally reachable.
- Runtime direct callers derive model type after `load_manifest`, so manifest validation protects loaded in-tree manifests. The mismatch remains only for direct public helper callers and tests/comments that bypass a manifest load.
- Stale direct-caller test/comment found outside reviewer ownership: `sparrow-engine-server/src/discover.rs` has `write_audio_classifier_manifest` using Mel + Softmax and comments that this is how `derive_model_type` returns `AudioClassifier`; this is already inconsistent with manifest validation.

## Decision
Recommend plan item (a): narrow public `derive_model_type` to the implemented audio matrix. Do not leave unchanged; the helper is public and currently advertises rejected combinations. Do not split strict/lenient helpers for this round; that adds API surface and still leaves the misleading public helper unless the old helper is renamed/deprecated, which is a larger API change.

## Exact proposed fix
1. In `sparrow-engine-types/src/model_type.rs`, change `derive_model_type` so only implemented audio pairs return audio model types:
   - `(MelSpectrogram, Sigmoid)` -> `AudioDetector`
   - `(RawAudio, Softmax)` -> `AudioClassifier`
   - Unsupported audio pairs fall back to generic `Classifier` for Softmax and generic `Detector` otherwise.
   - Suppress `OverheadDetector` promotion for any audio preprocessing method, including unsupported audio pairs, so the existing doc claim remains true: overhead only affects vision detectors.
2. Update model_type unit tests:
   - Mel + Softmax should no longer expect `AudioClassifier`; expect generic `Classifier` for both subtypes.
   - RawAudio + Sigmoid should no longer expect `AudioDetector`; expect generic `Detector` and assert overhead does not promote.
   - Mel + Yolo/other unsupported audio preprocess pairs should not promote to `OverheadDetector`.
   - Keep RawAudio + Softmax -> `AudioClassifier` and Mel + Sigmoid -> `AudioDetector`.
3. Update direct-caller tests/comments as ownership permits:
   - `sparrow-engine-cpu/src/engine.rs` unit test `model_type_from_preprocess_postprocess`: replace the Mel + Softmax `AudioClassifier` assertion with RawAudio + Softmax.
   - `sparrow-engine-server/src/discover.rs` test helper `write_audio_classifier_manifest`: rewrite fixture to RawAudio + Softmax (with `labels`) and update comment. This file is outside the provided reviewer-owned list, so it needs lead ownership expansion/reassignment before edit.
4. No `types.rs` changes needed.
5. No `manifest.rs` behavior change needed; it already enforces the accepted matrix. Optional: update/add a manifest test comment to cross-reference `derive_model_type` only if touching nearby tests.

## Tests to run after the reviewer applies the fix
- From `sparrow-engine/`: `cargo test -p sparrow-engine-types model_type --quiet`
- From `sparrow-engine/`: `cargo test -p sparrow-engine-types test_audio_rejects_unsupported_preprocess_postprocess_pairs --quiet`
- From `sparrow-engine/`: `cargo test -p sparrow-engine-cpu model_type_from_preprocess_postprocess --quiet`
- From `sparrow-engine/`: `cargo test -p sparrow-engine-cpu --test integration_reexports derive_model_type_reachable_at_crate_root --quiet`
- If ownership expands to `server/src/discover.rs`: `cargo test -p sparrow-engine-server discover_catalog_excludes_pipeline_with_audio_classifier_after_image_classifier --quiet`
- Then run the audit-fix round's normal verification command/script if defined by the reviewer lead.

## Risks
- Behavior change for unsupported direct helper inputs. This is intentional but public: external direct callers may see generic `Detector`/`Classifier` instead of `AudioDetector`/`AudioClassifier` for manifest-rejected pairs.
- If only `model_type.rs` is changed, existing direct-caller tests outside the owned list can fail (`cpu/src/engine.rs`; likely `server/src/discover.rs` fixture/comment). Coordinate ownership before applying.
- Because `derive_model_type` returns `ModelType` rather than `Result`, it still cannot report unsupported combinations. Narrowing prevents false audio advertising but does not turn the helper into validation; docs should state `ModelManifest::load` is the validation gate.
