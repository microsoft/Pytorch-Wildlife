# Inquisitor review — round 1

## Approval decisions recap

Phase 1: 13 APPROVED + 1 MODIFY out of 14 plan items.
- MODIFY: ITEM-REV-003 — plan's anchors (CPU ffi.rs:528 / GPU ffi.rs:537) pointed at V1 builders; the dangling-`Vec::as_ptr()`-on-empty bug actually existed in BOTH V1 and V2 builders symmetrically. I required the fix to land on all 4 sites (CPU V1+V2, GPU V1+V2). The reviewer accepted and applied accordingly.

Phase 2 verification confirms: every applied edit lines up against the cited plan items; one minor deviation on ITEM-REV-002 (model_type.rs left lenient; manifest parser is the sole enforcement gate). All other items match the plan. Ledger grew from 23 → 25 entries via the anti-narrowing APPEND clause (Python `__init__.py`, server `handlers/audio.rs`); both new entries are correctly stamped `added_round: 1` and are deferred to round 2 for source edits — but they ARE in this round's verification scope.

## Fix verification (per ledger file)
<a name="sparrow-engine-sparrow-engine-cli-src-main-rs"></a>
### sparrow-engine/sparrow-engine-cli/src/main.rs

VERIFIED — ITEM-REV-009 applied at main.rs:1513. New helper `audio_visualize_output_filter_threshold(cli_threshold, manifest_threshold) -> Option<f32>` collapses correctly: returns None when manifest has no threshold (thresholdless softmax/Perch 2 → no post-filter), otherwise CLI override > manifest. Call site at :1561-1569 gates BOTH the inference-time threshold lowering (was unconditional `Some(0.0)`) AND the post-filter branch on `output_filter_threshold.is_some()`. This correctly removes the CLI-only 0.5 surprise for Perch 2 while preserving the diagnostic-viz dance for MD_AudioBirds_V1.

<a name="sparrow-engine-sparrow-engine-core-src-preprocess-audio-rs"></a>
### sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs

VERIFIED — ITEM-AUD-001 + ITEM-AUD-002 applied at preprocess_audio.rs:311 (compute_segment_offsets) and :334 (segment_time_range). Both `pub fn`, both with explicit doc comments documenting the inclusive-tail termination contract and the `min(total_samples)` clamp. Body of compute_segment_offsets is byte-identical to the three deleted hand-rolled copies. segment_time_range preserves the clamp. No new tests added — pre-existing GPU `segment_offsets_match_cpu_loop` now exercises the lifted helper (verified at gpu/src/models/audio.rs:1559-1576).

<a name="sparrow-engine-sparrow-engine-core-src-viz-rs"></a>
### sparrow-engine/sparrow-engine-core/src/viz.rs

VERIFIED (no change) — Not touched in the diff (git stat shows 0 LOC). Auditor plan documented intentional skip: the new `AudioSegment.classes` field is intentionally unused by the heatmap renderer (reads only `confidence`). No new cross-impact surfaced from the round-1 edits — `AudioSegment` struct shape is unchanged, only behaviour/policy around it. OK.

<a name="sparrow-engine-sparrow-engine-core-tests-audio-heatmap-e2e-rs"></a>
### sparrow-engine/sparrow-engine-core/tests/audio_heatmap_e2e.rs

VERIFIED (no change) — Not touched. Auditor plan documented intentional skip: `classes: Vec::new()` fixture pattern doesn't merit a builder. Round-1 changes to AudioSegment surface preserved field shape so the fixture compiles unchanged. OK.

<a name="sparrow-engine-sparrow-engine-cpu-src-classify-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/classify.rs

VERIFIED (no change) — Not touched. Reviewer plan documented intentional skip: dispatch guard already rejects audio preprocess methods. No cross-impact from FFI/manifest changes that would require touching the image classifier path. OK.

<a name="sparrow-engine-sparrow-engine-cpu-src-detect-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/detect.rs

VERIFIED (no change) — Not touched. Reviewer plan documented intentional skip; image detector unaffected by audio-only edits. OK.

<a name="sparrow-engine-sparrow-engine-cpu-src-detect-audio-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs

VERIFIED — ITEMs AUD-001/002/004/005 applied. Confirmed: line 373 + line 548 call the new `preprocess_audio::compute_segment_offsets`; lines 479-489 + 642-652 use the new `preprocess_audio::segment_time_range`; line 217-220 uses idiomatic `RawAudio { window_samples, .. }` (the `sample_rate: _` dead-bind is gone); lines 675-685 carry the 11-line doc block explaining why the local softmax/top_k_indices duplicate `try_softmax` (refs cross-scope finding #2). No behaviour-affecting edits beyond the dedup — the loop bodies are intact in their new home.

<a name="sparrow-engine-sparrow-engine-cpu-src-ffi-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/ffi.rs

VERIFIED — ITEM-REV-003 applied to BOTH V1 (lines 555-560) and V2 (lines 646-650) builders per the Phase-1 MODIFY. Both now: `header.data = if segments.is_empty() { ptr::null() } else { segments.as_ptr() }`. Matches the nested `classes` arena's existing null-on-empty pattern. Symmetric with GPU. No regressions in the surrounding ownership semantics (combined Box + AudioResultWithOwner / AudioResultV2WithOwner intact).

<a name="sparrow-engine-sparrow-engine-cpu-src-preprocess-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/preprocess.rs

VERIFIED (no change) — Not touched. Reviewer plan documented intentional skip. No FFI/manifest cross-impact requiring preprocess.rs edits. OK.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-ffi-symbols-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs

VERIFIED — ITEM-REV-004 applied. Link-smoke test at line 33-47 now pins 7 symbols including `sparrow_engine_detect_audio_v2` + `sparrow_engine_audio_result_v2_free`. nm test at lines 135-151 now asserts BOTH directions: missing (`expected.difference(&actual)`) AND extra (`actual.difference(&expected)`). 34-count assertion preserved at :154-159, plus `assert_eq!(actual.len(), expected.len())` at :160 as a tripwire against accidental duplicate-name parsing. Set type switched HashSet→BTreeSet for stable diff output. Correct.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-perch2-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs

VERIFIED — ITEM-REV-005 applied. New ignored test `perch2_detect_audio_v2_preserves_top5_classes_over_ffi` at line 232. Gated `#[ignore]` + `#[cfg(feature = "ffi")]`. Uses the same bundle-resolution path. Goes through `sparrow_engine_engine_new` → `sparrow_engine_load_model` → `sparrow_engine_detect_audio_v2` → ... → `sparrow_engine_audio_result_v2_free` → `unload_model` → `engine_free`. Asserts len=2, sample_rate=32000, classes_len=5 per segment, top-1 label non-null + UTF-8, and `confidence ≈ classes[0].probability` (parity with Rust path) using `< f32::EPSILON` tolerance — correct since both values originate from the same softmax slice in the same f32 evaluation.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-reexports-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_reexports.rs

VERIFIED — ITEM-REV-006 applied at line 90. `let _ac: sparrow_engine::AudioClass = sparrow_engine::AudioClass { class_idx: 0, label: None, probability: 0.0 };` exercises the crate-root re-export with `label: None` to cover the optional-label edge. Compile-time reachability proof — same shape as the other reexport assertions in this file. OK.

<a name="sparrow-engine-sparrow-engine-gpu-src-classify-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/classify.rs

VERIFIED (no change) — Not touched. OK.

<a name="sparrow-engine-sparrow-engine-gpu-src-detect-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/detect.rs

VERIFIED (no change) — Not touched. OK.

<a name="sparrow-engine-sparrow-engine-gpu-src-detect-audio-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs

VERIFIED — ITEM-AUD-003 applied at lines 222/228/234: `classes: Vec::new(),` re-indented from column 12 to column 16, now aligned with sibling fields. Pure formatting. OK.

<a name="sparrow-engine-sparrow-engine-gpu-src-ffi-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/ffi.rs

VERIFIED — ITEM-REV-003 applied symmetric to CPU. V1 builder at lines 564-569 and V2 builder at lines 655-659 both gate `header.data` on `segments.is_empty()`. Matches CPU exactly. OK.

<a name="sparrow-engine-sparrow-engine-gpu-src-models-audio-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/models/audio.rs

VERIFIED — ITEMs AUD-001 + AUD-002 applied. The previous private `compute_segment_offsets` (with its self-confessed `Mirror of sparrow-engine-cpu` doc) is GONE from the source. Five call sites (lines 569, 1144, plus three test sites 1559/1571/1576) now use `preprocess_audio::compute_segment_offsets`. Line 1376 uses `preprocess_audio::segment_time_range`. Drift admission removed; single home for the windowing termination invariant.

<a name="sparrow-engine-sparrow-engine-gpu-src-models-classifier-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs

VERIFIED (no change) — Not touched. Auditor plan documented intentional skip (two defense-in-depth audio-rejection arms are correct as-is). OK.

<a name="sparrow-engine-sparrow-engine-python-src-lib-rs"></a>
### sparrow-engine/sparrow-engine-python/src/lib.rs

VERIFIED — ITEM-REV-007 applied. Three convert_audio_segment tests now exist: original single-class @ 1617, multi-class order + None labels @ 1639 (preserves top-1 ordering, mixes Some/None labels, asserts confidence == classes[0].probability), and empty-classes @ 1677. Pure test addition. Coverage matches plan acceptance.

<a name="sparrow-engine-sparrow-engine-server-src-response-rs"></a>
### sparrow-engine/sparrow-engine-server/src/response.rs

VERIFIED — ITEM-REV-008 applied. New `audio_class_opt(class_idx, Option<&str>, probability)` helper at line 214 makes labels optional; the original `audio_class(...)` now wraps it. A new multi-class test (lines 240+) mixes labeled/unlabeled entries and asserts class_idx + probability remain serialized when label is None (note: `label` field is `Option<String>` with `#[serde(skip_serializing_if = "Option::is_none")]` semantics per response.rs:148, so the JSON shape correctly omits the `label` key for None entries while keeping `class_idx` and `probability`). OK.

<a name="sparrow-engine-sparrow-engine-types-src-manifest-rs"></a>
### sparrow-engine/sparrow-engine-types/src/manifest.rs

VERIFIED — ITEMs REV-001 + REV-002 applied. (1) RawAudio numeric validation (line 649-665) now only checks sample_rate>0 and window_samples>0 (the segment_duration_s consistency check correctly moved out). (2) Sliding-window parsing (line 767-789) adds `is_finite()` to both timing fields. (3) New is_audio→SlidingWindow gate at lines 800-804. (4) The RawAudio×SlidingWindow consistency check now runs AFTER strategy parsing at lines 806-824, so it can no longer be silently skipped. (5) New audio combination matrix at lines 899-910 rejects everything except (Mel, Sigmoid) and (RawAudio, Softmax). New tests at 2031 (sliding_window required), 2040 (mismatch), 2049 (±1 tolerance), 2063 (non-finite rejection), 2084 (unsupported combos). Reviewer self-reports 66 manifest tests pass.

<a name="sparrow-engine-sparrow-engine-types-src-model-type-rs"></a>
### sparrow-engine/sparrow-engine-types/src/model_type.rs

VERIFIED with DEVIATION-NOTED — Reviewer report (and my read of the diff) confirms the function body of `derive_model_type` is unchanged: `(RawAudio, Sigmoid) → AudioDetector` and `(MelSpectrogram, Softmax) → AudioClassifier` still hold. Only doc-message + rustfmt churn (~20 LOC, all whitespace/format/string-literal). The plan's acceptance bullet (`Update model_type tests so unsupported combos are not advertised as AudioDetector/AudioClassifier by the public helper`) was NOT executed; reviewer explicitly justified the deviation: enforce only at the manifest parser. WARNING: `derive_model_type` is `pub` (visibility was deliberately widened — see header comment) and IS re-exported at the crate root (verified earlier via `sparrow_engine::derive_model_type` in integration_reexports.rs:154). External direct callers (Python/server/CLI consumers bypassing ModelManifest::load) would still see the misleading AudioDetector for (RawAudio, Sigmoid). Not blocking — manifest is the live runtime gate — but should be tightened or re-documented in round 2. The new test assertion message at line 286 honestly says: "derives AudioDetector for legacy direct callers but is rejected by manifest validation" — acceptable as a self-documenting tripwire.

<a name="sparrow-engine-sparrow-engine-types-src-types-rs"></a>
### sparrow-engine/sparrow-engine-types/src/types.rs

VERIFIED (no change) — Not touched. Reviewer plan documented intentional skip (no-label binary-detector classes mismatch is emitted upstream in detect_audio.rs, not in types.rs itself). No struct-shape change required this round. OK.

<a name="sparrow-engine-sparrow-engine-python-python-sparrow-engine-init-py"></a>
### sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py

VERIFIED — APPENDED THIS ROUND (added_round=1 via reviewer-flagged cross-impact). Not modified by either editor this round. Inspection confirms the issue: line 15-16 imports `AudioResult, AudioSegment` from the native module but NOT `AudioClass`; `__all__` at line 35-60 lists `AudioSegment` + `AudioResult` (line 59-60) but no `AudioClass`. Python users receive AudioClass instances at runtime (via convert_audio_class in lib.rs:452-458) but cannot reference the symbol by name for isinstance/annotations. Real issue; intentionally deferred — must be fixed in round 2.

<a name="sparrow-engine-sparrow-engine-server-src-handlers-audio-rs"></a>
### sparrow-engine/sparrow-engine-server/src/handlers/audio.rs

VERIFIED — APPENDED THIS ROUND (added_round=1 via reviewer-flagged cross-impact). Not modified by either editor this round. Inspection confirms the issue at line 109: `let labels: Vec<String> = vec![model_id.clone(); response.segments.len()];` — drift labels use `model_id` for every segment, ignoring the per-segment top-1 class label now available in `AudioSegment.classes[0].label`. Pre-Perch-2 this was correct (single-class detector); post-Perch-2 it loses class identity for store=true drift metrics on multi-class softmax models. Real issue; intentionally deferred — must be fixed in round 2.


## Missed issues

None new this round at the source level. Three observations carried forward:

1. **model_type.rs deviation** (WARNING, not blocking). Plan ITEM-REV-002 asked to narrow `derive_model_type`'s audio arms + tests. Reviewer chose to enforce only at the manifest parser and left `derive_model_type` lenient. The function is `pub` + crate-root re-exported, so external callers bypassing `ModelManifest::load` still see misleading `AudioDetector` for `(RawAudio, Sigmoid)` and `AudioClassifier` for `(MelSpectrogram, Softmax)`. The new assertion message at model_type.rs:286 honestly documents this. Round 2 should either (a) narrow the arms to match the manifest's accepted matrix, or (b) split into `derive_model_type_lenient` and `derive_model_type_strict` and route public callers through the strict variant.

2. **Two appended ledger files have known issues, fix deferred** (`__init__.py` AudioClass re-export gap; `handlers/audio.rs` drift-label uses model_id). Both correctly entered the ledger via the anti-narrowing APPEND clause with `added_round: 1` and are flagged in the reviewer report's cross-scope section. They MUST be addressed by round 2.

3. **`cargo test` was not run end-to-end on the auditor side** due to a host-local glibc 2.38 / ort_sys symbol mismatch (`__isoc23_strtoll`). Reviewer separately ran package-scoped tests through `scripts/test.sh` and all passed (per verification.txt: 399 total). Compatibility note documented; not a regression.

## Cross-impact

- **REV-001/REV-002 (manifest tightening) ↔ in-tree golden manifests.** I confirmed only via inspection (not by running `cargo test --package sparrow-engine-types`) that in-tree audio manifests already declare `strategy = "sliding_window"`; the reviewer's own test run reports 66 manifest tests pass, which implicitly covers the existing `models/audiobirds.toml` parse path. No regression observed.
- **AUD-001/AUD-002 (helper lift to core) ↔ GPU `models/audio.rs`.** Previously-private `compute_segment_offsets` in GPU is fully removed; five call sites (including three test sites) now consume the core helper. Termination + clamp semantics are byte-preserved. No behavioural drift expected; the pre-existing GPU `segment_offsets_match_cpu_loop` test still serves as a tripwire.
- **REV-003 (V1+V2 null-on-empty) ↔ V2 FFI Perch 2 test (REV-005).** REV-005 asserts `!header.data.is_null()` for the populated 2-segment case (line 287) — consistent with the new null-on-empty contract (data is null only when len=0). Symmetric.
- **REV-009 (CLI threshold helper) ↔ runtime Perch 2 path in detect_audio.rs.** The CLI helper correctly observes that Perch 2 (Softmax) has no manifest threshold → `audio_confidence_threshold()` returns None → helper returns None → no inference-time lowering, no post-filter. Self-consistent with `prepare_audio_detection` line 174 which sets the Softmax-arm default threshold to 0.0.

## Verification results

- **Both editor commits cleanly land.** Auditor: `91bf51e`. Reviewer: `53b65bf`. Diff stat across the 14 modified files matches the report claims (sparrow-engine-types/manifest.rs +245 LOC, sparrow-engine-cpu/ffi.rs +171 LOC, etc.).
- **No regression evidence.** `verification.txt` reports 399 cargo tests passed across the cpu workspace; clippy `-D warnings` clean across cpu/gpu/python/server/cli.
- **Smoke-test golden symlink note** in verification.txt is environmental (fixtures live in the sibling `sparrow-engine-dev` repo) — not a code regression.
- **Anti-narrowing posture preserved.** Ledger grew by APPEND only (23 → 25), zero entries removed.

## Coverage analysis

- Ledger files: 25
- Verified this round: 25
- Cumulative verified (scope_check.sh): 25/25
- Uncovered: []

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=25/25 UNCOVERED=[] NEW=14
