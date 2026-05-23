# Auditor plan — round 2

Scope: structural review of the 7 owned audio-pipeline backbone files
post-round-1. Findings restricted to simplification, dead code, dedup
(only when 3+ occurrences), naming/readability. Behavioral issues are
deferred to round-3 cross-scope.

Round 1 outcome recap (drives this round): all 5 round-1 items
(ITEM-AUD-001..005) were APPROVED + applied; none REJECTED → no
`-V2` re-proposals. Most structural smells in the owned set were
harvested already. Round-2 surface is therefore small.

---

<a name="ITEM-AUD-R2-001"></a>
ITEM-AUD-R2-001 | sparrow-engine/sparrow-engine-core/src/viz.rs:1057, 1083, 1102, 1115, 1135, 1150, 1168 | Re-indent the six `classes: Vec::new(),` lines inside the viz.rs tests so they align with their sibling struct-literal fields. Currently sit at column 8 while sibling fields (`start_time_s`/`end_time_s`/`confidence`) are at column 12, visually breaking the struct literal exactly like AUD-003 caught in `sparrow-engine-gpu/src/detect_audio.rs:222,228,234` in round 1 (col-12 vs col-16). Pure rustfmt-equivalent readability fix; the well-formatted `make_seg_full` helper at viz.rs:1326-1333 demonstrates the intended indent. | 6 occurrences of the same bug pattern that round-1 AUD-003 fixed elsewhere — same root cause (line was inserted by a tooling pass that didn't re-format the parent block). Affected sites: `heatmap_inverted_alpha_panics_in_debug` (1057), `heatmap_*_release` siblings (1083, 1102), `audio_segments_to_annotations_normalizes_time` (1115), `heatmap_segment_beyond_duration_no_panic` (1135), `heatmap_invalid_pow_panics_in_debug` (1150), `heatmap_invalid_pow_returns_input_in_release` (1168). No behaviour change; struct literal is semantically identical regardless of leading whitespace. The sibling test file `audio_heatmap_e2e.rs:69-71` uses a one-line literal style — no indent issue there.

---

Files NOT modified (reviewed, no in-scope structural finding):

- `sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs` —
  helpers from round 1 (`compute_segment_offsets`, `segment_time_range`)
  in place at lines 311 / 334, doc comments present, no new
  3-occurrence-rule dedup surfaces. A double-blank-line gap between
  `segment_time_range`'s closing brace (line 344) and the next
  `MelFilterbank` block header (line 347) is cosmetic-only;
  `cargo fmt` does not normalize that gap. Skip.
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs` — post-round-1
  state clean. `softmax` / `top_k_indices` doc tripwire still planted
  at line 675-685 referencing the cross-scope `try_softmax` reuse
  blocker. The `outputs.len() == 0` candidate at line 434 was reviewed
  again — `ort::SessionOutputs` does not expose `.is_empty()` directly
  (verified: clippy run was clean), so the `len() == 0` idiom stays.
  No new 3+ dedup opportunities. The 3 `pin_session()` + lock + run
  patterns at lines 271 / 427 / 583 each have different output-shape
  extraction logic; not a clean dedup target.
- `sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs` — round-1
  AUD-003 indentation fix applied; otherwise this file is the
  `AudioModel::detect`/`detect_streaming` thin dispatch + the
  `merge_segments` mirror discussed in cross-scope below. No
  in-scope structural finding within the 3+ rule.
- `sparrow-engine/sparrow-engine-gpu/src/models/audio.rs` — round-1
  helper centralization applied (5 sites now call
  `preprocess_audio::compute_segment_offsets`; 1 site calls
  `segment_time_range`). `extract_audio_params` at line 1312-1345 is
  a single-call helper, not a duplication target. The
  `collect_segments` function at line 1357 contains a defensive
  `debug_assert!(finite)` followed by `if !finite { continue; }` —
  intentional belt-and-suspenders (debug panics on misuse, release
  silently skips); not dead code, not refactor-worthy.
- `sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs` — only
  Perch 2 touch points are the two audio-rejection arms at lines 467
  & 780 (defense-in-depth load-vs-classify pair). No structural
  cleanup needed; same disposition as round 1.
- `sparrow-engine/sparrow-engine-core/tests/audio_heatmap_e2e.rs` —
  three test fixtures use `classes: Vec::new()` on one line (lines
  69-71); indentation is correct, no churn.

---

Cross-scope findings (deferred to round 3 — outside owned-file set or
out of structural scope):

1. **CPU↔GPU `merge_segments` / `merge_segments_with_class`
   verbatim copy.** Sites: `sparrow-engine-cpu/src/detect_audio.rs:751-793`
   and `sparrow-engine-gpu/src/detect_audio.rs:128-165`. Round 1
   already flagged this; GPU file's own doc literally says
   "Verbatim from sparrow-engine-cpu (no GPU dependency)" — the same
   self-confessed drift admission pattern that round-1 AUD-001
   caught for `compute_segment_offsets`. Still only 2 callers, below
   the 3+ dedup threshold this round. Natural home would be
   `sparrow-engine-types` (alongside `AudioRange`) or
   `sparrow-engine-core` (alongside `preprocess_audio`), but BOTH
   target crates are outside auditor's owned-file set this round
   (sparrow-engine-types is reviewer-owned; sparrow-engine-core is
   owned for `preprocess_audio.rs`/`viz.rs`/tests only — adding a new
   `postprocess_audio.rs` module is an architectural step beyond
   structural simplification). Defer to round 3; if a third caller
   appears in the interim (e.g. a future `python` flavor of the
   audio pipeline) the 3+ rule trips and the lift becomes
   unconditional.
2. **`postprocess::try_softmax` still cannot be reused from the
   raw-audio path** — carried forward from round 1 cross-scope #2.
   `postprocess.rs` is outside the audit-fix owned-file set;
   tripwire doc remains in CPU detect_audio.rs:675-685.
3. **`resolve_classifier_output` fallback behavior vs docstring**
   (CPU detect_audio.rs:267). Docstring says "single-output
   classifiers fall back to output 0", but `unwrap_or(0)` actually
   fires for ANY multi-head classifier missing a `"label"`-named
   output. Behavioural edge — out of structural scope.
4. **Cross-scope reviewer findings carried from round 1 that touch
   detect_audio.rs** (behavioural, deferred to round 3 unless
   coordinated reviewer/auditor cross-scope work is opened):
   - RawAudio `segment_duration_s` opt override is ignored
     (window_samples wins).
   - RawAudio top-K batching validates only flattened output
     length; should require rank-2 `[batch, num_classes]` shape.

STATUS: PLAN-READY
