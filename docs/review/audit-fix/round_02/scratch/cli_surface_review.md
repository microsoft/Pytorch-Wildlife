# CLI surface review notes

Round 2 Step 1 read-only reviewer sub-agent.

Status: initialized; required reads and owned-file inspection pending.

## Context reads
- Initialized notes and read required audit-fix context files.
- Round 1 carried forward: Python `AudioClass` export gap, server audio drift labels use `model_id`, model_type lenient direct helper warning, CLI threshold fix verified in R1.

## Reviewer-owned surface inspection

### CLI threshold visualization path
- `sparrow-engine-cli/src/main.rs:1513-1518` keeps the R1 helper: returns `Some(cli_or_manifest)` only when manifest has a threshold; returns `None` for thresholdless softmax models.
- `main.rs:1561-1569` only lowers inference to `Some(0.0)` when `--visualize` and that output filter exists. Thresholdless Perch 2 no longer gets a synthetic output filter.
- `main.rs:1608-1628` only post-filters `output_view` when `output_filter_threshold` exists. No current regression found for default Perch 2 visualization.

### CLI class display/serialization
- Raw JSON includes `classes` only when `s.classes.len() > 1` (`main.rs:1748-1761`); tests cover empty/single/multi-class (`main.rs:3103-3144`).
- Default merged JSON/CSV uses top-1 label via `merge_segments_with_class` (`main.rs:1796-1798`), so Perch 2 class identity survives merged output when labels exist.
- Raw CSV intentionally preserves old 6-column schema and does not expose class labels/probabilities (`main.rs:1768-1781`, test `main.rs:3256-3281`). Observation only: this is a class-display gap for `--raw-segments --format csv` on multi-class audio, but the test names it as intentional old-schema compatibility.

### No-change image paths
- CPU classify/detect and GPU classify/detect still reject audio manifests via `PreprocessMethod::MelSpectrogram | RawAudio` guards before image inference.
- CPU preprocess still rejects audio preprocessing methods in image `preprocess()`.
- No behavioral regression found in these owned image paths.

## Findings

### F1 — Python package exposes AudioClass in native module but not top-level package
- `sparrow-engine-python/src/lib.rs:1419` registers `AudioClass` in `_sparrow_engine_core`.
- `sparrow-engine-python/python/sparrow_engine/__init__.py:14-27` imports `AudioResult` and `AudioSegment` but not `AudioClass`; `__all__` at lines 35-67 also omits it.
- Runtime `AudioSegment.classes` therefore yields objects whose type cannot be imported as `sparrow_engine.AudioClass`, blocking documented type annotation / isinstance use at the top-level package.

### F2 — Server store=true drift labels still discard top-1 AudioClass label
- `sparrow-engine-server/src/handlers/audio.rs:91-100` builds response segments from native result.
- Store path at `handlers/audio.rs:103-115` sets `labels` to `vec![model_id.clone(); response.segments.len()]` for every segment.
- For multi-class softmax audio (Perch 2), `AudioSegment.classes[0].label` is available on the native/result response path, but drift/category metrics receive only model-id buckets. This was the R1 carry-forward issue and remains present.

### F3 — Public derive_model_type remains lenient for unsupported audio combinations
- `sparrow-engine-types/src/manifest.rs:899-910` rejects unsupported audio preprocess/postprocess combos at manifest-load time.
- `sparrow-engine-types/src/model_type.rs:24-36` still maps RawAudio+Sigmoid to `AudioDetector` and MelSpectrogram+Softmax to `AudioClassifier` for direct public callers; test at lines 284-290 documents the legacy behavior.
- This is the R1 carry-forward warning; if round 2 is tightening public surface consistency, this remains unresolved.
