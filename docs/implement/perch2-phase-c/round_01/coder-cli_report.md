# coder-cli report — cli-multiclass-display

<a name="cli-multiclass-display"></a>

## Files modified
- `sparrow-engine/sparrow-engine-cli/src/main.rs`

## Final commit
- `7bfa2d6b79346411d224c2573c9f5e47521ee6bb`

## SIGNATURE phrase
- `audio_raw_json_classes_and_class_aware_merge`

## Tests written + results
- Added unit test: `audio_raw_json_classes_and_class_aware_merge`.
- `cargo build -p sparrow-engine-cli --features cpu --bin spe` — PASS.
- `cargo test -p sparrow-engine-cli --features cpu --bins` — PASS, 59 tests.
- `cargo test -p sparrow-engine-cli --features cpu` — PASS, 59 unit tests; 2 ignored integration tests.
- Note: the requested `cargo test -p sparrow-engine-cli --features cpu --lib --tests --bins` command is invalid for this bin-only package (`no library targets found`).
- Release smoke build `cargo build --release -p sparrow-engine-cli --features cpu --bin spe` — PASS.
- Code-auditor review — PASS (`STATUS: OK`).

## Smoke test output
Command:

```bash
SPARROW_ENGINE_MODEL_DIR=/home/miao/repos/PW_refactor/sparrow-engine-dev/.zenodo-staging \
./target/release/spe detect-audio --model perch-v2 \
  --model-dir /home/miao/repos/PW_refactor/sparrow-engine-dev/.zenodo-staging \
  --print --raw-segments --format json \
  sparrow-engine-core/tests/fixtures/audio/medium_10s.wav | jq .
```

Output:

```json
{
  "file": "sparrow-engine-core/tests/fixtures/audio/medium_10s.wav",
  "model_id": "perch-v2",
  "duration_s": 10,
  "sample_rate": 32000,
  "segments": [
    {
      "start_time_s": 0,
      "end_time_s": 5,
      "confidence": 0.52798575,
      "classes": [
        {
          "class_idx": 335,
          "label": "Alarm",
          "probability": 0.52798575
        },
        {
          "class_idx": 13497,
          "label": "Telephone",
          "probability": 0.03146214
        },
        {
          "class_idx": 1542,
          "label": "Bell",
          "probability": 0.017377894
        },
        {
          "class_idx": 14462,
          "label": "Wind",
          "probability": 0.015735175
        },
        {
          "class_idx": 14344,
          "label": "Vehicle",
          "probability": 0.01414722
        }
      ]
    },
    {
      "start_time_s": 5,
      "end_time_s": 10,
      "confidence": 0.24102566,
      "classes": [
        {
          "class_idx": 335,
          "label": "Alarm",
          "probability": 0.24102566
        },
        {
          "class_idx": 14462,
          "label": "Wind",
          "probability": 0.0327828
        },
        {
          "class_idx": 14344,
          "label": "Vehicle",
          "probability": 0.022398587
        },
        {
          "class_idx": 5160,
          "label": "Fire",
          "probability": 0.020709578
        },
        {
          "class_idx": 13497,
          "label": "Telephone",
          "probability": 0.017958969
        }
      ]
    }
  ]
}
```

## Cross-boundary issues
- None.

STATUS: DONE COMMIT=7bfa2d6b79346411d224c2573c9f5e47521ee6bb SIGNATURE="audio_raw_json_classes_and_class_aware_merge" ITEM=cli-multiclass-display
