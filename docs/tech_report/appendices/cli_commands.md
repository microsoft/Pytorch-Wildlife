# CLI Command Inventory

`spe` binary commands, flags, and exit codes. 11 commands total (stable through Phase 4.4).

## Commands

| Command | Purpose |
|---------|---------|
| `spe detect` | Image detection (single file, list, or directory) |
| `spe classify` | Image classification |
| `spe detect-audio` | Audio detection (WAV files) |
| `spe pipeline` | Ad-hoc detect → crop → classify pipeline |
| `spe models list` | List available model IDs in `{model_dir}/` |
| `spe models info <id>` | Show manifest metadata for a model |
| `spe models verify [--write]` | Verify model checksums against manifest; `--write` to bootstrap |
| `spe device` | Show active device (compile-time check; see § 06) |
| `spe init` | Explicit engine initialization (usually auto on first use) |
| `spe hash <path>` | SHA-256 file hash |
| `spe day-night <path>` | Day/night classification from an image |

## Flags (inference commands)

Applied to `detect`, `classify`, `detect-audio`, `pipeline` unless noted.

| Flag | Argument | Behavior |
|------|----------|----------|
| `--device` | `auto\|cpu\|cuda[:N]` | Override default device resolution |
| `--format` | `json\|csv` | Output format for inline stdout |
| `--threshold` | float | Detection / classification confidence threshold |
| `--max-detections` | int | Cap detections per image |
| `--recursive` | — | Descend into subdirectories (symlink-cycle-safe via `canonicalize`) |
| `--visualize` | — | Render annotations onto images (requires `--output-dir`) |
| `--output-dir` | path | Destination for visualized outputs |
| `--export-format` | `megadet\|coco\|csv` | Export to file (suppresses inline output) |
| `--export-output` | path | File path for export; parent dir created if missing |
| `--summary` | — | Print per-category stats to stderr (detect only) |

## Pipeline-specific flags

| Flag | Argument | Behavior |
|------|----------|----------|
| `--detector` | model_id | Detector model in pipeline |
| `--classifier` | model_id | Classifier model in pipeline |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Per-file error on inference (batch continues; some files failed) |
| 1 | Empty input (no files matched the pattern) |
| 1 | Other runtime errors |
| 141 | SIGPIPE (e.g., `spe detect \| head`) — clean exit |

## Flag validation

- `--visualize` requires `--output-dir` (clap `requires`).
- `--export-format` implies `--export-output` (else errors before inference).
- `--device cuda` when no GPU is present errors at engine creation, not later.

## Source

`sparrow-engine/sparrow-engine-cli/src/main.rs` — clap `Parser` definitions and handlers for all 11 commands.

## References

- `05_implementation_details.md § CLI behavior` — deeper behavior notes
- `docs/master_plan.md § Phase 3` — flag matrix source
