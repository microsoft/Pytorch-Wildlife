# Sparrow Engine

A Rust ML inference engine for camera-trap and bioacoustic data.
Drop-in for MegaDetector v6, DeepFaune, HerdNet, OWL-T, SpeciesNet, and
MD_AudioBirds_V1; model-agnostic via TOML manifests.

## Quickstart

> NOTE: install URLs below are RFC-2606 placeholders pending public
> hosting per `docs/release_dev_plan.md § R1` + `§ R3`. Today's
> supported lead form is to clone the repo and run the wrapper locally;
> the `curl | sh` and `iwr | iex` one-liners are documented for
> post-R3.

**Today (supported)** — clone the repo and run the wrapper locally
(CWD = repo root):

```bash
# Linux / macOS
bash installer/sparrow-engine-install.sh
```

```powershell
# Windows PowerShell
installer\sparrow-engine-install.ps1
```

**Post-R3 (deferred; not yet supported)** — once GH Releases publish
the canonical URL, the stdin-piped one-liner will become the lead form:

```bash
# Linux / macOS (post-R3)
curl -LsSf https://sparrow-engine.example/install.sh | sh
```

<!-- TODO: replace with canonical sparrow-engine URL when public hosting fires per release_dev_plan.md § R3 -->

```powershell
# Windows PowerShell (post-R3)
iwr https://sparrow-engine.example/install.ps1 -useb | iex
```

The stdin-piped form fails today because the wrapper resolves
`probe.sh` relative to `dirname "$0"`; under `curl | sh`, `$0` is
`bash`, not the script path. See `docs/install.md § Troubleshooting`.

The wrapper probes hardware once, picks the right CPU or GPU build, and
installs the matching CLI binary plus the Python wheel into `~/.sparrow_engine/`.
Pass `--flavor cpu` or `--flavor gpu` to skip the probe. Pass `--docker`
to install the HTTP-server image instead.

System prerequisites for GPU: NVIDIA driver ≥550.x, CUDA 12.6 runtime,
and **cuDNN ≥9.10** (cuDNN 9.8 has a Conv-engine bug on sm_89 — see
`docs/install.md § Troubleshooting`).

See `docs/install.md` for the full install reference (per-platform
commands, air-gapped path, exit-code catalog 0–14, troubleshooting).

## Architecture

Sparrow Engine is engine-only: it loads ONNX models and runs inference. Annotation,
training, data versioning, model registry, drift detection, and
deployment orchestration live in sibling repos. See
`docs/design/architecture.md` for the canonical 5-component layout.

Core invariants:

- ONNX for all models (vision + audio)
- NCHW layout mandatory
- Normalized bbox `[0,1]` at all public API boundaries
- TOML manifests (one per model)
- NMS in the ONNX graph, never in the Sparrow Engine
- `Engine` is a singleton (ORT is process-global)

See `CLAUDE.md` for the full set of locked-in design decisions.

## Documentation

- `docs/install.md` — User-facing install guide (canonical)
- `docs/master_plan.md` — Phase status (Phase 1 → Phase 4.1)
- `docs/design/architecture.md` — 5-component architecture
- `docs/benchmarks.md` — Benchmark results + methodology
- `docs/tech_report/` — Public technical report

## License

`<!-- TODO: license details when public release work fires per release_dev_plan.md § R3 -->`
