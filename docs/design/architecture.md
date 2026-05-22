# Sparrow Engine — Architecture

**Status**: ACTIVE — canonical highest-level architecture for the sparrow-engine project
**Date**: 2026-04-29 (initial; tracks Phase 3.7 Track A consensus)
**Supersedes**: `docs/design/v4/libsparrow_engine/design_report.md` and `consensus_design_revised.md` (3-consumer architecture, partial supersession)
**Source of decisions**: Phase 3.7 Track A research (`docs/research/phase3.7/track_a/round_04/inquisitor_review.md`) + recommendation (`docs/design/phase3.7/mlops_planning.md` rev 2). The Phase 3.7 doc-update consensus round artifacts (formerly under `docs/design/phase3.7-doc-update/round_01/`) were retired in the 2026-05-07 doc consolidation; the consensus outputs are absorbed into the canonical `phase3.7/` docs and the round artifacts remain recoverable via git history.
**Lifecycle**: Updated when an architectural decision changes. Per-phase implementation specs live under `docs/design/phaseN/`.

---

## TL;DR

Sparrow Engine is the inference engine. It does one thing: load ONNX models and run inference, fast. Everything else — annotation, training, data versioning, model registry, drift detection, deployment orchestration — lives in **sibling repos**, not inside sparrow-engine. The architecture is **5 components**:

```
                    Sparrow Studio
                  (annotation, GUI)
                          │
                          ▼
                  ┌───────────────┐
                  │   sparrow-data  │                          ← data substrate
                  │ (storage,     │                   (sibling repo, deferred)
                  │  versioning,  │
                  │  preprocess,  │
                  │  inf. logs)   │
                  └──────┬────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      ┌────────────────┐   ┌──────────────────┐
                  │ sparrow-engine   │ │ fine-tuning repo│  ← inference engine + training
      │    (engine)      │ │   (Docker env)   │    Sparrow Engine here; fine-tuning = colleague's repo
      └───────┬────────┘   └─────────┬────────┘
              │                      │
              ▼                      ▼
         inference         sparrow-engine-compatible ONNX
         results           with version + provenance
              │                      │
              └──────────┬───────────┘
                         ▼
                  ┌───────────────┐
                  │   sparrow-ops   │                          ← model ops layer
                  │ (registry,    │                   (sibling repo, deferred)
                  │  drift Tier3, │
                  │  CI/CD,       │
                  │  monitoring)  │
                  └───────────────┘
```

Sparrow Engine today is a single Cargo workspace with 7 crates (`sparrow-engine-types`, `sparrow-engine-core`, `sparrow-engine-cpu`, `sparrow-engine-gpu` (fully populated post-Step-1 + Step-2 + Phase C — image GPU pipeline + audio GPU pipeline + cdylib via `ffi` feature), `sparrow-engine-server`, `sparrow-engine-cli`, `sparrow-engine-python`) after Phase 3.8 Phase A landed 2026-05-02 (pre-Phase-A: 4 crates with `libsparrow_engine` as the monolith). A future workspace crate (`sparrow-utils`) is sketched for Phase 4+ to host stateless utilities (visualization, format conversion, summary stats, day/night classification, public file hashing). The two sibling repos (`sparrow-data`, `sparrow-ops`) get their own design rounds — and their own Docker images — when forcing functions trigger their construction; sibling-repo construction is held until sparrow-engine is fully built (per `feedback_dev_first_release_last.md`). **Phase 3.8 Phase A** carved `libsparrow_engine` into `sparrow-engine-types` + `sparrow-engine-core` + `sparrow-engine-cpu` + new empty-stub `sparrow-engine-gpu` (executed 2026-05-02 per `docs/changelog.md`). **Step 1** (image GPU pipeline) + **Step 2** (audio GPU pipeline) + **Phase C Waves 1-5** (Engine dispatch glue + consumer wiring + dual cdylibs / wheels / Docker images + acceptance gates) brought the GPU pipeline online and wired consumers (substantively complete 2026-05-04 / 2026-05-05 / 2026-05-06); 5/5 acceptance gates PASS per `docs/review/phase3.8-phase-c/round_01/acceptance_gates.md`. Phase A implementation plan canonical at `docs/design/phase3.8/phase_a/implementation_plan.md`.

---

## Component responsibilities

### Sparrow Studio
- Interactive annotation UI (Local + Web variants)
- Ground-truth labeling workflows
- User-facing project management
- Owns its own UI rendering (does NOT consume sparrow-engine's viz)

### sparrow-engine (this repo)
- Inference engine: ORT singleton, manifest-driven model loading, preprocessing, postprocessing, detection / classification / audio / pipeline inference
- C ABI for Sparrow Studio Local consumption (FFI cdylib)
- HTTP service (`sparrow-engine-server`) for Sparrow Web consumption
- CLI binary (`sparrow-engine-cli`) and Python package (`sparrow-engine-python`) for direct user consumption
- Manifest schema definition (TOML) — including the Phase 4 `[provenance]` pointer fields + `[drift_reference]` per-class frequency that sibling repos populate
- **Phase 4 sibling-integration seam (substantively complete 2026-05-07)**: `?store=true` + `halt_on_store_failure` per-request HTTP query params on detect/classify/audio/pipeline; `InferenceLogSink` trait + default `StderrJsonLinesSink`; `compute_drift_metrics` (PSI eps=1e-4, nearest-rank percentile) emits Tier-1/2 per request. The `InferenceLogRecord` JSON-line wire format (`SCHEMA_VERSION="1.0"`) is the sparrow-engine-side contract that `sparrow-data` ingests and `sparrow-ops` reads for Tier-3 drift consumption (both siblings deferred construction). Detail: `docs/design/phase4/{schema.md, README.md}`.

### sparrow-data (sibling, deferred)
- Continuous-cadence data substrate: storage, content-addressable snapshot versioning, project-script preprocessing orchestration, inference logging for drift tracking
- Sparrow Engine writes inference results back to sparrow-data; sparrow-ops queries sparrow-data for drift metrics + ground truth
- The Phase 4 `InferenceLogSink` trait in `sparrow-engine-server` is the plug point: a future `SparrowEngineDataHttpSink` impl wraps an HTTP POST in `tokio::task::spawn_blocking` (or upgrades the trait to `async fn emit`) without sparrow-engine-side changes

### fine-tuning repo (colleague's repo, separate project)
- Repo-and-project-agnostic training Docker environment
- Each ingested project provides a preprocessing script (same script sparrow-data orchestrates)
- Output: sparrow-engine-compatible ONNX with version + manifest metadata
- NOT owned by the sparrow-engine team; cross-team contract via TOML manifest schema + ONNX format

### sparrow-ops (sibling, deferred)
- Event-driven model operations layer: deployment-event log, model registry, CI/CD for model updates, drift detection Tier-3, monitoring/dashboard, retraining orchestrator, notifications
- Tier-1/2 drift metrics live in `sparrow-engine-server` (per-request, stateless emit); Tier-3 (reference distributions, CUSUM, alarm path) lives in sparrow-ops

---

## Workspace crate vs sibling repo — decision rule

When deciding where a new module lives, apply these criteria:

| Criterion | Workspace crate (in sparrow-engine monorepo) | Sibling repo |
|---|---|---|
| **State** | Stateless (pure functions / library) | Stateful (owns DB, storage, files, artifacts) |
| **Cadence** | Locked to sparrow-engine's | Independent (continuous or event-driven separately) |
| **Independent users** | Only consumed by sparrow-engine's own crates, OR by sibling repos via published Cargo dep | Consumed by other repos with their own service stacks |
| **Deployment artifact** | Compiled into sparrow-engine's already-shipped binaries | Ships its own Docker image / service / package |
| **Boundary enforcement** | Soft (can grow tight coupling to internals) | Hard (only depends on published API) |
| **Dependency direction** | Lower-level dep | Higher-level service / consumer |

**Application**:
- `sparrow-utils` (viz, stats, export, daynight, hash) → workspace crate. Stateless, no own deployment, library only. Published to crates.io for sibling-repo consumption when needed.
- `sparrow-data` → sibling repo. Stateful, continuous cadence, own Docker image, cross-project consumption (fine-tuning + inference paths).
- `sparrow-ops` → sibling repo. Stateful, event-driven cadence, own Docker image, cross-deployment scope.
- `sparrow-engine-cli`, `sparrow-engine-python` → workspace crates. Tight API co-evolution with the Sparrow Engine crates (`sparrow-engine-types`/`sparrow-engine-core`/`sparrow-engine-cpu`); same team.

**Rejected merges** (and why):
- sparrow-utils into sparrow-ops → would invert the dependency direction (sparrow-engine-cli would operationally depend on sparrow-ops repo).
- fine-tuning repo into sparrow-ops → fine-tuning is repo-agnostic, owned by separate team; would force non-sparrow-engine ML projects to install sparrow-ops.
- sparrow-data and sparrow-ops into one repo → different cadences (continuous vs event-driven), different state shapes.
- sparrow-engine-cli/sparrow-engine-python into a separate "sparrow-engine-interface" sibling → API co-evolution forces lockstep with the engine crates; coordination cost > benefit at current scale.

---

## Dependency direction (must respect)

```
                                       Sparrow Studio
                                              │
                                              ▼
                                        sparrow-engine (engine)
                                       ▲      ▲
                                       │      │
                                 sparrow-data  ▼
                                       ▲   sparrow-utils (workspace crate)
                                       │      ▲
                                       │      │
                                       └─ sparrow-ops (consumer of utils + data)
                                              ▲
                                              │
                                       fine-tuning repo ↛ sparrow-ops
                                       (fine-tuning is invoked BY ops via Docker
                                        socket; doesn't depend on it)
```

Rules:
- The Sparrow Engine crates (`sparrow-engine-types` ← `sparrow-engine-core` ← {`sparrow-engine-cpu`, `sparrow-engine-gpu`}; post-Phase-3.8: Phase A 2026-05-02, Step 1 2026-05-04, Step 2 2026-05-05, Phase C 2026-05-06) depend only on Rust stdlib + `ort` + image/audio crates (+ `cudarc` + `nvjpeg-sys` for `sparrow-engine-gpu`). No siblings. Phase 3.8 Phase A renamed `libsparrow_engine` → `sparrow-engine-cpu` and extracted leaf crates `sparrow-engine-types` + `sparrow-engine-core`; Phase C Wave 4b made `sparrow-engine-gpu` parallel to `sparrow-engine-cpu` (both ship cdylibs as `libsparrow_engine.so`). Dependency direction preserved across the carve. See `docs/design/phase3.8/final_design.md §2` + `phase_a/implementation_plan.md`.
- `sparrow-engine-server`, `sparrow-engine-cli`, `sparrow-engine-python` depend on `sparrow-engine-cpu` OR `sparrow-engine-gpu` via mutually-exclusive Cargo features (`--features cpu` default / `--features gpu`) and (when extracted) `sparrow-utils`. No siblings.
- `sparrow-utils` depends on Rust stdlib + `image` + `ab_glyph` (for visualization). No siblings, no engine crates.
- `sparrow-data` consumes Sparrow Studio's annotation API + sparrow-engine manifests + (optionally) `sparrow-utils`. Does NOT depend on sparrow-ops or sparrow-engine.
- `sparrow-ops` consumes sparrow-engine manifests + `sparrow-utils` + sparrow-data's HTTP API + invokes fine-tuning Docker env. Does NOT depend on Sparrow Studio.
- fine-tuning repo consumes whatever data its preprocessing script asks for (sparrow-data is one option). Does NOT depend on any sparrow-engine repo at compile time; cross-repo contract is the manifest schema + ONNX format.

---

## Locked-in decisions from Phase 3.7 Track A

These supersede or extend the v4 locked-in decisions:

1. **Sparrow Engine is engine-only.** No MLOps surface inside sparrow-engine. Confirmed by 4-round Track A research + lead re-evaluation.
2. **Two sibling repos, not one.** Data lifecycle (continuous) and model lifecycle (event-driven) have different cadences and different state shapes; coupling them in one repo conflates concerns.
3. **sparrow-utils as workspace crate, not sibling.** Stateless utilities consumed by sparrow-engine's own CLI/Python + sibling repos via published Cargo crate. Sibling-repo placement would invert dependency direction (sparrow-engine-cli → sparrow-ops repo).
4. **Phase 4 manifest `[provenance]` pointer fields** (formerly tagged "Phase 5a"; folded into Phase 4 per `docs/master_plan.md:360`). Three optional `[provenance]` fields (`training_dataset_id`, `training_experiment_id`, `training_repo_commit`) added to the manifest TOML schema. ~50 LOC. Pre-positions sibling-repo integration without committing to a stack. Cheap insurance against unrecoverable provenance loss.
5. **Defer sibling construction.** Build sparrow-engine fully first, then sparrow-data + sparrow-ops siblings (per `feedback_dev_first_release_last.md`). Forcing-function triggers determine when each sibling activates.
6. **Per-project preprocessing scripts.** fine-tuning repo is repo-and-project-agnostic; each ingested project ships a preprocessing script that sparrow-data orchestrates at ingestion AND fine-tuning consumes at training time. Same script = guaranteed train/inference parity.
7. **Drift detection split.** Tier-1/2 (per-request stateless metrics) lives in sparrow-engine-server. Tier-3 (reference distributions + CUSUM + alarm path) lives in sparrow-ops.
8. **Reject MLflow + DVC adoption.** Both are Apache-2.0 OSS (verified) but operational cost > value at sparrow-engine's scale. GitHub primitives (Releases for ONNX artifacts, git for manifest history) cover the model-registry gap; DVC's value-add is to the training repo, not sparrow-engine.
9. **No public release until development done.** No PyPI, no GH Releases (beyond the existing CLI distribution), no CI push-tests until sparrow-engine + siblings reach internal-delivery quality. Development first; release last.

---

## Forcing-function triggers (when each subsystem activates)

| Trigger | Subsystem activated | Lives in |
|---|---|---|
| First production deployment ships | Data ingestion + storage + inference logging | sparrow-data |
| First model retrain needed (drift OR scheduled cadence) | Snapshot versioning + retraining orchestrator | sparrow-data + sparrow-ops |
| Wrong-model-deployed incident | Model registry + rollback metadata | sparrow-ops |
| First p95-confidence-decay incident across cameras | Drift detection (Tier-1/2 in sparrow-engine-server, Tier-3 in sparrow-ops) | sparrow-engine-server + sparrow-ops |
| Sparrow Web operator load > 10 cameras + p99 latency complaints | Monitoring dashboard | sparrow-ops |
| Model updates exceed once/quarter | CI/CD pipeline | sparrow-ops |

Detail in `docs/design/phase3.7/mlops_planning.md` §3.2.

---

## Sparrow Engine-side preservation discipline (do this NOW)

Until siblings exist, every sparrow-engine design decision considers their eventual integration. The discipline is documented in the sparrow-ops integration feedback memo and codified in `mlops_planning.md` §6.2. Summary:

- **Manifest schema**: stay loader-spec; `#[serde(default)]` on optional fields; reserve `[provenance]` and similar prefixes; no `deny_unknown_fields`.
- **FFI contract**: stable C ABI for any function siblings might call.
- **Inference output**: keep the result format stable so sparrow-data can log results without sparrow-engine-internal changes.
- **Tracing events**: treat as a public API (sparrow-ops monitoring will subscribe).
- **CLI commands**: if a new `spe` subcommand starts feeling like deployment-tracking or registry-management, push back — that's sparrow-ops territory.
- **Engine state**: keep inside sparrow-engine's process boundary. No persistent deployment-tracking state in sparrow-engine.

---

## Where to go next

| Reader | Read this |
|---|---|
| Newcomer / external researcher | `CLAUDE.md` (project root) → `docs/master_plan.md` Project Goal + Architecture → here |
| Sparrow Studio team member | here → `docs/design/v4/libsparrow_engine/consensus_design_revised.md` (FFI contract, locked-in v4 internals) → `docs/review/sparrow-studio/` |
| fine-tuning repo maintainer (colleague) | here → `sparrow-engine/sparrow-engine-types/src/manifest.rs` (schema source; was `sparrow-engine/libsparrow_engine/src/manifest.rs` pre-Phase-A) → `docs/design/phase3.7/sibling_scope_sketches.md` §3 (sparrow-data ingestion contract) |
| Future sparrow-data developer | here → `docs/design/phase3.7/sibling_scope_sketches.md` §2 → `docs/design/phase3.7/mlops_planning.md` §3 → `docs/design/phase5/README.md` (sibling-project pre-input stub; historical `phase5/` directory name preserved for stable URLs) |
| Future sparrow-ops developer | here → `docs/design/phase3.7/sibling_scope_sketches.md` §3 → `docs/design/phase3.7/mlops_planning.md` §3+§4 → `docs/design/phase5/README.md` (sibling-project pre-input stub) |
| Sibling-design-round participant (was: "Phase 5 design-round participant" before 2026-04-30 reframing) | `docs/design/phase3.7/mlops_planning.md` (rev 2 — full recommendation) → `sibling_scope_sketches.md` → `codebase_separation_survey.md` → `docs/research/phase3.7/track_a/round_04/inquisitor_review.md` |

---

## Lifecycle of this document

- **Updated when** an architectural decision changes (workspace crate ↔ sibling repo, new component, removed component, dependency direction flip, etc.).
- **NOT updated when**: per-phase implementation details change (those live in `docs/design/phaseN/`); locked-in decisions get refined within their existing scope (those live in `CLAUDE.md § Locked-In Design Decisions`).
- **Single source of truth** for: 5-component architecture, dependency direction, repo boundaries, workspace-crate-vs-sibling-repo decision rule, locked-in decisions from Phase 3.7 onward.
- **Cross-references**: `mlops_planning.md` for the long-form Phase 3.7 Track A recommendation; `master_plan.md` for live phase status; `CLAUDE.md` for the per-project context summary.

When reversing or significantly amending a decision, file a new design round (e.g., `docs/design/phase3.7-amendment/` or higher version), update this doc with a SUPERSEDED-IN-PART banner pointing forward, and preserve the prior content as historical record.
