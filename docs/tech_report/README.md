# Sparrow Engine Tech Report

Master technical report for the Sparrow Engine project. Source of detail for all derived reports (executive summary, blog posts, academic write-ups, internal design docs).

## Purpose

1. **Primary documentation** for the Sparrow Engine library and platform, covering all phases from the Python-era origins through Phase 4.
2. **Consolidated reference** — the canonical place where design decisions, gotchas, benchmarks, and lessons are captured with citations to source.
3. **Derivation source** for shorter audience-targeted reports. See `STYLE.md` § Derived-report usage.

This is a snapshot, not the system of record. `docs/master_plan.md`, `docs/ideas.md`, `docs/lessons.md`, and the code itself remain authoritative for live state.

## Audience

Written for:
- **Internal tech team and future maintainers** — design rationale, gotchas, enforcement points.
- **Stakeholders (AI for Good Lab leadership, product owners)** — outcomes, benchmarks, schedule, risk.
- **External readers (community, academic)** — architecture, methodology, results vs prior art.

See `STYLE.md` for tone and evidence-citation rules all sections follow.

## Layout

| File | Section | Audience focus |
|------|---------|----------------|
| `STYLE.md` | Style guide for this report and derived reports | Writers |
| `00_executive_summary.md` | 1-2 pg: what sparrow-engine is + headline results | All |
| `01_background_motivation.md` | Problem space: Python-era limits, Triton, Sparrow Studio integration need | Stakeholders, external |
| `02_project_timeline.md` | Phase 0 → 4, key dates and decisions | All |
| `03_architecture.md` | 4-crate workspace pre-Phase-3.8-Phase-A; superseded by 7-crate workspace post-2026-05-02 (`docs/design/architecture.md` canonical) | Tech team, external |
| `04_design_decisions.md` | ORT-only, NCHW mandate, engine singleton, NMS-in-graph, TOML manifests, GPU default | Tech team |
| `05_implementation_details.md` | Models (current set), viz pipeline, export, FFI safety, Python GIL, workers | Tech team |
| `06_gotchas_and_constraints.md` | cuDNN 9.8 Conv bug, ORT CUDA-EP + NHWC, SIGPIPE, compile-time device probe, fork() + singleton | Tech team, external |
| `07_benchmarks.md` | GPU/CPU numbers, vs Triton, cold start, pipeline, detection parity, test counts | All |
| `08_validation_and_testing.md` | Audit-fix methodology, 5-round convergence, manual testing 69/69, consumer audit | Tech team |
| `09_sparrow_studio_integration.md` | Sparrow Studio Local (C# P/Invoke) + Sparrow Web (HTTP), 3-worker architecture | Tech team, stakeholders |
| `10_lessons_learned.md` | Rust pivot ROI, audit-fix value, doc reorg, multi-teammate writing failure modes | All |
| `11_phase_3.5_planned.md` | 12 items from `docs/ideas.md`, MT-17 priority + PyTorch head-to-head benchmark + audio heatmap test | Tech team, stakeholders |
| `13_fine_tuning_pipeline.md` | Placeholder — fine-tuning repo is colleague's separate repo per Phase 3.7 Track A; detailed section deferred until that repo reaches alpha | Tech team |
| `appendices/` | Raw benchmark logs, FFI export list, CLI inventory, API endpoint list, open questions | Tech team |

## Status by section

> **Status snapshot anchored at 2026-04-29..30** — all chapter content predates Phase 3.8 Phase A (2026-05-02), Step 1/2 (2026-05-04/05), Phase C (2026-05-06), Phase 4 (2026-05-07), Phase 4.1 (2026-05-12), Phases 4.2/4.3/4.4 (2026-05-13), and audit-fix-chain-2 (2026-05-13). Per-chapter body refresh is a future tech-report project; chapters 12 + 14 are PRUNE CANDIDATES — see Cross-Boundary Findings in `docs/review/doc-fix-cleaning-2/round_01/documenter_report.md`. Live sources (always authoritative over this snapshot): `docs/master_plan.md`, `CLAUDE.md § Locked-In Design Decisions`, `docs/design/architecture.md`, `docs/lessons.md`, `docs/changelog.md`.

| File | Status | Last verified |
|------|--------|---------------|
| STYLE.md | DRAFT | 2026-04-21 |
| 00_executive_summary.md | DRAFT | 2026-04-29 (R9 + R11 F4 + R12 N-R10-1) |
| 01_background_motivation.md | DRAFT | 2026-04-21 |
| 02_project_timeline.md | DRAFT | 2026-04-29 (R9 C11 + R12 N-R10-1+N-R10-3) |
| 03_architecture.md | DRAFT | 2026-04-29 (R11 F5 + R12 N-R10-4) |
| 04_design_decisions.md | DRAFT | 2026-04-29 (R11 F1 + F2 + F8) |
| 05_implementation_details.md | DRAFT | 2026-04-29 (R12 SW-1) |
| 06_gotchas_and_constraints.md | DRAFT | 2026-04-29 (R12 N-R10-1 + SW-3) |
| 07_benchmarks.md | DRAFT | 2026-04-29 (R11 F7 + F9) |
| 08_validation_and_testing.md | DRAFT | 2026-04-29 (R12 SW-2 + SW-4 + R13 NEW-R12-1) |
| 09_sparrow_studio_integration.md | DRAFT | 2026-04-29 (R10-CARRY-1 + R11 F6 + R13 NEW-R12-2) |
| 10_lessons_learned.md | DRAFT | 2026-04-21 |
| 11_phase_3.5_planned.md | DRAFT — Phase 3.5 COMPLETE 2026-04-28 | 2026-04-29 |
| 13_fine_tuning_pipeline.md | DRAFT — placeholder, detail deferred post-Phase 4 | 2026-04-21 |
| appendices/ | DRAFT | 2026-04-21 |

Status values: STUB / DRAFT / FINAL / NEEDS-REFRESH. All sections currently DRAFT pending review; promote to FINAL after user sign-off.

_Status-table last refresh: 2026-04-30 (R15 — 10 rows bumped to mirror chapter freshness footers per R12 inquisitor §4.1 N1; R13 + NEW-R13-1 substantive edits to chs 08 + 09 + 11 + 12 reflected; R14 inq §4.2 Option A mirror-footer semantic preserved — ch 08/09/12 cells set to 2026-04-29 mirroring chapter footers; ch 11 cell + footer kept at 2026-04-29 per Option A "annotate, don't bump" convention)._

## Derivation recipes (preview)

Executive summary: `00_executive_summary.md` + `07_benchmarks.md` intro table + `03_architecture.md` § Consumer model summary. ~2 pages.

Blog post: `00_executive_summary.md` + `06_gotchas_and_constraints.md` § cuDNN 9.8 story + `07_benchmarks.md` + `10_lessons_learned.md` § Rust pivot. ~3 pages, narrative-lead.

Academic write-up: `03_architecture.md` + `04_design_decisions.md` + `06_gotchas_and_constraints.md` + `07_benchmarks.md` + `08_validation_and_testing.md`. Strip process narrative; add related-work section on inference runtimes.

Internal design doc: this report minus `00_executive_summary.md`.

## Maintenance

See `STYLE.md` § Maintenance. Summary:

- Update `master_plan.md` / `ideas.md` / `lessons.md` first; reconcile this report after.
- Phase transitions promote `NN_phase_N_planned.md` → `NN_phase_N.md` and trigger a confidence-block refresh.
- Quarterly `/doc-fix` sweep on `docs/tech_report/` to catch drift.

## References

- `docs/master_plan.md` — canonical phase list and architecture
- `docs/ideas.md` — Phase 3.5 backlog
- `docs/lessons.md` — durable lessons
- `docs/benchmarks.md` — benchmark results + methodology
- `docs/design/` — design docs per phase
- `docs/review/` — audit and review artifacts
- `~/.claude/projects/-home-miao-repos-PW-refactor-sparrow-engine-dev/memory/project_tech_report_notes.md` — drafting notes (collected during Phase 3 manual testing)
