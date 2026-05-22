# Fine-Tuning Pipeline

**Status.** Not started in sparrow-engine. Separate workstream, colleague-owned. Per Phase 3.7 Track A (2026-04-29), this workstream is now framed as the **`fine-tuning repo`** sibling repo — a repo-and-project-agnostic Docker training environment that consumes per-project preprocessing scripts and outputs sparrow-engine-compatible ONNX with manifest provenance metadata. NOT merged into sparrow-engine (engine-only constraint) or sparrow-ops (cross-team ownership; serves multiple ML projects). See `docs/design/architecture.md` for the 5-component placement and `docs/design/phase3.7/sibling_scope_sketches.md` for the cross-repo contract sketch. Detailed content deferred until that repo reaches alpha.

## Placeholder scope

Fine-tuning is outside sparrow-engine's current scope. v3 design D-v3-9 explicitly locked this in — sparrow-engine is inference-only. Adding fine-tuning would mean:

- Training loop, optimizer, loss functions.
- Dataset abstractions (data loaders, augmentation).
- Longer-running jobs (hours to days) vs inference (milliseconds).
- Different hardware profile (GPU-bound for longer durations, multi-node scaling).
- Different deployment shape (job scheduler vs inference service).

These are legitimately a different system from sparrow_engine.

## What the fine-tuning pipeline will likely be

Anticipated, based on AI-for-biodiversity project context:

- A separate repository (not part of sparrow-engine).
- Consumes sparrow-engine-compatible ONNX models — outputs sparrow-engine-compatible ONNX models.
- Integrates with Sparrow Studio's annotation storage (Phase 4) as a data source.
- Produces fine-tuned model + updated TOML manifest + updated labels file as an atomic unit.
- May integrate with AML (Azure ML) for scheduling and artifact management.

## What sparrow-engine needs from the fine-tuning pipeline

If fine-tuning produces a new model, that model flows into sparrow-engine via:

1. ONNX export.
2. TOML manifest (including `onnx_sha256`, `onnx_size_bytes` added in Phase 3 catalog).
3. Optional: updated labels file.
4. Drop into `{model_dir}/{model_id}/`.
5. `spe models verify` checks integrity.
6. spe loads on next inference call.

This is the existing sparrow-engine model-onboarding path — no sparrow-engine-side code change needed to support fine-tuned models that follow the convention.

## What the fine-tuning pipeline needs from sparrow_engine

Minimal. At most:

- Clear documentation of the manifest schema and tolerance bounds for golden-output regression.
- A `spe tools fine-tuning-smoke-test <manifest>` utility that runs inference on a small fixture set and reports tolerance against the fine-tuned model's intended outputs.
- Possibly: access to Phase 4 annotation storage via SQL read access (out of sparrow-engine's current scope but Phase 4 could add a read-only export).

None of these are blocking sparrow-engine work.

## Scheduling

Section will be rewritten as a proper chapter (covering actual architecture, colleague's contribution surface, dataset provenance, evaluation protocols, scoping for conservation-specific concerns) once:

1. Phase 4 is complete (annotation storage schema is frozen).
2. Colleague's fine-tuning work reaches alpha.

Until then this file is a placeholder to hold the slot in the tech-report structure.

## Questions to resolve when the section is written

- How much of the fine-tuning pipeline reuses sparrow-engine's preprocessing? If fine-tuning uses different augmentation, does it produce models that remain correct when sparrow-engine preprocesses inference-time images differently?
- Label space evolution. If fine-tuning adds a new species class to SpeciesNet, how do Sparrow Studio annotations made against the old label space migrate?
- Active learning loop. Does the fine-tuning pipeline pull annotations from Sparrow Studio Phase 4 storage? If so, what triggers re-training?

## Confidence

**Confidence**: LOW across all dimensions. This is a placeholder.
- Factual accuracy: LOW — content is speculative; colleague-owned work not yet visible
- Completeness: LOW — placeholder only
- Freshness: LOW — defer refresh until Phase 4 complete

## References

- `docs/design/v3/final_decisions.md` D-v3-9 — "No fine-tuning in the API"
- `docs/master_plan.md § Phase 4` — data layer that fine-tuning will likely consume
- `04_design_decisions.md` D-v3-9
