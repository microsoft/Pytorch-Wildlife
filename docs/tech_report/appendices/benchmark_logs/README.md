# Benchmark Logs

Raw benchmark output from Phase 2 and Phase 2.5 runs. Currently this folder is a pointer to the canonical source; individual log files will be snapshot here when benchmarks are re-run for tech-report freshness updates.

## Canonical source

`docs/benchmarks.md` — primary benchmark document with full methodology and numbers.

## Snapshot plan

Benchmarks will be re-run and snapshotted in this folder at phase milestones:

| Trigger | Snapshot content |
|---------|------------------|
| Phase 3 close (done) | Already reflected in `docs/benchmarks.md` |
| Phase 3.5 close | Post-Phase-3.5 benchmark snapshot (especially relevant if item #6 changes output volume) |
| Phase 4 close | Post-Phase-4 benchmark snapshot (with `?store=true` overhead measured) |
| Any hardware change | Snapshot with new hardware specs |

Each snapshot lands as a dated file:

```
appendices/benchmark_logs/
  2026-04-14_phase2_http_bench.md      — snapshot from Phase 2 close
  2026-04-14_phase2.5_direct_bench.md  — snapshot from Phase 2.5 close
  YYYY-MM-DD_<milestone>_*.md          — future snapshots
```

## References

- `docs/benchmarks.md` — current canonical source
- `07_benchmarks.md` — main tech-report benchmarks section
- `docs/master_plan.md § Benchmark Results` — summary copied into master plan for quick reference
