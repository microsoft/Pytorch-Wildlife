# Style Guide — Sparrow Engine Tech Report

Scope: this style applies to all files under `docs/tech_report/`. Derived reports (executive summary, blog post, academic abstract) should inherit or deliberately deviate from these rules and state the deviation in their front matter.

## Voice

- **Direct and factual.** State what is, what was decided, what the tradeoffs were. Cite evidence (file paths, commits, benchmark tables, issue numbers) for every non-trivial claim.
- **No marketing language.** Banned words: *powerful, elegant, beautifully, game-changer, revolutionary, seamlessly, blazing, rock-solid, simply, effortlessly*. Banned phrases: *the beauty of this approach, this elegantly solves, leveraging the power of, best-in-class*.
- **Neutral tone.** No selling. When listing options, present pros and cons flatly. Do not advocate.
- **Past tense for decisions and outcomes.** Present tense for current state. Future tense for planned work.
- **First-person plural ("we") sparingly.** Prefer passive or subject-focused sentences. ("The engine rejects a second `Engine::new()` call" beats "We designed the engine to reject...".)

## Evidence citation

Every claim that is not trivially true must carry evidence.

- **File path + line number** for code claims: `sparrow-engine/sparrow-engine-cpu/src/engine.rs:222` (post-Phase-3.8-Phase-A path; pre-2026-05-02 was `sparrow-engine/libsparrow_engine/src/engine.rs:222`)
- **Commit SHA** for history claims: `commit 09ee0aa`
- **Benchmark row** with hardware + workload for performance claims
- **Issue/PR link** for bugs: ORT issues #27912, #12288
- **Round report path** for audit findings: `docs/review/phase3-final/round_04/inquisitor_review.md`

Unverified claims: mark as `UNVERIFIED` inline, or cut. No speculation presented as fact.

## Structure of every doc

```
# Title

[1-2 sentence summary of what this doc covers.]

## Why (optional when obvious)
Why this doc exists. What question does it answer.

## Content
[Body, usually with subsections.]

## References
[Bullet list of file paths, commits, external docs relied on.]
```

- Headings: `##` for major sections, `###` for subsections. No `####` if avoidable.
- Lists over prose when enumerating ≥3 items.
- Tables for comparisons, configuration matrices, benchmark data.
- Code blocks for: API signatures, manifest snippets, FFI structs, sample commands.

## Tables

- Always have header row with pipe-alignment.
- Column headers describe what the row entries mean, not just "Name" / "Value".
- If a table is wider than 120 chars, split into two tables with a shared key column.

## Code blocks

- Annotate the language: ` ```rust `, ` ```toml `, ` ```bash `.
- Keep blocks under 20 lines. For longer samples, link to the source file.
- Show actual code from the repo when possible, not illustrative pseudo-code. If illustrative, label it `// illustrative — not in-repo`.

## Why / What / How format for design decisions

When explaining a design decision:
```
**Why**: <motivation — the constraint, risk, or requirement>
**What**: <the decision as a concrete statement>
**How to apply / enforce**: <code path, manifest field, test, or doc that makes it load-bearing>
```

Avoid vague hand-waves. If "why" is not clear enough to state in one sentence, the decision shouldn't be in the report.

## Benchmarks

- **Always cite hardware**: CPU model, GPU model, RAM, OS, driver, ORT version, CUDA/cuDNN version.
- **Always cite workload**: number of images, resolution, model, batch size, warm vs cold.
- **Report mean + median** when timings vary. Raw logs go in `appendices/`.

## Confidence assessment (end of each section)

Every section concludes with:

```
**Confidence**: [HIGH/MEDIUM/LOW]
- Factual accuracy: [score] — [verified/memory/inferred]
- Completeness: [score] — [what might be missing]
- Freshness: [score] — [when last verified]
```

For planned-work sections (Phase 3.5, Phase 4, fine-tuning pipeline), confidence is naturally LOW on "freshness" since the work hasn't happened yet; state that explicitly.

## What to cut vs. what to keep

- Keep: decisions and their rationale, gotchas and their root cause, benchmarks with methodology, API surface, locked-in constraints.
- Cut: process narrative ("first we tried X, then Y, eventually Z"), internal politics, praise or blame of individuals or teams, exhaustive listing of refactor diffs.
- Keep in appendices: raw benchmark logs, FFI export lists, complete CLI command inventory.

## Cross-references

- Link to companion files in `docs/tech_report/` with relative paths.
- Link to source-of-truth design docs under `docs/design/` when the full design belongs there. This report consolidates; it does not replace.
- Link to review artifacts under `docs/review/` for audit details.

## Derived-report usage

This report is the master. Derived reports should:

- **Executive summary (~2 pages)**: pull from `00_executive_summary.md` + selected benchmarks + 1-paragraph architecture summary.
- **Blog post (~3 pages)**: pull from `00_executive_summary.md` + one gotcha + benchmarks + a narrative hook.
- **Academic write-up**: emphasize `03_architecture.md`, `04_design_decisions.md`, `06_gotchas_and_constraints.md`, `07_benchmarks.md`.
- **Internal design doc**: the whole report minus `00_executive_summary.md`.

When deriving, strip confidence blocks and cut citations to one per claim. Keep tone identical.

## Open questions to resolve before publishing

Any open question in the report should be tagged `QUESTION: {open issue}` and listed in `appendices/open_questions.md`. Closed questions get their resolution cited and the tag removed.

## Maintenance

- Update `master_plan.md`, `ideas.md`, `lessons.md` **first**, then reconcile the tech report with those canonical sources. The tech report is a snapshot, not the system of record.
- After any phase transition (e.g., Phase 3.5 → Phase 4), update `02_project_timeline.md`, the relevant `NN_phase_N_planned.md` promotes to `NN_phase_N.md`, and confidence blocks refresh.
- A quarterly sweep should run `/doc-fix` on `docs/tech_report/` to catch drift vs code.

## References

- `~/.claude/rules/persona.md` — parent style rules (this doc inherits and specializes)
- `~/.claude/rules/confidence.md` — confidence-block format
- `~/.claude/rules/anti-hallucination.md` — evidence requirement
- `docs/master_plan.md` — canonical phase list
