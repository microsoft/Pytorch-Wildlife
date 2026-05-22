# Open Questions

Items flagged `QUESTION:` in the main tech-report sections awaiting resolution. Update this file whenever a question is closed (remove it) or a new one is flagged (add it).

## Currently open

None at the moment. The tech-report draft pass closed all questions inline or routed them to Phase 3.5 / Phase 4 items.

## Resolved (archive)

| Question | Resolution | Closed in |
|----------|------------|-----------|
| Should R2 rewrite or delete CLAUDE.md Docs Structure tree block? | Rewrite (option a). CLAUDE.md is top-level AI-context; inline tree gives quick orientation. | doc-fix-reorg R2, `commit 09ee0aa` |
| Should the MI-2 fix use `tracing::warn!` or `eprintln!`? | `eprintln!` + `HashSet<u32>` dedup. `tracing` is not a libsparrow_engine dependency; adding it for one warn site is over-scoped. | Phase 3 final audit-fix R4, `commit ee01898` |
| Should `uv.lock` be committed for `sparrow-engine-client`? | Gitignored. sparrow-engine-client is a library; convention is not to pin lock files for libraries. | Phase 3 post-audit housekeeping, `commit 7fed112` |

## Conventions

New questions should be flagged inline in the main section with:

```markdown
QUESTION: <concise statement>
```

And listed here with:

- Statement (one sentence).
- Section + file where raised.
- Proposed resolution paths (A, B, C ...) if applicable.

When closed, move to "Resolved" with the final decision and the commit or session that closed it.
