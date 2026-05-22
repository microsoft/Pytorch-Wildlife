# Lessons Learned

Durable lessons captured during design, implementation, and validation. Meant for future sparrow-engine work and transferrable to other projects.

## The Rust pivot did not buy faster inference

**Claim.** Rust wins over Python were real, but the largest wins were not on raw inference speed.

**Evidence.**

- Raw isolated inference: Rust ~85 ms, Python ~85 ms. Same ORT C library underneath. ~0% difference.
- Direct-inference benchmark (GPU, MDv6, 100 images): Rust 71.5 ms/img mean, Python 89.2 ms/img. Rust is 1.25× faster on the non-inference portion (image decode, preprocessing). ORT inference itself is the same.
- Cold start: ~4.3 s (Python) → ~348 ms (Rust). 12× faster.
- Docker image CPU: ~650 MB (Python) → 163 MB (Rust). 4× smaller.
- Total pipeline (startup + preprocess + single-image infer): ~3214 ms → 91.4 ms. 2.87× faster.

**Takeaway.** When evaluating Rust vs Python for ML serving, the wins are: cold start, memory, container size, GIL-free concurrency, deployment-shape. Raw inference is unchanged because everyone calls the same ORT / PyTorch C libraries. Don't sell the rewrite on "inference is faster" — sell it on deployment economics.

**Reference.** § 01, § 07.

## Audit-fix pays for itself

**Claim.** The multi-round audit-fix cycle caught bugs that first-round reviewers missed. The iterative discipline was worth the compute cost.

**Evidence.**

- Phase 3 final audit-fix R2: inquisitor's independent scan caught a HIGH sparrow-engine-client SRV1 wire-compat break. R1 tests used stale fixtures; the break would have shipped in the Phase 3 release.
- Phase 3 final audit-fix R3: inquisitor's independent scan caught MI-1 (stats all-NaN sentinel leak) and MI-2 (COCO label_id namespace collision). Neither reviewer found either in R1/R2 despite having audited the relevant files multiple times.
- Phase 3 final audit-fix R4: inquisitor's approval-gate caught an infeasible proposed fix. R3 had sketched `tracing::warn!` for MI-2 without verifying `tracing` was a libsparrow_engine dependency. R4 inquisitor caught it during plan review; reviewer respec'd to Option A (`eprintln!` + `HashSet<u32>`).

**Takeaway.** Independent fresh-eye scanning finds things reviewers who "already know the code" miss. The cost is about 1–2 extra rounds of agent compute per phase. The benefit is catching silent-data-corruption bugs (MI-2) and wire-compat breaks (SRV1) before users hit them.

**Reference.** § 08.

## Fresh team per round, always

**Claim.** Agents that just audited and approved code are not independent reviewers of the resulting change. The audit-fix skill enforces fresh team per round for exactly this reason.

**Evidence.**

- R3 documenter and reviewer both reported CONVERGED ("zero findings, zero changes"). R3 inquisitor — different agent, independent scan, started from the code state without knowing the other agents' conclusions — found MI-1 and MI-2.
- R5 auditor respawn: original R5 auditor spawn failed silently (config issue during spawn). The respawned R5 auditor, with no inheritance from the failed instance, did the R5 structural scan cleanly.
- Doc-fix R2 (post-reorg): R1 count claim was "25 edits" but R2 documenter re-counted to "24". Minor but illustrates that fresh count ≠ inheriting prior claim.

**Takeaway.** Inter-round continuity lives in the reports on disk, not in agent state. Every iterative skill in `~/.claude/skills/` (audit-fix, doc-fix, code-review, research) enforces this. Projects adopting this pattern should too.

**Reference.** § 08 — audit-fix methodology.

## Interface ownership: sparrow-engine defines, Sparrow adapts

**Claim.** In a multi-consumer system where one library (sparrow-engine) is consumed by multiple products (Sparrow Studio Local, Web, CLI, Python, external), the library should own the interface and consumers should adapt. The alternative (each consumer maintains its own glue and workarounds) produces exactly the silent divergence that caused the original Python-era rewrite.

**Evidence.**

- Pre-sparrow-engine: Python stack, Sparrow Studio Local's C# reimplementation, and Sparrow Studio Web's worker glue all reimplemented parts of preprocessing. Correctness fixes landed three times.
- Post-v4: libsparrow_engine owns pre/post entirely. TOML manifests make it declarative. Correctness fixes land once.
- MT-11 example: CLI exposed extended `ModelInfo` fields but `ModelResponse` (sparrow-engine-server) didn't. Fix was extending sparrow-engine's type, not adding Sparrow-side enrichment. This is interface ownership in practice.
- Phase 2.5 consumer audit spent 4 rounds specifically checking that feature rollout was consistent across CLI + Python + FFI + HTTP.

**Takeaway.** When a library has N consumers, the marginal cost of consumer divergence is `N × fix_cost`. Centralizing the interface makes it 1× but requires discipline: every feature request goes through the library, no consumer-side workarounds allowed without an issue in the library's tracker.

**Reference.** § 03 interface ownership, § 09 Sparrow Studio.

## Documentation reorg was necessary

**Claim.** Project documentation accumulated 45+ ad-hoc review directories at `docs/review/` over 4+ phases. Names like `r1`, `deep_r2`, `final_pass_r1`, `post_windows_r2`, `audit-fix-r3` had lost meaning. A reorg into 4 canonical buckets + `historical/` with themed sub-buckets restored navigability.

**Evidence.**

- Before reorg: 45+ directories at one flat level, no taxonomy.
- After reorg: 4 canonical buckets (`phase3-final`, `phase3-testing`, `phase2.5-consumer-audit`, `sparrow-studio`) + 1 historical archive + 1 active doc-fix cycle.
- Doc-fix-reorg cycle: 3 rounds. R1 fixed 26 stale path references (docs referencing old review-dir names). R2 fixed CLAUDE.md's Docs Structure block. R3 CONVERGED.
- Secondary finding: during reorg, discovered that `phase2.5-consumer-audit/` had incomplete content — Round 2 and Round 4 were in `sparrow-engine/docs/review/` (teammates writing from wrong cwd during earlier audit-fix cycles). Even the rounds that existed in both locations were fragmented: Round 1 had the inquisitor report in one location and auditor + reviewer + verification in the other. Consolidation merged all 4 rounds + complete role reports into the canonical location.

**Takeaways.**

1. **Enforce folder-name taxonomy from the start.** Naming convention for review folders should be enforced by convention and by a CLAUDE.md-level rule, not by ad-hoc choice per round.
2. **Teammates run from the right cwd.** Past audit-fix cycles spawned teammates that sometimes wrote to `sparrow-engine/docs/` instead of `docs/`. Future cycles set cwd explicitly in the spawn config.
3. **Deprecate rather than accumulate.** After a phase closes, move its review artifacts to `historical/<category>/<phase>/` immediately, not years later. The longer the delay, the more content drift between active and historical.

**Reference.** `docs/review/README.md`, `docs/review/doc-fix-reorg/`.

## Test infrastructure for singletons requires `serial_test`

**Claim.** Rust tests that exercise process-global state (ORT `OrtEnv`, sparrow-engine's `ENGINE_EXISTS` AtomicBool) cannot rely on `--test-threads=1` alone for isolation. Intermittent flakes in the libsparrow_engine engine tests resisted diagnosis until we adopted `serial_test`.

**Evidence.** Pre-fix: 3–5 of the 18 engine tests failed non-deterministically on the full test suite despite `--test-threads=1`. All passed in isolation. Post-fix: all 173 libsparrow_engine tests pass deterministically. Commit `7fed112`.

**Takeaway.** Any Rust project with process-global state should use `serial_test` from day one. The flake diagnostic cost later is higher than the `dev-dependencies` addition cost upfront.

**Reference.** § 06 engine singleton test order-dependency.

## Scope expansion is expected during audit-fix

**Claim.** First-round reviewers in a fresh audit-fix cycle often find items outside the announced scope. Lead should expect scope expansion and have a decision rubric ready.

**Evidence.** Phase 3 final audit-fix R1 reviewer flagged 4 out-of-scope findings during the round:

- MT-SERVER (SERIOUS): sparrow-engine-server `ModelResponse` missing Phase 3 extended fields.
- MT-NHWC (FLAG): manifest parser silently accepts "nhwc" layout.
- EX-M2 (MEDIUM): COCO `label_id = 0` pycocotools convention.
- PC3 (design ambiguity): `visualize()` batch-accumulate semantics.

Lead decisions made mid-round:

- MT-SERVER: expand scope, fix this round (consistency rule — MT-11 motivation applies).
- MT-NHWC: fix this round (silent-accept + runtime crash is worst failure mode).
- EX-M2: doc-only fix (code change would double-encode for compliant users).
- PC3: leave as-is, add doc comment (observable behavior matches spec).

**Takeaway.** Don't treat scope as sacred. Out-of-scope findings often indicate where the original scope was too narrow. Have a decision pattern: "fix in this round" / "document-only" / "defer with issue" / "leave as-is with note." Don't kick everything to next round; that dilutes convergence.

**Reference.** `docs/review/phase3-final/round_01/reviewer_report.md` cross-boundary section.

## Architectural constraints from upstream libraries

**Claim.** Upstream library bugs and constraints (ORT CUDA EP + NHWC, cuDNN 9.8 Conv bug, Rust default SIGPIPE) have shaped sparrow-engine's design at multiple points. These are not theoretical risks — they manifested during development and shipped as locked-in rules.

**Evidence.**

- NCHW mandate: locked in after MT-10 (SpeciesNet NHWC + ORT CUDA EP = SafeInt overflow crash).
- cuDNN 9.10+ requirement: locked in after MT-15 (cuDNN 9.8 Conv bug on sm_89).
- SIGPIPE fix: locked in after MT-16 (`spe ... | head` printed fatal error).
- `serial_test`: locked in after intermittent engine test flakes.

**Takeaway.** Upstream-library gotchas will continue to appear. Keep gotchas documented (§ 06) so new maintainers don't re-debug them. Consider an explicit "upstream bug tracking" section for long-lived ORT / cuDNN issues.

## Generate your own golden outputs

**Claim.** Using an upstream reference implementation (PyTorchWildlife / PIL) to generate golden outputs for validation couples your correctness bar to upstream changes. Generating from your own library keeps the reference self-consistent.

**Evidence.**

- v4 design decision D-v4-9 locked this in.
- Sparrow Engine's Rust `image` crate letterbox rounds padding top-left, PIL distributes symmetrically. 1-detection difference on some edge cases.
- Using PIL as reference would make sparrow-engine's correctness bar change every time PIL ships a new version.

**Takeaway.** Self-generated golden outputs. Tolerance accounts for f32/f64 precision gaps but not for upstream preprocessing changes.

## Over-delegation is a failure mode

**Claim.** Iterative skills delegate work to sub-agents. Too much delegation produces agents that orchestrate but never synthesize, or that miss what a direct read would catch immediately.

**Evidence.** Lead constraint in `~/.claude/rules/agent-coordination.md`: "Lead manages and summarizes — NEVER does deep research, web search, or investigation." The corollary is that the lead also should not skip direct reads when a spot-check would be faster than spawning a sub-agent.

Observed during this session: the R3 inquisitor in Phase 3 final audit-fix did sub-agent delegation for the independent scan but ALSO did direct grep + Read for each finding. This produced verified-at-source findings (MI-1, MI-2) rather than "sub-agent reported X" which could have been a hallucination.

**Takeaway.** Don't let delegation become ritual. Spot-check sub-agent claims with direct reads. When the delegation adds no value (e.g., you already know where to look), skip it.

## HTTP schema changes are never safe

**Claim.** Even "additive" HTTP schema changes (adding optional response fields) can break strict downstream consumers. Treat every schema change as breaking until tests prove otherwise.

**Evidence.** Phase 3 SRV1 extended `ModelResponse` with 5 new fields. The new `default: bool` field is always serialized (not `Option<bool>`). `sparrow-engine-client`'s `ModelInfo(**m)` dataclass unpack was strict — it rejected any extra field. R1 tests used stale fixtures that didn't serialize `default`. The break would have shipped without R2 inquisitor's independent scan.

**Takeaway.**

- Test fixtures should be regenerated from the current schema at test time, not hand-written once and reused.
- Any `ModelResponse` change triggers the `sparrow-engine-client` test suite as part of the audit-fix gate.
- Long-term: consider relaxing strict unpack (accept extra fields) to make clients forward-compatible.

## Process narrative is cheap; cut it

**Claim.** Tech reports, design docs, and review artifacts often include process narrative ("we first tried X, then switched to Y, finally settled on Z"). It is cheap to write during development and expensive for readers.

**Evidence.** This report's STYLE.md explicitly bans process narrative. Readers want decisions and their rationale, not the iteration history. History lives in git log.

**Takeaway.** Reports capture decisions and gotchas. Git log captures process. Don't duplicate.

## Freshness anchor (2026-05-13)

**Lessons from 2026-04 onward live primarily in `docs/lessons.md`** (the authoritative system-of-record per `STYLE.md § Maintenance`). For the post-2026-04-21 set (audit-fix-2 anti-drift discipline; Phase 4 substantive scope creep with explicit `feedback_no_excuses` logging; Phase 3.8 Phase A C8 `[lib] name = "sparrow-engine"` invariant; dual-cdylib workspace-test collision; flavor-strict `Device::Auto` post-MT-4.1-2; etc.), read `docs/lessons.md` directly. This chapter snapshot is anchored at 2026-04-21.

## Confidence

**Confidence**: HIGH
- Factual accuracy: HIGH — every lesson cites specific evidence from commits, audit-fix reports, MT logs, or rules files
- Completeness: MEDIUM — covers high-leverage lessons; more granular lessons (e.g., specific toolchain quirks) defer to `docs/lessons.md`
- Freshness: HIGH — 2026-04-21

## References

- `docs/lessons.md` — durable cross-session lessons (a subset of these + lower-level entries)
- `docs/review/phase3-final/` — audit-fix round reports with the in-context findings
- `docs/review/doc-fix-reorg/` — doc-fix cycle after reorg
- `~/.claude/rules/agent-coordination.md` — lead constraint and delegation discipline
- `~/.claude/skills/audit-fix/SKILL.md` — iterative skill definition
