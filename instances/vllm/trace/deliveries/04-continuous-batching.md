# Ch04 — Continuous Batching (v6, FIRST full-pipeline chapter)

- **Type**: delivery (canonical)
- **Chapter**: 04-continuous-batching
- **Status**: published_v6
- **Published**: 2026-05-06
- **Source commit**: vLLM 98661fe
- **Reviewer**: reviewer-2 (APPROVED)
- **Snapshot**: `trace/snapshots/04-continuous-batching/v6-2026-05-06/`
- **Narrative**: `instances/vllm/artifacts/04-continuous-batching/narrative/chapter.md` (712 lines, ~3064 words)
- **Tags**: v6-baseline, full-pipeline-first, part1-complete

## Posterity note — first chapter under v6 standards

Ch04 is the FIRST chapter where the entire pipeline (implementer → tester → writer → reviewer → archivist) ran end-to-end producing a v6-grade artifact. Establishes the baseline for Ch05–Ch28 rewrites. Subsequent chapters should be benchmarked against this delivery.

## Quality gate evidence

| Check | Result |
|---|---|
| `lint_formulas.py` | PASS — 0 blocking. 4 non-blocking inline-density warnings on lines 92, 118, 150, 171 (single-symbol vars: `$P_i$`, `$O_i$`, `$B$`, `$N$`, `$R_k$`, `$n_r$` — W06-allowed, NOT REJECT-worthy) |
| `lint_source_grounding.py` | PASS — all checks green |
| Tests | 48/48 pass (`pytest --ignore=tests/_legacy -q`) |
| Demo runnable | Verified — output (step 17/18/19, KV reclaimed 200/200, speedup 20.80×) matches narrative §4.5 verbatim |
| Source mapping table | 13 rows (≥ 5 required) |
| `impl-notes.md` source files | 5 (≥ 3 required): `scheduler.py`, `output.py`, `request_queue.py`, `request.py`, `kv_cache_manager.py` |
| `# REFERENCE:` comments | 65 across 6 impl files (scheduler.py: 38, request_queue.py: 8, output.py: 7, request.py: 6, kv_cache_manager.py: 5, demo.py: 1) |
| 5-step rhythm | PASS — verified across all 7 sections (§4.1–§4.7) |
| Reference format | All `{source_dir}/path:Lline` or `implementation/file.py:Lline` per CLAUDE.md |

## Numerical fidelity (demo ↔ narrative)

| Quantity | Demo output | Narrative §4.5 |
|---|---|---|
| Static batching steps | 416 | 416 |
| Continuous batching steps | 20 | 20 |
| Speedup | 20.80× | 20.80× |
| KV reclaimed (preempt path) | 200/200 | 200/200 |
| Bubble formula `T_static = max(P_i) + max(O_i)` | 400+16=416 | derived in §4.1 |

## Diagrams decision (precedent for future chapters)

**APPROVED without new diagrams.** Reviewer judgment: §4.5's 19-step verbatim demo replay (with admission/preemption/finish callouts) + §4.1's cafeteria analogy + §4.2's running-first reasoning + §4.4's tables convey all abstract concepts. None of budget allocation, two-phase scheduling, preempt-tail rule, or chunked-prefill on/off reads as confusing without a figure. **Precedent**: if 5-step rhythm + numbered demo replay + tables suffice, no figure is mandatory.

## Novel insights verified

1. Bubble math uses actual demo workload (P=400/64/16, O=4/8/16) — formula `T_static = 400 + 16 = 416` ties directly to demo output `static: 416 steps`.
2. Three asserts framing as "final correctness gate" (§4.2.1) correctly tied to `scheduler.py:L848-L853`.
3. Phase 1 `continue` vs Phase 2 `break` semantic difference is the closing teaching point of §4.2.4 — accurate per `scheduler.py:L446-L462`.
4. Minimal preempt-trigger config (`num_gpu_blocks=2`) in §4.6 is reproducible.
5. F01 forward-pointer to Ch20 in §4.3.4 explains `finished_req_ids` gracefully without breaking flow.
6. All source paths use full `{source_dir}/path:Lline` form on first reference.

## v6 baseline for Ch05–Ch28

Future chapter rewrites must meet or exceed:
- ≥ 5 source files in `impl-notes.md`
- ≥ 60 `# REFERENCE:` comments distributed across impl files
- Source mapping table ≥ 10 rows (Ch04 had 13)
- 5-step rhythm (Source Trail → Bridge → Theory Deep Dive → Implementation → Source Diff) per major section
- Demo output numbers must appear verbatim in narrative
- Both linters: `formula` 0 blocking, `source-grounding` all green
- Test pass rate: 100% (Ch04: 48/48)

## Cross-references

- Trace decision: `trace/decisions/2026-05-05_ch05-ch28-directory-remap-to-outline-ids-+-legacy-snapshot.md`
- Auto-record: `trace/deliveries/2026-05-05_ch04-continuous-batching-ch04-continuous-batching-v6-published-—-first-full-pipeline-.md`
- Reviewer status: `/tmp/book-factory/04-continuous-batching/reviewer-status.json` (ephemeral, also archived in snapshot dir)
