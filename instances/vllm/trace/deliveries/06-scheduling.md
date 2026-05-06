# Ch06 — Scheduling Policy (v6, THIRD v6-baseline chapter)

- **Type**: delivery (canonical)
- **Chapter**: 06-scheduling
- **Status**: published_v6
- **Published**: 2026-05-06
- **Source commit**: vLLM 98661fe
- **Reviewer**: reviewer-2 (APPROVED, 1 review cycle)
- **Snapshot**: `trace/snapshots/06-scheduling/v6-2026-05-06/`
- **Narrative**: `instances/vllm/artifacts/06-scheduling/narrative/chapter.md` (655 lines, ~3351 words)
- **Tags**: v6-cadence-holds, n3-baseline, part2-progress, scope-discipline-confirmed

## Posterity note — third v6-baseline chapter, cadence holds at N=3

Ch06 is the THIRD chapter under v6 standards. **The v6 cadence holds at N=3** with Ch06 strictly exceeding both predecessors on mapping density. The pattern is not bouncing between two outliers — it's a stable production rhythm. Future chapters can be benchmarked against this floor with confidence.

## Quality gate evidence — Ch04↔Ch05↔Ch06 comparison

| Check | Ch04 (N=1) | Ch05 (N=2) | **Ch06 (N=3)** | trend |
|---|---|---|---|---|
| `lint_formulas.py` | 0 blocking, 4 non-blocking | 0 blocking, 0 non-blocking | **0 blocking, 1 non-blocking** | clean |
| `lint_source_grounding.py` | PASS | PASS | **PASS** | = |
| Tests | 48/48 | 74/74 | **97/97 in 0.11s** | growing |
| Demo numerics fidelity | VERIFIED | VERIFIED | **VERIFIED** (4 spot-checks) | = |
| Source mapping table rows | 13 | 21 | **29 (+11 mini = 40)** | densest yet |
| `impl-notes.md` source files | 5 | 7 | **6** | ≥ floor |
| `# REFERENCE:` total | 65 | 61 | **60** | ≥ floor |
| `# REFERENCE:` files | 6 | 7 | **6** | = |
| 5-step rhythm | PASS | PASS | **PASS** | = |
| Review cycles | 1 | 1 | **1** | = |
| Lines | 712 | 757 | **655** | tighter |
| Words | 3064 | 3849 | **3351** | balanced |

**Note on length:** Ch06 is shorter than Ch05 (655 vs 757 lines) yet has DENSER source grounding (40 crossrefs vs 21). This is the scope-discipline payoff: Ch06 references Ch04 §4.3.2 by section number rather than re-deriving the schedule() loop. Density up, redundancy down.

## Demo-to-narrative numerical fidelity (4 spot-checks reproduce verbatim)

| Quantity | Demo output | Narrative |
|---|---|---|
| Head-of-line factor | 8.00x | §6.2.1 |
| Recompute/swap/abort @ 8K | 164/62/5000 ms (and 10.2/3.9/5000, 41/15.6/5000, 655.4/250.0/5000 at smaller L) | §6.3.1 |
| Priority order under aging | [B, D, C, A] base; [B, A, C, D] aged | §6.4.3, §6.4.4 |
| p95 TTFT sweep | 800/400/200/100/50 ms with ★ at (max_seqs=128, B=8192, threshold=0) | §6.5.2 |
| 16× ratio | 800ms / 50ms = 16.0 | §6.5.3 |

Every number in the demo output appears verbatim in the narrative. Same fidelity discipline as Ch04, Ch05.

## Source files referenced (impl-notes §1)

8 source-file:line references covering 6 distinct files (Ch04: 5, Ch05: 7):
1. `vllm/v1/core/sched/request_queue.py:L13-L17` — `SchedulingPolicy` Enum
2. `vllm/v1/core/sched/request_queue.py:L75-L130` — `FCFSRequestQueue(deque)`
3. `vllm/v1/core/sched/request_queue.py:L131-L200` — `PriorityRequestQueue(heapq)`
4. `vllm/v1/core/sched/scheduler.py:L67-176` (init), `L372-374`, `L478-504` (priority preempt), `L568`, `L1561-1565`, `L1567-1577` (`_select_waiting_queue_for_scheduling`)
5. `vllm/v1/core/sched/interface.py` — `SchedulerInterface` ABC
6. `vllm/config/scheduler.py` — policy config
7. `vllm/v1/core/sched/scheduler.py:L952-L972` — `_preempt_request`
8. `vllm/v1/kv_offload/cpu/gpu_worker.py:L319` — swap_blocks_batch (**reference-only**, NOT a preempt path — this is the language-trap point)
9. `vllm/v1/core/sched/scheduler.py:L1750-L1811` — `finish_requests` / `FINISHED_ABORTED`

## `# REFERENCE:` distribution across impl

| File | Count |
|---|---|
| `request_queue.py` | 27 |
| `policy.py` | 15 |
| `preemption_strategy.py` | 9 |
| `pareto.py` | 4 |
| `starvation_analysis.py` | 4 |
| `demo.py` | 1 |
| **Total** | **60** |

## Diagrams decision (precedent reaffirmed at N=3)

**APPROVED without diagrams.** Reviewer-2 evaluated TWO team-lead diagram candidates and rejected both:
- §6.3 trade-off matrix → 7-axis × 3-strategy ASCII matrix at lines 181-189 is clearer than a figure for that many cells side-by-side. A 2D graphic would have to drop 5 of the 7 axes.
- §6.5 Pareto frontier → 7-row sweep table at lines 478-484 with ★ Pareto markers + saturation column conveys the trade-off plane MORE precisely than a 2D scatter would. Critically, §6.5.4 explicitly warns readers "1 Pareto point is a model simplification artifact, not vLLM truth" — a real scatter plot would obscure that caveat by visually committing to it.

The N=3 diagrams-not-mandatory pattern is now a stable v6 trait: when ASCII tables + numbered demo replays + multi-axis trade-off matrices carry the data load, no figure is required.

## Novel strengths beyond Ch04+Ch05 baselines

1. **Two-tier mapping table**: §6.6 main (29 rows) + §6.3.5 mini-table (6 rows, preemption-strategy → vLLM source) + §6.5.5 mini-table (5 rows, pareto-config → vLLM scheduler internals). Total 40 source crossrefs without overloading the main table. Useful pattern when source surface is broad.
2. **Tester framing guidance applied with surgical precision**: §6.3.2 leads with "swap is faster, recompute is simpler" (literal: "不要说 recompute 更快"); §6.4.1 leads with K18 invariant (`prepend_request == add_request`); §6.5.3 explicitly notes 16× is sweep-pair (P05), not threshold's single-knob effect. Sets precedent for tester→writer framing-guidance loop.
3. **Strong language-trap callouts × 4**: §6.3.2, §6.3.4, §6.4.5, §6.5.4 each have explicit "don't say X" framing. Interview-level rigor.
4. **Honest framing in §6.5.4**: "1 Pareto point is a model simplification artifact, not vLLM truth" — academic integrity move that prevents readers from over-fitting to demo numbers.
5. **Cross-chapter coherence**: Front-matter cites Ch04 §4.3.2 by section number; closing forward-link to Ch07 with "substrate → ecosystem" framing that signals book architecture.
6. **Scope discipline**: §6.1 is 49 lines, opens with a 5-line Ch04 §4.3.2 quote, identifies 3 policy hot-spots, then lays out the §6.2-§6.5 roadmap — DOES NOT re-derive schedule() loop mechanics. §6.3.5 explicitly states "Ch04 已覆盖，Ch06 只引用回去". This is the v6 scope-discipline payoff.

## Patterns now eligible for v6 baseline propagation (Ch07+)

The following patterns appear in `state.json:v6_compliance.patterns_promoted_to_baseline`:
- **two_tier_mapping** — main table + per-section mini-tables when source surface is broad
- **tester_framing_guidance** — testers apply Knowledge facts as direct narrative-shaping guidance for the writer; reviewer verifies the framing landed
- **language_trap_callouts** — explicit "don't say X" for common misconceptions
- **honest_demo_caveats** — flag what's a model artifact vs vLLM truth

These are NOT new floors (Ch04, Ch05 published without them). They're patterns to reach for when the chapter has the relevant trigger conditions.

## v6 cadence floor (3-chapter aggregate)

Floor thresholds confirmed at N=3:

| Threshold | Floor | Ch04 | Ch05 | Ch06 |
|---|---|---|---|---|
| Source files in `impl-notes.md` | ≥ 5 | 5 | 7 | 6 |
| `# REFERENCE:` comments | ≥ 60 | 65 | 61 | 60 |
| Source mapping table rows | ≥ 10 | 13 | 21 | 29 |
| Tests pass rate | 100% | 48/48 | 74/74 | 97/97 |
| Lint formula blocking | 0 | 0 | 0 | 0 |
| Lint source-grounding | all green | PASS | PASS | PASS |
| Demo numerics in narrative | verbatim | ✓ | ✓ | ✓ |
| 5-step rhythm | every major § | ✓ | ✓ | ✓ |
| Review cycles to APPROVE | 1 (typical) | 1 | 1 | 1 |

## Process / handoff observations from Ch06 pipeline

- Pipeline ran end-to-end with **1 review cycle, no REVISE** — same as Ch04 + Ch05.
- Implementer→tester→writer→reviewer→archivist handoffs each required explicit `SendMessage` (handoff-protocol candidate rule confirmed at N=3-within-instance, but per team-lead's strict reading of CLAUDE.md "2+ repos", still not eligible for `wisdom/architecture.md` promotion — N=3 chapters within ONE repo ≠ 2+ repos).
- **Two scripts/learn.py framework bugs surfaced and captured during this session**:
  1. Append-mode produces malformed double-prefix headings (`## K??: K??:` / `## K??: P??:`). 6 occurrences across scheduler.md (3) + preemption.md (3). Trace at `trace/cross-chapter/learn-py-append-id-bug.md`.
  2. `_parse_module_file` returns `[]`, making `compact()` non-functional. Already captured in `trace/cross-chapter/handoff-protocol-2026-05-06.md` companion candidate.
- scheduler.md compacted twice in this session: first pass K05-K09 (post-Ch04 v5 era; preserved external citations from test code), second pass K10-K14 (after Ch06 work pushed it back over cap). Final state: 12 top-level facts, 10 IDs preserved as `### KXX:` subheadings.
- preemption.md created (new module for Ch06); 5 facts P01-P05; double-prefix labels cleaned post-Ch06.

## Cross-references

- Trace decision: `trace/decisions/2026-05-05_ch05-ch28-directory-remap-to-outline-ids-+-legacy-snapshot.md`
- Ch05 baseline delivery: `trace/deliveries/05-memory-management.md`
- Ch04 baseline delivery: `trace/deliveries/04-continuous-batching.md`
- Auto-record: `trace/deliveries/2026-05-06_ch06-scheduling-ch06-scheduling-v6-published-—-third-v6-baseline-chapter,-ca.md`
- Reviewer status JSON: `/tmp/book-factory/06-scheduling/reviewer-status.json` (also archived in snapshot dir)
- Ch06 implementer brief: `trace/briefs/06-scheduling-implementer-2026-05-06.md`
- Cross-chapter candidate (learn.py append-id bug): `trace/cross-chapter/learn-py-append-id-bug.md`
- Cross-chapter candidate (handoff protocol, N=3 within instance): `trace/cross-chapter/handoff-protocol-2026-05-06.md`
- Knowledge: `knowledge/modules/scheduler.md` (K01-K22, with K05-K09 + K10-K14 compacted), `knowledge/modules/preemption.md` (P01-P05)
