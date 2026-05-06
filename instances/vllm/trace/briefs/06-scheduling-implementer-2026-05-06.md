# Rehydration Brief — Ch06 Scheduling (Implementer)

- **Chapter**: `06-scheduling` (slug-aligned ID; legacy ID was `07-scheduling`)
- **Title**: 请求调度系统 — 策略、优先级与公平性
- **Outline level**: core (Part 2)
- **Status**: `needs_rewrite` — full v6-grade rewrite per Ch04/Ch05 baseline (now confirmed at N=2)
- **Dependencies**: `04-continuous-batching` (Scheduler.schedule() mechanics — DO NOT re-derive); `05-memory-management` (KVCacheManager, watermark, eviction — DO NOT re-derive)
- **Dependents downstream**: `12-kv-offload` (preemption swap path)
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/`
- **Brief generated**: 2026-05-06 by archivist
- **Recipient**: book-editor-2 → forward to implementer-2 with full context

---

## 1. Project state context

- Part 1 complete (Ch01-Ch04 published). Part 2 underway: Ch05 published_v6 yesterday; Ch06 is the next.
- v6 cadence is now confirmed at N=2 (Ch04, Ch05). The baseline is reproducible — Ch06 should match or exceed it without surprises.
- **Handoff protocol candidate** (held at candidate per team-lead — not yet wisdom-grade because CLAUDE.md requires N=2 across REPOS, not chapters): every pipeline stage MUST `SendMessage` next-stage on completion (`TaskUpdate` alone is bookkeeping, not a wake signal). Implementer-2 must SendMessage tester-2 explicitly when impl is ready. team-lead is operationally enforcing this in *this* book regardless of formal promotion. See `trace/cross-chapter/handoff-protocol-2026-05-06.md`.

## 2. Chapter scope (what Ch06 actually covers — and what it does NOT)

**Core question**: when multiple requests compete for limited GPU resources, *who goes first, who gets evicted, and how is fairness preserved*?

**The spine of the chapter** (legacy intro lines 1-25, written from the right angle):
1. `vllm/v1/core/sched/request_queue.py:L13-L17` — `SchedulingPolicy` Enum has only TWO values: `FCFS` and `PRIORITY`. The whole chapter follows from this binary choice.
2. **FCFS world** (`request_queue.py:L75 FCFSRequestQueue` extends `deque`): O(1) add/pop/prepend. Preempt-tail rule (covered in Ch04 — REFERENCE BACK, do NOT re-derive).
3. **Priority world** (`request_queue.py:L131 PriorityRequestQueue` uses `heapq` keyed on `(priority, arrival_time)`): O(log n). Tiebreaker on arrival_time prevents non-determinism.
4. **Preemption mechanics**: swap-out (legacy/v0) vs recompute (vLLM-v1 default) vs abort. `scheduler.py:L478-L482` is where priority mode chooses the lowest-priority request for preemption (NOT the tail).
5. **Starvation analysis**: vLLM has NO built-in anti-starvation in priority mode. A continuous stream of high-priority requests can indefinitely starve low-priority ones. Why is this OK in practice? (Workload assumption.)
6. **Pareto frontier**: latency-throughput trade-offs across policies, with measured numbers from a small demo workload.

**CRITICAL — what is OUT of scope** (do NOT re-cover; reference back to prior chapters):
- `Scheduler.schedule()` two-phase mechanics (Phase 1 running, Phase 2 waiting) → fully covered in Ch04 §4.2-4.3. Reference, don't re-walk.
- Token budget arithmetic, bubble math → Ch04 §4.1.
- KVCacheManager allocate_slots / watermark eviction → Ch05 §5.4-5.5.
- Memory profiling / KV byte budget → Ch05 §5.1-5.3.

If implementer-2 finds themselves explaining `schedule()` from scratch — STOP. The chapter has gone wrong. Reference and move on.

## 3. Source modules to reference (commit 98661fe verified)

| File | Lines | What |
|---|---|---|
| `instances/vllm/source/vllm/v1/core/sched/request_queue.py` | L13-L17 | `SchedulingPolicy` Enum (the whole chapter pivots on this) |
| `instances/vllm/source/vllm/v1/core/sched/request_queue.py` | L20-L74 | `RequestQueue` ABC |
| `instances/vllm/source/vllm/v1/core/sched/request_queue.py` | L75-L130 | `FCFSRequestQueue(deque)` — O(1) operations |
| `instances/vllm/source/vllm/v1/core/sched/request_queue.py` | L131-L200 | `PriorityRequestQueue(heapq)` — keyed on `(priority, arrival_time)` |
| `instances/vllm/source/vllm/v1/core/sched/request_queue.py` | L201+ | `create_request_queue(policy)` factory |
| `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L159-L169 | Policy initialization in `Scheduler.__init__` (reference scheduler.md K08 for full line ranges) |
| `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L478-L482 | Preempt-lowest-priority logic (priority mode) |
| `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L568-L569 | `step_skipped_waiting = create_request_queue(self.policy)` — policy applied per-step |
| `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L1568+ | Conditional FCFS branch (verify in source) |
| `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L952-L972 | `_preempt_request()` — invoked under both policies |
| `instances/vllm/source/vllm/v1/core/sched/output.py` | L181-L258 | `SchedulerOutput` (already covered in Ch04; reference for completeness) |
| `instances/vllm/source/vllm/v1/core/sched/utils.py` | full | Helper functions (small; review for relevance) |

**Implementer-2 MUST verify these line numbers** against `git rev-parse HEAD` in `instances/vllm/source/` matching `98661fe` before writing references. Use `scheduler.md K08` as the authoritative line-range table for `scheduler.py` (K08 was just compacted but ID-anchored — grep `### K08:` in the module file).

## 4. Knowledge module recommendation

- **Extend `knowledge/modules/scheduler.md`** (currently 12/15 facts post-2026-05-06 compaction; can absorb 3 more before next cap event). Ch06 is naturally on the same source surface as Ch04, so co-located facts make sense.
- **OR create `knowledge/modules/preemption.md`** if preemption logic and policies deserve a separate module (debatable — argument for: Ch12 kv-offload depends on it, so a separate module would be reusable; argument against: preemption is intertwined with scheduling and a separate module risks duplication).
- Recommendation: **extend `scheduler.md`** for Ch06; if preemption-specific facts pile up, split into `preemption.md` at Ch12 time.
- Update `knowledge/INDEX.md` if a new module is created.

## 5. Strengths to preserve from Ch04 + Ch05 (the v6 baseline)

These are confirmed precedents — replicate them in Ch06:

1. **Demo with reproducible numerics** (Ch04: 416/20/20.80x; Ch05: 35148 blocks/275 concurrent/163.8ms recompute). For Ch06, propose a workload like 10 requests across two priority levels: measure tail latency, mean wait time, throughput under FCFS vs PRIORITY. Numbers must appear verbatim in narrative.
2. **5-step rhythm per major section** (Source Trail → Bridge → Theory Deep Dive → Implementation → Source Diff). Ch04 had 7 sections; Ch05 had 7. Ch06 natural sections from outline: §6.1 the two-policy choice, §6.2 FCFS deque mechanics, §6.3 PriorityQueue heapq mechanics, §6.4 preemption modes (swap/recompute/abort), §6.5 starvation analysis, §6.6 Pareto trade-offs.
3. **Source mapping table density** (Ch04: 13; Ch05: 21). Ch06 should aim for ≥10 (preferably ≥20 like Ch05). Source surface is narrower than Ch05 — don't inflate.
4. **`# REFERENCE:` saturated** (Ch04: 65 across 6; Ch05: 61 across 7). Target ≥60.
5. **Engineering-rationale call-outs** at interview-question level (Ch05 had deque-vs-doubly-linked, null block, free-doesn't-clear-cache). Ch06 candidates: heapq-vs-sortedcontainers (raised in legacy intro), tiebreaker-on-arrival-time-prevents-non-determinism, why-no-built-in-anti-starvation.
6. **Version-history disambiguation** (Ch05 v0/v1 swap). Ch06 candidate: vLLM v0 had `SwapOutPolicy`; v1 defaults to recompute. Mention to prevent reader confusion with older blog posts / Kwon et al. paper.
7. **Cross-chapter forward/back pointers** (Ch04 F01 to Ch20; Ch05 to Ch20 + Ch06). Ch06 should: (a) back-pointer to Ch04 §4.3 for `schedule()` mechanics, (b) back-pointer to Ch05 §5.4-5.5 for kv_cache eviction, (c) forward-pointer to Ch12 for offload-style preemption (kv-offload chapter).
8. **Diagrams optional when demo + tables suffice** (Ch04, Ch05 both shipped without). Ch06 candidate diagrams: priority-queue-vs-deque ops timeline, starvation-scenario timeline. If a numbered demo replay or ASCII trade-off matrix carries the load, no figure needed.
9. **Chinese narrative + math-block split** (per K16): never put Chinese inside `$$` blocks.

## 6. Implementation skeleton hints

The implementer's `implementation/` for Ch06 likely needs:
- `simple_request_queue.py` — minimal `FCFSRequestQueue(deque)` + `PriorityRequestQueue(heapq)` matching the source 1:1. Both share an ABC.
- `simple_preemption.py` — three preemption modes (`recompute`, `swap`, `abort`). Recompute is the default; the other two are reference-only. Ties into Ch05's KVCacheManager via the `free_blocks` API (do NOT re-implement that — import it from Ch05's impl, or use a thin SimpleKVCacheManager stub that mirrors the Ch05 interface).
- `simple_scheduler_policy.py` — a thin policy-aware shim around Ch04's Scheduler that switches queue type and preemption choice based on `SchedulingPolicy`. Imports `Scheduler` core from Ch04 impl and parametrizes it.
- `demo.py` — runs both policies on the same workload, prints policy comparison: tail latency, mean wait, throughput. Should EXACTLY match numbers in narrative §6.5-6.6.
- `impl-notes.md` — list ≥ 5 source files (request_queue.py, scheduler.py, output.py, utils.py, plus one of: kv_cache_manager.py for preemption-touching-cache OR config/scheduler.py for policy config).

The legacy `implementation/_legacy/scheduling_policies.py` (281 lines) is reference-only — review the structure (especially how policy and queue are bound) but rewrite from current source under v6 standards. Legacy chapter intro at `narrative/_legacy/chapter.md:L1-L70` is well-pitched and worth re-reading for tone.

## 7. Process / handoff requirements (candidate rule, operationally enforced)

Per `trace/cross-chapter/handoff-protocol-2026-05-06.md` (held at candidate; team-lead operationally enforces):
1. Implementer-2 MUST SendMessage tester-2 when implementation is ready (with paths, gate checks, ETA).
2. Tester-2 MUST SendMessage writer-2 with test-pass evidence (`X/X tests pass in Ys`).
3. Writer-2 MUST SendMessage reviewer-2 when narrative is ready (line count, lint status if pre-checked).
4. Reviewer-2 MUST SendMessage archivist-2 with verdict + status-JSON path on APPROVE.
5. Each stage's `TaskUpdate` → completed BEFORE the SendMessage, so the next agent sees consistent task state.
6. book-editor-2: do NOT rely on summary-text "[to X] do Y" — that's invisible to teammates. Use SendMessage.

If you (implementer-2) finish but tester-2 doesn't ack within ~10min, ping team-lead — that's the stall pattern this candidate rule was written to prevent.

## 8. Knowledge / wisdom queries before starting

```
# Knowledge — Ch06 builds on scheduler module
cat knowledge/modules/scheduler.md
  # Especially: K01 (schedule entry), K02 (preempt-pop semantics),
  # K05 (chunked prefill), K06 (FCFS pop-tail), K11 (continue-not-break Phase 1),
  # K10 (allocate_slots signature)

# Knowledge — Ch05 leaves K-facts in memory.md (newly created)
cat knowledge/modules/memory.md  # if it exists from Ch05 AFTER WORK extract

# Wisdom for implementer (priority: debugging > architecture > testing > writing)
cat wisdom/debugging.md   # any heap/deque gotchas
cat wisdom/architecture.md  # W04 backpressure, W08 lateral, W12 pipeline
cat instances/vllm/trace/cross-chapter/handoff-protocol-2026-05-06.md  # candidate handoff rule, operationally enforced
cat wisdom/testing.md     # W02 preemption test design (must apply for §6.4 tests)
```

## 9. Open questions for book-editor / team-lead

1. **Knowledge module split**: extend `scheduler.md` or create `preemption.md`? (Recommendation: extend; split at Ch12 time if preemption facts pile up.)
2. **Demo workload**: how complex? Ch04 used 3 requests (P=400/64/16), Ch05 used a tiny model. For Ch06 starvation demo, a stream of 10+ requests across 2-3 priority levels is enough to show the effect. Open question: should the demo include a "what-if-anti-starvation-was-added" comparison? (Probably not — keeps scope tight.)
3. **Topology**: default `linear` should suffice — Ch06 is conceptually parallel to Ch04 and the implementer can lean on Ch04's impl. Reconsider if implementer hits unknowns in priority-mode preemption.
4. **Section-count discipline**: Ch04 had 7 sections, Ch05 had 7. Ch06 outline lists 5 subsections, mapping cleanly to §6.1-§6.6. Don't pad to hit 7 — 5-6 is fine if each is dense.

## 10. Deliverables checklist for implementer-2

- [ ] `implementation/simple_request_queue.py` with ≥ 15 `# REFERENCE:` comments
- [ ] `implementation/simple_preemption.py` with ≥ 15 `# REFERENCE:` comments
- [ ] `implementation/simple_scheduler_policy.py` with ≥ 15 `# REFERENCE:` comments (parametrize Ch04 Scheduler, do NOT duplicate)
- [ ] `implementation/demo.py` with two-policy comparison (FCFS vs PRIORITY)
- [ ] `implementation/impl-notes.md` listing ≥ 5 source files
- [ ] Knowledge update: 3+ facts appended to `scheduler.md` (or new `preemption.md`)
- [ ] On completion: `TaskUpdate #11 → completed` AND **explicit `SendMessage` to tester-2** (handoff-protocol candidate rule, operationally enforced)

## 11. Tasks expected (book-editor-2 should create on kickoff)

Following the Ch04/Ch05 pattern:
- #11 Ch06 implementer: rewrite scheduling-policy implementation
- #12 Ch06 tester: validate scheduling-policy implementation
- #13 Ch06 writer: produce narrative chapter.md (v6 standard)
- #14 Ch06 reviewer: gate APPROVE/REVISE on chapter.md
- #15 Ch06 archivist: record delivery and update state

---

**This brief should be forwarded by book-editor-2 to implementer-2 with the full body. Reviewer-2 and writer-2 may also benefit from skimming §2 (scope boundaries — what NOT to re-cover) and §5 (strengths-to-preserve catalog).**
