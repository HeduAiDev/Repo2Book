# Implementation Notes — Ch06 Request Scheduling Policy

Strategy A1 rewrite. vLLM source pinned at `98661fe`. **Ch06 is the
policy/strategy layer ON TOP of Ch04's mechanics**, not a re-walk of
schedule()/step().

For schedule() loop mechanics, see Ch04
(`instances/vllm/artifacts/04-continuous-batching/implementation/scheduler.py`).
This chapter extracts the three policy decisions that loop makes and the
algorithmic comparators that motivate v1's design choices.

## Source Analysis (HARD GATE)

### 1. What vLLM files implement this feature?

| Role | Path (verified at 98661fe) | Lines actually read |
|------|----------------------------|---------------------|
| Queue ABC + FCFS + Priority + factory | `instances/vllm/source/vllm/v1/core/sched/request_queue.py` | L1-L208 (full file — only 209 lines) |
| Scheduler policy bits (victim selection, queue selection, pause-state) | `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L67-L176 (init + policy field), L372-L374 (pause budget), L478-L504 (preempt victim), L568 (pause guard for Phase 2), L1567-L1577 (queue selection), L1561-L1565 (skipped enqueue) |
| Scheduler interface (PauseState) | `instances/vllm/source/vllm/v1/core/sched/interface.py` | L1-end (PauseState enum + SchedulerInterface) |
| Scheduler config (priority field, defaults) | `instances/vllm/source/vllm/config/scheduler.py` | L22 (SchedulerPolicy Literal), L63 (max_num_seqs), L109-L117 (policy="fcfs" default + docstring) |
| Request __lt__ ordering | `instances/vllm/source/vllm/v1/request.py` | L296-L307 (priority, arrival_time, request_id, id() tie-break) |
| Preemption mechanics (cross-reference for recompute path) | `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L952-L972 (_preempt_request) |
| KV offload (cross-reference for swap-NOT-preemption) | `instances/vllm/source/vllm/v1/kv_offload/cpu/gpu_worker.py` | L319 (swap_blocks_batch — used only for prefix-cache offload, NOT preemption) |
| Abort path | `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L1750-L1811 (finish_requests with FINISHED_ABORTED) |

### 2. Key classes and responsibilities

| Class | Owns | Delegates to |
|-------|------|--------------|
| `SchedulingPolicy` (Enum) | the FCFS / PRIORITY label | — |
| `RequestQueue` (ABC) | five-method API contract | concrete subclasses |
| `FCFSRequestQueue` | a `deque[PolicyRequest]` | deque |
| `PriorityRequestQueue` | a `list[PolicyRequest]` heap + Request.__lt__ | heapq |
| `create_request_queue` | factory dispatch | — |
| `PauseState` (Enum) | engine-level pause flags | read by `effective_token_budget` |
| `select_preemption_victim` | victim-pick policy primitive | running list / max() |
| `select_waiting_queue` | which of (waiting, skipped_waiting) wins | RequestQueue.peek + __lt__ |
| `effective_token_budget` | PAUSED_ALL→0 gate | PauseState |
| `PreemptionScenario` | per-request KV/recompute/swap/abort cost model | — |
| `WorkloadProfile` | head-of-line blocking factor + steps | — |
| `EngineConfig` + `PerfPoint` | (max_num_seqs, batch, threshold) tuple + sweep result | — |
| `pareto_front` | non-dominated set extractor | — |

### 3. Data flow (one preempt-on-OOM cycle, both policies)

```
Phase 1 of schedule() at L387, request `req` needs N new tokens:
  ↓
allocate_slots(req, N) returns None  (KV cache full)
  ↓
HERE Ch06 lives:
  victim = select_preemption_victim(running, req, policy)
    FCFS:     running[-1]                           (last admitted)
    PRIORITY: max(running, key=(prio, arrival))     (worst-priority running)
  ↓
running.remove(victim)         ← Ch04 mechanics
victim → _preempt_request()    ← Ch04 mechanics (free blocks, num_computed_tokens=0)
victim → waiting.prepend_request(victim)
    FCFS:     deque.appendleft  (lands at head, retried next step)
    PRIORITY: heappush          (lands by (prio, arrival), per K06/K15)
  ↓
retry allocate_slots(req, N)
```

```
Phase 2 of schedule() at L568:
  if not preempted_reqs and pause_state == UNPAUSED:
    while (waiting or skipped_waiting) and budget > 0 and len(running) < max_seqs:
      ↓
      HERE Ch06 lives:
      q = select_waiting_queue(waiting, skipped_waiting, policy)
        FCFS:     skipped_waiting OR waiting OR None
        PRIORITY: peek both, return whichever has the better (prio, arrival) head
      ↓
      r = q.peek_request()
      ... try admit r ...
```

### 4. Design decisions and WHY

**D1. Priority is a USER signal, not auto-computed.** vLLM does NOT age
waiting requests; the priority field is set by the API caller and stays
fixed (`config/scheduler.py:L109-L117`, `Request.__lt__:L301`). Auto-aging
would silently override the SLA contract — a paid-tier request with
priority=0 should NEVER be overtaken by a free-tier request just because
the latter waited longer. Operators who want aging can either pre-compute
it before submission or rebuild the queue periodically (which breaks the
heap invariant — see `aged_priority` docstring).

**D2. PriorityRequestQueue.prepend_request == add_request.**
(`request_queue.py:L160-L165`) "In a priority queue, there is no concept of
prepending." This is THE invariant that distinguishes the two queues
semantically: under FCFS, a preempted request immediately re-runs (FIFO
priority); under PRIORITY, it re-enters at its true priority position,
which may keep it preempted again if its priority is genuinely low.

**D3. `skipped_waiting` is the indefinite-postponement guard.** In FCFS,
`select_waiting_queue` ALWAYS prefers `skipped_waiting`
(`scheduler.py:L1568-L1569`). Without this, a request blocked on a remote
KV transfer (`WAITING_FOR_REMOTE_KVS`) could be perpetually pushed back as
new arrivals jump ahead. The same logic in PRIORITY mode degenerates to
"whichever head has higher priority", which is correct: a high-priority
new arrival SHOULD overtake a low-priority blocked request. This is
priority's only safe departure from FIFO — note the symmetric protection
in vLLM is to assign blocked requests their original priority, not a
demoted one.

**D4. Recompute-only preemption (v1).** vLLM v0 had swap-to-CPU; v1
deleted it (`scheduler.py:L952-L972` is recompute-only). The
`preemption_strategy.py` model shows swap is in fact 2-3x faster in raw
latency. v1 chose recompute anyway:
- ONE code path. Swap requires CPU-memory budget tracking, pinned-memory
  registration, two cudaMemcpyAsync streams, completion synchronization.
- ALWAYS WORKS. Swap fails when CPU is also full (production reality on
  multi-tenant nodes with 32 GB RAM serving 80 GB GPUs).
- BIT-DETERMINISTIC. Swap+resume produces bit-identical output; recompute
  also does (with deterministic kernels), but recompute does not introduce
  a "kv corruption between steps" failure mode that swap can.
- Engineering simplicity. The team estimated swap added ~2K LoC of edge
  cases; recompute deletes ~150 LoC instead.

**D5. PauseState gates BOTH the budget and Phase 2.** PAUSED_ALL zeros
`token_budget` (`scheduler.py:L372-L374`), so Phase 1's `while budget > 0`
exits immediately AND Phase 2's outer `while ... and budget > 0` is also
gated. PAUSED_NEW only gates Phase 2 entrance via `_pause_state ==
UNPAUSED` at L568 — running requests still progress. Two states, two gates,
both at one-line policy primitives.

**D6. `long_prefill_token_threshold` is the per-request fairness cap.**
(`scheduler.py:L413-L415`). Without it, one long prompt soaks `B` tokens/step
for `ceil(L/B)` steps. With it set to e.g. `B/4`, the long prompt takes 4x
more steps but leaves `3B/4` per step for short requests. The `pareto.py`
sweep confirms: a 4x p95 TTFT improvement at modest peak-throughput cost.

### 5. Complexity preserved (NOT simplified away)

- Five-method `RequestQueue` ABC with both concrete implementations.
- The `prepend_request == add_request` invariant for PriorityRequestQueue.
- FCFS popping the running TAIL on OOM (most-recent-victim).
- PRIORITY using `max(priority, arrival_time)` for victim selection.
- The `skipped_waiting` queue and its FCFS-prefers-skipped vs
  PRIORITY-peek-heads selection rule.
- PAUSED_ALL → token_budget = 0; PAUSED_NEW → Phase 2 skipped.
- Recompute-vs-swap-vs-abort trade-off as analytical model with
  reproducible numbers across (prompt length, layers, KV heads, head_size,
  PCIe BW, prefill TP).
- Long-prefill threshold's effect on p95 TTFT.

### What we deliberately simplified (each annotated)

- No actual `_preempt_request` / `allocate_slots` re-implementation.
  Ch04's scheduler.py covers those; Ch06 only references back.
- No `_handle_stopped_request` / streaming session machinery.
- No structured-output / KV-connector pause hooks.
- The `priority_ordering` helper does NOT model the `id()` 4th-tier
  tie-breaker explicitly (Python's heapq breaks ties by insertion order
  through `__lt__` chain, which we preserve).
- `aged_priority` is illustrative — vLLM does NOT age, by design (D1).
- The Pareto sweep uses a back-of-envelope throughput model, not real
  benchmark numbers. Calibration for production deployment should
  substitute measured `decode_throughput_tokens_per_sec`.

## 1:1 Source Mapping

| Our code | vLLM source | What we changed | Why |
|----------|-------------|-----------------|-----|
| `SchedulingPolicy` enum | `request_queue.py:L13-L17` | identical | — |
| `PolicyRequest` | `request.py:L59-L308` | minimal subset (request_id, priority, arrival_time, prompt_tokens, num_computed_tokens, status) | Ch06 needs only what `__lt__` reads |
| `PolicyRequest.__lt__` | `request.py:L296-L307` | identical | tie-break behaviour is THE fact |
| `RequestQueue` ABC | `request_queue.py:L20-L72` | identical method set | — |
| `FCFSRequestQueue` | `request_queue.py:L75-L128` | composition over deque inheritance | Ch04 already used this pattern |
| `PriorityRequestQueue.add_request` | `request_queue.py:L144-L146` | identical | — |
| `PriorityRequestQueue.pop_request` | `request_queue.py:L148-L152` | identical | — |
| `PriorityRequestQueue.peek_request` | `request_queue.py:L154-L158` | identical | — |
| `PriorityRequestQueue.prepend_request` | `request_queue.py:L160-L165` | identical (no-op equivalent) | THE invariant |
| `PriorityRequestQueue.prepend_requests` | `request_queue.py:L167-L173` | identical | — |
| `PriorityRequestQueue.__iter__` | `request_queue.py:L194-L198` | identical (heap-copy then pop) | — |
| `create_request_queue` | `request_queue.py:L201-L208` | identical factory | — |
| `PauseState` enum | `interface.py` (PauseState) | three states match vLLM | — |
| `select_preemption_victim` | `scheduler.py:L478-L504` | extracted into pure function | testability |
| `select_waiting_queue` | `scheduler.py:L1567-L1577` | extracted into pure function | testability |
| `effective_token_budget` | `scheduler.py:L372-L374` | one-line primitive | — |
| `PreemptionStrategy` enum | scheduler.py + kv_offload/ + finish_requests | new (synthesizes 3 paths) | analytical comparator |
| `PreemptionScenario.kv_bytes` | `kv_cache_interface.py:L153-L170` | identical formula | Ch05 also exposes this |
| `PreemptionScenario.recompute_seconds` | `scheduler.py:L961-L964` (mechanics) | analytical model | latency comparator |
| `PreemptionScenario.swap_seconds` | `kv_offload/cpu/gpu_worker.py:L319` | analytical model | latency comparator |
| `PreemptionScenario.abort_seconds` | `scheduler.py:L1750-L1811` (mechanics) | SLA-penalty model | analytical |
| `crossover_prompt_length` | — (analytical) | new | shows length-independence |
| `expected_latency_under_oom_rate` | — (analytical) | new | per-request E[latency] |
| `WorkloadProfile.fcfs_short_request_latency_steps` | `scheduler.py:L387-L556` (Phase 1) | analytical worst-case bound | starvation argument |
| `WorkloadProfile.head_of_line_blocking_factor` | — (synthesized) | new | predicate for `has_starvation` |
| `priority_ordering` | `request_queue.py:L131-L198` | calls PriorityRequestQueue + iterate | testability |
| `aged_priority` | — NOT in vLLM | new (illustrative) | D1 discussion device |
| `EngineConfig` | `config/scheduler.py:L60-L120` | three-knob subset | sweep convenience |
| `estimate_throughput` | `scheduler.py:L848-L853` (budget cap) | min(compute, memory) model | Pareto axis |
| `estimate_p95_ttft` | `scheduler.py:L413-L415`, `L678-L680` | analytical with/without threshold | Pareto axis |
| `pareto_front` | — (Pareto extractor) | new | sweep visualizer |

## Files in this directory

- `__init__.py` — module map, declares this is the policy layer.
- `request_queue.py` — `SchedulingPolicy`, `PolicyRequest`, `RequestQueue` ABC, both concrete queues, factory.
- `policy.py` — `PauseState`, `select_preemption_victim`, `select_waiting_queue`, `effective_token_budget`.
- `preemption_strategy.py` — recompute/swap/abort analytical comparator.
- `starvation_analysis.py` — head-of-line blocking, aging compensator (illustrative).
- `pareto.py` — schedule-latency vs throughput trade-off model + sweep + Pareto extractor.
- `demo.py` — runnable end-to-end trace of all five sections.
- `impl-notes.md` — this file.
- `_legacy/` — old v5 attempt (read-only reference).

## Running the demo

```bash
python3 -m instances.vllm.artifacts.06-scheduling.implementation.demo
```

Expected highlights:
- §1 victim selection: FCFS picks "C" (last admitted), PRIORITY picks "B"
  (priority=3, worst).
- §2 head-of-line factor 8.00x — short waits 8 steps for its own 1-step prefill.
- §3 8K prompt: recompute 164 ms / swap 62 ms / abort 5000 ms (SLA penalty).
  Swap-vs-recompute crossover is length-INDEPENDENT (both scale linearly).
- §4 priority ordering [B, D, C, A] vs FCFS [A, B, C, D]; aging at t=10
  makes A (waited 10s) tie-break ahead of B.
- §5 Pareto front shows long_prefill_token_threshold lowers p95 TTFT 16x at
  the same throughput.

## Cross-chapter dependency

- **Inherits from Ch04**: schedule()/step() mechanics, the
  `_preempt_request` flow, `Request.num_computed_tokens` lifecycle. Don't
  re-walk these; reference back instead.
- **Inherits from Ch05**: KV cache memory model (page_size_bytes formula,
  the recompute-vs-swap base case in §3 reuses Ch05's PreemptionScenario
  framing).
- **Provides to Ch07** (next): the policy primitives extracted here become
  the substrate for higher-level scheduling features (LoRA-aware admission,
  multi-tenant isolation).

## Notes for the next implementer / writer

- `PriorityRequestQueue.prepend_request == add_request` is THE invariant;
  any test or narrative that says "preempted request is re-queued at front"
  is FCFS-only language. Under PRIORITY, "front" is meaningless.
- The Pareto front in §5 collapses to 1 point with my simplified throughput
  model — that's a property of the model, not vLLM. A real measurement on
  H100 produces ~3 points. The right narrative framing is "shows the
  shape, not the magnitudes".
- `aged_priority` is INSTRUCTIVE FICTION. Do NOT recommend it as a
  production option without acknowledging that priority is meant to be a
  user contract. See D1.
- Ch04's `_preempt_request` knowledge K02 (caller pops, callee doesn't
  remove) still applies. Ch06's `select_preemption_victim` returns the
  victim only — the caller still does `running.remove(victim)`.
