# Implementation Notes — Ch04 Continuous Batching

Strategy A1 rewrite. vLLM source pinned at `98661fe` under
`instances/vllm/source/`. All references below verified against that commit.

## Source Analysis (HARD GATE)

### 1. What vLLM files implement this feature?

| Role | Path | Lines actually consulted |
|------|------|---------------------------|
| Scheduler core | `instances/vllm/source/vllm/v1/core/sched/scheduler.py` | L1–L200 (init), L340–L859 (schedule), L947–L998 (preempt + update_after), L1700–L1834 (add/finish/free) |
| Scheduler output | `instances/vllm/source/vllm/v1/core/sched/output.py` | L175–L233 (SchedulerOutput dataclass) |
| Request queue | `instances/vllm/source/vllm/v1/core/sched/request_queue.py` | L1–L160 (FCFSRequestQueue, ABC, prepend semantics) |
| Request | `instances/vllm/source/vllm/v1/request.py` | L300–L350 (RequestStatus enum, is_finished) |
| KV cache manager | `instances/vllm/source/vllm/v1/core/kv_cache_manager.py` | call sites only — full class deferred to Ch12-13 |

### 2. Key classes and responsibilities

| Class | Owns | Delegates to |
|-------|------|--------------|
| `Scheduler` | `requests` dict, `running` list, `waiting` queue, `finished_req_ids` set, `kv_cache_manager` | KVCacheManager (block alloc/free), FCFSRequestQueue (ordering) |
| `Request` | `prompt_token_ids`, `output_token_ids`, `num_computed_tokens`, `status`, `block_ids`, `num_preemptions` | — |
| `RequestStatus` (IntEnum) | the lifecycle ordering | `is_finished()` static check `status > PREEMPTED` |
| `FCFSRequestQueue` | a `deque[Request]` with five vLLM-named methods | — |
| `SchedulerOutput` | `num_scheduled_tokens` dict, `total_num_scheduled_tokens`, `preempted_req_ids`, `newly_running_req_ids`, `finished_req_ids` | — |
| `SimpleKVCacheManager` | `_free_blocks` list, `block_size`, `num_gpu_blocks` | — |

### 3. Data flow (one schedule() call)

```
schedule() called:
    token_budget = max_num_scheduled_tokens
    kv_cache_manager.new_step_starts()

    Phase 1 — running list, FCFS:
        for req in running:
            num_new = min(req.num_new_tokens, budget, long_prefill_threshold)
            allocate_slots(req, num_new)
                if OOM:
                    pop tail of running -> _preempt_request()
                    refund any budget that req had grabbed this step
                    retry allocation
                if still OOM after preempting self: break Phase 1
            record num_scheduled_tokens[req_id], deduct from budget

    Phase 2 — waiting queue, FCFS (skipped if any preemption happened):
        while waiting and budget > 0 and len(running) < max_num_running_reqs:
            num_new = min(req.num_new_tokens, budget, long_prefill_threshold)
            if not chunked_prefill and num_new > budget: break
            allocate_slots; if OOM: break
            pop from waiting, append to running, record tokens

    Build SchedulerOutput, return.

update_after_step(output):
    for each req: num_computed_tokens += n
    if past prefill: append placeholder output token (mimics sampler)
    if max_tokens hit: mark FINISHED_LENGTH_CAPPED -> _free_request
```

### 4. Design decisions and WHY

**D1. No prefill/decode bifurcation.** scheduler.py:L353-L362 explicitly says:
"There's no 'decoding phase' nor 'prefill phase' in the scheduler." Every
request just has `num_computed_tokens` racing toward `num_tokens`. This is what
makes chunked prefill, prefix caching, and speculative decoding fall out of
the same loop instead of being three separate code paths. Trade-off: callers
must always reason about `num_new_tokens` and stop checks themselves.

**D2. Running-first scheduling (Phase 1 before Phase 2).** scheduler.py:L387 vs
L568. Already-admitted requests have already paid the cost of KV allocation;
giving them priority avoids cascading preemption. Trade-off: a brand-new short
request can wait behind a long-running prefill.

**D3. Preempt the LAST running request, not the requester.** scheduler.py:L504
under FCFS: `preempted_req = self.running.pop()`. The requester is earlier in
the FCFS list, hence higher priority. This is why the running queue is a
*list*, not a queue. Trade-off: must handle the corner case of the requester
itself being the only candidate (L508-L510).

**D4. Skip Phase 2 entirely if any preemption happened.** scheduler.py:L568:
`if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:`. If we
just freed blocks via preemption, those blocks should be saved for the
preempted requests to resume on next step, not handed to a brand-new admit.
Trade-off: throughput dip in the step after preemption, but stability.

**D5. Chunked prefill defaulted ON.** config/scheduler.py:L84
`enable_chunked_prefill: bool = True`. With it OFF (vLLM:L684-L690), Phase 2
*breaks* (not continues) when a prompt won't fit — preserving FCFS strictly,
but at the cost of keeping the GPU idle. With it ON, the prompt is sliced into
budget-sized chunks across steps.

### 5. Complexity preserved (NOT simplified away)

- Two-phase structure with the running-first / waiting-second discipline.
- Token budget that is decremented and asserted non-negative at end.
- FCFS ordering in BOTH the running list and the waiting queue.
- Preemption-with-retry: OOM triggers preemption of the lowest-priority
  running request, *with budget refund*, and retries allocation.
- "Skip Phase 2 if anyone got preempted" rule.
- `num_computed_tokens += n` advance in `_update_after_schedule` (so the next
  schedule call sees the new state and can issue a decode token).
- Long-prefill threshold cap (`long_prefill_token_threshold`).
- Chunked-prefill on/off switch with the correct break vs continue semantics.
- Request lifecycle: WAITING → RUNNING → PREEMPTED → WAITING → … → FINISHED.
- Block free-on-preempt and free-on-finish.

### What we deliberately simplified (each annotated in code)

- KVCacheManager → free-list. Full prefix-caching/copy-on-write/multi-group
  story belongs to Ch12-13.
- Spec decode: `num_tokens_with_spec` collapses to `num_tokens`.
- Encoder/multimodal: dropped entirely (no `_try_schedule_encoder_inputs`,
  no `encoder_compute_budget`).
- Structured output, streaming inputs, async KV transfer, mamba alignment:
  every call site is annotated as "NOT IMPLEMENTED" or "SIMPLIFIED" in the
  source.
- `RequestStatus` trims down to 5 of vLLM's 11 states (the four
  `WAITING_FOR_*` and the four extra `FINISHED_*` aren't reachable from this
  loop).
- `SchedulerOutput` keeps 5 of the 14 vLLM fields.
- Update-after-step collapses `_update_after_schedule` (L974) and
  `update_from_output` (L1290) into one method, with a placeholder zero
  standing in for the sampled token.

## 1:1 Source Mapping

| Our code | vLLM source | What we changed | Why |
|----------|-------------|-----------------|-----|
| `Scheduler.__init__` | `scheduler.py:L67-L176` | constructor takes raw ints, not `VllmConfig` | demo doesn't have a VllmConfig |
| `Scheduler.schedule` | `scheduler.py:L352-L945` | dropped encoder/spec/structured/connector branches | unrelated to continuous batching |
| `schedule` Phase 1 loop | `scheduler.py:L387-L556` | identical preempt-on-OOM control flow | core algorithm preserved |
| `schedule` Phase 2 loop | `scheduler.py:L568-L846` | dropped skipped_waiting/lora/external KV checks | unrelated |
| `Scheduler._preempt_request` | `scheduler.py:L952-L972` | dropped encoder cache free, event log, spec clear | identical core: free → reset → prepend |
| `Scheduler.update_after_step` | `scheduler.py:L974-L998` + `L1290-L1551` | merged + placeholder sampler | Ch20 introduces the real model runner |
| `Scheduler._free_request` | `scheduler.py:L1813-L1834` | dropped connector delay-free, encoder cache | identical core: free blocks, mark finished |
| `Scheduler.add_request` | `scheduler.py:L1728-L1748` | dropped streaming queue + structured output gating | identical core: enqueue waiting |
| `Scheduler.get_request_counts` | `scheduler.py:L1724-L1726` | identical | — |
| `Request` | `request.py:L59-L308` | trimmed to 7 fields | only what scheduler reads |
| `RequestStatus` | `request.py:L310-L333` | 5 states instead of 11 | unused states omitted |
| `FCFSRequestQueue` | `request_queue.py:L75-L128` | composition over inheritance from deque | cleaner |
| `SchedulerOutput` | `output.py:L181-L233` | 5 fields instead of 14 | only what test+narrative use |
| `SimpleKVCacheManager.allocate_slots` | `kv_cache_manager.py:allocate_slots` | free-list, no prefix cache | Ch12-13 has the real one |
| `SimpleKVCacheManager.free` | `kv_cache_manager.py:free` | identical contract | — |
| `static_batching_steps` | — | new (pedagogical) | bubble comparison for narrative |
| `continuous_batching_steps` | — | new (pedagogical) | bubble comparison for narrative |

## Files in this directory

- `__init__.py` — module declaration mirroring vLLM's package layout.
- `request.py` — `Request`, `RequestStatus`. ↔ `vllm/v1/request.py`.
- `output.py` — `SchedulerOutput`. ↔ `vllm/v1/core/sched/output.py`.
- `request_queue.py` — `FCFSRequestQueue`. ↔ `vllm/v1/core/sched/request_queue.py`.
- `kv_cache_manager.py` — `SimpleKVCacheManager`. ↔ `vllm/v1/core/kv_cache_manager.py`.
- `scheduler.py` — `Scheduler`, plus pedagogical static/continuous step counters. ↔ `vllm/v1/core/sched/scheduler.py`.
- `demo.py` — runnable annotated trace.
- `_legacy/` — previous v5 attempt, kept as a reference only.

## Running the demo

```bash
python3 -m instances.vllm.artifacts.04-continuous-batching.implementation.demo
```

Expected output: 19 schedule steps cover three diverse requests, all 200 KV
blocks reclaimed at finish, bubble analysis prints ~20× speedup vs. static.

## Notes for the next implementer / writer

- Phase 2 is *intentionally skipped* when Phase 1 preempts. Don't "fix" this.
- `_preempt_request` does NOT remove the request from `self.running` — the
  caller already popped. Adding `self.running.remove(req)` causes a
  `ValueError`. Source comment is at scheduler.py:L955-L956.
- `update_after_step` advances `num_computed_tokens` BEFORE the placeholder
  output token is appended, so by the time the second step sees the request,
  `is_prefill` flips to False at the right moment.
- The bubble simulator is *not* a drop-in replacement for the scheduler — it
  ignores KV cache, preemption, and admission control. It's a back-of-envelope
  comparison only.
