# Test Report — Ch04 Continuous Batching

**Tester**: tester@book-factory
**Date**: 2026-05-05
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer (Task #3)

## Summary

```
48 passed, 0 failed in 0.05s
```

| Module | Tests | Status |
|---|---|---|
| `test_request.py` | 7 | PASS |
| `test_request_queue.py` | 7 | PASS |
| `test_kv_cache_manager.py` | 9 | PASS |
| `test_scheduler.py` | 25 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/04-continuous-batching
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

## Coverage by behavior class

1. **Output shape**: empty schedule → empty output; `total_num_scheduled_tokens` matches sum.
2. **Arrival-order admission (FCFS waiting queue)**: first-added-first-running, capped by both `max_num_running_reqs` and token budget.
3. **Step-loop progress**: single-request completion, chunked prefill (80 tok prompt at 32 budget → 32+32+16), chunked-prefill-OFF blocks Phase 2 (no jump-ahead), 1-tok/step decode advance, `long_prefill_token_threshold` caps chunk size.
4. **Preemption / eviction**: FCFS pop-tail, post-preempt invariants (PREEMPTED status, num_computed=0, blocks freed, prepended to waiting front, num_preemptions++), Phase 2 skip on preemption, preempted requests eventually resume.
5. **Batch shape invariants**: token budget never overshot, running count bounded, per-request tokens ≤ num_new_tokens, running ∩ waiting = ∅.
6. **Integration**: all blocks reclaimed on drain, FINISHED_LENGTH_CAPPED set correctly, demo workload (400/64/16 prompts → 200 blocks → 10–30 steps) terminates cleanly.
7. **Bubble simulator**: static formula = longest_prompt + longest_output; continuous beats static by >5× on the demo workload.

## Wisdom applied

**W02 (preemption tests must be designed so preemption actually triggers)** — every preempt test uses `blocks=2, block_size=16`, two 16-token prompts that each need exactly 1 block. Both fit initially, then decode growth pushes past block boundaries and triggers OOM → preempt. We assert `preempted_req_ids` is non-empty before claiming "preemption tested".

## Fidelity findings

### F01 — `SchedulerOutput.finished_req_ids` is empty in our impl (severity: low)

vLLM (`scheduler.py:L919-L923`) populates `finished_req_ids` across step boundaries — the model runner uses it on step N to drop KV state for requests that finished during step N-1's `update_from_output`. The vLLM source even has a NOTE at L995-L997:

> We shouldn't do `self.finished_req_ids.clear()` here because it will also affect the scheduler output.

In our merged `update_after_step` (collapsing vLLM's split between `_update_after_schedule` and `update_from_output`), we add freshly-finished IDs to `self.finished_req_ids`, then clear the set at the END of the same call. The next `schedule()` builds `SchedulerOutput.finished_req_ids = set(self.finished_req_ids)` which is empty.

**Ch04 impact**: None. Ch04 has no model runner, no KV-eviction-driven follow-up step. The authoritative finish check is `request.is_finished()`.

**Documented in tests**: `test_finished_req_ids_field_is_simplified` asserts the field is a subset of the actually-finished set (passes today because all-empty is a valid subset; tightens to a real check if a future implementer fixes the carry).

**Recommendation for future chapters that wire a runner**: clear at the START of `schedule()` (matches vLLM's "between steps" semantics), not at the end of `update_after_step`. Or split `update_after_step` back into `_update_after_schedule` (advance) + `update_from_output` (finish + clear-old-finished).

## Next stage

Hand off to Writer (Task #3). Implementation deliverable validated; backpressure gate OPEN.
