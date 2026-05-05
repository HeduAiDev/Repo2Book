# Implementation Notes — Ch06 Request Scheduling System

## Source Analysis

### 1. What files implement this feature?

| File | Lines | Role |
|------|-------|------|
| `vllm/v1/core/sched/scheduler.py` | 2295 | `Scheduler` — main class. `__init__` (L68), `schedule` (L352, the two-phase algorithm), `_preempt_request` (L952), `_update_after_schedule` (L974), `update_from_output` (L1290), `_enqueue_waiting_request` (L1561), `_select_waiting_queue_for_scheduling` (L1567), `finish_requests` (L1750), `_free_request` (L1813) |
| `vllm/v1/core/sched/request_queue.py` | 209 | `SchedulingPolicy` (Enum: FCFS, PRIORITY). `RequestQueue` (ABC). `FCFSRequestQueue` (deque-based, O(1) append/popleft). `PriorityRequestQueue` (heap-based, O(log n) push/pop) |
| `vllm/v1/core/sched/interface.py` | 245 | `SchedulerInterface` (ABC). `PauseState` (IntEnum: UNPAUSED, PAUSED_NEW, PAUSED_ALL) |
| `vllm/v1/core/sched/output.py` | ~300 | `SchedulerOutput` — single dataclass carrying all decisions for one step (new_reqs, resumed_reqs, running_reqs, preempted_reqs, num_scheduled_tokens) |
| `vllm/v1/core/sched/async_scheduler.py` | ~200 | Async variant — overlaps schedule() with model forward |
| `vllm/v1/core/sched/utils.py` | ~80 | `check_stop()` — stop condition checking; `remove_all()` — bulk removal helper |

### 2. Key classes and their responsibilities

- **`Scheduler`** (`scheduler.py:L67`): The engine's decision-maker. Owns `running: list[Request]` (FIFO scheduled order), `waiting: RequestQueue` (policy-driven queue), `skipped_waiting: RequestQueue` (requests deferred this step). Calls `KVCacheManager.allocate_slots()` for each request; triggers preemption on OOM.

- **`SchedulingPolicy`** (`request_queue.py:L13`): Two values — `FCFS` (first-come-first-served, deque) and `PRIORITY` (min-heap by `(priority, arrival_time)`). Policy is chosen at `Scheduler.__init__` and fixed for the session.

- **`RequestQueue`** (`request_queue.py:L20`): ABC defining `add_request`, `pop_request`, `peek_request`, `prepend_request`, `remove_request`. The ABC lets `Scheduler` treat both queues uniformly.

- **`FCFSRequestQueue`** (`request_queue.py:L75`): Inherits from both `deque[Request]` and `RequestQueue`. `add_request = append` (tail), `pop_request = popleft` (head). O(1) for all queue ops.

- **`PriorityRequestQueue`** (`request_queue.py:L131`): Wraps a list as a min-heap (`heapq`). Requests compared by `Request.__lt__` which orders `(priority, arrival_time, request_id)` — smaller priority value = higher precedence. O(log n) push/pop.

- **`PauseState`** (`interface.py:L22`): `UNPAUSED` (normal), `PAUSED_NEW` (finish running, admit no new), `PAUSED_ALL` (schedule nothing). Used during live weight updates.

- **`SchedulerOutput`** (`output.py`): The batch decision for one step. Contains: `scheduled_new_reqs`, `scheduled_cached_reqs`, `num_scheduled_tokens`, `scheduled_spec_decode_tokens`, `grammar_bitmask`, `free_encoder_mm_hashes`, `structured_output_request_ids`, `preempted_req_ids`, `finished_req_ids`, `total_num_scheduled_tokens`.

### 3. The two-phase scheduling algorithm

```
schedule():
    token_budget = max_num_scheduled_tokens

    # Phase 1: Schedule RUNNING requests (running-first policy)
    for request in self.running:
        if token_budget == 0: break
        num_new = min(request.remaining_tokens, token_budget)
        while True:
            new_blocks = kv_cache_manager.allocate_slots(request, num_new)
            if new_blocks is not None: break
            # OOM — preempt lowest priority request
            if policy == PRIORITY:
                preempted = max(running, key=lambda r: (r.priority, r.arrival_time))
            else:  # FCFS
                preempted = running.pop()  # latest scheduled → preempt it
            _preempt_request(preempted)  # restore KV blocks, set WAITING
            if preempted == request: break  # nothing left to preempt
        if new_blocks is None: break
        scheduled_running_reqs.append(request)
        token_budget -= num_new

    # Phase 2: Admit WAITING requests (only if no preemption happened)
    if not preempted_reqs and _pause_state == UNPAUSED:
        while waiting and token_budget > 0:
            if len(running) == max_num_running_reqs: break
            request = waiting.peek_request()
            # Get cached blocks (prefix cache)
            computed_blocks, num_cached = kv_cache_manager.get_computed_blocks(request)
            num_new = min(request.num_tokens - num_cached, token_budget)
            new_blocks = kv_cache_manager.allocate_slots(request, num_new, ...)
            if new_blocks is None: break  # OOM — stop admission
            request = waiting.pop_request()
            running.append(request)
            token_budget -= num_new
```

Key invariants:
- **Running-first**: Already-scheduled requests get priority over new admissions. This prevents starvation of decode requests by prefill.
- **Preempt-on-OOM**: When `allocate_slots()` returns None, fall back to preemption rather than refusing the request outright.
- **No-mixed-admission**: If any preemption happened this step, skip waiting admission. This prevents thrashing (preempt A to admit B, next step preempt B to admit A).
- **Budget monotone decrease**: Each scheduled token subtracts from `token_budget`. When budget hits 0, stop.

### 4. FCFS vs Priority — the fairness tension

**FCFS** (deque):
- Pros: O(1) ops, no metadata, completely fair in arrival order
- Cons: Head-of-line blocking — a long prompt at the head stalls everyone behind it
- Preemption victim: `self.running.pop()` (last-scheduled → most recent joiner)

**Priority** (min-heap):
- Pros: High-priority requests (SLA, paid tier, interactive) always preferred
- Cons: Starvation — low-priority requests may never run under sustained high-priority load
- Preemption victim: `max(running, key=lambda r: (r.priority, r.arrival_time))` — preempts the one that *would be scheduled last* if the queue were rebuilt
- Starvation mitigation in production: priority aging (not in V1 default), or multi-queue weighted round-robin (not in V1)

Interesting asymmetry: for waiting queue, Priority uses heap order (smallest first); for preemption, Priority uses *reverse* order (largest priority value first — i.e., lowest "actual priority"). This makes sense: we admit the most important waiting request but preempt the least important running one.

### 5. Preemption: swap-out vs recompute vs abort

vLLM V1 implements **recompute-preemption** only:

```python
def _preempt_request(self, request, timestamp):
    # 1. Free all KV blocks (kv_cache_manager.free)
    # 2. Reset request state: RUNNING → PREEMPTED → WAITING
    # 3. Reset num_computed_tokens = 0 (prefix cache may recover it)
    # 4. Add back to waiting queue at HEAD (prepend_request)
    # 5. Increment preemption counter (for metrics)
```

**Why recompute over swap**:
- Swap requires CPU↔GPU copy (PCIe bandwidth ~64 GB/s for Gen4), often slower than recompute on H100
- Prefix cache means recompute isn't truly "from zero" — hit rate on resumed prefix often >80%
- Recompute has zero memory overhead (no CPU buffer pool to manage)

V0 (legacy) had both swap and recompute; V1 deleted swap. See `docs/design/v1/prefix_caching.md` for authors' reasoning.

**Why not abort**: Aborting would break SLA — the user's request would silently fail. vLLM prefers performance degradation (retry) over correctness loss (drop).

### 6. Design decisions and WHY

**Decision 1: Two-phase scheduling (running-first, waiting-second)**

Why: If we interleaved or alternated, each step would need to recompute relative priorities between running and waiting. Running-first gives us a clear rule: already-started work finishes first (decode-friendly). Waiting admission only on spare capacity.

Alternative considered: Strict FCFS across running+waiting. Problem: decodes (cheap, 1 token each) get starved by one slow prefill.

**Decision 2: `max_num_scheduled_tokens` as the single budget**

Instead of separate budgets for prefill/decode/encoder, vLLM uses one unified token budget. Prefill 512 + decode 64 both subtract from the same pool. This is what enables chunked prefill + decode co-scheduling (Ch04).

**Decision 3: Preemption preempts the LAST scheduled (in FCFS) or HIGHEST priority value (in Priority)**

FCFS: `running.pop()` — the request just added is the one that didn't "earn" its slot yet. Earlier requests have been processing longer.

Priority: reverse of admission order — we admit the smallest priority value, so we preempt the largest.

**Decision 4: No-preemption-and-admission in the same step**

Without this rule, you get thrashing: preempt A (OOM), admit B (fits because A freed), next step preempt B (OOM from A resuming), admit A, forever. The rule "if any preemption, skip waiting phase" breaks the cycle.

**Decision 5: `skipped_waiting` queue for deferred requests**

When a waiting request can't be scheduled this step (LoRA full, WAITING_FOR_REMOTE_KVS, etc.), it goes to `skipped_waiting`. Next step, `_select_waiting_queue_for_scheduling()` picks from `skipped_waiting` before `waiting`. This preserves deferred requests' position relative to new arrivals.

### 7. What complexity must our implementation preserve?

| Mechanism | Source | Why Preserve |
|-----------|--------|--------------|
| Two-phase schedule() | `scheduler.py:L352-L945` | The running-first ordering is the core throughput guarantee |
| SchedulingPolicy enum dispatch | `request_queue.py:L13-L18` | FCFS vs Priority is a semantic contract, not just a knob |
| FCFS preemption = running.pop() | `scheduler.py:L504` | Deterministic — no ambiguity about which victim |
| Priority preemption = max(priority, arrival_time) | `scheduler.py:L479-L488` | Inverse of admission order |
| Preempt-and-admission mutual exclusion | `scheduler.py:L568` | Prevents thrashing |
| Token budget monotone | `scheduler.py:L371, L521, L692` | Budget hits zero → stop, no overflow |
| Running-list append order = FIFO | `scheduler.py:L517` | Preserves "first scheduled = first preempted" invariant for FCFS |

### 8. What we simplify (and why it's OK for education)

| Original Complexity | Our Simplification | Justification |
|---------------------|-------------------|---------------|
| Multimodal (encoder_cache_manager, mm_budget) | Not implemented | Encoders are orthogonal to scheduling |
| Speculative decoding (use_eagle, num_spec_tokens, spec_token_ids) | Not implemented | Ch21 content |
| KV Connector (P/D disaggregation, WAITING_FOR_REMOTE_KVS) | Not implemented | Ch22-25 content |
| LoRA (max_loras constraint) | Not implemented | Orthogonal admission constraint |
| Structured output (grammar_bitmask) | Not implemented | Sampling-layer concern |
| Mamba block-aligned split | Not implemented | Architecture-specific |
| Encoder-decoder models | Not implemented | Most LLMs are decoder-only |
| Pipeline parallelism | Not implemented | Multi-GPU topology |
| CUDA Graph capture | Not implemented | Model runner concern |
| PauseState transitions beyond simple check | Only UNPAUSED / PAUSED_ALL | PAUSED_NEW is an operational knob |
| Async scheduler | Only sync variant | Same algorithm, different execution model |
| Routed experts capture | Not implemented | MoE-specific (Ch09, Ch27) |
| KV events, metrics collectors | Simplified stats struct | Observability layer |
| `long_prefill_token_threshold` | Not implemented | Chunked prefill tuning knob (Ch04) |
| `scheduler_reserve_full_isl` admission | Not implemented | Admission control refinement |

## Source Mapping Table

| Our Code | Original Source | What We Changed | Why |
|----------|----------------|-----------------|-----|
| `SchedulingPolicy` | `request_queue.py:L13-L18` | Identical | Core enum, no simplification |
| `RequestQueue` (ABC) | `request_queue.py:L20-L72` | Removed `prepend_requests`, `remove_requests` (bulk variants) | Not used in our minimal path |
| `FCFSRequestQueue` | `request_queue.py:L75-L128` | Kept core ops: `add_request`, `pop_request`, `peek_request`, `prepend_request`, `remove_request` | Full FCFS semantics |
| `PriorityRequestQueue` | `request_queue.py:L131-L199` | Kept heap-based ops | Full Priority semantics |
| `create_request_queue` factory | `request_queue.py:L201-L208` | Identical | Factory dispatch |
| `RequestStatus` enum | `vllm/v1/request.py:L310-L326` | 4 states (WAITING/RUNNING/PREEMPTED/FINISHED) vs original ~8 | Ch04 convention, sufficient for scheduling logic |
| `Request` dataclass | `vllm/v1/request.py:L59-L308` | ~10 fields vs ~60 fields | Removed encoder/spec/LoRA/multimodal/structured-output fields |
| `SchedulerOutput` | `output.py` | ~5 fields vs ~15 fields | Removed encoder/spec/grammar/KV-connector fields |
| `Scheduler.__init__` | `scheduler.py:L67-L300` | ~40 lines vs ~230 lines | Removed VllmConfig/connector/encoder/LoRA/Mamba/PP setup |
| `Scheduler.schedule()` | `scheduler.py:L352-L945` | ~100 lines vs ~600 lines | Kept two-phase + preemption core; removed encoder/spec/LoRA branches |
| `Scheduler._preempt_request()` | `scheduler.py:L952-L972` | ~15 lines | Core preemption logic preserved |
| `Scheduler.update_from_output()` | `scheduler.py:L1290-L1553` | ~40 lines vs ~260 lines | Removed encoder/spec/KV-connector update paths; kept token-append + finish detection |
| `Scheduler.add_request()` | `scheduler.py:L1728-L1748` | Simplified | Core `_enqueue_waiting_request` logic |
| `Scheduler.finish_requests()` | `scheduler.py:L1750-L1811` | Simplified | Abort flow |

## Key formulas

**Remaining tokens to compute**:
```
remaining = num_tokens_total - num_computed_tokens
```

**Token budget constraint**:
```
sum(num_scheduled_tokens[r] for r in scheduled_reqs) <= max_num_scheduled_tokens
```

**FCFS preemption order**:
```
preempt(FCFS) = running.pop()  # last appended = last scheduled
```

**Priority preemption order**:
```
preempt(PRIORITY) = argmax(r in running, key=(r.priority, r.arrival_time))
# Largest priority value + latest arrival = lowest actual priority
```

**Admission gate**:
```
admit iff (len(running) < max_num_running_reqs
          AND token_budget > 0
          AND no preemption happened this step
          AND pause_state == UNPAUSED)
```

## Running Example

Our demos use a pool of 3 requests:
- `r1`: priority=1 (highest), prompt=8 tokens, max_tokens=4
- `r2`: priority=2, prompt=6 tokens, max_tokens=3
- `r3`: priority=3 (lowest), prompt=12 tokens, max_tokens=5

With `max_num_running_reqs=2`, `max_num_scheduled_tokens=16`, and a pool of 4 KV blocks (block_size=8) to force preemption. The demos show:

1. **FCFS admission order**: r1 (t=0), r2 (t=1), r3 (t=2) → running = [r1, r2], r3 waiting
2. **FCFS preemption under pressure**: KV OOM → pop r2 (last scheduled), r2 moves to WAITING head
3. **Priority preemption**: with running = [r3, r2, r1], OOM → preempt r3 (highest priority value = lowest actual priority)
4. **Pareto frontier**: max_num_running_reqs=1 (low latency, low throughput) vs =4 (high throughput, higher tail latency)
