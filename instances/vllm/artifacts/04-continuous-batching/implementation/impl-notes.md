# Implementation Notes — Ch04 Continuous Batching

## Source Analysis

### 1. What files implement this feature?

The continuous batching scheduler spans ~2300 lines in vLLM's V1 engine. The core scheduling logic is in `scheduler.py`, with supporting infrastructure in the request queue, KV cache manager, and request data structure.

| File | Lines | Role |
|------|-------|------|
| `vllm/v1/core/sched/scheduler.py` | L67-L300 | `Scheduler.__init__()` — constructor, KV cache config, scheduling constraints, encoder setup |
| `vllm/v1/core/sched/scheduler.py` | L352-L950 | `schedule()` — **core algorithm**: running-first scheduling, waiting admission, KV cache allocation, preemption, output construction |
| `vllm/v1/core/sched/scheduler.py` | L952-L972 | `_preempt_request()` — evict a running request back to waiting |
| `vllm/v1/core/sched/scheduler.py` | L974-L998 | `_update_after_schedule()` — advance `num_computed_tokens` post-schedule |
| `vllm/v1/core/sched/scheduler.py` | L1290-L1500+ | `update_from_output()` — process model output, detect stops, update request states |
| `vllm/v1/core/sched/request_queue.py` | full | `RequestQueue`, `SchedulingPolicy` (FCFS/PRIORITY) — waiting request queue with priority ordering |
| `vllm/v1/request.py` | L59-L308 | `Request` dataclass — ~20 fields tracking token progress, status, multimodal inputs |
| `vllm/v1/request.py` | L310-L352 | `RequestStatus` enum — 11 status values (WAITING through FINISHED_REPETITION) |
| `vllm/v1/core/kv_cache_manager.py` | full | `KVCacheManager` — block-based KV cache allocation, `allocate_slots()`, `free()`, `get_computed_blocks()` |
| `vllm/v1/core/sched/output.py` | full | `SchedulerOutput`, `NewRequestData`, `CachedRequestData` — output data structures |
| `vllm/config/scheduler.py` | L56-L84 | `SchedulerConfig` — `max_num_scheduled_tokens`, `max_num_seqs`, `policy`, chunked prefill thresholds |

### 2. Key classes and responsibilities

- **Scheduler** (`scheduler.py:L67`): Central orchestrator. Maintains `self.waiting` (priority queue), `self.running` (list), `self.finished_req_ids` (set). The `schedule()` method is called once per model forward step and decides which requests get how many tokens.

- **RequestQueue** (`request_queue.py`): Priority-ordered queue for waiting requests. Supports `SchedulingPolicy.FCFS` (first-come-first-served) and `SchedulingPolicy.PRIORITY` (priority-based). Methods: `peek_request()`, `pop_request()`, `prepend_request()`.

- **KVCacheManager** (`kv_cache_manager.py`): Manages KV cache blocks using PagedAttention. Key method `allocate_slots(request, num_new_tokens)` returns `KVCacheBlocks` or `None` (OOM). Also handles prefix caching via `get_computed_blocks()`.

- **Request** (`request.py:L59`): Per-request state machine. Key fields: `num_computed_tokens` (tokens fed to model), `num_tokens` (total tokens including prompt + output), `status` (RequestStatus), `priority`, `arrival_time`.

- **SchedulingPolicy** (`request_queue.py`): Enum controlling which waiting request goes next. FCFS uses arrival order; PRIORITY uses `(priority, arrival_time, request_id)` tuple ordering.

### 3. Data flow (one scheduling step)

```
┌─────────────────────────────────────────────────────────────────┐
│ schedule() called once per model forward step                    │
│                                                                  │
│  1. For each RUNNING request:                                   │
│     ├─ if finished → free KV cache → FINISHED                    │
│     ├─ compute num_new_tokens (catch-up to num_tokens)           │
│     ├─ allocate KV blocks (retry + preempt if OOM)               │
│     └─ if allocated → advance num_computed_tokens, deduct budget │
│                                                                  │
│  2. For each WAITING request (while budget > 0):                 │
│     ├─ compute num_new_tokens (prompt + any pre-generated output)│
│     ├─ allocate KV blocks                                        │
│     ├─ if allocated → promote to RUNNING, deduct budget          │
│     └─ if OOM → break (don't preempt for new requests)           │
│                                                                  │
│  3. Build SchedulerOutput → pass to ModelRunner                  │
│                                                                  │
│  4. ModelRunner.forward() → sampled tokens, logprobs             │
│                                                                  │
│  5. update_from_output():                                        │
│     ├─ append new token IDs to request                           │
│     ├─ check stop conditions (max_tokens, EOS, structured output)│
│     └─ if stopped → mark FINISHED, free resources                │
└─────────────────────────────────────────────────────────────────┘
```

In our educational reimplementation, steps 3-5 are simplified into Phase 3 (simulate 1 token per eligible request per step).

### 4. Design decisions (from source with line references)

1. **Running-first scheduling** (`scheduler.py:L387`): RUNNING requests are always scheduled before WAITING requests. This maximizes continuity (fewer preemptions) and ensures that already-admitted requests make steady progress. The comment at L389 says "First, schedule the RUNNING requests."

2. **Token budget** (`scheduler.py:L371`): `max_num_scheduled_tokens` caps tokens per step. This prevents a single long-prompt request from monopolizing the entire step. Multiplied by model size, this determines the maximum memory for intermediate activations.

3. **Priority-based preemption** (`scheduler.py:L479-L484`): When KV cache is full during a running request's allocation, the lowest-priority running request is evicted. In PRIORITY mode: `max(running, key=lambda r: (r.priority, r.arrival_time))`. In FCFS mode: the last request in the running list is popped (L504: `self.running.pop()`).

4. **No prefill/decoding phase distinction** (`scheduler.py:L353-L362`): There is no explicit "prefill phase" or "decoding phase." Each request simply has `num_computed_tokens` catching up to `num_tokens_with_spec`. This unified model naturally handles chunked prefill (limit tokens per step), prefix caching (restore computed tokens from cache), and speculative decoding (extra tokens in `spec_token_ids`).

5. **Chunked prefill** (`scheduler.py:L413-L414`, `L678-L690`): If `long_prefill_token_threshold` is set and `num_new_tokens` exceeds it, the allocation is capped at the threshold. This breaks long prompts into multiple forward passes, interleaving them with decode tokens for better latency.

6. **Preempted-request requeue** (`scheduler.py:L972`): Preempted requests are prepended to the waiting queue (`prepend_request`), not appended. This ensures they get priority when rescheduling — a fairness mechanism.

### 5. Complexity preserved in our implementation

| Mechanism | Original Location | Why Preserved |
|-----------|------------------|---------------|
| Running-first scheduling | `scheduler.py:L387` | Core of continuous batching — running requests never starved |
| Token budget | `scheduler.py:L371` | Required for fair scheduling; prevents monopolization |
| KV cache block allocation + OOM preemption | `scheduler.py:L466-L510` | Demonstrates the resource-constrained scheduling problem |
| Priority-based victim selection | `scheduler.py:L479-L483` | Shows how preemption chooses victims (max priority number = lowest priority) |
| Request state machine (WAITING->RUNNING->PREEMPTED->FINISHED) | `request.py:L310-L326` | The backbone that all scheduling logic operates on |
| Preemption retry loop | `scheduler.py:L466-L510` | After preempting, re-attempt allocation for the current request |
| Prepend-on-preempt | `scheduler.py:L972` | Preempted requests get priority when resumed (fairness) |

## Source Mapping Table

| Our Code | Original Source | What We Changed | Why |
|----------|----------------|-----------------|-----|
| `ContinuousBatchingScheduler.__init__()` | `scheduler.py:L67-L300` | No KVConnector, no encoder-decoder, no speculative config, no structured output, no LoRA, no CUDA graph, no Mamba, no multi-GPU; simplified config to three numeric params | Educational clarity — ~30 lines vs ~230 |
| `schedule()` (3-phase) | `scheduler.py:L352-L950` | 3 explicit phases (running, waiting, model forward) vs original's two while-loops with nested allocation; removed encoder budget, speculative tokens, async KV loading, block hashing, common prefix computation | Easier to trace the algorithm flow; ~50 lines vs ~600 |
| `_preempt_request()` | `scheduler.py:L952-L972` | Simplified: no encoder cache free, no event recording, no spec token cleanup; num_computed_tokens NOT reset (modeled as prefix-cache-enabled, matching effective behavior after `get_computed_blocks()` restoration at L616-L647) | Core preemption concept: free KV cache, move to waiting, prepend for fairness |
| `_pick_preemption_victim()` | `scheduler.py:L479-L483` | Extracted as a separate method; uses `(priority, request_id)` ordering vs `(priority, arrival_time, request_id)` (we don't track arrival_time) | Single responsibility; priority-based victim selection clearly visible |
| `KVCache` | `kv_cache_manager.py` | Fixed block pool with cumulative allocation; no PagedAttention block table, no prefix caching, no block hashing; simplified: allocate(num_tokens) returns block count | PagedAttention details are covered in Ch03; we only need OOM→preemption trigger |
| `Request` | `request.py:L59-L308` | 6 fields vs ~20; no multimodal, structured output, KV transfer, block hashing, encoder inputs, streaming, spec decode, pooling, trace headers | Only the fields that drive scheduling decisions are kept |
| `RequestStatus` | `request.py:L310-L326` | 4 states vs 11; WAITING/RUNNING/PREEMPTED/FINISHED vs the full enum with WAITING_FOR_* variants and 6 FINISHED_* sub-states | The core lifecycle is preserved; sub-states are async/debug details |
| `StaticBatchSimulator` | N/A (educational addition) | Pure demo class: simulates static batching with max_batch_size, counts idle slots, computes utilization | Side-by-side comparison with continuous batching to show the bubble problem |
| `add_request()` | `request_queue.py` | Direct list append; no priority queue integration | Simpler, but the priority ordering is demonstrated via `_pick_preemption_victim()` |
| `abort_request()` | `scheduler.py` (via `finish_requests`) | Linear scan of running+waiting lists; original uses dict lookup and status-based removal | Adequate for educational context; real implementation uses `self.requests` dict for O(1) lookup |

### What we simplified (and why it's OK for education)

| Original Complexity | Our Simplification | Justification |
|---------------------|-------------------|---------------|
| Multi-GPU coordination (tensor/pipeline/context parallelism) | Single-device | Distributed scheduling is a separate concern (Ch11 DCP/PCP) |
| Encoder-decoder models + multimodal inputs | Text-only generative models | Encoder-decoder is a specialization; core scheduling is the same |
| Speculative decoding (draft tokens, rejection sampling) | None | Covered in Ch21; the scheduler treats spec tokens as extra `num_tokens_with_spec` |
| Structured output (grammar-constrained generation) | None | Separate feature; status checks add complexity without insight |
| KV Connectors (P/D disaggregation, offloading) | None | Covered in Ch12; scheduling-wise it's just async state transitions |
| Streaming requests (resumable input) | None | Edge case; streaming statuses add 3 WAITING_FOR_* variants |
| LoRA (adapter scheduling constraints) | None | max_loras constraint is simple but orthogonal to core algorithm |
| Prefix caching (block hashing, cache-aware scheduling) | Simplified: preempted requests keep `num_computed_tokens` | Effectively models prefix-cache-enabled behavior; full hashing covered in Ch05 |
| Chunked prefill (long_prefill_token_threshold) | Uniform token budget for all | The budget cap already demonstrates chunking; the threshold is a tuning detail |

### Key formula: how num_new_tokens works

In the original (scheduler.py:L408-L411):
```
num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
```

In our simplified version:
```
num_new_tokens = num_tokens_total - num_computed_tokens
              = (prompt_tokens + num_output_tokens) - num_computed_tokens
```

This represents "how many tokens do I need to feed the model so it can see all existing tokens?" For decode, this is typically 1 (one new output token). For prefill, this is the entire prompt. The scheduler caps this at the token budget and KV cache capacity.
