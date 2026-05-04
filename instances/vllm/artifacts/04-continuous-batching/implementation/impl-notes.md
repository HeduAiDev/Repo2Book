# Implementation Notes — Ch4 Continuous Batching

## Source Analysis (HARD GATE)

### 1. What vLLM files implement this feature?

| File | Path (relative to vllm/) | Role |
|------|--------------------------|------|
| Scheduler | `vllm/v1/core/sched/scheduler.py` | Core two-phase scheduling: schedule() L352, preempt L952, update L974, update_from_output L1290, free L1813 |
| Scheduler Output | `vllm/v1/core/sched/output.py` | SchedulerOutput dataclass L181, NewRequestData L31, CachedRequestData L112 |
| Request Queue | `vllm/v1/core/sched/request_queue.py` | FCFSQueue L75, PriorityQueue L131, SchedulingPolicy L13 |
| Request | `vllm/v1/request.py` | Request class L59, RequestStatus L310 |
| Scheduler Config | `vllm/config/scheduler.py` | SchedulerConfig L26: max_num_scheduled_tokens, max_num_seqs, enable_chunked_prefill, long_prefill_token_threshold |
| Scheduler Interface | `vllm/v1/core/sched/interface.py` | SchedulerInterface ABC, PauseState enum |
| KV Cache Manager | `vllm/v1/core/kv_cache_manager.py` | KVCacheManager: allocate_slots(), free(), get_computed_blocks() |

### 2. Key classes and their responsibilities

**Scheduler (scheduler.py:L67):**
- Owns: `self.running` (list), `self.waiting` (RequestQueue), `self.requests` (dict)
- Owns: `self.kv_cache_manager` (KVCacheManager)
- Owns: `self.finished_req_ids` (set), `self.prev_step_scheduled_req_ids` (set)
- Key methods: `schedule()` (L352), `_preempt_request()` (L952), `_update_after_schedule()` (L974), `update_from_output()` (L1290), `_free_request()` (L1813), `add_request()` (L1728)
- Delegates to: KVCacheManager for block allocation, RequestQueue for FCFS/Priority ordering

**Request (request.py:L59):**
- Owns: token IDs (prompt + output), status, computed tokens count
- Key properties: `num_tokens`, `num_tokens_with_spec`, `num_computed_tokens`, `num_output_tokens`
- State machine: WAITING → RUNNING → PREEMPTED → WAITING | FINISHED_*

**RequestQueue (request_queue.py):**
- FCFSRequestQueue (L75): deque-based, O(1) append/popleft
- PriorityRequestQueue (L131): heap-based, O(log n) push/pop

**KVCacheManager (kv_cache_manager.py):**
- Owns: block pools, prefix cache hash table, block-to-block mapping
- Key methods: `allocate_slots()`, `free()`, `get_computed_blocks()`, `get_blocks()`

### 3. Data flow

```
Client sends request
  → Scheduler.add_request() (L1728)
    → self.waiting.add_request(request) [FCFS or Priority]
  
Engine loop: schedule() → model_runner → update_from_output() → repeat

schedule() (L352):
  1. token_budget = max_num_scheduled_tokens
  2. Phase 1 — Running requests (L388-L556):
     For each running request (FCFS order):
       a. num_new = min(request.num_tokens - num_computed_tokens, budget)
       b. Try kv_cache_manager.allocate_slots(request, num_new, ...)
       c. If OOM: preempt lowest-priority running request (pop last),
          free its blocks, retry allocation
       d. If allocation succeeds: add to scheduled_running_reqs
  3. Phase 2 — Waiting requests (L568-L846):
     While waiting queue not empty AND budget > 0 AND running < max:
       a. Pop from waiting (FCFS)
       b. num_new = min(request.num_tokens - num_computed_tokens, budget)
       c. If chunked prefill disabled and doesn't fit: break
       d. Try allocate_slots()
       e. If allocation succeeds: append to running, set status=RUNNING
  4. Build SchedulerOutput (L871-L945) with:
     - new_reqs_data, cached_reqs_data
     - num_scheduled_tokens, total_num_scheduled_tokens
     - preempted_req_ids, finished_req_ids

_update_after_schedule() (L974):
  - Advance num_computed_tokens for all scheduled requests
  - Clear self.finished_req_ids for next step

update_from_output() (L1290):
  - Process sampled token IDs from model runner
  - Check stop conditions (check_stop → FINISHED_STOPPED/LENGTH_CAPPED)
  - Handle spec decode rejections (reduce num_computed_tokens)
  - Free encoder cache inputs
  - Build EngineCoreOutput per request

_free_request() (L1813):
  - Free KV cache blocks (self.kv_cache_manager.free(request))
  - Free encoder cache
  - Track in self.finished_req_ids
  - Remove from self.requests dict
```

### 4. Design decisions and WHY

**Decision 1: No separate prefill/decode phases (scheduler.py:L353-L362)**
- Why: Every request is just a stream of tokens. The scheduler advances `num_computed_tokens` toward `num_tokens_with_spec`. This naturally handles chunked prefill, prefix caching, and speculative decoding with the same mechanism.
- Trade-off: Requires careful tracking of computed vs. total tokens per request.
- Alternative: Separate prefill/decode mode switching. vLLM chose unified because it's more general.

**Decision 2: Running-first scheduling (Phase 1 before Phase 2)**
- Why: Running requests have already paid the overhead of KV cache allocation. Preempting them wastes work. FCFS order within running ensures fairness.
- Trade-off: Can starve waiting requests under heavy load. Mitigated by preemption (OOM preempts lowest-priority running request).
- Source: scheduler.py:L388-L846 — the two-phase structure is explicit.

**Decision 3: Preempt lowest-priority (last in FCFS), not the one being scheduled**
- Why: The request being scheduled is higher-priority (earlier in FCFS running list). Preempting a different request keeps the most "senior" running request alive.
- Trade-off: More complex code (retry loop). But prevents one request's OOM from cascading.
- Source: scheduler.py:L503-L504 — `preempted_req = self.running.pop()` for FCFS.

**Decision 4: Chunked prefill (scheduler_config.py:L84)**
- Why: Long prompts otherwise block the entire batch for many steps. Chunking interleaves prefill and decode work, keeping GPU utilization high.
- Trade-off: Slightly more scheduler complexity. Need to track `is_prefill_chunk` flag.
- Source: scheduler_config.py:L84 `enable_chunked_prefill: bool = True`

**Decision 5: Token budget limit (max_num_scheduled_tokens)**
- Why: Prevents out-of-memory in the model runner. Each forward pass has a maximum token capacity (max_num_batched_tokens).
- Trade-off: Can artificially limit throughput if set too low.
- Source: scheduler.py:L371 `token_budget = self.max_num_scheduled_tokens`

### 5. Complexity preserved in our implementation

- **Two-phase scheduling** (running first, then waiting) — the core architecture
- **Token budget management** — per-step cap on total scheduled tokens
- **FCFS ordering** — within both running and waiting queues
- **Chunked prefill** — long prompts split across steps when budget is tight
- **Preemption with retry** — OOM triggers preemption of lowest-priority running request, not just failure
- **Request state machine** — WAITING → RUNNING → PREEMPTED → WAITING | FINISHED
- **KV cache block lifecycle** — allocate on schedule, free on preemption/finish

### What we simplified (for pedagogy)

- **RequestQueue** — used Python list instead of deque/heap (same FCFS semantics)
- **KVCacheManager** — free-list-based allocation instead of prefix-cache-aware block management
- **update_from_output** — collapsed into update_after_step (no spec decode, no logprobs, no stop strings)
- **No encoder cache / multimodal support** — irrelevant to core continuous batching concept
- **No speculative decoding** — out of scope for this chapter
- **No skipped_waiting queue** — handles blocked statuses (WAITING_FOR_REMOTE_KVS, etc.)

## Source Mapping Table

| Our Code | vLLM Source | What We Changed | Why |
|----------|------------|-----------------|-----|
| `ContinuousBatchingScheduler.__init__()` | `scheduler.py:L67-L148` | Simplified constructor params | Remove VllmConfig dependency |
| `ContinuousBatchingScheduler.schedule()` | `scheduler.py:L352-L945` | Same two-phase structure; simplified preemption retry | Core algorithm unchanged |
| `schedule() Phase 1` | `scheduler.py:L388-L556` | Same FCFS iteration + preempt-on-OOM | Identical algorithm |
| `schedule() Phase 2` | `scheduler.py:L568-L846` | Same budget check + chunked prefill | Identical algorithm |
| `_preempt_request()` | `scheduler.py:L952-L972` | Same: free blocks, reset tokens, requeue | Identical algorithm |
| `update_after_step()` | `scheduler.py:L974-L998`, `L1290-L1551` | Merged _update_after_schedule + update_from_output | Simplified (no spec decode, logprobs, stop strings) |
| `_finish_request()` | `scheduler.py:L1813-L1834` | Same: free blocks, remove from running | Simplified (no connector, encoder cache) |
| `add_request()` | `scheduler.py:L1728-L1748` | Same: add to waiting queue | Simplified (no streaming, structured output) |
| `SimpleKVCacheManager.allocate_slots()` | `kv_cache_manager.py:allocate_slots()` | Free-list instead of block pool | Simplified for demo |
| `SimpleKVCacheManager.free()` | `kv_cache_manager.py:free()` | Same: return blocks to free list | Identical semantics |
| `Request` | `vllm/v1/request.py:L59-L308` | Minimal subset of fields | Only the fields scheduler needs |
| `RequestStatus` | `vllm/v1/request.py:L310-L337` | Same enum semantics | Trimmed finished variants |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py:L181-L200` | Simplified fields | Only what tests + narrative need |
| `static_batching_simulation()` | — | New (pedagogical) | Chapter narrative comparison |
| `continuous_batching_simulation()` | — | New (pedagogical) | Chapter narrative comparison |
