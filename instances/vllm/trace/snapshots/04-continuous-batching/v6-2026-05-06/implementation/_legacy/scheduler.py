"""
Continuous Batching Scheduler — Our Reimplementation.

REFERENCE sources (vLLM v1 scheduler):
    Scheduler.schedule():         vllm/v1/core/sched/scheduler.py:L352-L945
    RequestStatus:                vllm/v1/request.py:L310-L337
    SchedulerOutput:              vllm/v1/core/sched/output.py:L181-L200
    Chunked Prefill API:          vllm/v1/config/scheduler.py:L84-L90
    FCFS Queue:                   vllm/v1/core/sched/request_queue.py:L75-L128
    _preempt_request:             vllm/v1/core/sched/scheduler.py:L952-L972
    _update_after_schedule:       vllm/v1/core/sched/scheduler.py:L974-L998
    update_from_output:           vllm/v1/core/sched/scheduler.py:L1290-L1551
    _free_request:                vllm/v1/core/sched/scheduler.py:L1813-L1834
    add_request:                  vllm/v1/core/sched/scheduler.py:L1728-L1748
    alloc/free blocks:            vllm/v1/core/kv_cache_manager.py

THE KEY INSIGHT (from scheduler.py:L353-L362):
    "There's no 'decoding phase' nor 'prefill phase' in the scheduler.
     Each request just has num_computed_tokens and num_tokens_with_spec."

    This is the essence of continuous batching: the scheduler treats every
    request uniformly as a stream of tokens. It decides HOW MANY tokens to
    advance each request by, not WHICH PHASE the request is in.

STATIC BATCHING (old way):
    Step 1: Prefill all prompts together → wait for longest prompt
    Step 2: Decode one token at a time → GPU idle during decode

CONTINUOUS BATCHING (vLLM):
    Step 1: Schedule a MIX of requests — some doing prefill chunks,
            some doing decode. Fill token budget.
    Step 2: After forward pass, update state. Some requests finish,
            new requests enter.
    Step 3: Repeat.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# RequestStatus — vllm/v1/request.py:L310-L337
# ═══════════════════════════════════════════════════════════════════════════

class RequestStatus(IntEnum):
    """REFERENCE: vllm/v1/request.py:L310-L337"""
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5

    def is_finished(self) -> bool:
        """REFERENCE: request.py:L332-L333 — status > PREEMPTED means finished"""
        return self.value > RequestStatus.PREEMPTED.value


# ═══════════════════════════════════════════════════════════════════════════
# Request — simplified from vllm/v1/request.py:L59-L308
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Request:
    """
    Simplified Request matching vLLM's key fields.

    REFERENCE: vllm/v1/request.py:L59-L308 (full Request class)
    """
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    arrival_time: float

    # State tracking
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0           # How many tokens processed so far
    output_token_ids: List[int] = field(default_factory=list)

    # KV Cache tracking
    block_ids: List[int] = field(default_factory=list)  # Allocated block IDs

    @property
    def num_tokens(self) -> int:
        """Total tokens to process = prompt + output."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_new_tokens(self) -> int:
        """Tokens NOT yet processed = total - computed."""
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        """Still processing the prompt? (hasn't caught up yet).

        REFERENCE: request.py num_computed_tokens vs num_prompt_tokens.
        In vLLM this also accounts for spec tokens and placeholders.
        """
        return self.num_computed_tokens < len(self.prompt_token_ids)


# ═══════════════════════════════════════════════════════════════════════════
# SchedulerOutput — vllm/v1/core/sched/output.py:L181-L200
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SchedulerOutput:
    """What the scheduler decided this step.

    REFERENCE: vllm/v1/core/sched/output.py:L181-L200
    vLLM's SchedulerOutput includes scheduled_new_reqs + scheduled_cached_reqs,
    preempted_req_ids, finished_req_ids, num_scheduled_tokens, etc.
    """
    scheduled_requests: Dict[str, int]  # request_id → num_tokens to advance
    total_scheduled_tokens: int
    finished_req_ids: List[str]
    preempted_req_ids: List[str] = field(default_factory=list)
    newly_running_req_ids: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# SimpleKVCacheManager stub (see Chapter 2 for full implementation)
# REFERENCE: vllm/v1/core/kv_cache_manager.py:L106
# ═══════════════════════════════════════════════════════════════════════════

class SimpleKVCacheManager:
    """
    Simplified KV cache manager for scheduler demo.

    REFERENCE: vllm/v1/core/kv_cache_manager.py (full KVCacheManager).

    The real KVCacheManager handles:
      - Block hash-based prefix caching
      - Copy-on-write for shared blocks
      - Multiple attention groups (full, sliding window, cross-attention)
      - Block eviction with reference counting
      - allocate_slots() with num_lookahead_tokens, num_computed_blocks, etc.

    We simplify to: allocate fresh blocks from a free list; free returns them.
    """
    def __init__(self, num_gpu_blocks: int, block_size: int = 16):
        """REFERENCE: kv_cache_manager.py:KVCacheManager.__init__"""
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self._free_blocks = list(range(num_gpu_blocks))

    def allocate_slots(
        self, request_id: str, num_new_tokens: int
    ) -> Optional[List[int]]:
        """
        Allocate enough blocks for num_new_tokens.

        REFERENCE: kv_cache_manager.py:allocate_slots()
        The real version takes request, num_new_tokens, num_lookahead_tokens,
        num_new_computed_tokens, new_computed_blocks, etc.
        """
        blocks_needed = (num_new_tokens + self.block_size - 1) // self.block_size
        if blocks_needed > len(self._free_blocks):
            return None
        allocated = self._free_blocks[:blocks_needed]
        self._free_blocks = self._free_blocks[blocks_needed:]
        return allocated

    def free(self, block_ids: List[int]):
        """REFERENCE: kv_cache_manager.py:free()"""
        self._free_blocks.extend(block_ids)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_gpu_blocks - self.num_free_blocks


# ═══════════════════════════════════════════════════════════════════════════
# ContinuousBatchingScheduler
# REFERENCE: vllm/v1/core/sched/scheduler.py:Scheduler class (L67-L945)
# ═══════════════════════════════════════════════════════════════════════════

class ContinuousBatchingScheduler:
    """
    Simplified continuous batching scheduler.

    REFERENCE: vllm/v1/core/sched/scheduler.py:L67-L945

    Key design decisions from vLLM (from the comment at L353-L362):
    1. No separate prefill/decode phases — every request has
       num_computed_tokens. The scheduler advances it toward num_tokens.
    2. Phase 1: RUNNING requests scheduled first (FCFS order within list).
    3. Phase 2: WAITING requests fill remaining token budget.
    4. OOM in Phase 1 → preempt lowest-priority running request.
    5. Chunked prefill: long prompts split across steps.
    """

    def __init__(
        self,
        max_num_scheduled_tokens: int,
        max_num_running_reqs: int,
        num_gpu_blocks: int,
        block_size: int = 16,
        enable_chunked_prefill: bool = True,
    ):
        """
        REFERENCE: scheduler.py:L67-L148 (Scheduler.__init__)
        The real Scheduler takes VllmConfig, KVCacheConfig, etc.
        """
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.max_num_running_reqs = max_num_running_reqs
        self.block_size = block_size
        self.enable_chunked_prefill = enable_chunked_prefill

        self.kv_cache_manager = SimpleKVCacheManager(num_gpu_blocks, block_size)

        # REFERENCE: scheduler.py:L158-L176
        # In vLLM the waiting queue is a RequestQueue (FCFS or Priority).
        # We use a simple list for waiting and running.
        self.requests: Dict[str, Request] = {}
        self.waiting: List[Request] = []
        self.running: List[Request] = []

    def add_request(self, req: Request) -> None:
        """REFERENCE: scheduler.py:L1728-L1748 (Scheduler.add_request)

        In vLLM this handles:
          - Streaming input (queued updates for resumable requests)
          - Structured output grammar initialization
          - RequestQueue enqueue (FCFS or Priority)
        """
        self.requests[req.request_id] = req
        self.waiting.append(req)

    # ── The Core: schedule() ──────────────────────────────────────────
    # REFERENCE: scheduler.py:L352-L945
    #
    # Architecture:
    #   1. Phase 1 — Running requests (L388-L556)
    #      For each RUNNING request (in FCFS order):
    #        a. Compute num_new_tokens capped by remaining token_budget
    #        b. Try allocate_slots()
    #        c. If OOM → preempt lowest-priority running request, retry
    #        d. Record scheduled tokens, deduct from budget
    #
    #   2. Phase 2 — Waiting requests (L568-L846)
    #      While waiting queue non-empty AND budget > 0 AND running < max:
    #        a. Pop from waiting queue (FCFS)
    #        b. Compute num_new_tokens, cap by budget
    #        c. If chunked prefill OFF and can't fit → STOP
    #        d. Try allocate_slots()
    #        e. If OOM → STOP (no preemption for waiting requests)
    #        f. Move to RUNNING, record scheduled tokens
    #
    #   3. Build SchedulerOutput (L871-L945)

    def schedule(self) -> SchedulerOutput:
        """
        Main scheduling loop — the heart of continuous batching.

        REFERENCE: scheduler.py:L352-L945
        """
        token_budget = self.max_num_scheduled_tokens
        scheduled: Dict[str, int] = {}   # req_id → num_tokens this step
        finished: List[str] = []
        preempted_req_ids: List[str] = []
        newly_running_req_ids: List[str] = []

        # self.trace_log accumulates annotations for the runnable demo
        if not hasattr(self, 'trace_log'):
            self.trace_log = []

        step_header = f"\n  ── Schedule Step ──  budget={token_budget}"
        self.trace_log.append(step_header)

        # ── Phase 1: Running Requests ──
        # REFERENCE: scheduler.py:L388-L556
        self.trace_log.append(f"  Phase 1 (running={len(self.running)}):")

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            req = self.running[req_index]
            trace_note = f"    r{req.request_id}[computed={req.num_computed_tokens}, " \
                         f"new={req.num_new_tokens}]"

            num_new = min(req.num_new_tokens, token_budget)
            if num_new <= 0:
                self.trace_log.append(f"{trace_note} → skip (no new tokens)")
                req_index += 1
                continue

            # Try to allocate KV cache for this request
            new_blocks = self.kv_cache_manager.allocate_slots(
                req.request_id, num_new
            )

            # If OOM, keep preempting lowest-priority running requests
            # until allocation succeeds or we can't preempt anyone.
            # REFERENCE: scheduler.py:L466-L511
            while new_blocks is None:
                # Preempt lowest-priority request (last in running list for FCFS)
                # REFERENCE: scheduler.py:L503-L504
                if len(self.running) <= 1:
                    # Only this request in running — can't preempt anyone
                    self.trace_log.append(f"{trace_note} → OOM, no one to preempt")
                    break

                preempted_req = self.running.pop()
                if preempted_req == req:
                    # Can't preempt ourselves — put it back
                    self.running.append(preempted_req)
                    self.trace_log.append(f"{trace_note} → OOM, can't preempt self")
                    break

                self._preempt_request(preempted_req)
                preempted_req_ids.append(preempted_req.request_id)
                self.trace_log.append(
                    f"      → preempted r{preempted_req.request_id}"
                )

                # If the preempted request was already scheduled this step,
                # restore its tokens to the budget.
                # REFERENCE: scheduler.py:L485-L502
                if preempted_req.request_id in scheduled:
                    freed_tokens = scheduled.pop(preempted_req.request_id)
                    token_budget += freed_tokens
                    self.trace_log.append(
                        f"      → restored {freed_tokens} tokens to budget"
                    )

                # Retry allocation
                new_blocks = self.kv_cache_manager.allocate_slots(
                    req.request_id, num_new
                )

            if new_blocks is None:
                # Even after preemption, can't schedule this request
                self.trace_log.append(f"{trace_note} → FAILED (OOM)")
                break

            # Allocation succeeded — schedule the request
            req.block_ids.extend(new_blocks)
            scheduled[req.request_id] = num_new
            token_budget -= num_new
            req_index += 1
            self.trace_log.append(
                f"{trace_note} → scheduled {num_new} tokens "
                f"(budget left={token_budget})"
            )

        # ── Phase 2: Waiting Requests ──
        # REFERENCE: scheduler.py:L568-L846
        self.trace_log.append(
            f"  Phase 2 (waiting={len(self.waiting)}, running={len(self.running)}):"
        )

        # In vLLM, after preemption Phase 2 is skipped:
        # REFERENCE: scheduler.py:L568 "if not preempted_reqs"
        if not preempted_req_ids:
            while self.waiting and token_budget > 0 \
                    and len(self.running) < self.max_num_running_reqs:
                req = self.waiting[0]
                trace_note = f"    r{req.request_id}[prompt={len(req.prompt_token_ids)},\n" \
                             f"             max_out={req.max_tokens}]"

                # Skip finished requests in waiting queue
                if req.status.is_finished():
                    self.waiting.pop(0)
                    continue

                num_new = req.num_new_tokens

                # REFERENCE: scheduler.py:L684-L690
                # If chunked prefill is disabled and the request can't fit
                # in the remaining budget, stop admitting.
                if (not self.enable_chunked_prefill
                        and num_new > token_budget):
                    self.trace_log.append(
                        f"{trace_note} → can't fit (need {num_new}, "
                        f"budget {token_budget}), stop admitting"
                    )
                    break

                # REFERENCE: scheduler.py:L682-L692
                # Chunked prefill: cap at token_budget, otherwise break if
                # request doesn't fit (handled above).
                if self.enable_chunked_prefill:
                    num_new = min(num_new, token_budget)

                if num_new <= 0:
                    self.waiting.pop(0)
                    continue

                # Allocate KV cache
                # REFERENCE: scheduler.py:L744-L754
                blocks = self.kv_cache_manager.allocate_slots(
                    req.request_id, num_new
                )
                if blocks is None:
                    self.trace_log.append(
                        f"{trace_note} → OOM (cache full), stop admitting"
                    )
                    break  # No more KV cache — stop admitting

                # Admit the request to running
                req.block_ids = blocks
                req.status = RequestStatus.RUNNING
                scheduled[req.request_id] = num_new
                token_budget -= num_new
                newly_running_req_ids.append(req.request_id)

                self.waiting.pop(0)
                self.running.append(req)
                self.trace_log.append(
                    f"{trace_note} → admitted, scheduled {num_new} tokens "
                    f"(budget left={token_budget})"
                )

        # ── Build SchedulerOutput ──
        # REFERENCE: scheduler.py:L871-L945
        output = SchedulerOutput(
            scheduled_requests=scheduled,
            total_scheduled_tokens=sum(scheduled.values()),
            finished_req_ids=finished,
            preempted_req_ids=preempted_req_ids,
            newly_running_req_ids=newly_running_req_ids,
        )

        self.trace_log.append(
            f"  → Result: {len(scheduled)} reqs, "
            f"{output.total_scheduled_tokens} tokens"
        )
        return output

    def _preempt_request(self, req: Request) -> None:
        """
        Preempt a running request: free its KV blocks, reset computed tokens,
        move it back to the waiting queue.

        REFERENCE: scheduler.py:L952-L972

        NOTE: The request MUST be popped from the running queue OUTSIDE this
        method (caller does `self.running.pop()` before calling us), as in
        vLLM.  Do NOT remove it again here.

        In vLLM this also:
          - Frees encoder cache
          - Clears spec token IDs
          - Increments num_preemptions
          - Records PREEMPTED event for logging
        """
        # REFERENCE: scheduler.py:L961 — self.kv_cache_manager.free(request)
        self.kv_cache_manager.free(req.block_ids)
        req.block_ids = []
        req.status = RequestStatus.PREEMPTED
        # REFERENCE: scheduler.py:L964 — reset computed tokens to 0
        # (must recompute from scratch since KV cache is gone)
        req.num_computed_tokens = 0
        # NOTE: Request already popped from self.running by caller.
        # Do NOT call self.running.remove(req) here.

        # REFERENCE: scheduler.py:L972 — self.waiting.prepend_request(request)
        # Put at front of waiting queue so it's re-scheduled next
        self.waiting.insert(0, req)

    def update_after_step(self, output: SchedulerOutput) -> None:
        """
        Update request state after model forward pass.

        REFERENCE: scheduler.py:L974-L998 (_update_after_schedule)
                   + scheduler.py:L1290-L1551 (update_from_output)

        In vLLM, _update_after_schedule advances num_computed_tokens for
        ALL scheduled requests immediately after scheduling (before the
        forward pass). This allows the next schedule() call to see the
        updated state.

        update_from_output handles:
          - Sampled token IDs from the model runner
          - Stop checking (via check_stop in _update_request_with_output)
          - Spec decode rejection (reduces num_computed_tokens)
          - Freeing encoder cache inputs after use
          - Building EngineCoreOutput for each request

        KEY INSIGHT for our simplified version:
        After prefill completes, num_computed_tokens catches up to
        num_prompt_tokens but output_token_ids is empty, so num_new_tokens
        becomes 0. To simulate the model forward pass producing a decode
        token, we generate one output token per step for every request
        that has finished prefill and still has output budget.
        """
        # REFERENCE: scheduler.py:L985-L993 (_update_after_schedule)
        for req_id, num_tokens in output.scheduled_requests.items():
            req = self.requests[req_id]
            req.num_computed_tokens += num_tokens

            # REFERENCE: scheduler.py:L988-L990
            # is_prefill_chunk flag: true if num_computed_tokens hasn't
            # caught up to num_tokens + num_output_placeholders.

            # Simulate model forward pass output:
            # If request has finished prefill (num_computed_tokens >= prompt_len),
            # the model produces one output token per decode step.
            # REFERENCE: scheduler.py:L1622-L1638 (_update_request_with_output)
            if not req.is_prefill:
                # Only generate output if we haven't hit max_tokens yet.
                # This check mirrors check_stop() in vLLM:
                #   stopped = check_stop(request, self.max_model_len)
                # which checks num_output_tokens >= max_tokens.
                if len(req.output_token_ids) < req.max_tokens:
                    # Placeholder — real token comes from model runner
                    req.output_token_ids.append(0)

            # Check if finished (max_tokens reached)
            # REFERENCE: scheduler.py:L1634-L1637
            # (check_stop inside _update_request_with_output)
            if len(req.output_token_ids) >= req.max_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self._finish_request(req)

    def _finish_request(self, req: Request) -> None:
        """
        Free a finished request's resources and remove it from running.

        REFERENCE: scheduler.py:L1813-L1834 (_free_request)
        In vLLM, _free_request also:
          - Handles KV connector delay-free (async KV transfer)
          - Frees encoder cache
          - Tracks finished IDs for next SchedulerOutput
          - Removes from self.requests dict
        """
        # REFERENCE: scheduler.py:L1827 — free blocks
        self.kv_cache_manager.free(req.block_ids)
        req.block_ids = []
        if req in self.running:
            self.running.remove(req)


# ═══════════════════════════════════════════════════════════════════════════
# Static Batching Comparison (for Chapter narrative)
# ═══════════════════════════════════════════════════════════════════════════

def static_batching_simulation(
    requests: List[Tuple[int, int]],  # (prompt_len, max_output_len)
) -> int:
    """
    Simulate static batching: all prompts must finish prefill before any decode.

    Returns total steps needed (each step = one model forward pass).

    How static batching works:
      Phase 1 — Prefill: process all prompts simultaneously. The batch waits
                for the LONGEST prompt to finish. Shorter prompts' GPUs idle.
      Phase 2 — Decode: generate one token per request per step. All requests
                advance at the same rate. When a request finishes, its slot
                remains idle until all finish.

    This produces the "bubble" — GPU idle time — that continuous batching
    eliminates by interleaving prefill and decode work.
    """
    steps = 0
    prompt_lens = [r[0] for r in requests]
    remaining_output = [r[1] for r in requests]
    active = [True] * len(requests)

    # Prefill phase: do all prompts (batch them together)
    max_prompt = max(prompt_lens)
    steps += max_prompt  # Each token of the longest prompt is one step

    # Decode phase: generate one token at a time
    while any(active):
        for i in range(len(requests)):
            if active[i]:
                remaining_output[i] -= 1
                if remaining_output[i] <= 0:
                    active[i] = False
        steps += 1

    return steps


def continuous_batching_simulation(
    requests: List[Tuple[int, int]],
    max_tokens_per_step: int = 2048,
) -> int:
    """
    Simulate continuous batching: prefill and decode tokens are interleaved.

    Returns total steps needed.

    How continuous batching works:
      Each step fills a token budget with a MIX of work:
        1. One decode token per request in decode phase
        2. Remaining budget: fill with prefill tokens (chunked across steps)

      Since decode tokens are cheap (1 token = 1 forward pass) and prefill
      tokens fill the leftover budget, GPU utilization stays high.

    This is a simplified model — the real scheduler is more sophisticated
    (FCFS queues, preemption, prefix caching, etc.).
    """
    steps = 0
    prompt_lens = [r[0] for r in requests]
    remaining_prompt = list(prompt_lens)
    remaining_output = [r[1] for r in requests]
    active = [True] * len(requests)

    while any(active):
        tokens_this_step = 0
        # First: one decode token per active request in decode phase
        for i in range(len(requests)):
            if active[i] and remaining_prompt[i] <= 0:
                if tokens_this_step + 1 <= max_tokens_per_step:
                    remaining_output[i] -= 1
                    tokens_this_step += 1
                    if remaining_output[i] <= 0:
                        active[i] = False

        # Then: fill remaining budget with prefill tokens
        for i in range(len(requests)):
            if active[i] and remaining_prompt[i] > 0:
                can_do = min(
                    remaining_prompt[i], max_tokens_per_step - tokens_this_step
                )
                if can_do > 0:
                    remaining_prompt[i] -= can_do
                    tokens_this_step += can_do

        steps += 1

    return steps


def bubble_analysis():
    """
    Quantify the GPU idle time (bubble) in static batching.

    Static batching has two kinds of bubble:
    1. Prefill bubble: short prompts wait for longest prompt's prefill
    2. Decode bubble: during decode, each request generates ONE token per step
       → GPU spends most time at low utilization

    Continuous batching fills these bubbles with other work.
    """
    # Example: 8 requests, half long prompt (2048), half short (128)
    requests = [(2048, 256)] * 4 + [(128, 256)] * 4

    static_steps = static_batching_simulation(requests)
    continuous_steps = continuous_batching_simulation(requests)

    static_gpu_util = (
        sum(r[0] for r in requests) + sum(r[1] for r in requests)
    ) / (static_steps * len(requests))

    print(f"Static batching:       {static_steps} steps")
    print(f"Continuous batching:   {continuous_steps} steps")
    print(f"Speedup:               {static_steps / continuous_steps:.1f}x")
    print(f"Static GPU utilization: {static_gpu_util:.1%}")

    return static_steps, continuous_steps


# ═══════════════════════════════════════════════════════════════════════════
# Runnable Demo — Annotated Scheduling Trace
# ═══════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Demo: run the scheduler with a diverse workload and print annotated trace.

    This demonstrates the core continuous batching behaviors:
      - Running-first scheduling
      - Chunked prefill (splitting long prompts across steps)
      - Mixing prefill and decode work in the same step
      - Token budget management
      - Request lifecycle (WAITING → RUNNING → FINISHED)
    """
    print("=" * 70)
    print("Continuous Batching Scheduler — Annotated Trace")
    print("=" * 70)

    # Setup: limited budget to force chunked prefill
    sched = ContinuousBatchingScheduler(
        max_num_scheduled_tokens=512,
        max_num_running_reqs=16,
        num_gpu_blocks=1000,
        block_size=16,
        enable_chunked_prefill=True,
    )

    # Add a diverse workload
    # r1: long prompt (will be chunked across steps)
    # r2: medium prompt
    # r3: short prompt
    sched.add_request(Request("1", list(range(800)), max_tokens=20, arrival_time=0))
    sched.add_request(Request("2", list(range(200)), max_tokens=50, arrival_time=0))
    sched.add_request(Request("3", list(range(50)), max_tokens=100, arrival_time=0))

    print(f"\nWorkload:")
    for req in sched.waiting:
        print(f"  r{req.request_id}: prompt={len(req.prompt_token_ids)}, "
              f"max_out={req.max_tokens}")

    # Run scheduling loop
    for step in range(8):
        print(f"\n{'─' * 50}")
        print(f"Step {step + 1}")

        # Print pre-schedule state
        print(f"  State before: "
              f"running={len(sched.running)}, "
              f"waiting={len(sched.waiting)}, "
              f"free_blocks={sched.kv_cache_manager.num_free_blocks}")

        output = sched.schedule()
        sched.update_after_step(output)

        # Print post-schedule summary
        for rid, ntokens in output.scheduled_requests.items():
            req = sched.requests[rid]
            phase = "prefill" if req.is_prefill else "decode"
            print(f"  r{rid}: {ntokens} tokens ({phase}), "
                  f"computed={req.num_computed_tokens}/{req.num_tokens}")

        if output.preempted_req_ids:
            print(f"  PREEMPTED: {output.preempted_req_ids}")

        if any(sched.requests[rid].status.is_finished()
               for rid in output.scheduled_requests):
            finished = [rid for rid in output.scheduled_requests
                        if sched.requests[rid].status.is_finished()]
            print(f"  FINISHED: {finished}")

        budget_used = sum(output.scheduled_requests.values())
        print(f"  Budget: {budget_used}/{sched.max_num_scheduled_tokens}")

        # Check if we're done
        if not sched.waiting and not sched.running:
            print(f"\nAll requests finished after {step + 1} steps.")
            break

    # Print KV cache efficiency
    total_allocated = sched.kv_cache_manager.num_used_blocks
    print(f"\n{'=' * 70}")
    print(f"Final KV Cache: {total_allocated}/{sched.kv_cache_manager.num_gpu_blocks} "
          f"blocks used")
    print(f"Bubble analysis (static vs continuous):")
    bubble_analysis()


if __name__ == "__main__":
    run_demo()
