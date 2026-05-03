"""
Continuous Batching Scheduler — Our Reimplementation.

REFERENCE sources:
    Scheduler.schedule():  vllm/v1/core/sched/scheduler.py:L352
    RequestStatus:         vllm/v1/request.py:L310
    SchedulerOutput:       vllm/v1/core/sched/output.py:L181
    Chunked Prefill API:   vllm/v1/config/scheduler.py:L84
    FCFS Queue:            vllm/v1/core/sched/request_queue.py:L75

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
# RequestStatus — vllm/v1/request.py:L310
# ═══════════════════════════════════════════════════════════════════════════

class RequestStatus(IntEnum):
    """REFERENCE: vllm/v1/request.py:L310-L338"""
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5

    def is_finished(self) -> bool:
        """REFERENCE: request.py:L332 — status > PREEMPTED means finished"""
        return self.value > RequestStatus.PREEMPTED.value


# ═══════════════════════════════════════════════════════════════════════════
# Request — simplified from vllm/v1/request.py
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Request:
    """
    Simplified Request matching vLLM's key fields.

    REFERENCE: vllm/v1/request.py (full Request class)
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
        """Still processing the prompt? (hasn't caught up yet)."""
        return self.num_computed_tokens < len(self.prompt_token_ids)


# ═══════════════════════════════════════════════════════════════════════════
# SchedulerOutput — vllm/v1/core/sched/output.py:L181
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SchedulerOutput:
    """What the scheduler decided this step."""
    scheduled_requests: Dict[str, int]  # request_id → num_tokens to advance
    total_scheduled_tokens: int
    finished_req_ids: List[str]
    preempted_req_id: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# Simple KVCacheManager stub (see Chapter 2 for full implementation)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleKVCacheManager:
    """
    Simplified KV cache manager for scheduler demo.

    REFERENCE: vllm/v1/core/kv_cache_manager.py:L106 (full KVCacheManager)
    """
    def __init__(self, num_gpu_blocks: int, block_size: int = 16):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self._free_blocks = list(range(num_gpu_blocks))

    def allocate_slots(self, request_id: str, num_new_tokens: int) -> Optional[List[int]]:
        blocks_needed = (num_new_tokens + self.block_size - 1) // self.block_size
        if blocks_needed > len(self._free_blocks):
            return None
        allocated = self._free_blocks[:blocks_needed]
        self._free_blocks = self._free_blocks[blocks_needed:]
        return allocated

    def free(self, block_ids: List[int]):
        self._free_blocks.extend(block_ids)


# ═══════════════════════════════════════════════════════════════════════════
# ContinuousBatchingScheduler
# REFERENCE: vllm/v1/core/sched/scheduler.py:L352 — schedule()
# ═══════════════════════════════════════════════════════════════════════════

class ContinuousBatchingScheduler:
    """
    Simplified continuous batching scheduler.

    REFERENCE: vllm/v1/core/sched/scheduler.py:L67-L945

    Key design decisions from vLLM:
    1. No separate prefill/decode phases — every request has num_computed_tokens
    2. Running requests scheduled first (FCFS within running list)
    3. Waiting requests fill remaining token budget
    4. OOM → preempt lowest-priority (FCFS: last in running list)
    5. Chunked prefill: long prompts split across steps
    """

    def __init__(
        self,
        max_num_scheduled_tokens: int,
        max_num_running_reqs: int,
        num_gpu_blocks: int,
        block_size: int = 16,
        enable_chunked_prefill: bool = True,
    ):
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.max_num_running_reqs = max_num_running_reqs
        self.block_size = block_size
        self.enable_chunked_prefill = enable_chunked_prefill

        self.kv_cache_manager = SimpleKVCacheManager(num_gpu_blocks, block_size)

        # REFERENCE: scheduler.py:L158-L176
        self.requests: Dict[str, Request] = {}
        self.waiting: List[Request] = []
        self.running: List[Request] = []

    def add_request(self, req: Request):
        """REFERENCE: scheduler.py:L1728"""
        self.requests[req.request_id] = req
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        """
        Main scheduling loop.

        REFERENCE: scheduler.py:L352-L945

        Phase 1: Schedule RUNNING requests (advance existing tokens)
        Phase 2: Schedule WAITING requests (admit new, fill budget)
        Phase 3: Build output
        """
        token_budget = self.max_num_scheduled_tokens
        scheduled: Dict[str, int] = {}   # req_id → num_tokens this step
        finished: List[str] = []
        preempted: Optional[str] = None

        # ── Phase 1: Running Requests ──
        # REFERENCE: scheduler.py:L388-L556
        for req in self.running[:]:  # iterate copy — may remove during loop
            num_new = min(req.num_new_tokens, token_budget)
            if num_new <= 0:
                continue

            # Allocate KV cache for advance
            new_blocks = self.kv_cache_manager.allocate_slots(req.request_id, num_new)
            if new_blocks is None:
                # OOM → preempt this request
                # REFERENCE: scheduler.py:L478-L511
                self._preempt_request(req)
                preempted = req.request_id
                continue

            req.block_ids.extend(new_blocks)
            scheduled[req.request_id] = num_new
            token_budget -= num_new

        # ── Phase 2: Waiting Requests ──
        # REFERENCE: scheduler.py:L568-L846
        while self.waiting and token_budget > 0 and len(self.running) < self.max_num_running_reqs:
            if preempted is not None:
                break  # Don't admit new requests after preemption

            req = self.waiting[0]

            # Skip blocked requests
            if req.status.is_finished():
                self.waiting.pop(0)
                continue

            num_new = req.num_new_tokens

            # Chunked prefill: cap at token budget
            if self.enable_chunked_prefill:
                num_new = min(num_new, token_budget)
            elif num_new > token_budget:
                break  # Can't fit — stop admitting (REFERENCE: scheduler.py:L684-L690)

            if num_new <= 0:
                self.waiting.pop(0)
                continue

            # Allocate KV cache
            blocks = self.kv_cache_manager.allocate_slots(req.request_id, num_new)
            if blocks is None:
                break  # No more KV cache — stop admitting

            req.block_ids = blocks
            req.status = RequestStatus.RUNNING
            scheduled[req.request_id] = num_new
            token_budget -= num_new

            self.waiting.pop(0)
            self.running.append(req)

        # ── Phase 3: Build Output ──
        # REFERENCE: scheduler.py:L871-L945
        return SchedulerOutput(
            scheduled_requests=scheduled,
            total_scheduled_tokens=sum(scheduled.values()),
            finished_req_ids=finished,
            preempted_req_id=preempted,
        )

    def _preempt_request(self, req: Request):
        """REFERENCE: scheduler.py:L952-L972"""
        self.kv_cache_manager.free(req.block_ids)
        req.block_ids = []
        req.status = RequestStatus.PREEMPTED
        req.num_computed_tokens = 0  # Must recompute from scratch
        self.running.remove(req)

    def update_after_step(self, output: SchedulerOutput):
        """
        Update request state after model forward pass.

        REFERENCE: scheduler.py:L974-L1041 (_update_after_schedule)
                   + scheduler.py:L1290-L1551 (update_from_output)
        """
        for req_id, num_tokens in output.scheduled_requests.items():
            req = self.requests[req_id]
            req.num_computed_tokens += num_tokens

            # Decode: generate one output token per step
            if not req.is_prefill and req.num_new_tokens == 1:
                if len(req.output_token_ids) < req.max_tokens:
                    req.output_token_ids.append(0)  # Placeholder — real token from model

            # Check if finished
            if len(req.output_token_ids) >= req.max_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self._finish_request(req)

    def _finish_request(self, req: Request):
        """REFERENCE: scheduler.py:L1813 (_free_request)"""
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
    """
    # Simplified: each step fills token budget with a mix of prefill + decode
    # This is a rough model — real scheduler is more sophisticated
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
                can_do = min(remaining_prompt[i], max_tokens_per_step - tokens_this_step)
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

    # Total tokens that need processing:
    # Prefill: 4×2048 + 4×128 = 8704
    # Decode: 8×256 = 2048
    total_tokens = sum(r[0] for r in requests) + sum(r[1] for r in requests)

    # Static: long prefills dictate step count
    static_gpu_util = total_tokens / (static_steps * len(requests))  # rough

    print(f"Static batching:       {static_steps} steps")
    print(f"Continuous batching:   {continuous_steps} steps")
    print(f"Speedup:               {static_steps / continuous_steps:.1f}x")
    print(f"Static GPU utilization: {static_gpu_util:.1%}")

    return static_steps, continuous_steps


if __name__ == "__main__":
    bubble_analysis()
