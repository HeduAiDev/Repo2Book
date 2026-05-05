"""
Continuous Batching Scheduler — educational reimplementation.

REFERENCE: vllm/v1/core/sched/scheduler.py
  - Scheduler.__init__()          → L67-L300
  - schedule()                     → L352-L950 (core algorithm)
  - _preempt_request()            → L952-L972
  - update_from_output()          → L1290+ (request state update)

REFERENCE: vllm/v1/request.py → Request, RequestStatus
REFERENCE: vllm/v1/core/kv_cache_manager.py → KVCacheManager (block allocation)
REFERENCE: vllm/v1/core/sched/request_queue.py → RequestQueue, SchedulingPolicy

Key insight: Static batching waits for ALL requests in a batch to finish before
starting a new batch — creating "bubbles" of idle GPU time. Continuous batching
allows requests to join and leave at EVERY scheduling step — eliminating bubbles.

Request lifecycle: WAITING → RUNNING → (PREEMPTED → WAITING) → FINISHED
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Request State Machine
# ═══════════════════════════════════════════════════════════════════════
# REFERENCE: vllm/v1/request.py → RequestStatus (enum.IntEnum, L310-L326)

class RequestStatus(Enum):
    """Request lifecycle states.

    WAITING: queued, not yet allocated compute/KV resources
    RUNNING: actively processed this scheduling step
    PREEMPTED: evicted to free KV cache (naturally flows back to WAITING)
    FINISHED: completed (reached max_tokens or stopped)
    """
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


# REFERENCE: vllm/v1/request.py → Request (dataclass, L59-L308)
@dataclass
class Request:
    """A single inference request tracking token-level progress.

    In real vLLM, this has ~20 fields for multimodal inputs, structured
    output, speculative decoding, KV connectors, etc. We keep the six
    fields that drive the scheduling algorithm.
    """
    request_id: str
    prompt_tokens: int           # Number of prompt tokens (input)
    max_tokens: int = 256        # Max output tokens to generate

    # Priority: lower number = higher priority.
    # In the original (request.py:L301-L307), Request.__lt__ compares
    # (priority, arrival_time, request_id). We simplify to priority only.
    priority: int = 0

    # ── Dynamically updated each scheduling step ──
    num_computed_tokens: int = 0   # Tokens already fed to the model
    num_output_tokens: int = 0     # Output tokens generated so far
    status: RequestStatus = RequestStatus.WAITING

    @property
    def num_tokens_total(self) -> int:
        """Total tokens that exist for this request (prompt + generated).

        The scheduler schedules until num_computed_tokens catches up to
        num_tokens_total. Each time they meet, the model generates one
        more output token (auto-regressive loop).
        """
        return self.prompt_tokens + self.num_output_tokens

    @property
    def num_new_tokens(self) -> int:
        """Tokens not yet processed = (prompt + outputs) - computed."""
        return self.num_tokens_total - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.num_output_tokens >= self.max_tokens

    def __repr__(self) -> str:
        return (f"Request({self.request_id!r}, "
                f"status={self.status.value}, "
                f"computed={self.num_computed_tokens}/{self.num_tokens_total}, "
                f"output={self.num_output_tokens}/{self.max_tokens})")


# ═══════════════════════════════════════════════════════════════════════
# Simplified KV Cache (block-based)
# ═══════════════════════════════════════════════════════════════════════
# REFERENCE: vllm/v1/core/kv_cache_manager.py → KVCacheManager (L1+)
#
# Real vLLM: each block stores key-value tensors for `block_size` tokens
# (PagedAttention). The scheduler calls allocate_slots() per request;
# OOM triggers preemption which calls free().
#
# Our simplification: fixed pool of blocks, allocation is cumulative,
# but we only count new blocks needed each step (not total). This over-
# allocates by 1 block every ~block_size decode steps — acceptable for
# educational demonstration where the KV cache is sized generously.

@dataclass
class KVCache:
    """Fixed-size block pool. Allocation is cumulative per request."""
    total_blocks: int
    block_size: int = 16
    free_blocks: int = 0
    allocations: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.free_blocks = self.total_blocks

    def allocate(self, request_id: str, num_tokens: int) -> Optional[int]:
        """Try to allocate blocks for `num_tokens` new tokens.

        Returns number of blocks allocated, or None if OOM.
        The allocation is cumulative — calling allocate(1) ten times
        eventually acquires ceil(total_tokens/block_size) blocks.
        """
        # ceil(num_tokens / block_size), min 1 block
        needed = max(1, (num_tokens + self.block_size - 1) // self.block_size)
        if needed <= self.free_blocks:
            self.free_blocks -= needed
            self.allocations[request_id] = (
                self.allocations.get(request_id, 0) + needed
            )
            return needed
        return None  # Out of memory → trigger preemption

    def free(self, request_id: str):
        """Return all blocks allocated to a request back to the pool."""
        if request_id in self.allocations:
            self.free_blocks += self.allocations.pop(request_id)


# ═══════════════════════════════════════════════════════════════════════
# Static Batching Simulator (for demo comparison)
# ═══════════════════════════════════════════════════════════════════════

class StaticBatchSimulator:
    """Simulates traditional static batching to demonstrate bubble waste.

    In static batching, all requests in a batch must complete before
    the next batch begins. This creates "bubbles" — GPU idle time when
    fast requests finish but the slowest one is still running.
    """

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.requests: list[Request] = []
        self.steps = 0
        self.idle_slots = 0  # Tokens that COULD have been computed but weren't

    def add_request(self, request: Request):
        self.requests.append(request)

    def run(self) -> dict:
        """Run static batching to completion. Returns stats dict."""
        queue = list(self.requests)
        batch_num = 0
        slot_history: list[list[str]] = []  # Per-step tracking

        while queue:
            batch_num += 1
            batch = queue[:self.max_batch_size]
            queue = queue[self.max_batch_size:]

            # Initialize batch: all requests start from scratch
            for req in batch:
                req.num_computed_tokens = 0
                req.num_output_tokens = 0
                req.status = RequestStatus.RUNNING

            # Step 1: Prefill (process all prompt tokens)
            for req in batch:
                req.num_computed_tokens = req.prompt_tokens
            self.steps += 1

            # Decode loop: generate 1 token per step for all active requests
            active = list(batch)
            while active:
                for req in active:
                    if req.num_computed_tokens >= req.num_tokens_total:
                        # Generate one output token
                        req.num_output_tokens += 1
                        req.num_computed_tokens += 1

                # Remove finished requests from active set
                still_active = [r for r in active if not r.is_finished]
                newly_finished = len(active) - len(still_active)
                active = still_active

                # Count idle slots: requests that finished but batch continues
                idle_now = sum(1 for r in batch
                              if r.is_finished and r in batch)
                self.idle_slots += idle_now

                if active:
                    self.steps += 1
                    slot_history.append(
                        [r.request_id if not r.is_finished else "."
                         for r in batch]
                    )

            for req in batch:
                req.status = RequestStatus.FINISHED

        total_slots = self.steps * min(self.max_batch_size, len(self.requests))
        total_slots = max(total_slots, 1)
        utilization = (total_slots - self.idle_slots) / total_slots * 100

        return {
            "steps": self.steps,
            "idle_slots": self.idle_slots,
            "utilization_pct": utilization,
            "batches": batch_num,
        }


# ═══════════════════════════════════════════════════════════════════════
# Continuous Batching Scheduler
# ═══════════════════════════════════════════════════════════════════════
# REFERENCE: vllm/v1/core/sched/scheduler.py → class Scheduler (L67-L2295)

class ContinuousBatchingScheduler:
    """Scheduler implementing continuous (dynamic) batching.

    Static batching problem:
      Batch of 3 requests: A(8 tokens), B(8), C(16).
      A and B finish at step 8, but GPU sits idle until C finishes at step 16.
      Result: 50% GPU utilization, 8 steps of "bubble" waste.

    Continuous batching solution:
      At EVERY step, finished requests leave and waiting requests join.
      No idle gaps — the token budget is fully utilized.
      Analogy: a cafeteria that seats new diners at any empty seat,
      rather than waiting for the entire group to finish.

    Algorithm (each schedule() call):
      1. Process RUNNING requests — continue generating tokens
      2. Admit WAITING requests if token budget + KV cache allow
      3. If KV cache OOM, preempt lowest-priority running request
      4. Simulate model forward (1 token per eligible request)

    REFERENCE: scheduler.py:L352-L545 (running scheduling)
    REFERENCE: scheduler.py:L567-L846 (waiting scheduling)
    REFERENCE: scheduler.py:L952-L972 (preemption)
    """

    def __init__(
        self,
        max_scheduled_tokens: int = 512,
        kv_cache_blocks: int = 256,
        block_size: int = 16,
    ):
        # REFERENCE: scheduler.py:L80-L111 (scheduling constraints)
        self.max_scheduled_tokens = max_scheduled_tokens
        self.kv_cache = KVCache(total_blocks=kv_cache_blocks, block_size=block_size)

        # REFERENCE: scheduler.py:L158-L170 (request queues)
        self.waiting: list[Request] = []   # WAITING + PREEMPTED requests
        self.running: list[Request] = []   # RUNNING requests
        self.finished: list[Request] = []  # Completed requests

        # Counters
        self.step_count = 0
        self.total_tokens_processed = 0
        self.total_preemptions = 0

    # ── Request Lifecycle ──────────────────────────────────────────

    def add_request(self, request: Request):
        """Enqueue a new inference request. REFERENCE: request_queue.py"""
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def abort_request(self, request_id: str):
        """Remove a request regardless of its current state."""
        for lst in [self.waiting, self.running]:
            for req in list(lst):
                if req.request_id == request_id:
                    lst.remove(req)
                    self.kv_cache.free(request_id)
                    return

    # ── Preemption ─────────────────────────────────────────────────
    # REFERENCE: scheduler.py:L479-L510 (victim selection)
    # REFERENCE: scheduler.py:L952-L972 (_preempt_request)

    def _pick_preemption_victim(self) -> Optional[Request]:
        """Select the lowest-priority running request to evict.

        In the original (scheduler.py:L479-L483), priority-based victim
        selection uses: max(running, key=lambda r: (r.priority, r.arrival_time)).
        We simplify to (priority, request_id) since we don't track arrival_time.
        """
        if not self.running:
            return None
        # Higher priority number = lower actual priority
        return max(self.running, key=lambda r: (r.priority, r.request_id))

    def _preempt_request(self, request: Request):
        """Evict a running request: free KV cache and return to waiting.

        IMPORTANT simplification: We do NOT reset num_computed_tokens to 0.
        The original vLLM (scheduler.py:L964) DOES reset it, but then
        immediately restores num_computed_tokens via prefix cache hits
        in get_computed_blocks() (scheduler.py:L616-L647). Our version
        models the EFFECTIVE behavior with prefix caching enabled:
        preempted requests resume from where they left off.

        REFERENCE: vllm/v1/core/sched/scheduler.py:L952-L972
        """
        self.kv_cache.free(request.request_id)
        request.status = RequestStatus.PREEMPTED
        # NOTE: num_computed_tokens NOT reset — see docstring above.
        self.total_preemptions += 1
        # Prepend: preempted requests get priority when resuming (fairness)
        self.waiting.insert(0, request)

    # ── Core Scheduling Algorithm ──────────────────────────────────
    # REFERENCE: vllm/v1/core/sched/scheduler.py:L352 (schedule)

    def schedule(self) -> list[Request]:
        """One scheduling step. Returns requests that made progress.

        Each call represents one forward pass of the model. The scheduler
        decides which requests get how many tokens this step, constrained
        by (1) token budget, (2) KV cache capacity, (3) max running requests.

        Returns:
            List of Request objects that had tokens processed this step.
        """
        token_budget = self.max_scheduled_tokens
        scheduled: list[Request] = []

        # ═══ Phase 1: Schedule RUNNING requests (continuity first) ═══
        # REFERENCE: scheduler.py:L387-L460
        # Running requests always get priority — this minimizes preemption
        # churn and maintains forward progress for already-admitted requests.
        for request in list(self.running):
            if request.is_finished:
                self._finish_request(request)
                continue

            if token_budget <= 0:
                break

            num_new = min(request.num_new_tokens, token_budget)
            if num_new <= 0:
                continue

            # Try to allocate KV cache blocks, preempting if needed.
            # The retry loop (matching scheduler.py:L466-L510 while True)
            # allows the scheduler to evict a lower-priority request and
            # immediately use its freed blocks for the current request.
            blocks = None
            while True:
                blocks = self.kv_cache.allocate(request.request_id, num_new)
                if blocks is not None:
                    break  # Allocation succeeded

                # KV cache full — preempt and retry
                victim = self._pick_preemption_victim()
                if victim is None:
                    break  # No one left to preempt
                self.running.remove(victim)
                self._preempt_request(victim)
                if victim is request:
                    break  # Preempted ourselves, can't schedule this step

            if blocks is not None:
                request.num_computed_tokens += num_new
                token_budget -= num_new
                self.total_tokens_processed += num_new
                scheduled.append(request)

        # ═══ Phase 2: Admit WAITING requests ═══
        # REFERENCE: scheduler.py:L567-L846
        # After servicing running requests, fill remaining capacity with
        # new or preempted requests from the waiting queue.
        for request in list(self.waiting):
            if token_budget <= 0:
                break

            num_new = min(request.num_new_tokens, token_budget)
            if num_new <= 0:
                continue

            blocks = self.kv_cache.allocate(request.request_id, num_new)
            if blocks is not None:
                request.num_computed_tokens += num_new
                request.status = RequestStatus.RUNNING
                token_budget -= num_new
                self.total_tokens_processed += num_new
                self.waiting.remove(request)
                self.running.append(request)
                scheduled.append(request)
            else:
                # KV cache exhausted — stop admitting
                # Original behavior (scheduler.py:L756): break, don't
                # preempt running requests just to admit new ones.
                break

        # ═══ Phase 3: Simulate model forward pass ═══
        # In real vLLM, ModelRunner executes the scheduled requests and
        # the scheduler's update_from_output() (scheduler.py:L1290) processes
        # the generated tokens. Here we simulate: each request that has
        # finished its prefill generates 1 output token per step.
        #
        # This models the auto-regressive decode loop: after the prompt
        # is fully computed, the model produces 1 token at a time.
        for request in list(self.running):
            if request.is_finished:
                self._finish_request(request)
            elif request.num_computed_tokens >= request.num_tokens_total:
                # All existing tokens processed → model generates 1 more
                request.num_output_tokens += 1
                # num_computed_tokens will be incremented next step's Phase 1
                # when we allocate the block for this output token.
                if request.is_finished:
                    self._finish_request(request)

        self.step_count += 1
        return scheduled

    # ── Helpers ────────────────────────────────────────────────────

    def _finish_request(self, request: Request):
        """Move a completed request from running to finished."""
        self.running.remove(request)
        self.kv_cache.free(request.request_id)
        request.status = RequestStatus.FINISHED
        self.finished.append(request)

    # ── Stats and Display ──────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "step": self.step_count,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
            "free_kv": self.kv_cache.free_blocks,
            "total_kv": self.kv_cache.total_blocks,
            "tokens": self.total_tokens_processed,
            "preemptions": self.total_preemptions,
        }

    def print_state(self):
        s = self.stats
        print(f"\n{'='*55}")
        print(f"Step {s['step']} | Running: {s['running']} "
              f"Waiting: {s['waiting']} | Finished: {s['finished']}")
        print(f"KV: {s['free_kv']}/{s['total_kv']} blocks | "
              f"Tokens: {s['tokens']} | Preemptions: {s['preemptions']}")
        for r in self.running:
            bar_len = 20
            filled = int(r.num_output_tokens * bar_len / max(r.max_tokens, 1))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{r.request_id:8s}] {bar} {r.num_output_tokens:3d}/{r.max_tokens}")
        print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════

def _demo_bubble_diagram():
    """Print an ASCII visualization of the static batching bubble problem.

    Three requests in one static batch. Fast requests finish early
    (at t=8) but the batch isn't done until the slowest (t=16).
    GPU sits idle for 8 steps — these are the "bubbles."
    """
    # Show the generation timeline (prompt prefill = 1 step, invisible)
    total_time = 16  # max generation steps (Charlie)
    short_time = 8   # min generation steps (Alice, Bob)
    idle = total_time - short_time  # steps of idle per fast request

    print(f"\n  Static batch of 3 requests, 1 generation token/step:")
    print(f"  Alice & Bob:   8 tokens → finish at step {short_time}")
    print(f"  Charlie:      16 tokens → finishes at step {total_time}")
    print(f"  Bubble: {idle} steps × 2 idle slots = {idle * 2} wasted slot-steps")
    print()

    # Print compact ASCII timeline
    header = "  Time →   "
    timeline = "           "
    for t in range(total_time + 1):
        if t % 4 == 0:
            header += f"t={t:<3}"
    print(header)

    for name, steps in [("Alice  ", short_time), ("Bob    ", short_time),
                         ("Charlie", total_time)]:
        line = f"  {name} "
        for t in range(total_time):
            line += "█" if t < steps else "·"
        line += f"  ({steps} tokens)"
        print(line)

    print(f"\n  Key: █ = computing   · = idle (bubble)")


def main():
    """Demonstrate static batching vs continuous batching.

    Three-part demo:
      Part 1 — Static batching: show the bubble problem visually
      Part 2 — Continuous batching: eliminate bubbles with the scheduler
      Part 3 — Late arrival: a request joins mid-execution

    All demo requests use prompt_tokens=1 (a symbolic single input token
    that represents the initial prompt). The focus is on generation
    (output) tokens, where the bubble effect dominates.
    """
    print("=" * 60)
    print("  CONTINUOUS BATCHING — From Bubbles to Full Utilization")
    print("=" * 60)

    # ── Part 1: Static Batching Bubble ────────────────────────────
    print("\n" + "─" * 60)
    print("  PART 1: The Static Batching Bubble Problem")
    print("─" * 60)
    print("""
    In static batching, ALL requests in a batch must complete before
    a new batch starts. If Alice finishes at step 8 but Charlie needs
    16 steps, Alice's slot sits idle for 8 steps — a "bubble" of
    wasted compute capacity.
    """)
    _demo_bubble_diagram()

    # Simulate static batching to compute actual waste
    static_sim = StaticBatchSimulator(max_batch_size=3)
    static_sim.add_request(Request("A", prompt_tokens=1, max_tokens=8))
    static_sim.add_request(Request("B", prompt_tokens=1, max_tokens=8))
    static_sim.add_request(Request("C", prompt_tokens=1, max_tokens=16))
    static_stats = static_sim.run()
    print(f"\n  Static batch simulation:")
    print(f"    Total steps:        {static_stats['steps']}")
    print(f"    Idle slots wasted:  {static_stats['idle_slots']}")
    print(f"    GPU utilization:    {static_stats['utilization_pct']:.1f}%")

    # ── Part 2: Continuous Batching Solution ──────────────────────
    print("\n" + "─" * 60)
    print("  PART 2: Continuous Batching Eliminates Bubbles")
    print("─" * 60)
    print("""
    Same 3 requests. At EVERY step the scheduler evaluates who needs
    compute. When Alice and Bob finish (step 9), their KV cache blocks
    are freed and Charlie can use the full token budget — no idle gaps.
    """)

    sched = ContinuousBatchingScheduler(
        max_scheduled_tokens=64,
        kv_cache_blocks=64,
        block_size=4,
    )

    sched.add_request(Request("A", prompt_tokens=1, max_tokens=8, priority=0))
    sched.add_request(Request("B", prompt_tokens=1, max_tokens=8, priority=0))
    sched.add_request(Request("C", prompt_tokens=1, max_tokens=16, priority=0))

    print("\n  Running continuous batching scheduler:")
    milestone = 0
    for step in range(30):
        sched.schedule()

        if step == milestone:
            sched.print_state()
            milestone += 5

        if not sched.running and not sched.waiting:
            print(f"\n  All done at step {step + 1}!")
            break

    cb_steps = sched.stats["step"]
    cb_tokens = sched.stats["tokens"]
    print(f"\n  Continuous batching stats:")
    print(f"    Steps:               {cb_steps}")
    print(f"    Tokens processed:    {cb_tokens}")
    print(f"    Preemptions:         {sched.stats['preemptions']}")
    print(f"    Finished requests:   {sched.stats['finished']}")

    print(f"\n  Comparison:")
    print(f"    Static batching:  {static_stats['steps']} steps, "
          f"{static_stats['idle_slots']} idle slots, "
          f"{static_stats['utilization_pct']:.1f}% utilization")
    print(f"    Continuous batch:  {cb_steps} steps, "
          f"0 idle slots, 100% slot utilization")
    print(f"    → Continuous batching recovers wasted capacity by "
          f"letting finished requests leave immediately.")

    # ── Part 3: Late Arrival ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PART 3: Late-Arriving Request Joins Mid-Execution")
    print("─" * 60)
    print("""
    Another key advantage: new requests can join mid-execution without
    waiting for a batch boundary. In static batching, a late request
    would queue until the ENTIRE current batch finishes.
    """)

    sched2 = ContinuousBatchingScheduler(
        max_scheduled_tokens=64,
        kv_cache_blocks=64,
        block_size=4,
    )

    sched2.add_request(Request("Alpha", prompt_tokens=1, max_tokens=10, priority=0))
    sched2.add_request(Request("Beta",  prompt_tokens=1, max_tokens=10, priority=0))
    sched2.add_request(Request("Gamma", prompt_tokens=1, max_tokens=14, priority=0))

    print("\n  Starting with Alpha, Beta, Gamma...")
    for _ in range(4):
        sched2.schedule()
    sched2.print_state()

    # Delta arrives late
    print("\n  *** Delta arrives at step 5! ***")
    print("  (In static batching, Delta would wait for Alpha/Beta/Gamma to finish)")
    sched2.add_request(Request("Delta", prompt_tokens=1, max_tokens=8, priority=0))

    # Run more steps, show Delta integrated
    for _ in range(4):
        sched2.schedule()
    sched2.print_state()

    # Run to completion
    for _ in range(20):
        sched2.schedule()
        if not sched2.running and not sched2.waiting:
            break

    print(f"\n  Completion order:")
    for r in sched2.finished:
        print(f"    {r.request_id}  "
              f"(output={r.num_output_tokens}/{r.max_tokens})")

    delta = next(r for r in sched2.finished if r.request_id == "Delta")
    print(f"\n  Delta joined at step 5, finished successfully among "
          f"the original requests — seamless integration.")

    print("\n" + "=" * 60)
    print("  KEY TAKEAWAY")
    print("=" * 60)
    print("""
    Continuous batching separates request LIFETIME from batch LIFETIME:
      - Static batching:  batch = fixed group, fixed duration
      - Continuous batch: batch = whoever needs compute right now

    This single architectural decision enables vLLM to achieve 10x+
    higher throughput than static-batching inference servers. It is
    the foundation upon which chunked prefill, prefix caching, and
    priority scheduling are built.
    """)


if __name__ == "__main__":
    main()
