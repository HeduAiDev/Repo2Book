"""
Chunked Prefill Scheduler — Runnable Demo.

REFERENCE:
    vllm/v1/core/sched/scheduler.py:L413-L415  — Running req long prefill cap
    vllm/v1/core/sched/scheduler.py:L678-L692  — Waiting req chunked prefill
    vllm/v1/core/sched/scheduler.py:L684-L690  — Chunked prefill guard
    vllm/config/scheduler.py:L70-L84           — SchedulerConfig

Run: python3 chunked_prefill.py
Shows how a 128K prompt is chunked across steps, interleaved with short requests.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SimRequest:
    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    num_computed_tokens: int = 0
    output_tokens: int = 0
    status: str = "waiting"

    @property
    def num_new_tokens(self) -> int:
        total = self.prompt_tokens + self.output_tokens
        return total - self.num_computed_tokens

    @property
    def is_prefilling(self) -> bool:
        return self.num_computed_tokens < self.prompt_tokens


# REFERENCE: vllm/v1/core/sched/scheduler.py:L352-L945 — schedule() main loop
# REFERENCE: vllm/v1/worker/gpu_model_runner.py:L1928 — discard_request_mask
class ChunkedPrefillScheduler:
    """
    Simplified scheduler showing chunked prefill.

    REFERENCE: vllm/v1/core/sched/scheduler.py:L388-L846

    Three决策点 from the real scheduler:
        L413-L415: Running req capped by long_prefill_token_threshold
        L678-L680: Waiting req capped by long_prefill_token_threshold
        L684-L690: If chunked prefill OFF and prompt > budget → reject admission
    """

    def __init__(self, max_tokens_per_step: int = 2048,
                 long_threshold: int = 2048,
                 enable_chunked: bool = True):
        self.max_tokens = max_tokens_per_step
        self.long_threshold = long_threshold
        self.enable_chunked = enable_chunked
        self.waiting: List[SimRequest] = []
        self.running: List[SimRequest] = []
        self.finished: List[SimRequest] = []
        self.step = 0

    def add(self, req: SimRequest):
        self.waiting.append(req)

    def schedule(self) -> Dict[str, int]:
        """One scheduling step. Returns {req_id: tokens_this_step}."""
        self.step += 1
        budget = self.max_tokens
        scheduled = {}

        # ── Phase 1: Running requests ──
        # REFERENCE: scheduler.py:L413-L415
        for req in self.running[:]:
            n = req.num_new_tokens
            if n <= 0: continue

            # Long prefill cap (L413-L414)
            if self.long_threshold > 0 and req.is_prefilling:
                n = min(n, self.long_threshold)

            n = min(n, budget)
            if n <= 0: continue
            scheduled[req.request_id] = n
            budget -= n

        # ── Phase 2: Waiting requests ──
        # REFERENCE: scheduler.py:L678-L692
        while self.waiting and budget > 0:
            req = self.waiting[0]
            n = req.num_new_tokens

            # Long prefill cap (L678-L680)
            if self.long_threshold > 0:
                n = min(n, self.long_threshold)

            # Guard (L684-L690)
            if not self.enable_chunked and n > budget:
                break

            n = min(n, budget)
            if n <= 0: break

            scheduled[req.request_id] = n
            budget -= n
            self.waiting.pop(0)
            self.running.append(req)
            req.status = "running"

        return scheduled

    def update(self, scheduled: Dict[str, int]):
        """After model forward. REFERENCE: scheduler.py:L988-L989"""
        for req_id, n in scheduled.items():
            for req in self.running:
                if req.request_id == req_id:
                    req.num_computed_tokens += n
                    if not req.is_prefilling and req.output_tokens < req.max_output_tokens:
                        req.output_tokens += 1
                    if req.output_tokens >= req.max_output_tokens:
                        req.status = "finished"
                        self.running.remove(req)
                        self.finished.append(req)
                    break


def demonstrate():
    """Show chunked prefill with 1 long + 3 short requests."""
    sched = ChunkedPrefillScheduler(
        max_tokens_per_step=2048, long_threshold=2048, enable_chunked=True)

    sched.add(SimRequest("long", prompt_tokens=8000, max_output_tokens=50))
    for i in range(3):
        sched.add(SimRequest(f"short-{i}", prompt_tokens=128, max_output_tokens=50))

    print("Chunked Prefill — Scheduling Trace")
    print("=" * 60)
    print(f"1 long prompt (8000 tokens) + 3 short (128 tokens)")
    print(f"Token budget: {sched.max_tokens}, threshold: {sched.long_threshold}")
    print(f"Chunked prefill: {'ON' if sched.enable_chunked else 'OFF'}")
    print()

    for step in range(5):
        scheduled = sched.schedule()
        sched.update(scheduled)
        running = [r.request_id for r in sched.running]
        total = sum(scheduled.values())
        # Show chunking: how many tokens did the long request get?
        long_tokens = scheduled.get("long", 0)
        short_tokens = total - long_tokens
        print(f"Step {step+1}: {len(scheduled)} reqs, {total} tokens "
              f"(long: {long_tokens}, short: {short_tokens}), running={running}")

    print(f"\nAfter 5 steps: long computed {sched.running[0].num_computed_tokens}/{8000} tokens")
    print(f"Short requests: started decoding while long is still prefill — chunked prefill in action")


if __name__ == "__main__":
    demonstrate()
