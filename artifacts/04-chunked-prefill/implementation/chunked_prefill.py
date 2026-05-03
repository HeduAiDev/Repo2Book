"""
Chunked Prefill — Our Reimplementation.

REFERENCE sources:
    Running req long cap:       vllm/v1/core/sched/scheduler.py:L413-L415
    Waiting req long cap:       vllm/v1/core/sched/scheduler.py:L678-L692
    Chunked prefill guard:      vllm/v1/core/sched/scheduler.py:L684-L690
    is_prefill_chunk:           vllm/v1/core/sched/scheduler.py:L988-L989
    discard_request_mask:       vllm/v1/worker/gpu_model_runner.py:L1928-L1933
    SchedulerConfig:            vllm/config/scheduler.py:L70-L84
    scheduler_reserve_full_isl: vllm/v1/core/sched/scheduler.py:L753

Key concept:
    Chunked prefill splits a long prompt into multiple scheduling steps.
    Each step processes a chunk (capped by long_prefill_token_threshold),
    interleaved with decode tokens from other requests.

    Without chunked prefill: a 128K prompt blocks ALL other requests
    until prefill completes. TTFT for that request is fast (one shot),
    but throughput is terrible (all others wait).

    With chunked prefill: 128K is split into 64 × 2K chunks, distributed
    across 64 steps. TTFT increases (64 forward passes before first token),
    but throughput is much higher (other requests interleaved).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Chunked Prefill Configuration
# REFERENCE: vllm/config/scheduler.py:L70-L84
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkedPrefillConfig:
    """
    REFERENCE: vllm/config/scheduler.py — SchedulerConfig
    """
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0     # 0 = disabled; >0 = cap per step
    max_num_scheduled_tokens: int = 2048      # total token budget per step
    scheduler_reserve_full_isl: bool = True   # check full sequence fits in KV cache


# ═══════════════════════════════════════════════════════════════════════════
# Simplified Request (for scheduling demo)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimRequest:
    request_id: str
    prompt_tokens: int           # total prompt length
    max_output_tokens: int
    num_computed_tokens: int = 0
    output_tokens: int = 0
    is_prefill_chunk: bool = False  # REFERENCE: scheduler.py:L988
    status: str = "waiting"        # waiting, running, finished

    @property
    def num_new_tokens(self) -> int:
        """Tokens still to compute = total - already computed."""
        total = self.prompt_tokens + self.output_tokens
        return total - self.num_computed_tokens

    @property
    def is_prefilling(self) -> bool:
        return self.num_computed_tokens < self.prompt_tokens

    @property
    def ttft_steps(self) -> int:
        """How many scheduling steps before first output token."""
        if not self.is_prefilling:
            return 0
        remaining = self.prompt_tokens - self.num_computed_tokens
        return (remaining + 2047) // 2048  # rough estimate


# ═══════════════════════════════════════════════════════════════════════════
# Chunked Prefill Scheduler
# REFERENCE: vllm/v1/core/sched/scheduler.py:L352-L945
# ═══════════════════════════════════════════════════════════════════════════

class ChunkedPrefillScheduler:
    """
    Simplified scheduler demonstrating chunked prefill logic.

    REFERENCE: scheduler.py:L352 — schedule() main loop
               scheduler.py:L413-L415 — running request long prefill cap
               scheduler.py:L678-L692 — waiting request chunked prefill
    """

    def __init__(self, config: ChunkedPrefillConfig):
        self.config = config
        self.waiting: List[SimRequest] = []
        self.running: List[SimRequest] = []
        self.finished: List[SimRequest] = []
        self.step_count = 0
        self.schedule_log: List[Dict] = []

    def add_request(self, req: SimRequest):
        self.waiting.append(req)

    def schedule(self) -> Dict[str, int]:
        """
        One scheduling step. Returns {req_id: num_tokens_scheduled}.

        REFERENCE: scheduler.py:L388-L846
        """
        self.step_count += 1
        token_budget = self.config.max_num_scheduled_tokens
        scheduled: Dict[str, int] = {}

        # ── Phase 1: Running requests (continue prefills, decodes) ──
        for req in self.running[:]:
            num_new = req.num_new_tokens
            if num_new <= 0:
                continue

            # Long prefill cap (REFERENCE: L413-L414)
            threshold = self.config.long_prefill_token_threshold
            if threshold > 0 and req.is_prefilling:
                num_new = min(num_new, threshold)

            # Token budget cap
            num_new = min(num_new, token_budget)
            if num_new <= 0:
                continue

            scheduled[req.request_id] = num_new
            token_budget -= num_new

        # ── Phase 2: Waiting requests (new prefills, chunked) ──
        # REFERENCE: scheduler.py:L677-L692
        while self.waiting and token_budget > 0:
            req = self.waiting[0]
            num_new = req.num_new_tokens

            # Long prefill cap (REFERENCE: L678-L680)
            threshold = self.config.long_prefill_token_threshold
            if threshold > 0:
                num_new = min(num_new, threshold)

            # Chunked prefill guard (REFERENCE: L684-L690)
            if not self.config.enable_chunked_prefill:
                if num_new > token_budget:
                    break  # Prompt too large, can't admit

            # Token budget cap (REFERENCE: L692)
            num_new = min(num_new, token_budget)
            if num_new <= 0:
                break

            scheduled[req.request_id] = num_new
            token_budget -= num_new

            self.waiting.pop(0)
            self.running.append(req)
            req.status = "running"

        self.schedule_log.append(scheduled)
        return scheduled

    def update_after_step(self, scheduled: Dict[str, int]):
        """REFERENCE: scheduler.py:L974-L990"""
        for req_id, num_tokens in scheduled.items():
            for req in self.running:
                if req.request_id == req_id:
                    req.num_computed_tokens += num_tokens

                    # Set is_prefill_chunk (REFERENCE: L988-L989)
                    total = req.prompt_tokens + req.output_tokens
                    req.is_prefill_chunk = req.num_computed_tokens < total

                    # After prefill complete, generate output tokens
                    if not req.is_prefilling and req.output_tokens < req.max_output_tokens:
                        req.output_tokens += 1  # One decode token per step

                    # Check completion
                    if req.output_tokens >= req.max_output_tokens:
                        req.status = "finished"
                        self.running.remove(req)
                        self.finished.append(req)
                    break


# ═══════════════════════════════════════════════════════════════════════════
# Analysis: TTFT vs Throughput
# ═══════════════════════════════════════════════════════════════════════════

def ttft_vs_throughput_analysis(
    long_prompt_len: int = 128000,
    short_prompt_len: int = 128,
    num_short_requests: int = 8,
    max_tokens_per_step: int = 2048,
    long_threshold: int = 2048,
    output_len: int = 256,
) -> dict:
    """
    Quantify the TTFT-throughput trade-off of chunked prefill.

    Scenario: 1 long prompt (128K) + 8 short prompts (128 tokens each).

    WITHOUT chunked prefill:
        - Long prompt must be prefilled entirely (128K tokens) before any decode
        - Short prompts wait in queue
        - TTFT(long) = 128K/2048 ≈ 63 steps
        - Short TTFT(hort) = 63 + 128/2048 ≈ 63 steps (wait for long prefill)

    WITH chunked prefill:
        - Long prompt split into 64 × 2K chunks
        - Each step: 1 chunk (2K) + 8 decode tokens from short requests
        - Short requests start decoding much earlier
        - TTFT(long) ≈ 64 steps (same)
        - Short TTFT(hort) ≈ 1 step (immediately admitted!)
    """
    # Without chunked prefill: long prompt blocks everything
    long_prefill_steps = (long_prompt_len + max_tokens_per_step - 1) // max_tokens_per_step
    short_ttft_no_chunk = long_prefill_steps

    # With chunked prefill: long prompt interleaved with short decode
    chunks = (long_prompt_len + long_threshold - 1) // long_threshold
    # Each step: 1 chunk + up to num_short decode tokens
    # Short requests: admitted immediately, each generates 1 output token per step
    short_ttft_with_chunk = 1  # Immediately admitted, token in first step

    return {
        "long_prompt_len": long_prompt_len,
        "short_prompt_len": short_prompt_len,
        "num_short_requests": num_short_requests,
        "chunk_size": long_threshold,
        "num_chunks": chunks,
        "without_chunked_prefill": {
            "long_prefill_steps": long_prefill_steps,
            "short_ttft_steps": short_ttft_no_chunk,
            "total_steps_all_requests": long_prefill_steps + output_len,
        },
        "with_chunked_prefill": {
            "long_prefill_steps": chunks,
            "short_ttft_steps": short_ttft_with_chunk,
            "total_steps_all_requests": chunks + output_len,
            "throughput_gain": f"{short_ttft_no_chunk / short_ttft_with_chunk:.0f}× shorter TTFT for short requests",
        },
    }


def demonstrate():
    print("Chunked Prefill: TTFT vs Throughput")
    print("=" * 60)

    config = ChunkedPrefillConfig(
        enable_chunked_prefill=True,
        long_prefill_token_threshold=2048,
        max_num_scheduled_tokens=2048,
    )
    sched = ChunkedPrefillScheduler(config)

    # Add 1 long prompt + 3 short
    sched.add_request(SimRequest("long", prompt_tokens=8000, max_output_tokens=100))
    for i in range(3):
        sched.add_request(SimRequest(f"short-{i}", prompt_tokens=128, max_output_tokens=50))

    # Run scheduling
    for step in range(5):
        scheduled = sched.schedule()
        sched.update_after_step(scheduled)
        running = [r.request_id for r in sched.running]
        finished = [r.request_id for r in sched.finished]
        total_tokens = sum(scheduled.values())
        print(f"  Step {step+1}: {len(scheduled)} reqs, {total_tokens} tokens, "
              f"running={running}, finished={finished}")

    print()
    analysis = ttft_vs_throughput_analysis()
    print(f"Without chunked prefill: short TTFT = {analysis['without_chunked_prefill']['short_ttft_steps']} steps")
    print(f"With chunked prefill:    short TTFT = {analysis['with_chunked_prefill']['short_ttft_steps']} step")
    print(f"Gain: {analysis['with_chunked_prefill']['throughput_gain']}")


if __name__ == "__main__":
    demonstrate()
