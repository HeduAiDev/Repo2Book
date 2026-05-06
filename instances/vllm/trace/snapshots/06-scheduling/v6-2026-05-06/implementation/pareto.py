# REFERENCE: instances/vllm/source/vllm/config/scheduler.py:L60-L120
# (max_num_seqs, max_num_batched_tokens, policy, long_prefill_token_threshold)
"""Schedule latency vs throughput Pareto frontier.

Three knobs, one trade-off curve:

    max_num_seqs                  ↔ vllm/config/scheduler.py:L63
    max_num_batched_tokens        ↔ vllm/config/scheduler.py:L?? (computed)
    long_prefill_token_threshold  ↔ vllm/config/scheduler.py (caps Phase 1
                                    new tokens per request per step;
                                    used at scheduler.py:L413-L414)

The Pareto frontier:

    LATENCY axis: P95 time-to-first-token (TTFT) for a small request
        sharing the engine with one long-prefill request.
    THROUGHPUT axis: tokens/sec across the whole engine.

Increasing `max_num_seqs` improves throughput (more concurrency) but hurts
TTFT (more contention for token budget). `long_prefill_token_threshold`
caps any single request's per-step share, improving fairness but hurting
peak prefill throughput on large prompts.

This module gives an analytical model — not a real benchmark — so a writer
can ground the narrative in numbers a reader can reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    """The three knobs that define an operating point."""

    # REFERENCE: instances/vllm/source/vllm/config/scheduler.py:L63
    max_num_seqs: int                         # config.scheduler.max_num_seqs
    # REFERENCE: instances/vllm/source/vllm/config/scheduler.py — max_num_batched_tokens
    max_num_batched_tokens: int               # the per-step token budget B
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L413-L414
    long_prefill_token_threshold: int = 0     # 0 = disabled (no per-req cap)


@dataclass
class PerfPoint:
    """One operating point on the Pareto curve."""

    config: EngineConfig
    throughput_tokens_per_sec: float
    p95_ttft_seconds: float
    saturation: float  # 0..1, fraction of token budget actually used at steady state


def estimate_throughput(
    cfg: EngineConfig,
    avg_seq_len: int,
    decode_throughput_tokens_per_sec: float = 50_000.0,
) -> float:
    """Throughput model. Two regimes:

    1. Compute-bound (small batch, low concurrency):
        throughput ~= decode_throughput * min(running, max_num_seqs)
       (each running request adds one decode-token per step).

    2. Memory-bound (saturated batch):
        throughput ~= max_num_batched_tokens (budget is the cap).

    The minimum of the two regimes is the achievable throughput.

    REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L848-L853
    (post-loop assertion `total_num_scheduled_tokens <= max_num_scheduled_tokens`
     — the budget IS the per-step throughput cap)
    """
    compute_bound = decode_throughput_tokens_per_sec * min(cfg.max_num_seqs, 256)
    memory_bound = float(cfg.max_num_batched_tokens) * 1000  # *1000 because budget is per-step ms-scale
    # Simplification: treat the smaller as the bottleneck.
    return min(compute_bound, memory_bound)


def estimate_p95_ttft(
    cfg: EngineConfig,
    long_prompt_tokens: int,
    short_prompt_tokens: int,
) -> float:
    """Time-to-first-token for a short request when a long request is
    prefilling on the same engine.

    Without `long_prefill_token_threshold`, the long prompt can soak the
    entire budget for `ceil(long_prompt_tokens / max_num_batched_tokens)`
    steps before the short request gets a slice. With the threshold set,
    each step the long request takes at most `threshold` tokens, leaving
    `B - threshold` for the short request.

    REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L413-L415
    (`if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
       num_new_tokens = self.scheduler_config.long_prefill_token_threshold`)
    REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L678-L680
    (same threshold applied in Phase 2 for waiting requests)
    """
    if cfg.long_prefill_token_threshold == 0 or cfg.long_prefill_token_threshold >= cfg.max_num_batched_tokens:
        # Long prompt can occupy the full budget; short waits for it to drain.
        steps_to_wait = (long_prompt_tokens + cfg.max_num_batched_tokens - 1) // cfg.max_num_batched_tokens
    else:
        # Each step long takes `threshold`, short can fit in the leftover.
        leftover = cfg.max_num_batched_tokens - cfg.long_prefill_token_threshold
        steps_to_wait = (short_prompt_tokens + leftover - 1) // leftover
    # Each scheduler step is ~50 ms in a real engine (model fwd + sample).
    step_seconds = 0.050
    return steps_to_wait * step_seconds


def estimate_saturation(
    cfg: EngineConfig,
    avg_seq_len: int,
    arrival_rate_per_sec: float = 10.0,
) -> float:
    """Fraction of `max_num_batched_tokens` that the engine actually uses
    at steady state. Too high → tail-heavy. Too low → underutilized.
    """
    demand_tokens_per_sec = arrival_rate_per_sec * avg_seq_len
    # Scheduler steps at ~20 Hz; budget is per step.
    supply_tokens_per_sec = cfg.max_num_batched_tokens * 20
    return min(1.0, demand_tokens_per_sec / supply_tokens_per_sec)


def sweep(
    knobs: list[EngineConfig],
    long_prompt_tokens: int = 8192,
    short_prompt_tokens: int = 64,
    avg_seq_len: int = 512,
) -> list[PerfPoint]:
    """Evaluate all configs; caller plots the Pareto frontier."""
    points = []
    for cfg in knobs:
        points.append(
            PerfPoint(
                config=cfg,
                throughput_tokens_per_sec=estimate_throughput(cfg, avg_seq_len),
                p95_ttft_seconds=estimate_p95_ttft(
                    cfg, long_prompt_tokens, short_prompt_tokens
                ),
                saturation=estimate_saturation(cfg, avg_seq_len),
            )
        )
    return points


def pareto_front(points: list[PerfPoint]) -> list[PerfPoint]:
    """Non-dominated set. A point dominates another if it has strictly
    higher throughput AND strictly lower p95_ttft.
    """
    front = []
    for p in points:
        if not any(
            q.throughput_tokens_per_sec >= p.throughput_tokens_per_sec
            and q.p95_ttft_seconds <= p.p95_ttft_seconds
            and (
                q.throughput_tokens_per_sec > p.throughput_tokens_per_sec
                or q.p95_ttft_seconds < p.p95_ttft_seconds
            )
            for q in points
        ):
            front.append(p)
    front.sort(key=lambda p: p.throughput_tokens_per_sec)
    return front
