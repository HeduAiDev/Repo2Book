# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py
"""Annotated runnable trace for Ch06 — the policy/strategy layer.

Usage:
    python3 -m instances.vllm.artifacts.06-scheduling.implementation.demo

Five sections (matching outline):

    [1] Decision-loop survey: `schedule() → step() → update_states()`
        Pointer to Ch04 for mechanics; we only verify policy primitives.
    [2] FCFS fairness: head-of-line blocking analysis (long blocks short).
    [3] Preemption strategies: recompute vs swap vs abort with numbers.
    [4] Priority + aging compensator (the latter is pedagogical, NOT vLLM).
    [5] Pareto frontier sweep over (max_num_seqs, max_num_batched_tokens,
        long_prefill_token_threshold).
"""

from __future__ import annotations

from .pareto import EngineConfig, pareto_front, sweep
from .policy import (
    PauseState,
    effective_token_budget,
    select_preemption_victim,
    select_waiting_queue,
)
from .preemption_strategy import (
    PreemptionScenario,
    PreemptionStrategy,
    crossover_prompt_length,
    expected_latency_under_oom_rate,
)
from .request_queue import (
    FCFSRequestQueue,
    PolicyRequest,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from .starvation_analysis import (
    WorkloadProfile,
    aged_priority,
    has_starvation,
    priority_ordering,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print("=" * 64)


def section_1_decision_loop() -> None:
    _section("[1] schedule() → step() → update_states()")
    print("    Loop mechanics: see Ch04 §4.x (artifacts/04-continuous-batching).")
    print("    Ch06 only inspects POLICY primitives:")
    print()

    # Verify select_preemption_victim works for both policies.
    running = [
        PolicyRequest("A", priority=1, arrival_time=0.0),
        PolicyRequest("B", priority=3, arrival_time=1.0),  # worst priority
        PolicyRequest("C", priority=2, arrival_time=2.0),  # last to arrive
    ]
    requester = PolicyRequest("D", priority=1, arrival_time=3.0)

    fcfs_victim = select_preemption_victim(running, requester, SchedulingPolicy.FCFS)
    prio_victim = select_preemption_victim(running, requester, SchedulingPolicy.PRIORITY)
    print(f"    FCFS victim:     {fcfs_victim.request_id}  (running.pop() = last admitted)")
    print(f"    PRIORITY victim: {prio_victim.request_id}  (max by priority,arrival)")

    # Verify pause-state gating.
    print()
    print(f"    budget=2048 + UNPAUSED:    -> {effective_token_budget(2048, PauseState.UNPAUSED):>4d}")
    print(f"    budget=2048 + PAUSED_NEW:  -> {effective_token_budget(2048, PauseState.PAUSED_NEW):>4d}  (only blocks admit; running progresses)")
    print(f"    budget=2048 + PAUSED_ALL:  -> {effective_token_budget(2048, PauseState.PAUSED_ALL):>4d}  (engine fully frozen)")

    # Verify select_waiting_queue (skipped wins under FCFS).
    print()
    waiting = create_request_queue(SchedulingPolicy.FCFS)
    skipped = create_request_queue(SchedulingPolicy.FCFS)
    waiting.add_request(PolicyRequest("new-1", arrival_time=10.0))
    skipped.add_request(PolicyRequest("blocked-1", arrival_time=2.0))
    chosen = select_waiting_queue(waiting, skipped, SchedulingPolicy.FCFS)
    print(f"    FCFS select_waiting_queue: chose {chosen.peek_request().request_id}  (skipped wins)")


def section_2_fcfs_fairness() -> None:
    _section("[2] FCFS head-of-line blocking — long prompt starves short")
    profile = WorkloadProfile(
        n_long=2,
        prompt_long=4096,
        n_short=8,
        prompt_short=64,
        token_budget=512,
        max_running=4,
    )
    short_wait = profile.fcfs_short_request_latency_steps()
    long_complete = profile.fcfs_long_request_completion_steps()
    factor = profile.head_of_line_blocking_factor()
    print(f"    Workload: {profile.n_long} long(4K) + {profile.n_short} short(64)")
    print(f"    Token budget B = {profile.token_budget}")
    print(f"    Long-request prefill steps:    {long_complete}")
    print(f"    Short-request worst-case wait: {short_wait} steps")
    print(f"    Head-of-line blocking factor:  {factor:.2f}x")
    print(f"    FCFS starves under this profile?    {has_starvation(profile, SchedulingPolicy.FCFS)}")
    print(f"    PRIORITY (well-assigned) starves?   {has_starvation(profile, SchedulingPolicy.PRIORITY)}")


def section_3_preemption_strategies() -> None:
    _section("[3] Preemption: recompute (v1) vs swap (v0) vs abort")
    scenarios = [
        ("short prompt (512)", PreemptionScenario(prompt_tokens=512, num_layers=32, num_kv_heads=8, head_size=128)),
        ("medium prompt (2K)", PreemptionScenario(prompt_tokens=2048, num_layers=32, num_kv_heads=8, head_size=128)),
        ("long prompt (8K)",   PreemptionScenario(prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128)),
        ("xlong prompt (32K)", PreemptionScenario(prompt_tokens=32768, num_layers=32, num_kv_heads=8, head_size=128)),
    ]
    print(f"    {'scenario':<22}  {'recompute':>10}  {'swap':>8}  {'abort':>8}  {'winner':>10}  {'kv (GiB)':>10}")
    for label, s in scenarios:
        kv_gib = s.kv_bytes / (1024**3)
        print(
            f"    {label:<22}  "
            f"{s.recompute_seconds() * 1000:>8.1f} ms  "
            f"{s.swap_seconds() * 1000:>6.1f} ms  "
            f"{s.abort_seconds() * 1000:>6.1f} ms  "
            f"{s.winner().value:>10}  "
            f"{kv_gib:>10.3f}"
        )

    # Crossover insight.
    base = scenarios[0][1]
    crossover = crossover_prompt_length(base)
    if crossover == -1:
        print(f"    Crossover analysis: prefill_tp={base.prefill_throughput_tokens_per_sec:.0f} tok/s,")
        print(f"                        bw={base.pcie_bandwidth_bytes_per_sec / 1024**3:.0f} GiB/s")
        print(f"                        → recompute always wins (TP > threshold)")
    else:
        print(f"    Crossover: swap always faster regardless of prompt length")
    print(f"    vLLM v1's choice: {PreemptionStrategy.RECOMPUTE.value} for ALL prompts.")
    print(f"    Reason: simplicity, OOM-safety, no CPU-memory dependency,")
    print(f"            bit-deterministic resume — outweighs ~2-3x latency penalty.")

    # Expected latency over a workload with sporadic OOM.
    print()
    print("    Expected per-request latency at 5% OOM/step, 100 steps/req:")
    s = scenarios[2][1]  # 8K prompt
    for strat in PreemptionStrategy:
        e = expected_latency_under_oom_rate(s, strat, 0.05, 100)
        print(f"      {strat.value:<10} -> {e * 1000:.0f} ms")


def section_4_priority_and_aging() -> None:
    _section("[4] PRIORITY policy + (pedagogical) aging compensator")
    requests = [
        PolicyRequest("A", priority=3, arrival_time=0.0),  # low priority
        PolicyRequest("B", priority=1, arrival_time=1.0),  # highest
        PolicyRequest("C", priority=2, arrival_time=2.0),  # mid
        PolicyRequest("D", priority=1, arrival_time=3.0),  # tied with B, later
    ]

    fcfs_q = FCFSRequestQueue()
    for r in requests:
        fcfs_q.add_request(r)
    fcfs_order = [r.request_id for r in fcfs_q]

    prio_order = [r.request_id for r in priority_ordering(requests)]
    print(f"    arrival order (FCFS):   {fcfs_order}")
    print(f"    priority queue order:   {prio_order}")
    print(f"      → B before D because earlier arrival on tied priority")

    # Aging at t=10 — the late, low-priority A becomes effectively highest.
    now = 10.0
    aged = sorted(requests, key=lambda r: aged_priority(r, now, aging_rate=1.0))
    aged_ids = [r.request_id for r in aged]
    print(f"    aged order at t={now}:    {aged_ids}")
    print(f"      A's effective priority at t=10: {aged_priority(requests[0], now)}")
    print(f"      (NOTE: vLLM does NOT auto-age; this is illustrative only.)")


def section_5_pareto_frontier() -> None:
    _section("[5] schedule-latency vs throughput Pareto frontier")
    knobs = [
        EngineConfig(max_num_seqs=8,   max_num_batched_tokens=512),
        EngineConfig(max_num_seqs=16,  max_num_batched_tokens=1024),
        EngineConfig(max_num_seqs=32,  max_num_batched_tokens=2048),
        EngineConfig(max_num_seqs=64,  max_num_batched_tokens=4096),
        EngineConfig(max_num_seqs=128, max_num_batched_tokens=8192),
        # With long_prefill_token_threshold caps:
        EngineConfig(max_num_seqs=32,  max_num_batched_tokens=2048, long_prefill_token_threshold=512),
        EngineConfig(max_num_seqs=64,  max_num_batched_tokens=4096, long_prefill_token_threshold=1024),
    ]
    points = sweep(knobs, long_prompt_tokens=8192, short_prompt_tokens=64, avg_seq_len=512)
    front = pareto_front(points)
    front_ids = {id(p) for p in front}

    print(f"    {'max_seqs':>10} {'budget':>8} {'long_thr':>10}  "
          f"{'tput (Mtok/s)':>14} {'p95 TTFT':>11} {'sat':>6} {'pareto?':>8}")
    for p in points:
        cfg = p.config
        marker = "★" if id(p) in front_ids else " "
        print(
            f"    {cfg.max_num_seqs:>10d} {cfg.max_num_batched_tokens:>8d} "
            f"{cfg.long_prefill_token_threshold:>10d}  "
            f"{p.throughput_tokens_per_sec / 1e6:>14.2f} "
            f"{p.p95_ttft_seconds * 1000:>9.0f}ms "
            f"{p.saturation:>6.2f} {marker:>8}"
        )
    print()
    print(f"    Pareto frontier has {len(front)} non-dominated points (★).")
    print(f"    Setting long_prefill_token_threshold dramatically improves p95 TTFT")
    print(f"    at modest throughput cost — see rows where threshold > 0.")


def run_demo() -> None:
    print("=" * 64)
    print("Ch06 — Request Scheduling Policy — annotated trace")
    print("=" * 64)
    print("Mechanics layer: Ch04 §4.x. This chapter is the POLICY/strategy layer.")
    section_1_decision_loop()
    section_2_fcfs_fairness()
    section_3_preemption_strategies()
    section_4_priority_and_aging()
    section_5_pareto_frontier()


if __name__ == "__main__":
    run_demo()
