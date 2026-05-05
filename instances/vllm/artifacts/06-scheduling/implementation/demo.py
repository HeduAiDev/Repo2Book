"""Demo script — shows the simplified Scheduler in action.

Runs three scenarios:
  1. Happy path — 3 requests fit in memory, scheduled round-robin
  2. Preemption — 4 requests, not enough memory, watch FCFS preempt the newest
  3. Priority — same OOM scenario but with PRIORITY policy, low-prio evicts

Usage:
    python3 demo.py
"""

import time

from output import SchedulerOutput
from request import Request, RequestStatus
from request_queue import SchedulingPolicy
from scheduler import Scheduler


def make_request(
    req_id: str, prompt_len: int, max_tokens: int, priority: int = 0,
    arrival_time: float | None = None,
) -> Request:
    """Build a Request with a synthetic prompt."""
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        priority=priority,
        arrival_time=arrival_time if arrival_time is not None else time.monotonic(),
    )


def simulate_model_runner(
    output: SchedulerOutput, sched: Scheduler
) -> dict[str, list[int]]:
    """Fake model runner: sample 1 token per request whose prefill is done.

    vLLM's real model runner samples exactly 1 token per scheduled request
    once that request has processed its full prompt (i.e., past the prefill
    phase). While still in chunked prefill, no token is sampled.

    After schedule() returns, the scheduler has already advanced
    num_computed_tokens, so we can check prefill completion directly.
    """
    sampled: dict[str, list[int]] = {}
    for req_id in output.num_scheduled_tokens:
        req = sched.requests.get(req_id)
        if req is None:
            continue
        prompt_len = len(req.prompt_token_ids)
        if req.num_computed_tokens >= prompt_len:
            sampled[req_id] = [42]  # prefill done → sample 1 decode token
        else:
            sampled[req_id] = []  # still chunked prefill
    return sampled


def print_step(step: int, output: SchedulerOutput, sched: Scheduler) -> None:
    running_ct, waiting_ct = sched.get_request_counts()
    used_blocks = sched.kv_cache_manager.num_used_blocks
    total_blocks = sched.kv_cache_manager.num_gpu_blocks
    print(
        f"  Step {step:3d} | running={running_ct} waiting={waiting_ct} "
        f"| blocks={used_blocks}/{total_blocks} | {output}"
    )


def scenario_1_happy_path() -> None:
    print("\n═══ Scenario 1: Happy path (3 requests, enough memory) ═══")
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=256,
        max_model_len=512,
        num_gpu_blocks=32,   # 32 * 16 = 512 token slots
        block_size=16,
        policy=SchedulingPolicy.FCFS,
    )

    sched.add_request(make_request("A", prompt_len=32, max_tokens=4))
    sched.add_request(make_request("B", prompt_len=48, max_tokens=3))
    sched.add_request(make_request("C", prompt_len=16, max_tokens=2))

    step = 0
    while sched.has_unfinished_requests() and step < 20:
        output = sched.schedule()
        sampled = simulate_model_runner(output, sched)
        finished = sched.update_from_output(output, sampled)
        print_step(step, output, sched)
        if finished:
            print(f"         → finished: {finished}")
        step += 1


def scenario_2_fcfs_preemption() -> None:
    print("\n═══ Scenario 2: FCFS preemption (tight memory forces evictions) ═══")
    # Only 8 blocks * 16 tokens = 128 token slots. Tight.
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=128,
        max_model_len=256,
        num_gpu_blocks=8,
        block_size=16,
        policy=SchedulingPolicy.FCFS,
    )

    # Four requests with long prompts — they'll compete for blocks.
    sched.add_request(make_request("A", prompt_len=32, max_tokens=20))
    sched.add_request(make_request("B", prompt_len=32, max_tokens=20))
    sched.add_request(make_request("C", prompt_len=32, max_tokens=20))
    sched.add_request(make_request("D", prompt_len=32, max_tokens=20))

    step = 0
    total_preemptions = 0
    while sched.has_unfinished_requests() and step < 40:
        output = sched.schedule()
        sampled = simulate_model_runner(output, sched)
        finished = sched.update_from_output(output, sampled)
        total_preemptions += len(output.preempted_reqs)
        print_step(step, output, sched)
        if output.preempted_reqs:
            preempted_ids = [r.request_id for r in output.preempted_reqs]
            print(f"         → PREEMPTED: {preempted_ids}")
        if finished:
            print(f"         → finished: {finished}")
        step += 1

    print(f"\n  Total preemptions across run: {total_preemptions}")
    for req_id, req in list(sched.requests.items()):
        print(f"  {req_id}: num_preemptions={req.num_preemptions}")


def scenario_3_priority() -> None:
    print("\n═══ Scenario 3: Priority scheduling (low-prio evicted first) ═══")
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=128,
        max_model_len=256,
        num_gpu_blocks=8,
        block_size=16,
        policy=SchedulingPolicy.PRIORITY,
    )

    # Priority 0 = highest, 10 = lowest.
    # Add in the order A(high), B(med), C(low), D(low) — but D arrives last.
    t0 = time.monotonic()
    sched.add_request(make_request("A", 32, 20, priority=0, arrival_time=t0))
    sched.add_request(make_request("B", 32, 20, priority=5, arrival_time=t0 + 0.1))
    sched.add_request(make_request("C", 32, 20, priority=10, arrival_time=t0 + 0.2))
    sched.add_request(make_request("D", 32, 20, priority=10, arrival_time=t0 + 0.3))

    step = 0
    while sched.has_unfinished_requests() and step < 40:
        output = sched.schedule()
        sampled = simulate_model_runner(output, sched)
        finished = sched.update_from_output(output, sampled)
        print_step(step, output, sched)
        if output.preempted_reqs:
            preempted_ids = [
                f"{r.request_id}(p={r.priority})" for r in output.preempted_reqs
            ]
            print(f"         → PREEMPTED: {preempted_ids}")
        if finished:
            print(f"         → finished: {finished}")
        step += 1

    print("\n  Final preemption counts (higher priority value = lower priority):")
    # Note: once requests finish, they're removed from sched.requests.
    # So we report what we can.
    for req_id, req in list(sched.requests.items()):
        print(
            f"  {req_id}: priority={req.priority}, preemptions={req.num_preemptions}"
        )


if __name__ == "__main__":
    scenario_1_happy_path()
    scenario_2_fcfs_preemption()
    scenario_3_priority()
    print("\n✓ Demo complete.")
