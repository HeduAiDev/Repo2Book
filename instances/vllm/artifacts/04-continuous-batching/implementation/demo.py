# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py
"""Annotated runnable trace of continuous batching.

Usage:
    python3 instances/vllm/artifacts/04-continuous-batching/implementation/demo.py

The demo creates three requests with very different prompt lengths and runs
the scheduler step-by-step, printing what got scheduled, who got preempted,
and the running KV-cache occupancy. The intent is to make the two-phase
discipline (running-first, waiting-second) and chunked prefill visible.
"""

from __future__ import annotations

from .request import Request, RequestStatus
from .scheduler import (
    Scheduler,
    continuous_batching_steps,
    static_batching_steps,
)


def _print_step(step: int, sched: Scheduler, output) -> None:
    print(f"\n── step {step} ────────────────────────────────────────────")
    print(
        f"  budget={sched.max_num_scheduled_tokens}  "
        f"running={len(sched.running)}  "
        f"waiting={len(sched.waiting)}  "
        f"free_blocks={sched.kv_cache_manager.num_free_blocks}"
    )
    for req_id, n in output.num_scheduled_tokens.items():
        req = sched.requests[req_id]
        phase = "prefill" if req.is_prefill else "decode "
        print(
            f"    r{req_id}: {n:>4} tok ({phase})  "
            f"computed={req.num_computed_tokens}/{req.num_tokens}  "
            f"blocks={len(req.block_ids)}"
        )
    if output.preempted_req_ids:
        print(f"    PREEMPTED: {sorted(output.preempted_req_ids)}")
    if output.newly_running_req_ids:
        print(f"    ADMITTED:  {output.newly_running_req_ids}")
    if output.finished_req_ids:
        print(f"    FINISHED:  {sorted(output.finished_req_ids)}")


def run_demo() -> None:
    print("=" * 64)
    print("Continuous Batching Scheduler — annotated trace")
    print("=" * 64)

    # Tight budget forces chunked prefill on r1.
    sched = Scheduler(
        max_num_scheduled_tokens=128,
        max_num_running_reqs=4,
        num_gpu_blocks=200,
        block_size=16,
        enable_chunked_prefill=True,
    )

    sched.add_request(Request("1", list(range(400)), max_tokens=4, arrival_time=0.0))
    sched.add_request(Request("2", list(range(64)), max_tokens=8, arrival_time=0.0))
    sched.add_request(Request("3", list(range(16)), max_tokens=16, arrival_time=0.0))

    print("\nWorkload:")
    for r in list(sched.waiting):
        print(f"  r{r.request_id}: prompt={r.num_prompt_tokens}, max_out={r.max_tokens}")

    for step in range(1, 30):
        if not sched.running and not sched.waiting:
            break
        output = sched.schedule()
        sched.update_after_step(output)
        _print_step(step, sched, output)

    # Final state.
    print("\n" + "=" * 64)
    finished = [
        rid
        for rid, r in sched.requests.items()
        if RequestStatus.is_finished(r.status)
    ]
    print(f"Finished requests: {sorted(finished)}")
    print(
        f"KV blocks reclaimed: "
        f"{sched.kv_cache_manager.num_free_blocks}/"
        f"{sched.kv_cache_manager.num_gpu_blocks}"
    )

    # Bubble analysis: same workload under both regimes.
    print("\nBubble analysis (same 3-request workload):")
    workload = [(400, 4), (64, 8), (16, 16)]
    static = static_batching_steps(workload)
    cont = continuous_batching_steps(workload, token_budget=128)
    print(f"  static     : {static} steps")
    print(f"  continuous : {cont} steps")
    if cont:
        print(f"  speedup    : {static / cont:.2f}x")


if __name__ == "__main__":
    run_demo()
