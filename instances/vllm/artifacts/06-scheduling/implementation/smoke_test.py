"""Smoke tests — verify basic invariants of the simplified Scheduler.

Not exhaustive. The full test suite is written by the Tester agent in
artifacts/06-scheduling/tests/. This file exists so `python3 smoke_test.py`
passes before handoff to the tester.
"""

from output import SchedulerOutput
from request import Request, RequestStatus
from request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from scheduler import Scheduler


def _req(req_id: str, prompt_len: int = 16, max_tokens: int = 2,
         priority: int = 0, arrival_time: float = 0.0) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        priority=priority,
        arrival_time=arrival_time,
    )


def test_fcfs_queue_order():
    q = FCFSRequestQueue()
    q.add_request(_req("A"))
    q.add_request(_req("B"))
    q.add_request(_req("C"))
    assert q.pop_request().request_id == "A"
    assert q.pop_request().request_id == "B"
    assert q.pop_request().request_id == "C"
    assert not q


def test_fcfs_prepend():
    q = FCFSRequestQueue()
    q.add_request(_req("A"))
    q.add_request(_req("B"))
    q.prepend_request(_req("P"))  # preempted request
    assert q.pop_request().request_id == "P"
    assert q.pop_request().request_id == "A"


def test_priority_queue_order():
    q = PriorityRequestQueue()
    q.add_request(_req("C", priority=10))
    q.add_request(_req("A", priority=0))
    q.add_request(_req("B", priority=5))
    # Lower priority value = scheduled first
    assert q.pop_request().request_id == "A"
    assert q.pop_request().request_id == "B"
    assert q.pop_request().request_id == "C"


def test_priority_tiebreak_by_arrival():
    q = PriorityRequestQueue()
    q.add_request(_req("late", priority=5, arrival_time=2.0))
    q.add_request(_req("early", priority=5, arrival_time=1.0))
    assert q.pop_request().request_id == "early"


def test_factory():
    assert isinstance(create_request_queue(SchedulingPolicy.FCFS), FCFSRequestQueue)
    assert isinstance(
        create_request_queue(SchedulingPolicy.PRIORITY), PriorityRequestQueue
    )


def test_scheduler_basic_decode():
    sched = Scheduler(
        max_num_running_reqs=2,
        max_num_scheduled_tokens=64,
        max_model_len=128,
        num_gpu_blocks=16,
        block_size=16,
    )
    sched.add_request(_req("A", prompt_len=16, max_tokens=2))

    # Step 0: prefill schedules all 16 tokens. After schedule(), num_computed=16.
    # The model runner sees prefill is done → produces first decode token.
    out = sched.schedule()
    assert out.num_scheduled_tokens == {"A": 16}
    assert len(out.scheduled_new_reqs) == 1
    sched.update_from_output(out, {"A": [42]})

    # Step 1: decode schedules 1 token (num_tokens=17, num_computed=16).
    # Model produces second token → hits max_tokens=2 → finishes.
    out = sched.schedule()
    sched.update_from_output(out, {"A": [42]})
    assert not sched.has_unfinished_requests()


def test_fcfs_preempts_newest():
    """When OOM, FCFS preempts the last (newest) running request."""
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=128,
        max_model_len=256,
        num_gpu_blocks=4,  # only 64 token slots
        block_size=16,
        policy=SchedulingPolicy.FCFS,
    )
    # 4 requests, each needing 1 block during prefill → only 4 fit.
    # After decode, each needs a 2nd block (for generated tokens) → preemption.
    for name in "ABCD":
        sched.add_request(_req(name, prompt_len=16, max_tokens=16))

    out = sched.schedule()
    # All 4 prefilled in step 0 (each takes 1 block).
    assert len(out.scheduled_new_reqs) == 4

    # Simulate a decode step. Each request now needs a 2nd block (position 16).
    # But only 4 blocks total → OOM → preemption.
    sched.update_from_output(out, {r: [42] for r in "ABCD"})
    out2 = sched.schedule()
    assert len(out2.preempted_reqs) >= 1
    # FCFS preempts from the end: D should be in the preempted set.
    preempted_ids = {r.request_id for r in out2.preempted_reqs}
    assert "D" in preempted_ids, f"expected D to be preempted, got {preempted_ids}"


def test_priority_preempts_lowest():
    """Under PRIORITY, preempt the request with the highest priority value."""
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=128,
        max_model_len=256,
        num_gpu_blocks=4,
        block_size=16,
        policy=SchedulingPolicy.PRIORITY,
    )
    sched.add_request(_req("high", prompt_len=16, max_tokens=16,
                           priority=0, arrival_time=1.0))
    sched.add_request(_req("med", prompt_len=16, max_tokens=16,
                           priority=5, arrival_time=2.0))
    sched.add_request(_req("low1", prompt_len=16, max_tokens=16,
                           priority=10, arrival_time=3.0))
    sched.add_request(_req("low2", prompt_len=16, max_tokens=16,
                           priority=10, arrival_time=4.0))

    out = sched.schedule()
    sched.update_from_output(out, {r.request_id: [42] for r in out.scheduled_new_reqs})
    out2 = sched.schedule()

    preempted_ids = {r.request_id for r in out2.preempted_reqs}
    assert preempted_ids, "expected at least one preemption"
    # Highest priority value = lowest priority → low1 or low2 should be evicted
    # (not "high" or "med").
    assert preempted_ids.issubset({"low1", "low2"}), (
        f"priority preempted wrong requests: {preempted_ids}"
    )


def test_preempted_request_resumes():
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=128,
        max_model_len=256,
        num_gpu_blocks=4,
        block_size=16,
        policy=SchedulingPolicy.FCFS,
    )
    for name in "ABCD":
        sched.add_request(_req(name, prompt_len=16, max_tokens=4))

    # Run until everyone finishes.
    steps = 0
    while sched.has_unfinished_requests() and steps < 100:
        out = sched.schedule()
        sampled: dict[str, list[int]] = {}
        for rid in out.num_scheduled_tokens:
            r = sched.requests.get(rid)
            if r and r.num_computed_tokens >= len(r.prompt_token_ids):
                sampled[rid] = [42]
            else:
                sampled[rid] = []
        sched.update_from_output(out, sampled)
        steps += 1
    assert not sched.has_unfinished_requests()


def test_finish_request_aborts():
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=64,
        max_model_len=128,
        num_gpu_blocks=8,
    )
    sched.add_request(_req("A"))
    sched.add_request(_req("B"))
    assert sched.get_num_unfinished_requests() == 2

    sched.finish_requests("A")
    assert sched.get_num_unfinished_requests() == 1
    assert "A" not in sched.requests


def test_schedule_output_invariants():
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=48,
        max_model_len=128,
        num_gpu_blocks=16,
    )
    sched.add_request(_req("A", prompt_len=32, max_tokens=2))
    sched.add_request(_req("B", prompt_len=32, max_tokens=2))

    out = sched.schedule()
    assert out.total_num_scheduled_tokens <= sched.max_num_scheduled_tokens
    assert out.total_num_scheduled_tokens == sum(out.num_scheduled_tokens.values())


def run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n  {len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
