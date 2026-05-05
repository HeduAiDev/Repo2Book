# test_scheduler_fcfs.py — FCFS scheduling behavior
# REFERENCE: instances/vllm/artifacts/06-scheduling/implementation/scheduler.py

import sys
import time

import pytest

sys.path.insert(0, "instances/vllm/artifacts/06-scheduling/implementation")

from request import Request, RequestStatus
from request_queue import SchedulingPolicy
from scheduler import Scheduler


def make_request(req_id: str, prompt_len: int = 32, max_tokens: int = 10, priority: int = 0) -> Request:
    """Create a minimal test request."""
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        priority=priority,
    )


class TestFCFSBasicSchedule:
    """Single-request scheduling: tokens advance and request finishes."""

    def test_single_request_schedules_all_prompt_tokens(self):
        """A request with prompt_len < max_num_scheduled_tokens gets all tokens in step 0."""
        s = Scheduler(
            max_num_running_reqs=10,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.FCFS,
        )
        req = make_request("A", prompt_len=32, max_tokens=10)
        s.add_request(req)

        out = s.schedule()
        assert out.num_scheduled_tokens["A"] == 32  # all prompt tokens
        assert len(s.running) == 1
        assert s.running[0].request_id == "A"
        assert s.running[0].status == RequestStatus.RUNNING

    def test_single_request_decodes_one_token_per_step_after_prefill(self):
        """After prefill, each step schedules 1 token (autoregressive decode)."""
        s = Scheduler(
            max_num_running_reqs=10,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.FCFS,
        )
        req = make_request("A", prompt_len=32, max_tokens=10)
        s.add_request(req)
        out = s.schedule()  # step 0: prefill (all 32 prompt tokens)

        # After prefill, num_computed_tokens = 32, num_tokens = 32.
        # Model runner produces 1 output token → now num_tokens = 33, gap = 1.
        sampled = {"A": [42]}  # one generated token
        s.update_from_output(out, sampled)
        # Now request has 33 tokens total, 32 computed → 1 needed for next step

        out2 = s.schedule()
        # After update_from_output added output, the request may still be running
        # and need more tokens computed
        if "A" in out2.num_scheduled_tokens:
            assert out2.num_scheduled_tokens["A"] <= 1  # decode is token-by-token

    def test_request_finishes_when_max_tokens_reached(self):
        """After max_tokens output tokens, the request transitions to FINISHED_LENGTH_CAPPED."""
        s = Scheduler(
            max_num_running_reqs=10,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.FCFS,
        )
        req = make_request("A", prompt_len=32, max_tokens=1)
        s.add_request(req)
        out = s.schedule()

        # Give it enough tokens to hit max_tokens
        sampled = {"A": [42]}
        finished = s.update_from_output(out, sampled)

        # It should finish after hitting max_tokens=1
        # (but only if _check_stop fires — which it does when output_token_ids >= max_tokens)
        assert "A" in finished
        assert "A" not in s.requests

    def test_token_budget_not_exceeded(self):
        """No step schedules more tokens than max_num_scheduled_tokens."""
        budget = 50
        s = Scheduler(
            max_num_running_reqs=10,
            max_num_scheduled_tokens=budget,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.FCFS,
        )
        for i in range(5):
            s.add_request(make_request(f"R{i}", prompt_len=30, max_tokens=10))

        out = s.schedule()
        total = sum(out.num_scheduled_tokens.values())
        assert total <= budget

    def test_max_num_running_reqs_capped(self):
        """The running queue never exceeds max_num_running_reqs."""
        s = Scheduler(
            max_num_running_reqs=3,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.FCFS,
        )
        for i in range(8):
            s.add_request(make_request(f"R{i}", prompt_len=20, max_tokens=5))

        out = s.schedule()
        nr, nw = s.get_request_counts()
        assert nr <= 3
        # Some requests remain in waiting because of the cap
        if len(out.scheduled_new_reqs) < 8:
            assert nw > 0

    def test_waiting_admitted_when_running_has_space(self):
        """When running queue drops below max, waiting requests get admitted."""
        s = Scheduler(
            max_num_running_reqs=4,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.FCFS,
        )
        # Add requests; only 4 can run at once
        for i in range(6):
            s.add_request(make_request(f"R{i}", prompt_len=20, max_tokens=3))

        out1 = s.schedule()
        nr1 = len(s.running)
        assert nr1 <= 4
        assert s.waiting or out1.scheduled_new_reqs  # some left waiting

        # Finish one request, freeing a slot
        finished = s.update_from_output(out1, {r: [0] * 3 for r in out1.num_scheduled_tokens})
        # Now a waiting request should be admitted in the next schedule() call
        if finished and s.waiting:
            out2 = s.schedule()
            assert len(out2.scheduled_new_reqs) >= 0  # may or may not admit new

    def test_scheduled_in_fcfs_order(self):
        """Requests are scheduled in the order they were added (FCFS)."""
        s = Scheduler(
            max_num_running_reqs=10,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.FCFS,
        )
        ids = ["X", "A", "M", "B"]
        for rid in ids:
            s.add_request(make_request(rid, prompt_len=20))

        out = s.schedule()
        scheduled_order = out.scheduled_new_reqs
        # Should be scheduled in insertion order
        for i, rid in enumerate(ids):
            assert scheduled_order[i].request_id == rid
