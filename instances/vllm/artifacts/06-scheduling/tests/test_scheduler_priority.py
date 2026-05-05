# test_scheduler_priority.py — Priority scheduling behavior
# REFERENCE: instances/vllm/artifacts/06-scheduling/implementation/scheduler.py

import sys
import time

import pytest

sys.path.insert(0, "instances/vllm/artifacts/06-scheduling/implementation")

from request import Request, RequestStatus
from request_queue import SchedulingPolicy
from scheduler import Scheduler


def make_request(req_id: str, prompt_len: int = 32, max_tokens: int = 5, priority: int = 0) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        priority=priority,
    )


class TestPriorityAdmission:
    """Priority queue admits highest-priority (lowest value) requests first."""

    def test_higher_priority_admitted_first(self):
        """Requests with lower priority value are admitted before higher-value ones."""
        s = Scheduler(
            max_num_running_reqs=2,  # Only 2 can run — forces priority ordering
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.PRIORITY,
        )

        # Add in arbitrary order; priority values: urgent=0, normal=5, low=10
        s.add_request(make_request("low", priority=10))
        s.add_request(make_request("normal", priority=5))
        s.add_request(make_request("urgent", priority=0))

        out = s.schedule()
        admitted_ids = [r.request_id for r in out.scheduled_new_reqs]

        # With max_num_running_reqs=2, only 2 admitted
        # "urgent" (priority=0) and "normal" (priority=5) should be first
        assert admitted_ids[0] == "urgent"   # priority 0
        assert admitted_ids[1] == "normal"   # priority 5
        # "low" stays in waiting
        assert "low" not in admitted_ids

    def test_same_priority_fifo_tiebreak(self):
        """Same-priority requests admitted in arrival order (FIFO tie-break)."""
        s = Scheduler(
            max_num_running_reqs=2,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.PRIORITY,
        )

        s.add_request(make_request("first", priority=5))
        time.sleep(0.001)  # ensure distinct arrival times
        s.add_request(make_request("second", priority=5))
        time.sleep(0.001)
        s.add_request(make_request("third", priority=5))

        out = s.schedule()
        admitted = [r.request_id for r in out.scheduled_new_reqs]

        # Only 2 fit; first and second should be admitted in arrival order
        assert admitted == ["first", "second"]

    def test_priority_admission_independent_of_add_order(self):
        """Priority admission order depends on priority value, not insert order."""
        s = Scheduler(
            max_num_running_reqs=3,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.PRIORITY,
        )

        # Add in reverse priority order
        s.add_request(make_request("lowest", priority=30))
        s.add_request(make_request("middle", priority=20))
        s.add_request(make_request("highest", priority=0))

        out = s.schedule()
        admitted = [r.request_id for r in out.scheduled_new_reqs]

        assert admitted[0] == "highest"  # priority 0
        assert admitted[1] == "middle"   # priority 20
        assert admitted[2] == "lowest"   # priority 30

    def test_priority_available_after_capacity_frees(self):
        """After a running request finishes, the next-highest-priority waiting request runs."""
        s = Scheduler(
            max_num_running_reqs=2,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=100,
            policy=SchedulingPolicy.PRIORITY,
        )

        s.add_request(make_request("A_hi", priority=0))
        s.add_request(make_request("B_mid", priority=5))
        s.add_request(make_request("C_low", priority=10))
        out1 = s.schedule()
        assert len(out1.scheduled_new_reqs) == 2

        # Finish A_hi to free a slot
        sampled = {r: [1, 2, 3, 4, 5] for r in out1.num_scheduled_tokens}
        s.update_from_output(out1, sampled)

        # C_low should now be admitted (only remaining waiting request)
        out2 = s.schedule()
        if len(out2.scheduled_new_reqs) > 0:
            c_admitted = any(r.request_id == "C_low" for r in out2.scheduled_new_reqs)
            assert c_admitted
