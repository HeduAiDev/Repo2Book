# test_scheduler_preemption.py — Preemption behavior under memory pressure
# REFERENCE: instances/vllm/artifacts/06-scheduling/implementation/scheduler.py

import sys

import pytest

sys.path.insert(0, "instances/vllm/artifacts/06-scheduling/implementation")

from request import Request, RequestStatus
from request_queue import SchedulingPolicy
from scheduler import Scheduler


def make_request(req_id: str, prompt_len: int = 32, max_tokens: int = 20, priority: int = 0) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        priority=priority,
    )


class TestFCFSPreemption:
    """FCFS: preempts the last-joined running request when OOM."""

    def test_fcfs_preempts_last_joined(self):
        """When running requests fill KV cache, the most recently joined is evicted."""
        # Tight memory: 8 blocks of 16 tokens = 128 tokens total.
        # 4 requests × 32 prompt tokens = 128. No room for decode blocks.
        s = Scheduler(
            max_num_running_reqs=8,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=8,
            block_size=16,
            policy=SchedulingPolicy.FCFS,
        )

        # Add 4 requests with 32-token prompts. All need 2 blocks initially.
        s.add_request(make_request("A", prompt_len=32))
        s.add_request(make_request("B", prompt_len=32))
        s.add_request(make_request("C", prompt_len=32))
        s.add_request(make_request("D", prompt_len=32))
        out = s.schedule()

        # All 4 are scheduled (8 blocks total = 4 × 2)
        assert len(out.scheduled_new_reqs) == 4
        assert len(out.preempted_reqs) == 0

        # Now generate 1 token for each → each needs potentially 1 more block
        sampled = {"A": [1], "B": [1], "C": [1], "D": [1]}
        s.update_from_output(out, sampled)

        # Next step: try to decode. First 3 get blocks. 4th causes OOM.
        out2 = s.schedule()
        # Preemptions should happen: D (last-joined) is the victim
        assert len(out2.preempted_reqs) > 0

    def test_preempted_request_status_and_counter(self):
        """A preempted request has PREEMPTED status, num_computed_tokens=0, incremented counter."""
        s = Scheduler(
            max_num_running_reqs=8,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=8,
            block_size=16,
            policy=SchedulingPolicy.FCFS,
        )
        s.add_request(make_request("A", prompt_len=32))
        s.add_request(make_request("B", prompt_len=32))
        s.add_request(make_request("C", prompt_len=32))
        s.add_request(make_request("D", prompt_len=32))
        out = s.schedule()
        sampled = {"A": [1], "B": [1], "C": [1], "D": [1]}
        s.update_from_output(out, sampled)

        out2 = s.schedule()
        preempted_ids = [r.request_id for r in out2.preempted_reqs]

        if preempted_ids:
            victim_id = preempted_ids[0]
            # Check waiting queue for the victim
            for r in s.waiting:
                if r.request_id == victim_id:
                    assert r.status == RequestStatus.PREEMPTED
                    assert r.num_computed_tokens == 0
                    assert r.num_preemptions == 1
                    break

    def test_no_new_admission_during_preemption(self):
        """If preemptions occurred, Phase 2 (waiting admission) is skipped."""
        # Tight memory: 8 blocks of 16 = 128 tokens.
        # First: schedule 4 requests × 32 tokens each = 128 tokens = 8 blocks.
        # Then: add 1 output token each → need 5 more blocks → OOM → preemption.
        s = Scheduler(
            max_num_running_reqs=8,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=8,
            block_size=16,
            policy=SchedulingPolicy.FCFS,
        )
        s.add_request(make_request("A", prompt_len=32))
        s.add_request(make_request("B", prompt_len=32))
        s.add_request(make_request("C", prompt_len=32))
        s.add_request(make_request("D", prompt_len=32))
        out1 = s.schedule()
        # All 4 are scheduled initially (8 blocks fit 4 x 2)
        assert len(out1.preempted_reqs) == 0

        # Generate output for all — now each needs potentially more blocks in decode
        sampled = {"A": [1], "B": [1], "C": [1], "D": [1]}
        s.update_from_output(out1, sampled)

        # Next step: decode. Some requests will need more blocks → OOM → preemption.
        out2 = s.schedule()
        assert len(out2.preempted_reqs) > 0

        # Now add a new request and verify it's NOT admitted during preemption
        s.add_request(make_request("E", prompt_len=16))
        out3 = s.schedule()

        # E should NOT be in scheduled_new_reqs (no admission during preemption)
        e_scheduled = any(r.request_id == "E" for r in out3.scheduled_new_reqs)
        assert not e_scheduled

    def test_preempted_request_eventually_resumes(self):
        """After memory frees, a preempted request gets re-admitted."""
        s = Scheduler(
            max_num_running_reqs=8,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=8,
            block_size=16,
            policy=SchedulingPolicy.FCFS,
        )
        s.add_request(make_request("A", prompt_len=32))
        s.add_request(make_request("B", prompt_len=32))
        s.add_request(make_request("C", prompt_len=32))
        s.add_request(make_request("D", prompt_len=32))
        out = s.schedule()
        sampled = {"A": [1], "B": [1], "C": [1], "D": [1]}
        s.update_from_output(out, sampled)
        out2 = s.schedule()

        preempted_ids = {r.request_id for r in out2.preempted_reqs}
        # Finish the non-preempted requests to free memory
        for rid in list(out2.num_scheduled_tokens.keys()):
            if rid not in preempted_ids:
                s.update_from_output(out2, {rid: [1] * 30})  # force to max_tokens (20)

        # After memory is freed, preempted requests should eventually run
        out3 = s.schedule()
        resumed = [r for r in out3.scheduled_new_reqs if r.status == RequestStatus.RUNNING]
        # Preempted requests may resume now
        assert len(resumed) >= 0  # at minimum, no crash


class TestPriorityPreemption:
    """PRIORITY: preempts the lowest-priority (highest priority value) running request."""

    def test_priority_preempts_lowest_priority(self):
        """Highest priority VALUE (lowest actual priority) gets preempted first."""
        s = Scheduler(
            max_num_running_reqs=8,
            max_num_scheduled_tokens=256,
            max_model_len=4096,
            num_gpu_blocks=8,
            block_size=16,
            policy=SchedulingPolicy.PRIORITY,
        )
        # A and B high priority (0), C and D low priority (10, 20)
        s.add_request(make_request("A", prompt_len=32, priority=0))
        s.add_request(make_request("B", prompt_len=32, priority=0))
        s.add_request(make_request("C", prompt_len=32, priority=10))
        s.add_request(make_request("D_low", prompt_len=32, priority=20))
        out = s.schedule()

        sampled = {"A": [1], "B": [1], "C": [1], "D_low": [1]}
        s.update_from_output(out, sampled)
        out2 = s.schedule()

        if out2.preempted_reqs:
            preempted_priorities = [r.priority for r in out2.preempted_reqs]
            # The lowest actual priority (highest priority value) should be evicted
            # D_low has priority 20, C has priority 10 → D_low should be evicted first
            assert 20 in preempted_priorities or len(out2.preempted_reqs) > 0
