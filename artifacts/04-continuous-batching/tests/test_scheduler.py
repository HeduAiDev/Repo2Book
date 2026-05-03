"""Tests — Ch4 Continuous Batching."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.scheduler import (
    Request, RequestStatus, SchedulerOutput,
    ContinuousBatchingScheduler, SimpleKVCacheManager,
    static_batching_simulation, continuous_batching_simulation,
)


class TestRequestStatus:
    def test_finished_check(self):
        assert not RequestStatus.WAITING.is_finished()
        assert not RequestStatus.RUNNING.is_finished()
        assert not RequestStatus.PREEMPTED.is_finished()
        assert RequestStatus.FINISHED_STOPPED.is_finished()

    def test_preempted_is_boundary(self):
        assert not RequestStatus.PREEMPTED.is_finished()
        assert RequestStatus.FINISHED_STOPPED.is_finished()


class TestRequest:
    def test_num_new_tokens(self):
        req = Request("r1", prompt_token_ids=list(range(100)),
                      max_tokens=50, arrival_time=0)
        assert req.num_new_tokens == 100
        req.num_computed_tokens = 60
        assert req.num_new_tokens == 40

    def test_is_prefill(self):
        req = Request("r1", prompt_token_ids=list(range(50)),
                      max_tokens=20, arrival_time=0)
        assert req.is_prefill
        req.num_computed_tokens = 50
        assert not req.is_prefill


class TestScheduler:
    def test_schedule_single_request(self):
        sched = ContinuousBatchingScheduler(
            max_num_scheduled_tokens=2048, max_num_running_reqs=128,
            num_gpu_blocks=1000, block_size=16,
        )
        sched.add_request(Request("r1", list(range(100)), max_tokens=20, arrival_time=0))

        out = sched.schedule()
        assert out.scheduled_requests == {"r1": 100}
        assert out.total_scheduled_tokens == 100
        assert len(sched.running) == 1
        assert sched.running[0].status == RequestStatus.RUNNING

    def test_chunked_prefill_splits_long_prompt(self):
        """With chunked prefill, long prompts should be split across steps."""
        sched = ContinuousBatchingScheduler(
            max_num_scheduled_tokens=500, max_num_running_reqs=128,
            num_gpu_blocks=1000, block_size=16, enable_chunked_prefill=True,
        )
        sched.add_request(Request("r1", list(range(1000)), max_tokens=50, arrival_time=0))

        # Step 1: should schedule 500 tokens (token budget)
        out1 = sched.schedule()
        assert out1.scheduled_requests["r1"] == 500
        sched.update_after_step(out1)
        assert sched.running[0].num_computed_tokens == 500

        # Step 2: remaining 500 tokens
        out2 = sched.schedule()
        assert out2.scheduled_requests["r1"] == 500
        sched.update_after_step(out2)
        assert sched.running[0].num_computed_tokens == 1000

    def test_multiple_requests_interleaved(self):
        """Running + waiting requests should be scheduled together."""
        sched = ContinuousBatchingScheduler(
            max_num_scheduled_tokens=2048, max_num_running_reqs=128,
            num_gpu_blocks=1000, block_size=16,
        )
        sched.add_request(Request("r1", list(range(500)), max_tokens=100, arrival_time=0))
        sched.add_request(Request("r2", list(range(200)), max_tokens=50, arrival_time=0))

        out = sched.schedule()
        # Both should be scheduled since token budget fits both
        assert len(out.scheduled_requests) == 2

    def test_truly_full_kv_cache_skips_waiting(self):
        """When KV cache is exhausted and no preemption possible, skip waiting."""
        sched = ContinuousBatchingScheduler(
            max_num_scheduled_tokens=2048, max_num_running_reqs=128,
            num_gpu_blocks=4, block_size=16,
        )
        # r1 takes all 4 blocks (64 tokens → 4 blocks)
        sched.add_request(Request("r1", list(range(64)), max_tokens=10, arrival_time=0))
        sched.schedule()
        sched.requests["r1"].num_computed_tokens = 64  # Skip preempt: mark as done

        # r2 needs 4 blocks but 0 free
        sched.add_request(Request("r2", list(range(64)), max_tokens=10, arrival_time=0))
        out = sched.schedule()
        # r2 can't be admitted — no way to get 4 blocks
        assert "r2" not in out.scheduled_requests

    def test_finished_request_freed(self):
        """When a request runs out of max_tokens, it finishes and frees blocks."""
        sched = ContinuousBatchingScheduler(
            max_num_scheduled_tokens=2048, max_num_running_reqs=128,
            num_gpu_blocks=100, block_size=16,
        )
        req = Request("r1", list(range(10)), max_tokens=0, arrival_time=0)
        sched.add_request(req)
        sched.schedule()
        sched.update_after_step(sched.schedule())

        # max_tokens=0 means it finished after prefill (output capped)
        assert len(sched.running) == 0  # finished, removed from running

    def test_status_lifecycle(self):
        sched = ContinuousBatchingScheduler(
            max_num_scheduled_tokens=2048, max_num_running_reqs=128,
            num_gpu_blocks=1000, block_size=16,
        )
        sched.add_request(Request("r1", list(range(10)), max_tokens=1, arrival_time=0))
        assert sched.waiting[0].status == RequestStatus.WAITING

        sched.schedule()
        assert sched.running[0].status == RequestStatus.RUNNING


class TestBatchingSimulation:
    def test_continuous_faster_than_static(self):
        requests = [(2048, 128)] * 2 + [(128, 256)] * 6
        s = static_batching_simulation(requests)
        c = continuous_batching_simulation(requests)
        assert c < s

    def test_single_request_equal(self):
        requests = [(100, 50)]
        s = static_batching_simulation(requests)
        c = continuous_batching_simulation(requests)
        assert c <= s  # Continuous should be same or better


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
