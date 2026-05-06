"""Unit tests for Request and RequestStatus.

Verifies the small state machine the scheduler depends on:
- Lifecycle ordering (anything past PREEMPTED is finished).
- `num_new_tokens`, `num_tokens`, `is_prefill` derived properties.
- Output token bookkeeping.
"""

from __future__ import annotations

from implementation.request import Request, RequestStatus


class TestRequestStatus:
    """Lifecycle ordering must match vLLM's IntEnum at request.py:L310-L333."""

    def test_status_ordering(self) -> None:
        """Enum values are monotonically increasing along the lifecycle."""
        assert RequestStatus.WAITING < RequestStatus.RUNNING
        assert RequestStatus.RUNNING < RequestStatus.PREEMPTED
        assert RequestStatus.PREEMPTED < RequestStatus.FINISHED_STOPPED
        assert RequestStatus.FINISHED_STOPPED < RequestStatus.FINISHED_LENGTH_CAPPED

    def test_is_finished_threshold(self) -> None:
        """`is_finished` returns True iff status > PREEMPTED."""
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
        assert not RequestStatus.is_finished(RequestStatus.WAITING)
        assert not RequestStatus.is_finished(RequestStatus.RUNNING)
        assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)


class TestRequestProperties:
    """Derived properties on Request — these drive the scheduler's decisions."""

    def test_num_prompt_and_new_tokens(self) -> None:
        """num_new_tokens = num_tokens - num_computed_tokens (no output yet)."""
        r = Request("r1", list(range(100)), max_tokens=50, arrival_time=0.0)
        assert r.num_prompt_tokens == 100
        assert r.num_output_tokens == 0
        assert r.num_tokens == 100
        assert r.num_new_tokens == 100

    def test_num_new_tokens_after_partial_compute(self) -> None:
        """Partial chunked prefill: 100-token prompt, 60 already computed."""
        r = Request("r1", list(range(100)), max_tokens=50, arrival_time=0.0)
        r.num_computed_tokens = 60
        assert r.num_new_tokens == 40

    def test_is_prefill_flips_on_full_prompt(self) -> None:
        """is_prefill is False once num_computed_tokens reaches num_prompt_tokens."""
        r = Request("r1", list(range(50)), max_tokens=20, arrival_time=0.0)
        assert r.is_prefill
        r.num_computed_tokens = 49
        assert r.is_prefill
        r.num_computed_tokens = 50
        assert not r.is_prefill

    def test_num_tokens_grows_with_output(self) -> None:
        """num_tokens = prompt_len + len(output_token_ids)."""
        r = Request("r1", list(range(10)), max_tokens=5, arrival_time=0.0)
        assert r.num_tokens == 10
        r.output_token_ids.extend([0, 0, 0])
        assert r.num_tokens == 13

    def test_is_finished_method(self) -> None:
        """Request.is_finished() delegates to RequestStatus.is_finished."""
        r = Request("r1", [1], max_tokens=1, arrival_time=0.0)
        assert not r.is_finished()
        r.status = RequestStatus.FINISHED_LENGTH_CAPPED
        assert r.is_finished()
