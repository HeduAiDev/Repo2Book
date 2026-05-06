"""Unit tests for starvation analysis.

Demo §2 numerics:
- 2 long(4K) + 8 short(64), B=512 → head-of-line factor 8.00x
- has_starvation(profile, FCFS) = True
- has_starvation(profile, PRIORITY) = False (assumes well-assigned priorities)

Demo §4:
- Arrival order [A, B, C, D] → priority-queue order [B, D, C, A]
  (B and D both priority=1; B arrived earlier; A has worst priority=3)
- aged_priority(A, t=10, rate=1.0) = priority(3) - 1.0*10 = -7
"""

from __future__ import annotations

from implementation.request_queue import PolicyRequest, SchedulingPolicy
from implementation.starvation_analysis import (
    WorkloadProfile,
    aged_priority,
    has_starvation,
    priority_ordering,
)


class TestWorkloadProfile:
    def test_demo_section_2_factor_is_8x(self) -> None:
        """Demo §2: 2 long(4K), 8 short(64), B=512 → head-of-line factor 8.00x."""
        p = WorkloadProfile(
            n_long=2, prompt_long=4096, n_short=8, prompt_short=64,
            token_budget=512, max_running=4,
        )
        assert p.fcfs_long_request_completion_steps() == 8  # 4096 / 512
        assert p.fcfs_short_request_latency_steps() == 8
        assert p.head_of_line_blocking_factor() == 8.0

    def test_short_prompt_smaller_than_budget_uses_one_step(self) -> None:
        """Short prompt of 64 with budget 512 → own_steps = ceil(64/512) = 1.
        head_of_line_factor = 8 / 1 = 8."""
        p = WorkloadProfile(
            n_long=1, prompt_long=4096, n_short=1, prompt_short=64,
            token_budget=512,
        )
        # Own steps clamped to 1 (max with the ceil division).
        assert p.head_of_line_blocking_factor() == 8.0

    def test_zero_long_prompt_no_blocking(self) -> None:
        """If long prompt = 0 tokens, no head-of-line blocking."""
        p = WorkloadProfile(
            n_long=0, prompt_long=0, n_short=10, prompt_short=64,
            token_budget=512,
        )
        # ceil(0/512) = 0, but max(1, ...) clamps own_steps; numerator is 0.
        assert p.fcfs_short_request_latency_steps() == 0
        assert p.head_of_line_blocking_factor() == 0.0

    def test_long_completion_ceil_divides(self) -> None:
        """Long prefill steps = ceil(prompt_long / B). Verify the ceiling."""
        p = WorkloadProfile(
            n_long=1, prompt_long=1025, n_short=1, prompt_short=64,
            token_budget=512,
        )
        # 1025 / 512 = 2.001 → ceil to 3.
        assert p.fcfs_long_request_completion_steps() == 3


class TestHasStarvation:
    def test_fcfs_starves_when_factor_above_one(self) -> None:
        """Demo §2: FCFS has factor=8 > 1 → starves."""
        p = WorkloadProfile(
            n_long=2, prompt_long=4096, n_short=8, prompt_short=64,
            token_budget=512,
        )
        assert has_starvation(p, SchedulingPolicy.FCFS) is True

    def test_fcfs_does_not_starve_when_factor_at_or_below_one(self) -> None:
        """When long_prefill_steps ≤ 1 and short_own_steps ≥ 1, factor ≤ 1.0."""
        p = WorkloadProfile(
            n_long=1, prompt_long=64, n_short=8, prompt_short=64,
            token_budget=512,
        )
        # Long prefill takes 1 step (64/512 ceil), short own_steps = 1 → factor = 1.0.
        assert has_starvation(p, SchedulingPolicy.FCFS) is False

    def test_priority_does_not_starve_when_well_assigned(self) -> None:
        """Demo §2: PRIORITY (well-assigned) returns False unconditionally
        in our predicate (assumes operator gave short jobs lower priority)."""
        p = WorkloadProfile(
            n_long=2, prompt_long=4096, n_short=8, prompt_short=64,
            token_budget=512,
        )
        assert has_starvation(p, SchedulingPolicy.PRIORITY) is False


class TestPriorityOrdering:
    def test_demo_section_4_order(self) -> None:
        """Demo §4: arrival order [A, B, C, D] → priority order [B, D, C, A].
        A: priority=3 (worst), B: priority=1, C: priority=2, D: priority=1."""
        requests = [
            PolicyRequest("A", priority=3, arrival_time=0.0),
            PolicyRequest("B", priority=1, arrival_time=1.0),
            PolicyRequest("C", priority=2, arrival_time=2.0),
            PolicyRequest("D", priority=1, arrival_time=3.0),
        ]
        order = [r.request_id for r in priority_ordering(requests)]
        # B and D tie on priority=1; B arrived earlier → B first.
        assert order == ["B", "D", "C", "A"]

    def test_empty_input_returns_empty(self) -> None:
        assert priority_ordering([]) == []

    def test_does_not_mutate_input(self) -> None:
        """Calling priority_ordering must NOT reorder the caller's list."""
        requests = [
            PolicyRequest("A", priority=3, arrival_time=0.0),
            PolicyRequest("B", priority=1, arrival_time=1.0),
        ]
        original_ids = [r.request_id for r in requests]
        priority_ordering(requests)
        # Caller's list unchanged.
        assert [r.request_id for r in requests] == original_ids


class TestAgedPriority:
    def test_demo_section_4_aged_a_is_minus_seven(self) -> None:
        """Demo §4: aged_priority(A: priority=3, arrival=0) at t=10, rate=1.0 = -7."""
        a = PolicyRequest("A", priority=3, arrival_time=0.0)
        assert aged_priority(a, now=10.0, aging_rate=1.0) == -7

    def test_recent_arrival_does_not_age(self) -> None:
        """A request that just arrived has wait_time=0 → effective = original priority."""
        r = PolicyRequest("R", priority=5, arrival_time=10.0)
        assert aged_priority(r, now=10.0, aging_rate=1.0) == 5

    def test_negative_wait_time_clamps_to_zero(self) -> None:
        """If `now` < `arrival_time` (clock skew?), wait_time clamps to 0."""
        r = PolicyRequest("R", priority=5, arrival_time=20.0)
        # now=10, arrival=20 → wait_time would be -10; clamped to 0.
        assert aged_priority(r, now=10.0, aging_rate=1.0) == 5

    def test_aging_rate_scales_linearly(self) -> None:
        """priority - rate*wait_time. rate=2 doubles the aging."""
        r = PolicyRequest("R", priority=10, arrival_time=0.0)
        # rate=1.0 at t=5 → 10 - 5 = 5.
        assert aged_priority(r, now=5.0, aging_rate=1.0) == 5
        # rate=2.0 at t=5 → 10 - 10 = 0.
        assert aged_priority(r, now=5.0, aging_rate=2.0) == 0
