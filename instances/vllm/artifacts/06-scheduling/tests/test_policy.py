"""Unit tests for policy primitives:
- select_preemption_victim (FCFS pops tail, PRIORITY uses max(priority, arrival))
- select_waiting_queue (FCFS prefers skipped, PRIORITY peeks both heads)
- effective_token_budget (PAUSED_ALL → 0)

vLLM references:
- scheduler.py:L478-L504 (victim selection)
- scheduler.py:L1567-L1577 (queue selection)
- scheduler.py:L372-L374 (pause budget)
"""

from __future__ import annotations

import pytest

from implementation.policy import (
    PauseState,
    effective_token_budget,
    select_preemption_victim,
    select_waiting_queue,
)
from implementation.request_queue import (
    FCFSRequestQueue,
    PolicyRequest,
    PriorityRequestQueue,
    SchedulingPolicy,
)


def _r(rid: str, priority: int = 0, arrival: float = 0.0) -> PolicyRequest:
    return PolicyRequest(request_id=rid, priority=priority, arrival_time=arrival)


class TestSelectPreemptionVictim:
    def test_fcfs_pops_tail(self) -> None:
        """vLLM L504: FCFS preempts running.pop() — the LAST element."""
        running = [_r("first", arrival=0), _r("mid", arrival=1), _r("last", arrival=2)]
        requester = _r("requester", arrival=3)
        victim = select_preemption_victim(running, requester, SchedulingPolicy.FCFS)
        assert victim.request_id == "last"

    def test_priority_picks_worst_priority_value(self) -> None:
        """K19: vLLM L478-L484 uses max(priority, arrival_time).
        max() returns the LARGEST priority value (= worst, lowest precedence)."""
        running = [
            _r("A", priority=1, arrival=0),  # high precedence, oldest
            _r("B", priority=3, arrival=1),  # WORST priority value
            _r("C", priority=2, arrival=2),
        ]
        requester = _r("requester", priority=1, arrival=3)
        victim = select_preemption_victim(running, requester, SchedulingPolicy.PRIORITY)
        # B has priority=3, the largest value → worst → preempt it.
        assert victim.request_id == "B"

    def test_priority_breaks_ties_with_latest_arrival(self) -> None:
        """When two requests share the worst priority, max() picks the
        LATER arrival (largest arrival_time tuple element)."""
        running = [
            _r("A", priority=5, arrival=0.0),  # tied worst-priority, EARLIEST
            _r("B", priority=1, arrival=1.0),  # high precedence
            _r("C", priority=5, arrival=2.0),  # tied worst-priority, LATEST
        ]
        requester = _r("requester", priority=1, arrival=3)
        victim = select_preemption_victim(running, requester, SchedulingPolicy.PRIORITY)
        # Among (A, C) both at priority=5, C arrived later → tail wins.
        assert victim.request_id == "C"

    def test_empty_running_raises(self) -> None:
        with pytest.raises(ValueError):
            select_preemption_victim([], _r("x"), SchedulingPolicy.FCFS)
        with pytest.raises(ValueError):
            select_preemption_victim([], _r("x"), SchedulingPolicy.PRIORITY)

    def test_demo_section_1_reproduces(self) -> None:
        """Demo §1: FCFS picks C (last admitted), PRIORITY picks B (worst prio)."""
        running = [
            _r("A", priority=1, arrival=0.0),
            _r("B", priority=3, arrival=1.0),  # worst priority
            _r("C", priority=2, arrival=2.0),  # last admitted
        ]
        requester = _r("D", priority=1, arrival=3.0)
        assert select_preemption_victim(running, requester, SchedulingPolicy.FCFS).request_id == "C"
        assert select_preemption_victim(running, requester, SchedulingPolicy.PRIORITY).request_id == "B"


class TestSelectWaitingQueue:
    def test_fcfs_prefers_skipped(self) -> None:
        """K20 / vLLM L1568-L1569: FCFS unconditionally returns skipped if non-empty."""
        waiting = FCFSRequestQueue()
        skipped = FCFSRequestQueue()
        waiting.add_request(_r("new"))
        skipped.add_request(_r("blocked"))
        chosen = select_waiting_queue(waiting, skipped, SchedulingPolicy.FCFS)
        assert chosen is skipped

    def test_fcfs_falls_back_to_waiting_when_skipped_empty(self) -> None:
        waiting = FCFSRequestQueue()
        skipped = FCFSRequestQueue()
        waiting.add_request(_r("new"))
        chosen = select_waiting_queue(waiting, skipped, SchedulingPolicy.FCFS)
        assert chosen is waiting

    def test_fcfs_returns_none_when_both_empty(self) -> None:
        waiting = FCFSRequestQueue()
        skipped = FCFSRequestQueue()
        assert select_waiting_queue(waiting, skipped, SchedulingPolicy.FCFS) is None

    def test_priority_returns_queue_with_better_head(self) -> None:
        """K20 / vLLM L1572-L1575: PRIORITY peeks both heads, returns whichever
        has the better (priority, arrival) tuple."""
        waiting = PriorityRequestQueue()
        skipped = PriorityRequestQueue()
        waiting.add_request(_r("new", priority=1, arrival=10))    # high precedence
        skipped.add_request(_r("blocked", priority=5, arrival=1))  # low precedence
        chosen = select_waiting_queue(waiting, skipped, SchedulingPolicy.PRIORITY)
        # waiting has the better head (priority=1 < priority=5) → pick waiting.
        assert chosen is waiting

    def test_priority_picks_skipped_when_skipped_has_better_head(self) -> None:
        waiting = PriorityRequestQueue()
        skipped = PriorityRequestQueue()
        waiting.add_request(_r("low-new", priority=10, arrival=0))
        skipped.add_request(_r("hi-blocked", priority=1, arrival=99))
        chosen = select_waiting_queue(waiting, skipped, SchedulingPolicy.PRIORITY)
        assert chosen is skipped

    def test_priority_returns_only_non_empty_queue(self) -> None:
        waiting = PriorityRequestQueue()
        skipped = PriorityRequestQueue()
        waiting.add_request(_r("only", priority=5, arrival=0))
        chosen = select_waiting_queue(waiting, skipped, SchedulingPolicy.PRIORITY)
        assert chosen is waiting

    def test_priority_returns_none_when_both_empty(self) -> None:
        waiting = PriorityRequestQueue()
        skipped = PriorityRequestQueue()
        assert select_waiting_queue(waiting, skipped, SchedulingPolicy.PRIORITY) is None


class TestEffectiveTokenBudget:
    def test_unpaused_returns_raw_budget(self) -> None:
        assert effective_token_budget(2048, PauseState.UNPAUSED) == 2048

    def test_paused_new_returns_raw_budget(self) -> None:
        """PAUSED_NEW only blocks Phase 2 admission via L568 guard;
        the BUDGET itself is unchanged. Running requests still progress."""
        assert effective_token_budget(2048, PauseState.PAUSED_NEW) == 2048

    def test_paused_all_zeros_budget(self) -> None:
        """vLLM L372-L374: PAUSED_ALL → token_budget = 0. Both phases halt."""
        assert effective_token_budget(2048, PauseState.PAUSED_ALL) == 0

    def test_paused_all_zeros_zero_input(self) -> None:
        """Idempotent: 0 stays 0."""
        assert effective_token_budget(0, PauseState.PAUSED_ALL) == 0

    def test_demo_pause_state_table_reproduces(self) -> None:
        """Demo §1: budget=2048, three pause states."""
        assert effective_token_budget(2048, PauseState.UNPAUSED) == 2048
        assert effective_token_budget(2048, PauseState.PAUSED_NEW) == 2048
        assert effective_token_budget(2048, PauseState.PAUSED_ALL) == 0
