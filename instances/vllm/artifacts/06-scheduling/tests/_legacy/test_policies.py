"""Tests — Ch6 Scheduling Policies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.scheduling_policies import (
    SchedulingPolicy, PauseState, ScheduledRequest,
    FCFSQueue, PriorityQueue, PreemptionPolicy, select_queue,
)


class TestScheduledRequest:
    def test_priority_ordering(self):
        """Lower priority value = higher priority (processed first)."""
        a = ScheduledRequest("A", priority=3, arrival_time=0)
        b = ScheduledRequest("B", priority=1, arrival_time=0)
        assert b < a  # B has lower priority value → higher priority

    def test_arrival_time_tiebreak(self):
        """Same priority → earlier arrival wins."""
        a = ScheduledRequest("A", priority=1, arrival_time=0)
        b = ScheduledRequest("B", priority=1, arrival_time=1)
        assert a < b

    def test_request_id_tiebreak(self):
        """Same priority and arrival → lexicographic ID wins."""
        a = ScheduledRequest("A", priority=1, arrival_time=0)
        b = ScheduledRequest("B", priority=1, arrival_time=0)
        assert a < b


class TestFCFSQueue:
    def test_fifo_order(self):
        q = FCFSQueue()
        q.add(ScheduledRequest("A"))
        q.add(ScheduledRequest("B"))
        q.add(ScheduledRequest("C"))
        assert q.pop().request_id == "A"
        assert q.pop().request_id == "B"
        assert q.pop().request_id == "C"

    def test_prepend(self):
        q = FCFSQueue()
        q.add(ScheduledRequest("A"))
        q.prepend(ScheduledRequest("B"))  # B goes to front
        assert q.pop().request_id == "B"
        assert q.pop().request_id == "A"


class TestPriorityQueue:
    def test_priority_order(self):
        q = PriorityQueue()
        q.add(ScheduledRequest("A", priority=3))
        q.add(ScheduledRequest("B", priority=1))
        q.add(ScheduledRequest("C", priority=2))
        assert q.pop().request_id == "B"  # priority 1 (highest)
        assert q.pop().request_id == "C"  # priority 2
        assert q.pop().request_id == "A"  # priority 3 (lowest)

    def test_prepend_falls_through(self):
        """In priority queue, prepend just does add — order is by priority."""
        q = PriorityQueue()
        q.add(ScheduledRequest("A", priority=1))
        q.prepend(ScheduledRequest("B", priority=3))
        q.prepend(ScheduledRequest("C", priority=0))  # top priority
        assert q.pop().request_id == "C"


class TestPreemption:
    def test_fcfs_preempts_most_recent(self):
        running = [
            ScheduledRequest("X", priority=1, arrival_time=0),
            ScheduledRequest("Y", priority=1, arrival_time=1),
            ScheduledRequest("Z", priority=1, arrival_time=2),
        ]
        victim = PreemptionPolicy.select_victim_fcfs(list(running))
        assert victim.request_id == "Z"

    def test_priority_preempts_worst_priority(self):
        running = [
            ScheduledRequest("X", priority=1, arrival_time=0),
            ScheduledRequest("Y", priority=5, arrival_time=1),
            ScheduledRequest("Z", priority=1, arrival_time=2),
        ]
        victim = PreemptionPolicy.select_victim_priority(running)
        assert victim.request_id == "Y"  # worst priority


class TestQueueSelection:
    def test_fcfs_prefers_skipped(self):
        """FCFS always drains skipped_waiting first."""
        waiting = FCFSQueue()
        skipped = FCFSQueue()
        waiting.add(ScheduledRequest("W"))
        skipped.add(ScheduledRequest("S"))
        chosen = select_queue(waiting, skipped, SchedulingPolicy.FCFS)
        assert chosen.peek().request_id == "S"

    def test_priority_compares_heads(self):
        """Priority mode: compare head elements, pick higher priority."""
        waiting = PriorityQueue()
        skipped = PriorityQueue()
        waiting.add(ScheduledRequest("W", priority=3))
        skipped.add(ScheduledRequest("S", priority=1))
        chosen = select_queue(waiting, skipped, SchedulingPolicy.PRIORITY)
        assert chosen.peek().request_id == "S"  # higher priority (1 < 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
