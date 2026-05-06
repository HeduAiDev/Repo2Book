"""Unit tests for SchedulingPolicy, PolicyRequest, both queues, and factory.

Critical invariants (vLLM source pinned at 98661fe):
- PolicyRequest.__lt__ tie-break: priority < arrival_time < request_id < id() (request.py:L296-L307)
- PriorityRequestQueue.prepend_request == add_request (K18 / request_queue.py:L160-L165)
- create_request_queue dispatches by SchedulingPolicy enum (request_queue.py:L201-L208)
- FCFS pop order = insertion order (deque-backed)
- Priority pop order = (priority asc, arrival_time asc) heap order
- Determinism: re-pushing the same items always yields the same pop order
"""

from __future__ import annotations

import pytest

from implementation.request_queue import (
    FCFSRequestQueue,
    PolicyRequest,
    PriorityRequestQueue,
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


def _r(rid: str, priority: int = 0, arrival: float = 0.0) -> PolicyRequest:
    return PolicyRequest(request_id=rid, priority=priority, arrival_time=arrival)


class TestPolicyRequestOrdering:
    """vLLM request.py:L296-L307 — 4-tier tie-break: priority, arrival, id, id()."""

    def test_priority_is_first_tier(self) -> None:
        a = _r("a", priority=5, arrival=100.0)
        b = _r("b", priority=1, arrival=100.0)
        # Smaller priority = higher precedence → b < a.
        assert b < a
        assert not (a < b)

    def test_arrival_is_second_tier_on_tied_priority(self) -> None:
        a = _r("a", priority=2, arrival=10.0)
        b = _r("b", priority=2, arrival=5.0)
        # Same priority → earlier arrival wins.
        assert b < a

    def test_request_id_is_third_tier(self) -> None:
        a = _r("xyz", priority=2, arrival=10.0)
        b = _r("abc", priority=2, arrival=10.0)
        # Same priority, same arrival → string ordering of ID.
        assert b < a

    def test_priority_smaller_means_higher(self) -> None:
        """vLLM convention (config/scheduler.py:L114): smaller is higher."""
        high = _r("h", priority=0)
        low = _r("l", priority=10)
        assert high < low


class TestFCFSRequestQueue:
    def test_add_pop_is_fifo(self) -> None:
        q = FCFSRequestQueue()
        a, b, c = _r("a", arrival=0), _r("b", arrival=1), _r("c", arrival=2)
        q.add_request(a)
        q.add_request(b)
        q.add_request(c)
        assert q.pop_request() is a
        assert q.pop_request() is b
        assert q.pop_request() is c

    def test_peek_does_not_consume(self) -> None:
        q = FCFSRequestQueue()
        q.add_request(_r("x"))
        before = len(q)
        q.peek_request()
        assert len(q) == before

    def test_peek_empty_raises(self) -> None:
        q = FCFSRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_prepend_request_jumps_to_head(self) -> None:
        """FCFS prepend = appendleft. The request lands at the FRONT.
        Contrast with PriorityRequestQueue which has no front concept."""
        q = FCFSRequestQueue()
        a, b = _r("a", arrival=0), _r("b", arrival=1)
        q.add_request(a)
        q.add_request(b)
        c = _r("c", arrival=99)  # late arrival but prepended
        q.prepend_request(c)
        assert q.pop_request() is c
        assert q.pop_request() is a
        assert q.pop_request() is b

    def test_truthiness_and_len(self) -> None:
        q = FCFSRequestQueue()
        assert not q
        assert len(q) == 0
        q.add_request(_r("x"))
        assert q
        assert len(q) == 1

    def test_iter_yields_in_fifo_order_without_consuming(self) -> None:
        q = FCFSRequestQueue()
        for i in range(3):
            q.add_request(_r(f"r{i}", arrival=i))
        ids = [r.request_id for r in q]
        assert ids == ["r0", "r1", "r2"]
        # Iteration must NOT consume.
        assert len(q) == 3


class TestPriorityRequestQueue:
    def test_pop_returns_smallest_priority_first(self) -> None:
        q = PriorityRequestQueue()
        # Insert in random order; expect priority-asc pop order.
        q.add_request(_r("low", priority=10, arrival=0))
        q.add_request(_r("high", priority=1, arrival=0))
        q.add_request(_r("mid", priority=5, arrival=0))
        assert q.pop_request().request_id == "high"
        assert q.pop_request().request_id == "mid"
        assert q.pop_request().request_id == "low"

    def test_arrival_breaks_priority_ties(self) -> None:
        q = PriorityRequestQueue()
        q.add_request(_r("late", priority=2, arrival=10.0))
        q.add_request(_r("early", priority=2, arrival=5.0))
        # Tied priority → earlier arrival wins.
        assert q.pop_request().request_id == "early"
        assert q.pop_request().request_id == "late"

    def test_peek_does_not_consume(self) -> None:
        q = PriorityRequestQueue()
        q.add_request(_r("x", priority=3))
        q.peek_request()
        assert len(q) == 1

    def test_peek_empty_raises(self) -> None:
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_pop_empty_raises(self) -> None:
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.pop_request()

    def test_prepend_request_equals_add_request(self) -> None:
        """K18 / vLLM L160-L165: THE invariant. prepend_request just heap-pushes;
        a low-priority preempted request does NOT jump to the head."""
        q = PriorityRequestQueue()
        q.add_request(_r("high", priority=1, arrival=0))
        q.add_request(_r("mid", priority=5, arrival=1))
        # Now "preempt" a low-priority request via prepend.
        q.prepend_request(_r("preempted", priority=10, arrival=2))
        # Pop order should still be priority-asc; the preempted request lands LAST.
        order = [q.pop_request().request_id for _ in range(3)]
        assert order == ["high", "mid", "preempted"]

    def test_prepend_requests_bulk(self) -> None:
        q = PriorityRequestQueue()
        q.add_request(_r("base", priority=5, arrival=0))
        # Build a queue to prepend.
        other = FCFSRequestQueue()
        other.add_request(_r("p1", priority=1, arrival=10))
        other.add_request(_r("p2", priority=10, arrival=11))
        q.prepend_requests(other)
        # All three are now in the heap; expect priority-asc pop order.
        ids = [q.pop_request().request_id for _ in range(3)]
        assert ids == ["p1", "base", "p2"]

    def test_iter_yields_priority_order_without_consuming(self) -> None:
        """vLLM L194-L198: iteration pops from a copy, original heap intact."""
        q = PriorityRequestQueue()
        q.add_request(_r("low", priority=10, arrival=0))
        q.add_request(_r("high", priority=1, arrival=0))
        q.add_request(_r("mid", priority=5, arrival=0))
        ids = [r.request_id for r in q]
        assert ids == ["high", "mid", "low"]
        assert len(q) == 3

    def test_determinism_across_reseeds(self) -> None:
        """The 3-tier tie-break (priority, arrival, request_id) gives a fully
        deterministic order — re-inserting the same set in any order gives the
        same pop order every time. id() is the 4th tier and only kicks in for
        truly identical (priority, arrival, id) triples; we don't test it here."""
        items = [
            _r("a", priority=1, arrival=0),
            _r("b", priority=1, arrival=1),
            _r("c", priority=2, arrival=0),
            _r("d", priority=2, arrival=2),
        ]
        # Insert in 3 different orders — pop order must be the same.
        orders = []
        for perm in [items, list(reversed(items)), [items[2], items[0], items[3], items[1]]]:
            q = PriorityRequestQueue()
            for r in perm:
                q.add_request(r)
            orders.append([q.pop_request().request_id for _ in range(4)])
        assert orders[0] == orders[1] == orders[2]


class TestFactory:
    def test_create_fcfs(self) -> None:
        q = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(q, FCFSRequestQueue)
        assert isinstance(q, RequestQueue)

    def test_create_priority(self) -> None:
        q = create_request_queue(SchedulingPolicy.PRIORITY)
        assert isinstance(q, PriorityRequestQueue)
        assert isinstance(q, RequestQueue)

    def test_unknown_policy_raises(self) -> None:
        # Construct a "fake" enum-like value to verify the dispatch raises.
        # Easier: pass None.
        with pytest.raises(ValueError):
            create_request_queue(None)  # type: ignore[arg-type]
