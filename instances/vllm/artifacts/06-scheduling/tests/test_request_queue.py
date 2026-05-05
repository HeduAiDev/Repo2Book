"""Tests for request_queue.py — FCFS and Priority queues."""
import sys
from pathlib import Path

# Make implementation importable
IMPL_DIR = Path(__file__).parent.parent / "implementation"
sys.path.insert(0, str(IMPL_DIR))

import pytest

from request import Request, RequestStatus
from request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


def _req(rid: str, priority: int = 0, arrival: float = 0.0) -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=[0, 1, 2],
        max_tokens=1,
        priority=priority,
        arrival_time=arrival,
    )


# ─── Factory ────────────────────────────────────────────────────────────────


class TestFactory:
    def test_fcfs_factory(self):
        q = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(q, FCFSRequestQueue)

    def test_priority_factory(self):
        q = create_request_queue(SchedulingPolicy.PRIORITY)
        assert isinstance(q, PriorityRequestQueue)


# ─── FCFSRequestQueue ───────────────────────────────────────────────────────


class TestFCFSRequestQueue:
    def test_empty_queue_is_falsy(self):
        q = FCFSRequestQueue()
        assert not q
        assert len(q) == 0

    def test_add_then_pop_preserves_order(self):
        q = FCFSRequestQueue()
        a, b, c = _req("A"), _req("B"), _req("C")
        q.add_request(a)
        q.add_request(b)
        q.add_request(c)
        assert len(q) == 3
        assert q.pop_request() is a
        assert q.pop_request() is b
        assert q.pop_request() is c
        assert not q

    def test_peek_does_not_remove(self):
        q = FCFSRequestQueue()
        a = _req("A")
        q.add_request(a)
        assert q.peek_request() is a
        assert len(q) == 1  # still there

    def test_peek_empty_raises(self):
        q = FCFSRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_prepend_goes_to_head(self):
        q = FCFSRequestQueue()
        a, b, c = _req("A"), _req("B"), _req("C")
        q.add_request(a)
        q.add_request(b)
        q.prepend_request(c)  # push to head
        assert q.pop_request() is c  # c first
        assert q.pop_request() is a
        assert q.pop_request() is b

    def test_remove_single_request(self):
        q = FCFSRequestQueue()
        a, b, c = _req("A"), _req("B"), _req("C")
        q.add_request(a)
        q.add_request(b)
        q.add_request(c)
        q.remove_request(b)
        assert len(q) == 2
        assert q.pop_request() is a
        assert q.pop_request() is c

    def test_bulk_remove(self):
        q = FCFSRequestQueue()
        a, b, c, d = _req("A"), _req("B"), _req("C"), _req("D")
        for r in [a, b, c, d]:
            q.add_request(r)
        q.remove_requests([b, d])
        remaining = list(q)
        assert remaining == [a, c]

    def test_iteration_preserves_fifo(self):
        q = FCFSRequestQueue()
        reqs = [_req(f"R{i}") for i in range(5)]
        for r in reqs:
            q.add_request(r)
        assert list(q) == reqs


# ─── PriorityRequestQueue ───────────────────────────────────────────────────


class TestPriorityRequestQueue:
    def test_empty_queue_is_falsy(self):
        q = PriorityRequestQueue()
        assert not q
        assert len(q) == 0

    def test_pop_returns_lowest_priority_value_first(self):
        """Lower priority value = higher precedence."""
        q = PriorityRequestQueue()
        q.add_request(_req("mid", priority=5))
        q.add_request(_req("hi", priority=1))
        q.add_request(_req("lo", priority=10))
        assert q.pop_request().request_id == "hi"
        assert q.pop_request().request_id == "mid"
        assert q.pop_request().request_id == "lo"

    def test_tiebreak_by_arrival_time(self):
        """Same priority → earlier arrival wins."""
        q = PriorityRequestQueue()
        q.add_request(_req("later", priority=5, arrival=2.0))
        q.add_request(_req("earlier", priority=5, arrival=1.0))
        q.add_request(_req("even_later", priority=5, arrival=3.0))
        assert q.pop_request().request_id == "earlier"
        assert q.pop_request().request_id == "later"
        assert q.pop_request().request_id == "even_later"

    def test_peek_does_not_remove(self):
        q = PriorityRequestQueue()
        a = _req("A", priority=1)
        q.add_request(a)
        assert q.peek_request() is a
        assert len(q) == 1

    def test_peek_empty_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_pop_empty_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.pop_request()

    def test_prepend_equivalent_to_add(self):
        """In a priority queue, prepend must not override priority ordering."""
        q = PriorityRequestQueue()
        q.add_request(_req("A", priority=5))
        # Prepending a lower-priority (higher value) request should NOT jump ahead.
        q.prepend_request(_req("B", priority=10))
        # A (priority 5) still comes before B (priority 10).
        assert q.pop_request().request_id == "A"
        assert q.pop_request().request_id == "B"

    def test_remove_request(self):
        q = PriorityRequestQueue()
        a = _req("A", priority=1)
        b = _req("B", priority=2)
        c = _req("C", priority=3)
        q.add_request(b)
        q.add_request(a)
        q.add_request(c)
        q.remove_request(b)
        assert len(q) == 2
        assert q.pop_request() is a
        assert q.pop_request() is c

    def test_bulk_remove(self):
        q = PriorityRequestQueue()
        a = _req("A", priority=1)
        b = _req("B", priority=2)
        c = _req("C", priority=3)
        d = _req("D", priority=4)
        for r in [a, b, c, d]:
            q.add_request(r)
        q.remove_requests([b, d])
        remaining = [q.pop_request() for _ in range(len(q))]
        assert remaining == [a, c]

    def test_iteration_is_in_priority_order(self):
        q = PriorityRequestQueue()
        q.add_request(_req("mid", priority=5))
        q.add_request(_req("hi", priority=1))
        q.add_request(_req("lo", priority=10))
        ordered = [r.request_id for r in q]
        assert ordered == ["hi", "mid", "lo"]
        # Iteration is destructive-on-copy — original still intact
        assert len(q) == 3

    def test_same_priority_and_arrival_no_crash(self):
        """When priority and arrival_time are identical, request_id is not
        used as tiebreaker here (dataclass eq=False), but heap ops still work."""
        q = PriorityRequestQueue()
        q.add_request(_req("A", priority=5, arrival=1.0))
        q.add_request(_req("B", priority=5, arrival=1.0))
        # Two requests pop without error — order between them is unspecified.
        p1 = q.pop_request()
        p2 = q.pop_request()
        assert {p1.request_id, p2.request_id} == {"A", "B"}
