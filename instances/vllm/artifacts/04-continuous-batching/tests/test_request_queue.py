"""Unit tests for FCFSRequestQueue.

Verifies the five methods the scheduler depends on (add_request, pop_request,
peek_request, prepend_request, prepend_requests) plus the iteration helpers.
The semantics MUST match vLLM (request_queue.py:L75-L128) so the scheduler's
control flow is portable.
"""

from __future__ import annotations

import pytest

from implementation.request import Request
from implementation.request_queue import FCFSRequestQueue


def _mk(rid: str) -> Request:
    return Request(rid, [0, 1, 2], max_tokens=1, arrival_time=0.0)


class TestFCFS:
    def test_add_then_pop_is_fifo(self) -> None:
        """add_request appends; pop_request returns the OLDEST first."""
        q = FCFSRequestQueue()
        a, b, c = _mk("a"), _mk("b"), _mk("c")
        q.add_request(a)
        q.add_request(b)
        q.add_request(c)
        assert q.pop_request() is a
        assert q.pop_request() is b
        assert q.pop_request() is c

    def test_peek_does_not_remove(self) -> None:
        """peek_request returns the head without consuming it."""
        q = FCFSRequestQueue()
        a = _mk("a")
        q.add_request(a)
        assert q.peek_request() is a
        assert len(q) == 1

    def test_peek_empty_raises(self) -> None:
        """peek on empty queue must raise IndexError (matches vLLM)."""
        q = FCFSRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_prepend_jumps_to_head(self) -> None:
        """prepend_request puts the request at the FRONT (used on preemption)."""
        q = FCFSRequestQueue()
        a, b, c = _mk("a"), _mk("b"), _mk("c")
        q.add_request(a)
        q.add_request(b)
        q.prepend_request(c)
        assert q.pop_request() is c
        assert q.pop_request() is a
        assert q.pop_request() is b

    def test_remove_request(self) -> None:
        """remove_request takes a request out of the middle."""
        q = FCFSRequestQueue()
        a, b, c = _mk("a"), _mk("b"), _mk("c")
        q.add_request(a)
        q.add_request(b)
        q.add_request(c)
        q.remove_request(b)
        assert q.pop_request() is a
        assert q.pop_request() is c

    def test_remove_requests_bulk(self) -> None:
        """remove_requests removes by identity (matches vLLM L109-L116)."""
        q = FCFSRequestQueue()
        a, b, c, d = _mk("a"), _mk("b"), _mk("c"), _mk("d")
        for r in (a, b, c, d):
            q.add_request(r)
        q.remove_requests([b, d])
        remaining = [r.request_id for r in q]
        assert remaining == ["a", "c"]

    def test_truthiness_and_length(self) -> None:
        """Empty queue is falsy; non-empty is truthy. Also __len__."""
        q = FCFSRequestQueue()
        assert not q
        assert len(q) == 0
        q.add_request(_mk("x"))
        assert q
        assert len(q) == 1
