# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py
"""FCFS request queue — same shape as vLLM, only the FCFS policy.

vLLM exposes both FCFSRequestQueue (deque-backed) and PriorityRequestQueue
(heap-backed) under a `RequestQueue` ABC. For continuous batching the FCFS
case is sufficient to demonstrate the running-first / waiting-second discipline.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from .request import Request


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L75-L128
class FCFSRequestQueue:
    """Deque-backed FIFO. Same method names and semantics as vLLM.

    Note: vLLM's FCFSRequestQueue *is* a `deque[Request]` via multiple
    inheritance; we wrap one instead, but expose the same five methods that
    the scheduler depends on (`add_request`, `pop_request`, `peek_request`,
    `prepend_request`, `prepend_requests`).
    """

    def __init__(self) -> None:
        self._dq: deque[Request] = deque()

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L78-L80
    def add_request(self, request: Request) -> None:
        self._dq.append(request)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L82-L84
    def pop_request(self) -> Request:
        return self._dq.popleft()

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L86-L90
    def peek_request(self) -> Request:
        if not self._dq:
            raise IndexError("peek from an empty queue")
        return self._dq[0]

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L92-L94
    def prepend_request(self, request: Request) -> None:
        self._dq.appendleft(request)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L96-L103
    def prepend_requests(self, requests: "FCFSRequestQueue") -> None:
        # vLLM uses `extendleft(requests)`, which reverses order — we mirror
        # that quirk so behaviour matches when the caller relies on it.
        self._dq.extendleft(requests._dq)

    def remove_request(self, request: Request) -> None:
        self._dq.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L109-L116
        to_remove = set(map(id, requests))
        keep = [r for r in self._dq if id(r) not in to_remove]
        self._dq.clear()
        self._dq.extend(keep)

    def __bool__(self) -> bool:
        return len(self._dq) > 0

    def __len__(self) -> int:
        return len(self._dq)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._dq)
