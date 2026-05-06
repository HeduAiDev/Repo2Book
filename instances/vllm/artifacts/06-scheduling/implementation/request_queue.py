# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py
"""Request queues — the data structure underneath the scheduling policy.

Ch04 implemented FCFSRequestQueue. Ch06 adds the FULL queue family:

    SchedulingPolicy   (Enum)              ↔ request_queue.py:L13-L17
    RequestQueue       (ABC)               ↔ request_queue.py:L20-L72
    FCFSRequestQueue   (deque-backed)      ↔ request_queue.py:L75-L128
    PriorityRequestQueue (heap-backed)     ↔ request_queue.py:L131-L198
    create_request_queue (factory)         ↔ request_queue.py:L201-L208

The two queue types differ in:
    - add complexity:   FCFS O(1)          / PRIORITY O(log n)
    - pop complexity:   FCFS O(1)          / PRIORITY O(log n)
    - prepend semantic: FCFS appendleft    / PRIORITY no-op (just add)

The "no-op prepend" for PRIORITY (`request_queue.py:L160-L165`) is the key
policy invariant: in a priority queue there is no concept of "front", only
priority order. Preempted requests re-enter and find their place by
(priority, arrival_time), not by FIFO position.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L13-L17
class SchedulingPolicy(Enum):
    """The two policies vLLM ships. `vllm/config/scheduler.py:L22` exposes
    these as a Literal["fcfs", "priority"] config field."""

    FCFS = "fcfs"
    PRIORITY = "priority"


# REFERENCE: instances/vllm/source/vllm/v1/request.py:L296-L307
@dataclass
class PolicyRequest:
    """Minimal Request stand-in for policy demos.

    Same `__lt__` rule as `Request.__lt__` in vLLM (request.py:L296). We
    cannot reuse Ch04's Request directly — that one lacks a `priority`
    field, which is the whole point of Ch06.
    """

    request_id: str
    priority: int = 0       # smaller = higher priority (vLLM convention)
    arrival_time: float = 0.0
    prompt_tokens: int = 0
    num_computed_tokens: int = 0
    status: str = "WAITING"

    # Pedagogical extras for starvation/Pareto analysis (NOT in vLLM's Request).
    finish_time: float = -1.0
    enqueue_index: int = -1  # used to break id() ties stably in tests

    def __lt__(self, other: "PolicyRequest") -> bool:
        # REFERENCE: instances/vllm/source/vllm/v1/request.py:L301-L307
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L20-L72
class RequestQueue(ABC):
    """ABC matches vLLM exactly. The five hooks are everything the scheduler
    calls during the loop."""

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L23-L26
    @abstractmethod
    def add_request(self, request: PolicyRequest) -> None: ...
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L28-L31
    @abstractmethod
    def pop_request(self) -> PolicyRequest: ...
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L33-L36
    @abstractmethod
    def peek_request(self) -> PolicyRequest: ...
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L38-L41
    @abstractmethod
    def prepend_request(self, request: PolicyRequest) -> None: ...
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L43-L47
    @abstractmethod
    def prepend_requests(self, other: "RequestQueue") -> None: ...
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L60-L62
    @abstractmethod
    def __bool__(self) -> bool: ...
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L64-L66
    @abstractmethod
    def __len__(self) -> int: ...
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L69-L71
    @abstractmethod
    def __iter__(self) -> Iterator[PolicyRequest]: ...


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L75-L128
class FCFSRequestQueue(RequestQueue):
    """Deque-backed FIFO. Same five methods as Ch04, restated here so this
    module is self-contained for Ch06's tests."""

    def __init__(self) -> None:
        self._dq: deque[PolicyRequest] = deque()

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L78-L80
    def add_request(self, request: PolicyRequest) -> None:
        self._dq.append(request)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L82-L84
    def pop_request(self) -> PolicyRequest:
        return self._dq.popleft()

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L86-L90
    def peek_request(self) -> PolicyRequest:
        if not self._dq:
            raise IndexError("peek from empty queue")
        return self._dq[0]

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L92-L94
    def prepend_request(self, request: PolicyRequest) -> None:
        self._dq.appendleft(request)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L96-L103
    def prepend_requests(self, other: RequestQueue) -> None:
        # vLLM uses `extendleft(other)`, which reverses; we mirror.
        self._dq.extendleft(iter(other))

    def __bool__(self) -> bool:
        return bool(self._dq)

    def __len__(self) -> int:
        return len(self._dq)

    def __iter__(self) -> Iterator[PolicyRequest]:
        return iter(self._dq)


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L131-L198
class PriorityRequestQueue(RequestQueue):
    """Heap-backed priority queue. `__lt__` on PolicyRequest defines order.

    KEY INVARIANT (vLLM:L160-L165): `prepend_request` is identical to
    `add_request` — there is NO concept of "prepending" in a priority queue.
    This matters for preemption: when `_preempt_request` re-queues the
    preempted request via `self.waiting.prepend_request(request)`, in
    PRIORITY mode it just heap-pushes; in FCFS mode it appendleft's.
    """

    def __init__(self) -> None:
        self._heap: list[PolicyRequest] = []

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L144-L146
    def add_request(self, request: PolicyRequest) -> None:
        heapq.heappush(self._heap, request)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L148-L152
    def pop_request(self) -> PolicyRequest:
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L154-L158
    def peek_request(self) -> PolicyRequest:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L160-L165
    def prepend_request(self, request: PolicyRequest) -> None:
        # Documented no-op equivalent to add_request. THE invariant.
        self.add_request(request)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L167-L173
    def prepend_requests(self, other: RequestQueue) -> None:
        for r in other:
            self.add_request(r)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[PolicyRequest]:
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L194-L198
        # Yield in priority order. We pop from a copy so the original heap
        # stays intact (matches vLLM's iter implementation).
        heap_copy = list(self._heap)
        while heap_copy:
            yield heapq.heappop(heap_copy)


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L201-L208
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Factory used everywhere in `scheduler.py` (`L167`, `L169`, `L569`)."""
    if policy is SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    if policy is SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    raise ValueError(f"Unknown scheduling policy: {policy}")
