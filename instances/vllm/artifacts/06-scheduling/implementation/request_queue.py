# Simplified RequestQueue — FCFS and Priority policies
# REFERENCE: vllm/v1/core/sched/request_queue.py

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from request import Request


class SchedulingPolicy(Enum):
    """Scheduling policy for the waiting queue.

    REFERENCE: request_queue.py:L13-L17
    """

    FCFS = "fcfs"           # First-come-first-served
    PRIORITY = "priority"   # Priority-based with (priority, arrival_time) ordering


class RequestQueue(ABC):
    """Abstract base class for request queues.

    REFERENCE: request_queue.py:L20-L72 — RequestQueue ABC

    Polymorphic interface so the scheduler does not branch on policy inside
    the hot path. FCFSRequestQueue and PriorityRequestQueue implement the
    same API but with different ordering.
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue per policy."""

    @abstractmethod
    def pop_request(self) -> Request:
        """Remove and return the highest-priority request."""

    @abstractmethod
    def peek_request(self) -> Request:
        """Look at the highest-priority request without removing it."""

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Put a request at the front (used for preempted requests)."""

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request (e.g., on abort)."""

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Bulk-remove specific requests."""

    @abstractmethod
    def __bool__(self) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Request]: ...


class FCFSRequestQueue(deque, RequestQueue):
    """First-come-first-served queue, backed by a deque.

    REFERENCE: request_queue.py:L75-L128 — FCFSRequestQueue(deque[Request], RequestQueue)

    Right end = tail (newest arrivals); left end = head (next to serve).
    Preempted requests are prepended (pushed to the left) so they resume
    before newer waiting requests in the next step.
    """

    def add_request(self, request: Request) -> None:
        self.append(request)  # push to right (tail)

    def pop_request(self) -> Request:
        return self.popleft()  # pop from left (head)

    def peek_request(self) -> Request:
        if not self:
            raise IndexError("peek from empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Preempted requests go back to the front of the queue."""
        self.appendleft(request)

    def remove_request(self, request: Request) -> None:
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Bulk remove: clear and rebuild without the removed requests.

        REFERENCE: request_queue.py:L109-L116 — same pattern.
        deque does not support in-place filtering.
        """
        to_remove = set(requests)
        remaining = [r for r in self if r not in to_remove]
        self.clear()
        self.extend(remaining)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """Priority queue backed by a min-heap.

    REFERENCE: request_queue.py:L131-L198 — PriorityRequestQueue

    Ordering defined by Request.__lt__: (priority, arrival_time).
    - Lower priority value = higher scheduling priority
    - Same priority → earlier arrival wins (FIFO within priority band)

    This is how vLLM prevents starvation: a request with priority 5 that
    arrived an hour ago still beats a priority 5 request that arrived now,
    so no "newer requests with same priority" can starve out older ones.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """In a priority queue, 'prepend' has no meaning — just re-add.

        REFERENCE: request_queue.py:L160-L165 — same comment in vLLM.
        Priority queues are re-ordered by (priority, arrival_time), not
        by insertion order, so prepending is equivalent to adding.
        """
        self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove + re-heapify. O(n) — acceptable for infrequent removals."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate in priority order (destructive on a heap copy).

        REFERENCE: request_queue.py:L194-L198 — same pattern.
        """
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Factory: build the queue matching the policy.

    REFERENCE: request_queue.py:L201-L208
    """
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
