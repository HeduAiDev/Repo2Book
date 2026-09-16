# SOURCE: vllm/v1/core/sched/request_queue.py
# 只做减法的忠实精简版：本章消费面 = waiting/skipped_waiting 双队列的
# add/pop/peek/prepend/remove（调度器窥队头与回侧队的物理载体）。
# SchedulingPolicy + RequestQueue ABC + FCFSRequestQueue 逐字。
# SUBTRACTED: SPDX 版权头；PriorityRequestQueue（heapq 双堆，L131-L199——
# 优先级调度域，ch11）与 create_request_queue 工厂的 priority 分支
# （L201-L205——精简版 FCFS-only，Scheduler 直接构造 FCFSRequestQueue）。
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Iterator, Iterable

if TYPE_CHECKING:
    from vllm.v1.request import Request


# SOURCE: vllm/v1/core/sched/request_queue.py:L13-L17 —— 逐字
class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


# SOURCE: vllm/v1/core/sched/request_queue.py:L20-L59 RequestQueue —— 逐字
class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L24（ABC）
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L29（ABC）
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L34（ABC）
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L39（ABC）
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L44（ABC）
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass


# SOURCE: vllm/v1/core/sched/request_queue.py:L75-L130 FCFSRequestQueue —— 逐字
class FCFSRequestQueue(deque, RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L78（FCFS）
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L82（FCFS）
    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L86（FCFS）
    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    # SOURCE: vllm/v1/core/sched/request_queue.py:L92（FCFS）
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L96（FCFS）
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L105（FCFS）
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L109（FCFS）
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L118（FCFS）
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    # SOURCE: vllm/v1/core/sched/request_queue.py:L122（FCFS）
    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L126（FCFS）
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()
