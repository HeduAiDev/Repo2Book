# SOURCE: vllm/v1/core/sched/request_queue.py
# waiting/skipped_waiting 的容器：FCFS 是 deque 子类（add=append、pop=popleft、
# peek 看队头、prepend=appendleft——ch11 抢占回队头靠最后这个）。
# SUBTRACTED: PriorityRequestQueue 堆实现（L131-L198）——PRIORITY 调度策略是
#   同构变体（dossier.delete 第 6 条批准，ch11 可再提）；默认 policy='fcfs'。
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from .request import Request


# SOURCE: vllm/v1/core/sched/request_queue.py:L13 SchedulingPolicy
class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


# SOURCE: vllm/v1/core/sched/request_queue.py:L20 RequestQueue
class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L23-L26
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L28-L31
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L33-L36
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L38-L41
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L43-L47
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L49-L52
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L54-L57
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L59-L62
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L64-L67
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    # SOURCE: vllm/v1/core/sched/request_queue.py:L69-L72
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


# SOURCE: vllm/v1/core/sched/request_queue.py:L75 FCFSRequestQueue
class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L78-L80
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L82-L84
    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L86-L90
    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    # SOURCE: vllm/v1/core/sched/request_queue.py:L92-L94
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L96-L103
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L105-L107
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L109-L116
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L118-L120
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    # SOURCE: vllm/v1/core/sched/request_queue.py:L122-L124
    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L126-L128
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


# SOURCE: vllm/v1/core/sched/request_queue.py:L201 create_request_queue
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    # SUBTRACTED: PRIORITY 分支（PriorityRequestQueue 堆实现，dossier.delete
    #   第 6 条批准）——本精简版只支持默认 FCFS。
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
