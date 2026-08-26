# SOURCE: vllm/v1/core/sched/request_queue.py
# waiting/skipped_waiting 的容器。FCFS 是 deque 子类：add=append、pop=popleft、
# peek 看队头、prepend=appendleft——被抢者『回队头』（_preempt_request）与跳过者
# 『插队头重排』（step_skipped_waiting → prepend_requests）的底层语义都在这四个
# 操作上。remove_requests 走过滤重建（deque 不支持原地过滤——finish_requests/
# stopped_preempted_reqs 从 waiting/skipped 摘除靠它）。
# SUBTRACTED: PriorityRequestQueue 堆实现（L131-L198）与 __lt__ 堆序——PRIORITY
#   调度策略是同环的另一『最不应保留』定义（dossier.delete 第 1 条批准，
#   抢占环的 PRIORITY 分支同批删）；默认 policy='fcfs'。
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from .request import Request


# SOURCE: vllm/v1/core/sched/request_queue.py:L13 SchedulingPolicy
class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L16-L17
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

    # SOURCE: vllm/v1/core/sched/request_queue.py:L78-L80 add_request
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L82-L84 pop_request
    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L86-L90 peek_request
    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    # SOURCE: vllm/v1/core/sched/request_queue.py:L92-L94 prepend_request
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L96-L103 prepend_requests
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L105-L107 remove_request
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L109-L116 remove_requests
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L118-L120 __bool__
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    # SOURCE: vllm/v1/core/sched/request_queue.py:L122-L124 __len__
    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L126-L128 __iter__
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


# SOURCE: vllm/v1/core/sched/request_queue.py:L201 create_request_queue
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    # SOURCE: vllm/v1/core/sched/request_queue.py:L203-L208
    # （PRIORITY 分支随 PriorityRequestQueue 删除——dossier.delete 第 1 条）
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
