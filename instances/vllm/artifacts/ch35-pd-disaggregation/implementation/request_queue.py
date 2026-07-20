# SPDX-License-Identifier: Apache-2.0
# Subtract-only companion for ch29《PD 分离的抽象与调度器集成》.
# 只做减法：与 vLLM 同名/同结构/同控制流，只删不增。
#
# 本文件是 vllm/v1/core/sched/request_queue.py 的子集：保留 SchedulingPolicy
# 二值枚举与 FCFS 队列的核心 deque 操作，供 _select_waiting_queue_for_scheduling
# 在 waiting / skipped_waiting 双队列间选取。
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from .request import Request


# SOURCE: vllm/v1/core/sched/request_queue.py:L13
class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


# SOURCE: vllm/v1/core/sched/request_queue.py:L20 (RequestQueue 抽象基类，精简为别名)
# SUBTRACTED: RequestQueue 抽象基类的完整 abstractmethod 清单（remove_request /
# remove_requests / __iter__ 等）与 PriorityRequestQueue（heapq 实现）本章不触碰；
# 精简版只保留 FCFS 一支，类型注解处用 FCFSRequestQueue 即可（原 L20-72, L131-199）。
RequestQueue = "FCFSRequestQueue"


# SOURCE: vllm/v1/core/sched/request_queue.py:L75
class FCFSRequestQueue(deque):
    """A first-come-first-served queue that supports deque operations."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L78
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L82
    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L86
    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    # SOURCE: vllm/v1/core/sched/request_queue.py:L92
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L96
    def prepend_requests(self, requests: "FCFSRequestQueue") -> None:
        """Prepend all requests from another queue to the front of this queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L109
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        self.clear()
        self.extend(filtered_requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L118
    def __bool__(self) -> bool:
        return len(self) > 0

    # SOURCE: vllm/v1/core/sched/request_queue.py:L126
    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()


# SOURCE: vllm/v1/core/sched/request_queue.py:L201
def create_request_queue(policy: SchedulingPolicy) -> FCFSRequestQueue:
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    # SUBTRACTED: PRIORITY → PriorityRequestQueue 分支（heapq 堆）本章不展开
    # （原 request_queue.py:L203-204）。
    raise ValueError(f"Unsupported scheduling policy: {policy}")
