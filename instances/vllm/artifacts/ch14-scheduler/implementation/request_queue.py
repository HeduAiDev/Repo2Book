# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版。验收：删掉 # SUBTRACTED 分支 ≈ 真实 vLLM。
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from request import Request


# SOURCE: vllm/v1/core/sched/request_queue.py:L13 (class SchedulingPolicy)
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


# SOURCE: vllm/v1/core/sched/request_queue.py:L75 (class FCFSRequestQueue)
# SUBTRACTED: 抽象基类 RequestQueue 与 PriorityRequestQueue —— 按
# subtraction_plan.delete「PRIORITY 策略整块」批准删除；精简版只演示 FCFS LIFO
# 抢占主线。FCFS 的 prepend_request=appendleft 是抢占回队头/skipped 重排的底层语义。
# 原 vllm/v1/core/sched/request_queue.py:L20 (RequestQueue), L131 (PriorityRequestQueue)。
class FCFSRequestQueue(deque):
    """A first-come-first-served queue that supports deque operations."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L78
    def add_request(self, request: Request) -> None:
        self.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L82
    def pop_request(self) -> Request:
        return self.popleft()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L86
    def peek_request(self) -> Request:
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

    # SOURCE: vllm/v1/core/sched/request_queue.py:L105
    def remove_request(self, request: Request) -> None:
        self.remove(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L109
    def remove_requests(self, requests: Iterable[Request]) -> None:
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


# SOURCE: vllm/v1/core/sched/request_queue.py:L201 (create_request_queue)
def create_request_queue(policy: SchedulingPolicy) -> FCFSRequestQueue:
    # SUBTRACTED: PRIORITY 分支返回 PriorityRequestQueue —— 见上，PRIORITY 整块已删。
    # 原 vllm/v1/core/sched/request_queue.py:L201-L208。
    return FCFSRequestQueue()
