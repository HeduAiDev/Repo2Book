# SOURCE: vllm/v1/core/sched/request_queue.py
# FCFS 请求队列 + create_request_queue 工厂。与真实文件同名同语义。
#
# SUBTRACTED: PriorityRequestQueue（PRIORITY 调度策略，dossier.delete 批准，保留 FCFS）
#   原 vllm/v1/core/sched/request_queue.py:L131-L199。
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum


# SOURCE: vllm/v1/core/sched/request_queue.py:L13
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


# SUBTRACTED: RequestQueue 抽象基类（仅定义接口契约，FCFSRequestQueue 直接
#   继承 deque 实现全部方法即可，删后行为不变）—— 原 request_queue.py:L20-L73。


# SOURCE: vllm/v1/core/sched/request_queue.py:L75
class FCFSRequestQueue(deque):
    """A first-come-first-served queue that supports deque operations."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L78
    def add_request(self, request) -> None:
        self.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L82
    def pop_request(self):
        return self.popleft()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L86
    def peek_request(self):
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    # SOURCE: vllm/v1/core/sched/request_queue.py:L92
    def prepend_request(self, request) -> None:
        self.appendleft(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L96
    def prepend_requests(self, requests) -> None:
        self.extendleft(requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L105
    def remove_request(self, request) -> None:
        self.remove(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L109
    def remove_requests(self, requests: Iterable) -> None:
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        # SOURCE: vllm/v1/core/sched/request_queue.py:L118
        return len(self) > 0

    def __iter__(self) -> Iterator:
        # SOURCE: vllm/v1/core/sched/request_queue.py:L126
        return super().__iter__()


# SOURCE: vllm/v1/core/sched/request_queue.py:L201
def create_request_queue(policy: SchedulingPolicy) -> FCFSRequestQueue:
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    # SUBTRACTED: PRIORITY 分支（dossier.delete 批准）
    raise ValueError(f"Unknown scheduling policy: {policy}")
