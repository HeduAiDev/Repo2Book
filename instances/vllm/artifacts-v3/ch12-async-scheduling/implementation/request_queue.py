# SOURCE: vllm/v1/core/sched/request_queue.py
# FCFS 等待队列——本章 schedule() 准入路径与 _preempt_request 回队头的消费面。
# PRIORITY 堆序（ch10/ch11 立过）不在本章 spine，FCFS 单策略镜像。
from __future__ import annotations

from .request import Request


# SOURCE: vllm/v1/core/sched/request_queue.py RequestQueue（FCFS 段）
class RequestQueue:
    # SOURCE: vllm/v1/core/sched/request_queue.py RequestQueue.__init__
    def __init__(self) -> None:
        # SOURCE: vllm/v1/core/sched/request_queue.py RequestQueue.__init__
        self.queue: list[Request] = []

    # SUBTRACTED: PRIORITY 模式（heapq 双堆）——policy 恒 fcfs。

    # SOURCE: vllm/v1/core/sched/request_queue.py add_request
    def add_request(self, request: Request) -> None:
        # SOURCE: vllm/v1/core/sched/request_queue.py add_request
        self.queue.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py prepend_request
    def prepend_request(self, request: Request) -> None:
        # SOURCE: vllm/v1/core/sched/request_queue.py prepend_request
        self.queue.insert(0, request)

    # SOURCE: vllm/v1/core/sched/request_queue.py prepend_requests
    def prepend_requests(self, requests) -> None:
        # SOURCE: vllm/v1/core/sched/request_queue.py prepend_requests
        self.queue[:0] = list(requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py pop_request
    def pop_request(self) -> Request:
        # SOURCE: vllm/v1/core/sched/request_queue.py pop_request
        return self.queue.pop(0)

    # SOURCE: vllm/v1/core/sched/request_queue.py peek_request
    def peek_request(self) -> Request:
        # SOURCE: vllm/v1/core/sched/request_queue.py peek_request
        return self.queue[0]

    # SOURCE: vllm/v1/core/sched/request_queue.py remove_requests
    def remove_requests(self, requests) -> None:
        # SOURCE: vllm/v1/core/sched/request_queue.py remove_requests
        requests = set(requests)
        self.queue = [req for req in self.queue if req not in requests]

    # SOURCE: vllm/v1/core/sched/request_queue.py __len__
    def __len__(self) -> int:
        # SOURCE: vllm/v1/core/sched/request_queue.py __len__
        return len(self.queue)

    # SOURCE: vllm/v1/core/sched/request_queue.py __bool__
    def __bool__(self) -> bool:
        # SOURCE: vllm/v1/core/sched/request_queue.py __bool__
        return len(self.queue) > 0

    # SOURCE: vllm/v1/core/sched/request_queue.py __iter__（prepend_requests
    # 收 RequestQueue 时的消费面）
    def __iter__(self):
        # SOURCE: vllm/v1/core/sched/request_queue.py __iter__
        return iter(self.queue)


# SOURCE: vllm/v1/core/sched/request_queue.py create_request_queue（FCFS 面）
def create_request_queue(policy: str = "fcfs") -> RequestQueue:
    # SUBTRACTED: PRIORITY 分支（policy 恒 fcfs——ch10/ch11 拥有堆序）。
    # SOURCE: vllm/v1/core/sched/request_queue.py create_request_queue
    return RequestQueue()
