# SOURCE: vllm/v1/core/sched/request_queue.py
# FCFSRequestQueue 最小镜像（ch15 同款切面）：deque + add/pop/prepend/
# prepend_requests/remove_requests——本章 waiting/skipped 双队列的载体。
# SUBTRACTED: RequestQueue 抽象基类与 PriorityRequestQueue（ch10/11 已建
#   全量优先级面）。
from collections import deque
from collections.abc import Iterable, Iterator

from .request import Request


# SOURCE: vllm/v1/core/sched/request_queue.py:L75 FCFSRequestQueue
class FCFSRequestQueue(deque[Request]):
    """A first-come-first-served queue that supports deque operations."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L78 add_request
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L79-L80
        self.append(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L82 pop_request
    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L83-L84
        return self.popleft()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L86 peek_request
    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L87-L90
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    # SOURCE: vllm/v1/core/sched/request_queue.py:L92 prepend_request
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L93-L94
        self.appendleft(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L96 prepend_requests
    def prepend_requests(self, requests: "FCFSRequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note that the requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        # SOURCE: vllm/v1/core/sched/request_queue.py:L102-L103
        self.extendleft(requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L105 remove_request
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L106-L107
        self.remove(request)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L109 remove_requests
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L110-L116
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    # SOURCE: vllm/v1/core/sched/request_queue.py:L118 __bool__
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L119-L120
        return len(self) > 0

    # SOURCE: vllm/v1/core/sched/request_queue.py:L122 __len__
    def __len__(self) -> int:
        """Get number of requests in queue."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L123-L124
        return super().__len__()

    # SOURCE: vllm/v1/core/sched/request_queue.py:L126 __iter__
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L127-L128
        return super().__iter__()


# create_request_queue（FCFS 支的装配面——policy 分派删）
# SOURCE: vllm/v1/core/sched/scheduler.py create_request_queue
def create_request_queue(policy=None) -> FCFSRequestQueue:
    # SUBTRACTED: SchedulingPolicy 分派（ch10/11）——FCFS 直供
    return FCFSRequestQueue()
