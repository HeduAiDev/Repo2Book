"""
Scheduling Policies — FCFS vs Priority.

REFERENCE sources:
    SchedulingPolicy:    vllm/v1/core/sched/request_queue.py:L13
    FCFSRequestQueue:    vllm/v1/core/sched/request_queue.py:L75
    PriorityRequestQueue:vllm/v1/core/sched/request_queue.py:L131
    Request.__lt__:      vllm/v1/request.py:L296
    Preemption selection: vllm/v1/core/sched/scheduler.py:L478
    Queue selection:     vllm/v1/core/sched/scheduler.py:L1567
    PauseState:          vllm/v1/core/sched/interface.py:L22
"""

import heapq
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict


class SchedulingPolicy(Enum):
    """REFERENCE: vllm/v1/core/sched/request_queue.py:L13"""
    FCFS = "fcfs"
    PRIORITY = "priority"


class PauseState(Enum):
    """REFERENCE: vllm/v1/core/sched/interface.py:L22"""
    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


@dataclass
class ScheduledRequest:
    """
    Simplified Request for scheduling demonstration.

    REFERENCE: vllm/v1/request.py — full Request class
    """
    request_id: str
    priority: int = 0         # lower = higher priority
    arrival_time: float = 0.0
    prompt_tokens: int = 0
    computed_tokens: int = 0
    status: str = "waiting"   # waiting, running, preempted, finished

    def __lt__(self, other: "ScheduledRequest") -> bool:
        """
        Priority ordering — identical tie-breaking to vLLM.

        REFERENCE: vllm/v1/request.py:L296-L307
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.request_id < other.request_id


# ═══════════════════════════════════════════════════════════════════════════
# FCFS Queue — vllm/v1/core/sched/request_queue.py:L75
# ═══════════════════════════════════════════════════════════════════════════

class FCFSQueue:
    """
    First-Come-First-Served request queue. Wraps deque.

    REFERENCE: vllm/v1/core/sched/request_queue.py:L75-L128
    """

    def __init__(self):
        self._deque: deque[ScheduledRequest] = deque()

    def add(self, req: ScheduledRequest):
        """Append to tail. REFERENCE: L78"""
        self._deque.append(req)

    def pop(self) -> ScheduledRequest:
        """Pop from head. REFERENCE: L82"""
        return self._deque.popleft()

    def prepend(self, req: ScheduledRequest):
        """
        Push to FRONT. Used for preempted requests.

        REFERENCE: L92 — prepend_request() calls appendleft()
        When a request is preempted, it goes to the front of the queue
        so it gets retried before any new arrivals.
        """
        self._deque.appendleft(req)

    def peek(self) -> Optional[ScheduledRequest]:
        return self._deque[0] if self._deque else None

    def remove(self, req: ScheduledRequest):
        """O(n) linear scan. REFERENCE: L105"""
        self._deque.remove(req)

    def __len__(self):
        return len(self._deque)

    def __bool__(self):
        return bool(self._deque)


# ═══════════════════════════════════════════════════════════════════════════
# Priority Queue — vllm/v1/core/sched/request_queue.py:L131
# ═══════════════════════════════════════════════════════════════════════════

class PriorityQueue:
    """
    Priority-based request queue. Uses heapq for O(log n) insert/pop.

    REFERENCE: vllm/v1/core/sched/request_queue.py:L131-L198

    Key insight from vLLM source:
        Smaller priority value = HIGHER priority (processed first).
        When priorities equal, earlier arrival_time wins.
        When both equal, lexicographic request_id wins.
        If all equal, id() tie-breaks.

    This is why the docstring says "requests with smaller value of
    priority are processed first" — priority=0 is top priority.
    """

    def __init__(self):
        self._heap: List[ScheduledRequest] = []

    def add(self, req: ScheduledRequest):
        """O(log n) heap push. REFERENCE: L144"""
        heapq.heappush(self._heap, req)

    def pop(self) -> ScheduledRequest:
        """O(log n) heap pop — returns minimum by __lt__. REFERENCE: L148"""
        return heapq.heappop(self._heap)

    def prepend(self, req: ScheduledRequest):
        """
        Fall through to add() — no prepend concept in priority queue.

        REFERENCE: L160-L165
        The source explicitly says:
            "In a priority queue, there is no concept of prepending.
             Just add to the queue and it will be sorted by priority."
        """
        self.add(req)

    def peek(self) -> Optional[ScheduledRequest]:
        return self._heap[0] if self._heap else None

    def remove(self, req: ScheduledRequest):
        """O(n) remove + re-heapify. REFERENCE: L175"""
        self._heap.remove(req)
        heapq.heapify(self._heap)

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)


# ═══════════════════════════════════════════════════════════════════════════
# Preemption Comparison
# REFERENCE: vllm/v1/core/sched/scheduler.py:L478-L511
# ═══════════════════════════════════════════════════════════════════════════

class PreemptionPolicy:
    """
    How to select which request to preempt when KV Cache is full.

    REFERENCE: scheduler.py:L478-L511

    FCFS:    self.running.pop() — preempt MOST RECENTLY admitted
    PRIORITY: max(running, key=lambda r: (r.priority, r.arrival_time))
              — preempt LOWEST PRIORITY (highest value, latest arrival)
    """

    @staticmethod
    def select_victim_fcfs(running: List[ScheduledRequest]) -> ScheduledRequest:
        """Pop last element = most recently scheduled. REFERENCE: L504"""
        return running.pop()

    @staticmethod
    def select_victim_priority(running: List[ScheduledRequest]) -> ScheduledRequest:
        """
        max() selects the WORST request by (priority, arrival_time).

        REFERENCE: L480-L488
        Because __lt__ means "higher priority" when (smaller priority,
        earlier arrival), max() gives us the "worst" request to preempt.

        The max key is (r.priority, r.arrival_time):
        - Higher priority value = lower priority → preempt first
        - Later arrival time among equal priority → preempt first
        """
        victim = max(running, key=lambda r: (r.priority, r.arrival_time))
        running.remove(victim)
        return victim


# ═══════════════════════════════════════════════════════════════════════════
# Queue Selection — vllm/v1/core/sched/scheduler.py:L1567-L1577
# ═══════════════════════════════════════════════════════════════════════════

def select_queue(
    waiting: FCFSQueue | PriorityQueue,
    skipped_waiting: FCFSQueue | PriorityQueue,
    policy: SchedulingPolicy,
) -> FCFSQueue | PriorityQueue | None:
    """
    Choose which queue to pull from next.

    REFERENCE: vllm/v1/core/sched/scheduler.py:L1567-L1577

    FCFS: Always prefer skipped_waiting — give blocked requests priority
          over new arrivals to avoid indefinite postponement.

    PRIORITY: Compare heads of both queues using __lt__. Pull from whichever
              has the higher-priority (lower value, earlier arrival) head.
    """
    if policy == SchedulingPolicy.FCFS:
        return skipped_waiting or waiting or None

    # PRIORITY mode: compare head elements
    if waiting and skipped_waiting:
        w_req = waiting.peek()
        s_req = skipped_waiting.peek()
        if w_req and s_req:
            # w_req < s_req means w_req has HIGHER priority (lower value)
            return waiting if w_req < s_req else skipped_waiting

    return waiting or skipped_waiting or None


# ═══════════════════════════════════════════════════════════════════════════
# Demonstration
# ═══════════════════════════════════════════════════════════════════════════

def demonstrate_policies():
    """Show how FCFS vs Priority ordering differs."""
    requests = [
        ScheduledRequest("A", priority=3, arrival_time=0),
        ScheduledRequest("B", priority=1, arrival_time=1),  # highest priority
        ScheduledRequest("C", priority=2, arrival_time=2),
        ScheduledRequest("D", priority=1, arrival_time=3),  # same priority as B, later
    ]

    print("FCFS ordering:")
    fcfs = FCFSQueue()
    for r in requests:
        fcfs.add(r)
    while fcfs:
        r = fcfs.pop()
        print(f"  {r.request_id}: priority={r.priority}, arrival={r.arrival_time}")

    print("\nPriority ordering:")
    pq = PriorityQueue()
    for r in requests:
        pq.add(r)
    while pq:
        r = pq.pop()
        print(f"  {r.request_id}: priority={r.priority}, arrival={r.arrival_time}")

    print("\nPreemption victim selection:")
    running = [
        ScheduledRequest("X", priority=1, arrival_time=0),
        ScheduledRequest("Y", priority=3, arrival_time=1),  # worst priority
        ScheduledRequest("Z", priority=1, arrival_time=2),
    ]
    # FCFS preempts most recent
    fcfs_victim = PreemptionPolicy.select_victim_fcfs(list(running))
    print(f"  FCFS victim: {fcfs_victim.request_id} (most recent)")
    # Priority preempts worst priority
    prio_victim = PreemptionPolicy.select_victim_priority(list(running))
    print(f"  PRIORITY victim: {prio_victim.request_id} (lowest priority)")


if __name__ == "__main__":
    demonstrate_policies()
