# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py
"""Policy primitives: who gets preempted, which queue gets pulled from.

These are the policy bits SCATTERED across `scheduler.py`. Ch04 walked through
the schedule() loop mechanics; here we extract the three policy decisions
that the loop makes:

    1. Victim selection on KV-cache OOM
       (FCFS pops tail, PRIORITY uses max(priority, arrival_time))
    2. Waiting-queue selection between `waiting` and `skipped_waiting`
       (FCFS prefers skipped, PRIORITY peeks both heads)
    3. Pause-state gating (`PauseState.PAUSED_ALL` zeros the budget)

If a function here ever starts looking like Ch04's schedule() loop, it has
slipped into mechanics — refactor back to a one-line policy primitive.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .request_queue import (
    FCFSRequestQueue,
    PolicyRequest,
    PriorityRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/interface.py — PauseState
class PauseState(Enum):
    """Engine-level pause flags. Read by `schedule()` at L372-L374:
       `if self._pause_state == PauseState.PAUSED_ALL: token_budget = 0`.

    The PAUSED_NEW state lets running requests progress but blocks new
    admission (used during in-flight model swaps in RLHF). PAUSED_ALL
    freezes the engine entirely (graceful drain).
    """

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/interface.py
    UNPAUSED = "unpaused"
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L568
    # (PAUSED_NEW lets Phase 1 RUNNING continue but skips Phase 2 admission)
    PAUSED_NEW = "paused_new"
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L372-L374
    # (PAUSED_ALL zeros the token_budget, halting both phases)
    PAUSED_ALL = "paused_all"


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L478-L504
def select_preemption_victim(
    running: list[PolicyRequest],
    requester: PolicyRequest,
    policy: SchedulingPolicy,
) -> PolicyRequest:
    """Pick which running request to evict so `requester` can be admitted.

    FCFS rule (`scheduler.py:L504`): `preempted_req = self.running.pop()` —
    the LAST element of running, i.e. the most-recently-admitted request.
    Why "last is lowest priority": running is FIFO order of admission;
    the requester being scheduled is earlier in the list, hence higher
    seniority.

    PRIORITY rule (`scheduler.py:L478-L484`):
        max(running, key=lambda r: (r.priority, r.arrival_time))
    Returns the request with the LARGEST priority value (numerically) and,
    among ties, the LATEST arrival time. That's the worst-priority running
    request — the right choice to preempt. Note `max` not `min` because
    smaller priority value = higher priority in vLLM's convention.

    Note: this function only PICKS the victim. It does not remove it from
    running, free its blocks, or re-queue it — those are mechanics covered
    in Ch04's `Scheduler._preempt_request`.
    """
    if not running:
        raise ValueError("Cannot preempt from empty running queue")

    if policy is SchedulingPolicy.FCFS:
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L504
        # `preempted_req = self.running.pop()` — pops the last (most-recent) entry.
        return running[-1]

    # PRIORITY
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L480-L483
    # REFERENCE: instances/vllm/source/vllm/v1/request.py:L301-L302
    # (priority field is the FIRST tier of __lt__ ordering)
    return max(running, key=lambda r: (r.priority, r.arrival_time))


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1567-L1577
def select_waiting_queue(
    waiting: RequestQueue,
    skipped_waiting: RequestQueue,
    policy: SchedulingPolicy,
) -> Optional[RequestQueue]:
    """Pick between `waiting` and `skipped_waiting`.

    `skipped_waiting` holds requests that were peeked-but-not-popped during a
    prior step (e.g. blocked waiting for remote KV transfers, or LoRA-cap-
    exceeded). They re-enter the schedule loop here.

    FCFS rule (`scheduler.py:L1568-L1569`):
        `return self.skipped_waiting or self.waiting or None`
    Skipped wins because they've already paid wait time — promoting them
    avoids indefinite postponement of blocked requests.

    PRIORITY rule (`scheduler.py:L1571-L1577`):
        Compare the heads of both queues by `__lt__`. Whichever head wins
        the priority-comparison is the queue to pull from.
    """
    if policy is SchedulingPolicy.FCFS:
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1568-L1569
        return skipped_waiting or waiting or None

    if waiting and skipped_waiting:
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1572-L1575
        # REFERENCE: instances/vllm/source/vllm/v1/request.py:L296-L307
        # (Request.__lt__ provides the comparison key for the head peek)
        w = waiting.peek_request()
        s = skipped_waiting.peek_request()
        return waiting if w < s else skipped_waiting

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1577
    return waiting or skipped_waiting or None


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L372-L374
def effective_token_budget(
    raw_budget: int, pause_state: PauseState
) -> int:
    """A one-line policy primitive: PAUSED_ALL zeros the budget.

    Trivial-looking but important: it's the ONE place pause-state interacts
    with scheduling. Without this, a PAUSED_ALL engine would still admit
    waiting requests (which might be in PAUSED_NEW transit).

    REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L568
    (Phase 2 also gated on `self._pause_state == PauseState.UNPAUSED`)
    """
    if pause_state is PauseState.PAUSED_ALL:
        return 0
    return raw_budget


__all__ = [
    "PauseState",
    "select_preemption_victim",
    "select_waiting_queue",
    "effective_token_budget",
    # Re-exported for tests and downstream chapters:
    "FCFSRequestQueue",
    "PriorityRequestQueue",
    "PolicyRequest",
    "RequestQueue",
    "SchedulingPolicy",
]
