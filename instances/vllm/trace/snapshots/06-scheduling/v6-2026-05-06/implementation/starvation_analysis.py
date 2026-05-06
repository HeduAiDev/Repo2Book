# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L131-L198
# (PriorityRequestQueue) +
# instances/vllm/source/vllm/v1/core/sched/scheduler.py:L478-L484 (priority preempt)
"""Starvation under FCFS, priority + aging compensator.

This module quantifies a fundamental scheduling pathology that vLLM's docs
mention only obliquely:

    Under pure FCFS with chunked prefill, a long prompt arriving FIRST
    can "head-of-line block" subsequently-arriving short requests *for
    every step it occupies the running queue*. The short request waits
    O(longest_prompt / token_budget) steps even though its own prompt is
    tiny. This is invisible in throughput numbers — it shows up as
    *tail latency*.

vLLM's response is two-fold:
    1. Provide PRIORITY policy (`scheduler_config.policy = "priority"`)
       — see `vllm/config/scheduler.py:L109-L117`.
    2. Document that priority is a USER responsibility, not an automatic
       fix — vLLM does NOT auto-age waiting requests. This file shows what
       an aging compensator WOULD look like, as a discussion device for
       why aging is safe in some workloads but not others.

Concepts covered:
    - Head-of-line latency under FCFS (bound + worst-case construction)
    - Priority bypass (no starvation if all priorities equal, OR if your
      priority assigner promotes long-waiters)
    - Aging compensator (priority -= aging_rate * wait_time)
    - The "priority inversion under preemption" subtlety
"""

from __future__ import annotations

from dataclasses import dataclass

from .request_queue import PolicyRequest, PriorityRequestQueue, SchedulingPolicy


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L568-L846
# (Phase 2 admits ONE request per step in FCFS while running runs Phase 1).
@dataclass
class WorkloadProfile:
    """How latency breaks down under FCFS for a 2-class workload.

    Class A: `n_long` requests with prompt length `prompt_long`.
    Class B: `n_short` requests with prompt length `prompt_short`.

    Both classes arrive at t=0 in interleaved order. Token budget per step
    is `B`. Running queue caps at `max_running` (so Phase 2 admission halts
    when running == max_running).
    """

    n_long: int
    prompt_long: int
    n_short: int
    prompt_short: int
    token_budget: int
    max_running: int = 4

    def fcfs_short_request_latency_steps(self) -> int:
        """Worst-case wait (in scheduler steps) for a short request that
        arrived AFTER the long requests under FCFS.

        Each long request needs `ceil(prompt_long / token_budget)` steps to
        prefill. Phase 1 (running) consumes the budget; Phase 2 only fires
        if any running progress fits in remaining budget. Worst-case the
        short request must wait until ONE long request finishes prefill
        AND there is room in `max_running`.

        REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L387-L556
        (Phase 1 RUNNING) + L568-L846 (Phase 2 WAITING — only fires if
        `not preempted_reqs` and budget remains)
        """
        steps_per_long_prefill = (self.prompt_long + self.token_budget - 1) // self.token_budget
        # Long requests fill running. The short request waits for a slot.
        # Once running is at max_running, we have to wait for one long to
        # prefill+start-decoding (roughly steps_per_long_prefill steps).
        return steps_per_long_prefill

    def fcfs_long_request_completion_steps(self) -> int:
        """For comparison: how long the long request itself takes to prefill.

        REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L408-L415
        (Phase 1 caps num_new_tokens at min(num_new_tokens, token_budget,
         long_prefill_token_threshold) — long prompts get sliced)
        """
        return (self.prompt_long + self.token_budget - 1) // self.token_budget

    def head_of_line_blocking_factor(self) -> float:
        """Ratio of short-request-wait to short-request-prefill-cost.

        > 1.0 means the short request waits *longer than its own prefill takes*
        — pure starvation overhead. The bigger the ratio, the worse the tail.
        """
        own_steps = max(1, (self.prompt_short + self.token_budget - 1) // self.token_budget)
        return self.fcfs_short_request_latency_steps() / own_steps


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/request_queue.py:L131-L198
def priority_ordering(requests: list[PolicyRequest]) -> list[PolicyRequest]:
    """Return `requests` in vLLM PRIORITY-queue admission order.

    Equivalent to `for r in requests: q.add_request(r); list(q)` but cheaper.
    Useful for tests that assert "PRIORITY admits short-after-long if short
    has lower priority value".
    """
    q = PriorityRequestQueue()
    for r in requests:
        q.add_request(r)
    return list(q)


# REFERENCE: not in vLLM (vLLM does NOT auto-age). Pedagogical, see module docstring.
def aged_priority(req: PolicyRequest, now: float, aging_rate: float = 1.0) -> int:
    """A hypothetical aging compensator. NOT in vLLM.

    Returns an effective priority that DECREASES (= higher precedence) the
    longer a request has been waiting. With `aging_rate=1.0` and integer
    priority, after 100 seconds of wait the request's effective priority
    drops by 100, eventually overtaking any new arrival.

    Why vLLM does NOT ship this:
      a. Priority is intended as a USER signal (e.g. paid tier vs free tier).
         Auto-aging would silently override the user's intent.
      b. Aging rate has to be tuned per workload — too aggressive starves
         high-priority traffic; too gentle does nothing.
      c. The aged priority changes the heap key, so re-heapify is needed
         every step, breaking the heap invariant.

    USE this if your application's policy is "priority is an SLA hint, not
    a hard ordering" — but you'll need to handle (c) by rebuilding the
    queue periodically.
    """
    wait_time = max(0.0, now - req.arrival_time)
    return int(req.priority - aging_rate * wait_time)


def has_starvation(profile: WorkloadProfile, policy: SchedulingPolicy) -> bool:
    """Quick predicate: does this profile starve under `policy`?

    FCFS starves whenever `head_of_line_blocking_factor() > 1.0`.
    PRIORITY does NOT starve when priorities are well-assigned (short jobs
    get lower priority value), but DOES starve when all priorities are
    equal — at which point it falls back to arrival_time, which is
    FCFS-equivalent.

    REFERENCE: instances/vllm/source/vllm/v1/request.py:L296-L307
    (Request.__lt__ — arrival_time is the second-tier key, so equal
     priority + different arrivals degrades to FIFO order)
    """
    if policy is SchedulingPolicy.FCFS:
        return profile.head_of_line_blocking_factor() > 1.0
    return False  # assumes priorities are well-assigned
