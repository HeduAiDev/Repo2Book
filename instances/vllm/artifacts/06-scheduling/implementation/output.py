# Simplified SchedulerOutput — the return value of scheduler.schedule()
# REFERENCE: vllm/v1/core/sched/output.py
# SIMPLIFIED: Removed GrammarOutput, CachedRequestData, encoder-related fields.
#   Focus on the core scheduling decision: {req_id: num_tokens}.

from dataclasses import dataclass, field


@dataclass
class SchedulerOutput:
    """Output of a single scheduling step.

    REFERENCE: vllm/v1/core/sched/output.py → SchedulerOutput

    Each scheduling step produces one SchedulerOutput. It tells the model runner:
    - Which requests to process (scheduled_new_reqs + scheduled_running_reqs)
    - How many tokens to compute for each (num_scheduled_tokens)
    - Which requests were preempted (preempted_reqs)
    - Which requests finished in the previous step (finished_req_ids)
    """

    # New requests being scheduled for the first time
    # REFERENCE: scheduler.py:L364 — scheduled_new_reqs
    scheduled_new_reqs: list = field(default_factory=list)

    # Running requests that continue to be scheduled
    # REFERENCE: scheduler.py:L366 — scheduled_running_reqs
    scheduled_running_reqs: list = field(default_factory=list)

    # Requests that were preempted this step (need to be rescheduled later)
    # REFERENCE: scheduler.py:L367 — preempted_reqs
    preempted_reqs: list = field(default_factory=list)

    # {req_id: num_tokens_to_compute} — the core scheduling decision
    # REFERENCE: scheduler.py:L370 — num_scheduled_tokens
    num_scheduled_tokens: dict = field(default_factory=dict)

    # Request IDs that finished in the previous step (model runner should clean up)
    # REFERENCE: scheduler.py:L176 — finished_req_ids, flushed each step
    finished_req_ids: set = field(default_factory=set)

    # Total tokens scheduled this step (for logging)
    total_num_scheduled_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether this scheduling step produced no work."""
        return len(self.num_scheduled_tokens) == 0

    def __repr__(self) -> str:
        return (
            f"SchedulerOutput(new={len(self.scheduled_new_reqs)}, "
            f"running={len(self.scheduled_running_reqs)}, "
            f"preempted={len(self.preempted_reqs)}, "
            f"tokens={self.num_scheduled_tokens}, "
            f"finished={self.finished_req_ids})"
        )
