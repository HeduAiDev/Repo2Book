# Simplified Request — the unit of scheduling
# REFERENCE: vllm/v1/request.py → Request dataclass
# SIMPLIFIED: Only fields relevant to scheduling. No multimodal, LoRA, spec decode.

import time
from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(Enum):
    """Lifecycle of a request through the scheduler.

    REFERENCE: vllm/v1/request.py → RequestStatus (values match vLLM)
    """

    WAITING = "waiting"          # In the waiting queue
    RUNNING = "running"          # Actively being processed
    PREEMPTED = "preempted"      # Temporarily evicted, will resume
    FINISHED_STOPPED = "finished_stopped"    # Completed normally
    FINISHED_ABORTED = "finished_aborted"    # User aborted
    FINISHED_LENGTH_CAPPED = "finished_length_capped"  # Hit max_tokens


@dataclass(eq=False)
class Request:
    """A single inference request.

    REFERENCE: vllm/v1/request.py → Request

    Key scheduling fields:
    - priority: Lower value = higher priority (used by PriorityRequestQueue)
    - arrival_time: Tiebreaker for same-priority requests (FIFO within priority band)
    - num_computed_tokens: How many tokens have been processed so far
    - num_tokens: Total tokens in the request (prompt + generated so far)
    """

    request_id: str
    prompt_token_ids: list[int]     # The input prompt
    max_tokens: int                  # Max new tokens to generate
    priority: int = 0                # Lower = higher priority
    arrival_time: float = field(default_factory=time.monotonic)

    # Dynamic state
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0     # Tokens already processed
    output_token_ids: list[int] = field(default_factory=list)  # Generated tokens
    num_preemptions: int = 0         # How many times this request was preempted

    @property
    def num_tokens(self) -> int:
        """Total tokens: prompt + generated output.

        REFERENCE: scheduler.py:L356 — num_tokens_with_spec =
            len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
        """
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_new_tokens_needed(self) -> int:
        """How many tokens still need to be computed for this request.

        REFERENCE: scheduler.py:L408-409 — num_new_tokens =
            request.num_tokens_with_spec - request.num_computed_tokens
        """
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        """Whether this request has reached a terminal state."""
        return self.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        )

    def __lt__(self, other: "Request") -> bool:
        """Priority ordering for PriorityRequestQueue.

        REFERENCE: request_queue.py:L131-138 — PriorityRequestQueue:
            requests with smaller priority value processed first;
            if same priority, earlier arrival_time first.
        """
        return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)
