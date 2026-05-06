# REFERENCE: instances/vllm/source/vllm/v1/request.py
"""Request and RequestStatus — the unit of scheduling.

Faithful to vLLM in shape (same field names, same enum order, same accessor
properties), but trimmed to the fields the scheduler actually reads in the
core continuous-batching loop. Multimodal/encoder/spec-decode/structured-output
fields are left out — every removal is annotated below.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# REFERENCE: instances/vllm/source/vllm/v1/request.py:L310-L333
class RequestStatus(enum.IntEnum):
    """Lifecycle states. Anything > PREEMPTED is finished.

    NOT IMPLEMENTED: the four extra WAITING_FOR_* and FINISHED_* variants in
    vLLM (structured output gating, async KV transfer, streaming inputs,
    abort/error/repetition reasons). The core scheduler only needs the five
    states below.
    """

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Anything past this point is finished.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # REFERENCE: instances/vllm/source/vllm/v1/request.py:L331-L333
        return status > RequestStatus.PREEMPTED


# REFERENCE: instances/vllm/source/vllm/v1/request.py:L59-L308
@dataclass
class Request:
    """A request in flight. Same key fields as vLLM's Request.

    Field names match vLLM exactly. The scheduler walks through requests
    advancing `num_computed_tokens` toward `num_tokens` one schedule step at
    a time.
    """

    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_time: float

    status: RequestStatus = RequestStatus.WAITING
    # Tokens already processed by the model. Advanced by
    # _update_after_schedule once a step's tokens are committed.
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)

    # KV cache block IDs allocated for this request. In vLLM this is owned
    # by KVCacheManager via `coordinator.get_blocks(request_id)`; we attach
    # it directly to Request for pedagogical clarity.
    block_ids: list[int] = field(default_factory=list)

    # Preemption counter — vLLM uses this for prefill metrics.
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L967
    num_preemptions: int = 0

    # ── Properties that match vLLM's Request API ──────────────────────────

    @property
    def num_prompt_tokens(self) -> int:
        # REFERENCE: instances/vllm/source/vllm/v1/request.py — same name in vLLM.
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total tokens this request will eventually have processed.

        In vLLM this is `num_prompt_tokens + num_output_tokens`. We keep the
        same definition. `num_tokens_with_spec` (vLLM) additionally counts
        speculative draft tokens — out of scope here.
        """
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def num_new_tokens(self) -> int:
        """Tokens still owed to the model = num_tokens - num_computed_tokens.

        REFERENCE: scheduler.py:L408-L412 computes this same quantity inline
        as `request.num_tokens_with_spec + request.num_output_placeholders -
        request.num_computed_tokens`.
        """
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        """True if we have not yet caught up with the prompt.

        Mirrors `is_prefill_chunk` in vLLM (set in _update_after_schedule).
        REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L988-L990
        """
        return self.num_computed_tokens < self.num_prompt_tokens

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)
