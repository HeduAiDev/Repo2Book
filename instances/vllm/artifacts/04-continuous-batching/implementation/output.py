# REFERENCE: instances/vllm/source/vllm/v1/core/sched/output.py:L181-L233
"""SchedulerOutput — what the scheduler decides each step.

vLLM's SchedulerOutput is a heavyweight dataclass carrying NewRequestData and
CachedRequestData lists for worker-side caching, encoder inputs, spec decode
tokens, KV connector metadata, and so on. We only keep the fields a single-GPU
test or trace needs. Each removed field is annotated.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/output.py:L181-L200
@dataclass
class SchedulerOutput:
    """Result of one Scheduler.schedule() call.

    Field name `num_scheduled_tokens` matches vLLM exactly so that the
    caller pattern `for req_id, n in output.num_scheduled_tokens.items()`
    transfers verbatim from vLLM (see scheduler.py:L984-L987).
    """

    # req_id -> num_scheduled_tokens this step.
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/output.py:L191-L193
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)

    # Sum of num_scheduled_tokens.values(). Cached for assertion at L849.
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/output.py:L194-L196
    total_num_scheduled_tokens: int = 0

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/output.py:L209-L212
    finished_req_ids: set[str] = field(default_factory=set)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/output.py:L217-L219
    preempted_req_ids: set[str] = field(default_factory=set)

    # Requests promoted from WAITING to RUNNING this step. vLLM splits this
    # into `scheduled_new_reqs` (first time scheduled) and
    # `scheduled_resumed_reqs` (resumed from preemption); we collapse the
    # two for the demo.
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L364-L365
    newly_running_req_ids: list[str] = field(default_factory=list)

    # NOT IMPLEMENTED: scheduled_new_reqs, scheduled_cached_reqs,
    # scheduled_spec_decode_tokens, scheduled_encoder_inputs,
    # num_common_prefix_blocks, free_encoder_mm_hashes,
    # has_structured_output_requests, kv_connector_metadata. Each maps to a
    # vLLM feature outside the scope of "core continuous batching loop".
