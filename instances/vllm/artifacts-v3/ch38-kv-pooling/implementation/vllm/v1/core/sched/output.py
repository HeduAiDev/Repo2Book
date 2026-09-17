# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""调度输出载体：_update_req_states 吃它拿增量块号。

# SOURCE: vllm/v1/core/sched/output.py:L34-L283
# SUBTRACTED: 结构化输出/投机解码草稿/多模态/EC 传输载荷字段——本章只保留
#   yield_req_data 消费的面（scheduled_new_reqs / scheduled_cached_reqs /
#   num_scheduled_tokens / finished_req_ids / preempted_req_ids）。
"""

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/v1/core/sched/output.py:L34-L113
@dataclass
class NewRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L34-L113
    """Request data structure used by SchedulerOutput."""

    req_id: str
    """The request ID of the request."""
    block_ids: tuple[list[int], ...] | None = None
    """The block IDs of the request. A tuple of block IDs per KV cache group."""
    num_computed_tokens: int = 0
    """Number of tokens computed before this request is scheduled."""
    resumed_from_preemption: bool = False
    """Whether it's a request resumed from preemption."""
    prefill_token_ids: list[int] | None = None
    """Tokens this prefill will compute KV for (v2 resumed requests)."""
    prompt_token_ids: list[int] | None = None
    """The original prompt tokens."""


# SOURCE: vllm/v1/core/sched/output.py:L115-L182
@dataclass
class CachedRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L115-L182
    """Request data structure for the requests in the cached request queue."""

    req_ids: list[str] = field(default_factory=list)
    """The request IDs of the requests."""
    new_block_ids: list[tuple[list[int], ...]] = field(default_factory=list)
    """The block IDs of the requests. A list of block IDs per request
    (per-group tuple)."""
    num_computed_tokens: list[int] = field(default_factory=list)
    """The number of tokens computed for each request."""
    resumed_req_ids: set[str] = field(default_factory=set)
    """The request IDs of the requests resumed from preemption."""


# SOURCE: vllm/v1/core/sched/output.py:L193-L283
@dataclass
class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L193-L283
    """The scheduler output for a step."""

    scheduled_new_reqs: list[NewRequestData] = field(default_factory=list)
    """New requests being scheduled for the first time."""
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    """Number of tokens scheduled for each request."""
    total_num_scheduled_tokens: int = 0
    """Total number of tokens scheduled."""
    scheduled_cached_reqs: CachedRequestData = field(default_factory=CachedRequestData)
    """Requests scheduled from the cached queue."""
    finished_req_ids: set[str] = field(default_factory=set)
    """Request ids that are finished."""
    preempted_req_ids: set[str] | None = None
    """Request ids that are preempted."""
    kv_connector_metadata: Any | None = None
    """The metadata from the KV connector."""
    scheduled_new_reqs_data: Any | None = None
    """Structure containing extra data for the new requests."""


__all__ = ["NewRequestData", "CachedRequestData", "SchedulerOutput"]
