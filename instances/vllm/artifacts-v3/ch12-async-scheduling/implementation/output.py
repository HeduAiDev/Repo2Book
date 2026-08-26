# SOURCE: vllm/v1/core/sched/output.py
# SchedulerOutput —— 本章两个 async 标志对（has_structured_output_requests /
# pending_structured_output_tokens，output.py:L235-L241）与 num_spec_tokens_
# to_schedule（占位列表定长）都在这里；NewRequestData/CachedRequestData 保
# worker 增量下发的契约面（ch10/ch18 已立完整版，此处精简字段）。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


# SOURCE: vllm/v1/core/sched/output.py:L18 NewRequestData（字段精简）
@dataclass
class NewRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L22-L27
    req_id: str
    # SUBTRACTED: mm_features/prompt_embeds/audio/token_type_ids——mm 面。
    # SOURCE: vllm/v1/core/sched/output.py:L29-L31
    prompt_token_ids: tuple[int, ...]
    # SOURCE: vllm/v1/core/sched/output.py:L33-L36
    block_ids: tuple[int, ...]
    # SOURCE: vllm/v1/core/sched/output.py:L38-L39
    num_computed_tokens: int

    # SOURCE: vllm/v1/core/sched/output.py:L41-L59 from_request
    @staticmethod
    def from_request(
        request,
        block_ids: tuple[int, ...],
        engine_start_time: float | None = None,
    ) -> "NewRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L43-L51
        return NewRequestData(
            req_id=request.request_id,
            prompt_token_ids=tuple(request.prompt_token_ids or ()),
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )


# SOURCE: vllm/v1/core/sched/output.py:L62 CachedRequestData（字段精简）
@dataclass
class CachedRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L66-L73
    req_ids: list[str] = field(default_factory=list)
    # SOURCE: vllm/v1/core/sched/output.py:L75 resumed_req_ids
    resumed_req_ids: set[str] = field(default_factory=set)
    # SUBTRACTED: new_token_ids 的 PP 回传用途（use_pp 分支——dossier.delete
    #   第 5 条批准）；字段保留原结构（非 PP 恒空列表——worker 自己缓存采样 token）。
    # SOURCE: vllm/v1/core/sched/output.py:L77-L78
    new_token_ids: list[list[int]] = field(default_factory=list)
    # SOURCE: vllm/v1/core/sched/output.py:L80-L82
    all_token_ids: dict[str, list[int]] = field(default_factory=dict)
    # SOURCE: vllm/v1/core/sched/output.py:L84-L86
    new_block_ids: list[tuple[list[int], ...] | None] = field(default_factory=list)
    # SOURCE: vllm/v1/core/sched/output.py:L88-L89
    num_computed_tokens: list[int] = field(default_factory=list)
    # SOURCE: vllm/v1/core/sched/output.py:L91-L93 num_output_tokens（async 下
    # 含占位——_make_cached_request_data 灌值处见 scheduler.py:L1451-L1457）
    num_output_tokens: list[int] = field(default_factory=list)

    # SOURCE: vllm/v1/core/sched/output.py:L95-L105 make_empty
    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L97-L105
        return cls()

    # SUBTRACTED: from_requests（V2 批量重建——第 4 条批准）。


# SOURCE: vllm/v1/core/sched/output.py:L193 SchedulerOutput
@dataclass
class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L194-L197 新请求全量打包
    scheduled_new_reqs: list[NewRequestData] = field(default_factory=list)
    # SOURCE: vllm/v1/core/sched/output.py:L198-L201 存量请求增量打包
    scheduled_cached_reqs: CachedRequestData = field(
        default_factory=CachedRequestData.make_empty
    )

    # SOURCE: vllm/v1/core/sched/output.py:L203-L208
    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int = 0
    # SOURCE: vllm/v1/core/sched/output.py:L209-L212 spec 草稿登记
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    # SUBTRACTED: scheduled_encoder_inputs（L213-L216——encoder，邻章）。
    # SOURCE: vllm/v1/core/sched/output.py:L217-L219
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be potentially used for cascade attention.
    num_common_prefix_blocks: list[int] = field(default_factory=list)

    # SOURCE: vllm/v1/core/sched/output.py:L221-L224
    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str] = field(default_factory=set)
    # SUBTRACTED: free_encoder_mm_hashes（L225-L227——encoder）。

    # SUBTRACTED: scheduled_encoder_input_stats（L229）/preempted_req_ids
    #   （L231-L233——V2 runner 专用，第 4 条批准）。

    # SOURCE: vllm/v1/core/sched/output.py:L235-L237 async 标志一
    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # SOURCE: vllm/v1/core/sched/output.py:L239-L241 async 标志二（deferred
    # sampling 的开关）
    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # SUBTRACTED: num_invalid_spec_tokens（L243-L244——spec 统计，第 6 条）、
    #   kv_connector_metadata/ec_connector_metadata/new_block_ids_to_zero/
    #   kv_cache_block_copies/partial_tail_offloads（L246-L265——connector/
    #   mamba/CoW 面）。

    # SOURCE: vllm/v1/core/sched/output.py:L267-L269 动态 spec 深度
    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0

    # SOURCE: vllm/v1/core/sched/output.py:L271-L283 make_empty
    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        # SOURCE: vllm/v1/core/sched/output.py:L273-L283
        return cls()


# SOURCE: vllm/v1/core/sched/output.py:L286-L292 GrammarOutput
@dataclass
class GrammarOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L288-L289
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # SOURCE: vllm/v1/core/sched/output.py:L290-L291
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"


# SUBTRACTED: EngineCoreOutput 伴生的 scheduler 侧观测结构（SchedulerStats/
#   PrefillStats/DecodeStats——可观测性）与 KVConnectorMetadata 族（connector）。
