# SOURCE: vllm/v1/core/sched/output.py
# 只做减法的忠实精简版。真实 SchedulerOutput 另有 KV/EC connector 元数据、编码器输入
# 调度、block 分配等几十个字段（调度章 ch13 范围），本章只保留与掩码装配/投机耦合
# 直接相关的字段。NewRequestData/CachedRequestData 等请求快照类型与本章无关，一并
# 省略。
#
# SUBTRACTED: SPDX 版权头、NewRequestData/CachedRequestData 两个 dataclass 及
# SchedulerOutput 的 scheduled_new_reqs/scheduled_cached_reqs/scheduled_encoder_inputs/
# num_common_prefix_blocks/finished_req_ids/free_encoder_mm_hashes/preempted_req_ids/
# kv_connector_metadata/ec_connector_metadata/new_block_ids_to_zero 字段——它们服务于
# KV cache/编码器/连接器，与本章掩码装配控制流无关。
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass
class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L181-241（精简到本章相关字段）
    # req_id -> num_scheduled_tokens
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    # req_id -> spec_token_ids（草稿位置；本章的 -1 补齐发生在这个字典的值里）
    scheduled_spec_decode_tokens: "dict[str, list[int]]" = field(default_factory=dict)

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: "dict[str, int] | None" = None


@dataclass
class GrammarOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L259-263
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"


@dataclass
class DraftTokenIds:
    # SOURCE: vllm/v1/outputs.py:L222-226
    # [num_reqs]
    req_ids: "list[str]"
    # num_reqs x num_draft_tokens
    draft_token_ids: "list[list[int]]"
