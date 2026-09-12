# SOURCE: vllm/v1/core/sched/output.py
# v3 ch31 脊柱③（调度输出切面）：GrammarOutput（L286-L291，逐字——掩码行序
# 权威 + ndarray 掩码同传的载体）+ SchedulerOutput 的本章消费字段
# （num_scheduled_tokens/scheduled_spec_decode_tokens/
# has_structured_output_requests/pending_structured_output_tokens/
# num_invalid_spec_tokens/num_spec_tokens_to_schedule）。
# SUBTRACTED：NewRequestData/CachedRequestData 的载荷细节与 kv_connector/
# ec_connector/encoder/routed_experts/preempted 等字段（L1-L283——ch05/ch16/
# ch17 的消费面）；make_empty 的等价保留。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# SUBTRACTED: NewRequestData/CachedRequestData/ScheduledEncoderInputStats 等
#   dataclass 定义（L10-L190）——装配载荷归 ch17；本章以 Any 型槽位承载
#   两个列表字段的占位语义。


# SOURCE: vllm/v1/core/sched/output.py:L195-L283 SchedulerOutput —— 消费字段切面
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[Any] = field(default_factory=list)
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: Any = None
    # SUBTRACTED: 真实为 CachedRequestData——载荷细节归 ch17/ch18。

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int = 0
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    # SUBTRACTED: scheduled_encoder_inputs/num_common_prefix_blocks/
    #   finished_req_ids/free_encoder_mm_hashes/scheduled_encoder_input_stats/
    #   preempted_req_ids（L209-L235——encoder/级联注意力/V2 抢占通知面，
    #   ch05/ch17/ch21）。

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # SUBTRACTED: kv_connector_metadata/ec_connector_metadata/
    #   new_block_ids_to_zero/kv_cache_block_copies/partial_tail_offloads
    #   （L246-L265——ch16/ch13）。

    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        # SOURCE: vllm/v1/core/sched/output.py:L271-L283 make_empty —— 等价保留
        #   （字段集为本切面子集）
        return cls()


@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L286-L291 GrammarOutput —— 逐字
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
