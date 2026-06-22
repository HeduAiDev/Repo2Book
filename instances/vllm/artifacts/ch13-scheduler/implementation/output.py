# SOURCE: vllm/v1/core/sched/output.py
# schedule() 一拍的产物数据结构：NewRequestData(首次全量) / CachedRequestData(增量) /
# SchedulerOutput(整拍打包)。与真实 vllm/v1/core/sched/output.py 同名同字段，
# 只删与本章无关的可选字段（spec/structured/KV-connector/v2 runner/PP）。
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

# SUBTRACTED: TYPE_CHECKING 下的 mm/lora/pooling/torch/connector 类型别名
#   （原 output.py:L8-L27）—— 本精简版不引入这些子系统类型。


# SOURCE: vllm/v1/core/sched/output.py:L30
@dataclass
class NewRequestData:
    # SUBTRACTED: mm_features / sampling_params / pooling_params / lora_request /
    #   prompt_embeds / prompt_is_token_ids / prefill_token_ids（mm/lora/pooling/
    #   v2 runner 字段，dossier.delete 批准）。保留‘首次全量’语义所需的核心三元组：
    #   prompt_token_ids（全量 prompt）+ block_ids（KV 块）+ num_computed_tokens。
    req_id: str
    prompt_token_ids: list[int] | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    # SOURCE: vllm/v1/core/sched/output.py:L46
    @classmethod
    def from_request(
        cls,
        request,
        block_ids: tuple[list[int], ...],
    ) -> "NewRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L46
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )


# SOURCE: vllm/v1/core/sched/output.py:L111
@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used
    # as the request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # SOURCE: vllm/v1/core/sched/output.py:L150
    @property
    def num_reqs(self) -> int:
        # SOURCE: vllm/v1/core/sched/output.py:L150
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        # SOURCE: vllm/v1/core/sched/output.py:L153
        return dict(zip(self.req_ids, self.num_output_tokens))

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L167
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


# SOURCE: vllm/v1/core/sched/output.py:L180
@dataclass
class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L180
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    scheduled_spec_decode_tokens: dict[str, list[int]]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]

    # Request IDs that are preempted in this step.
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    has_structured_output_requests: bool = False
    # Whether the scheduled requests have all the output tokens they need.
    pending_structured_output_tokens: bool = False

    # SUBTRACTED: scheduled_encoder_inputs / num_common_prefix_blocks /
    #   free_encoder_mm_hashes / num_invalid_spec_tokens / kv_connector_metadata /
    #   ec_connector_metadata / new_block_ids_to_zero（encoder/spec/KV-connector/
    #   mamba 字段，dossier.delete 批准；缺省 None/空，不影响 new/cached 二分骨架）。
