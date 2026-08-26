# SOURCE: vllm/v1/core/sched/output.py
# schedule() 一拍的产物数据结构：NewRequestData（首次全量——worker 缓存它）/
# CachedRequestData（增量——只发 diff）/ SchedulerOutput（整拍打包）。
# 与真实 vllm/v1/core/sched/output.py 同名同字段语义，删除项全部
# dossier.subtraction_plan.delete 批准（encoder/mm/connector/mamba/spec 统计）。
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


# SOURCE: vllm/v1/core/sched/output.py:L34 NewRequestData
@dataclass
class NewRequestData:
    # SUBTRACTED: mm_features / pooling_params / lora_request / prompt_embeds /
    #   prompt_is_token_ids / prefill_token_ids 字段（L38-L48——mm/lora/pooling/
    #   v2 runner，dossier.delete 批准）。保留『首次全量』语义的核心四元组。
    req_id: str
    prompt_token_ids: list[int] | None
    sampling_params: object | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    # SOURCE: vllm/v1/core/sched/output.py:L50-L69 from_request
    @classmethod
    def from_request(
        cls,
        request,
        block_ids: tuple[list[int], ...],
    ) -> "NewRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L57-L69
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )
        # SUBTRACTED: mm_features/pooling_params/lora_request/prompt_embeds/
        #   prompt_is_token_ids/prefill_token_ids 的透传（mm/lora/pooling/v2）。

    # SUBTRACTED: __repr__ / anon_repr（L71-L112）——日志脱敏可观测性输出。


# SOURCE: vllm/v1/core/sched/output.py:L115 CachedRequestData
@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as
    # the request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # MRV1-only: For requests not scheduled in the last step, propagate the token ids
    # to the connector. Won't contain requests scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # SUBTRACTED: anon_repr / __repr__（L132-L151）——日志脱敏可观测性输出。

    # SOURCE: vllm/v1/core/sched/output.py:L153-L155 num_reqs
    @property
    def num_reqs(self) -> int:
        # SOURCE: vllm/v1/core/sched/output.py:L154-L155
        return len(self.req_ids)

    # SOURCE: vllm/v1/core/sched/output.py:L157-L165 _req_id_to_num_output_tokens
    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        # SOURCE: vllm/v1/core/sched/output.py:L165
        return dict(zip(self.req_ids, self.num_output_tokens))

    # SOURCE: vllm/v1/core/sched/output.py:L167-L169 is_context_phase
    def is_context_phase(self, req_id: str) -> bool:
        # SOURCE: vllm/v1/core/sched/output.py:L168-L169
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    # SOURCE: vllm/v1/core/sched/output.py:L171-L181 make_empty
    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L173-L181
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


# SOURCE: vllm/v1/core/sched/output.py:L192 SchedulerOutput
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # SUBTRACTED: free_encoder_mm_hashes（L225-L227，encoder）与
    #   scheduled_encoder_input_stats（L229，encoder 统计）——dossier.delete 第 2/11 条；
    #   pending_structured_output_tokens（L240-L241，structured）——第 3 条；
    #   kv_connector_metadata / ec_connector_metadata / ec_manager_metadata
    #   （L246-L252，connector/encoder）——第 1/2 条；
    #   new_block_ids_to_zero（L253-L256，mamba 清零）——第 4 条；
    #   kv_cache_block_copies / partial_tail_offloads（L258-L265，connector CoW）——第 1 条；
    #   num_spec_tokens_to_schedule（L267-L269，动态 spec 调度）——第 9 条。
    #   均为默认 None/空的旁路字段，删后 new/cached 二分骨架不变。

    # SOURCE: vllm/v1/core/sched/output.py:L271-L283 make_empty
    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        # SOURCE: vllm/v1/core/sched/output.py:L273-L283
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
        )
        # SUBTRACTED: free_encoder_mm_hashes=[]（encoder，dossier.delete 批准）。

# SUBTRACTED: ScheduledEncoderInputStats（L184-L189，encoder 统计）与
#   GrammarOutput（L286-L292，structured grammar bitmask）——dossier.delete 批准。
