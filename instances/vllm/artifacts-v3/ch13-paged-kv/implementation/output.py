# SOURCE: vllm/v1/core/sched/output.py
# 过线协议数据（m7/m8）：NewRequestData（新请求全量块表随首帧过线）、
# CachedRequestData（在跑请求只带增量 new_block_ids）、SchedulerOutput
# （含 new_block_ids_to_zero 清零账——block_id 是调度器进程与 worker 进程
# 之间唯一共享键）。
# SUBTRACTED: mm/lora/pooling/prompt_embeds 字段面与 anon_repr（观测打印，
#   ch02/06）、prefill_token_ids（V2 runner，dossier.delete 第 12 条）、
#   ScheduledEncoderInputStats/encoder 字段（ch14 跨注意力族）、GrammarOutput
#   （ch30）、spec/connector/CoW/partial-tail 字段（第 5/7/9 条 → ch16/33）。
from dataclasses import dataclass

from .request import Request


# SOURCE: vllm/v1/core/sched/output.py:L34 NewRequestData
@dataclass
class NewRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L36-L42
    req_id: str
    prompt_token_ids: list[int] | None
    # SUBTRACTED: mm_features/sampling_params/pooling_params/lora_request/
    #   prompt_embeds/prompt_is_token_ids（L38-L45——ch02/06 消费面；字段
    #   保留 req_id/prompt_token_ids/block_ids/num_computed_tokens 四件）
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    # SUBTRACTED: prefill_token_ids（L47-L48——v2 model runner 专用，第 12 条）。

    # SOURCE: vllm/v1/core/sched/output.py:L50 from_request
    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> "NewRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L57-L69（mm/sampling/pooling/
        #   lora/prompt 实参删——对应字段已删；block_ids/num_computed_tokens
        #   过线契约逐字保留）
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )

    # SUBTRACTED: __repr__/anon_repr（L71-L112——打印观测）。


# SOURCE: vllm/v1/core/sched/output.py:L115 CachedRequestData
@dataclass
class CachedRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L117-L130
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
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

    # SUBTRACTED: anon_repr/__repr__（L132-L151——打印观测）。

    # SOURCE: vllm/v1/core/sched/output.py:L153 num_reqs
    @property
    def num_reqs(self) -> int:
        # SOURCE: vllm/v1/core/sched/output.py:L155
        return len(self.req_ids)

    # SUBTRACTED: _req_id_to_num_output_tokens/is_context_phase（L157-L169
    #   ——iteration 观测细账）。

    # SOURCE: vllm/v1/core/sched/output.py:L171 make_empty
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


# SUBTRACTED: ScheduledEncoderInputStats（L184-L189——encoder 统计，第 4 条）。


# SOURCE: vllm/v1/core/sched/output.py:L192 SchedulerOutput —— 跨进程载体
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    # SOURCE: vllm/v1/core/sched/output.py:L197 scheduled_new_reqs
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    # SOURCE: vllm/v1/core/sched/output.py:L201 scheduled_cached_reqs
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    # SOURCE: vllm/v1/core/sched/output.py:L203-L208
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # SUBTRACTED: scheduled_spec_decode_tokens/scheduled_encoder_inputs/
    #   free_encoder_mm_hashes/scheduled_encoder_input_stats/preempted_req_ids
    #   （L209-L233——spec/encoder/V2 runner，第 4/5/12 条）；kv_connector_
    #   metadata/ec_connector_metadata/ec_manager_metadata（L246-L252——第
    #   7 条 → ch16）；kv_cache_block_copies（L258-L259——第 9 条 CoW）；
    #   partial_tail_offloads（L261-L265——第 7 条）；num_spec_tokens_to_
    #   schedule（L267-L269——第 5 条）；has_structured_output_requests/
    #   pending_structured_output_tokens/num_invalid_spec_tokens（L235-L244
    #   ——ch12/30/33）。num_common_prefix_blocks（L217-L219——第 11 条）。

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    # SOURCE: vllm/v1/core/sched/output.py:L221-L224 finished_req_ids
    finished_req_ids: set[str]

    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    # SOURCE: vllm/v1/core/sched/output.py:L253-L256 new_block_ids_to_zero
    new_block_ids_to_zero: list[int] | None = None

    # SOURCE: vllm/v1/core/sched/output.py:L271 make_empty
    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        # SOURCE: vllm/v1/core/sched/output.py:L273-L283
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
        )
