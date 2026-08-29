# SOURCE: vllm/v1/core/sched/output.py
# SchedulerOutput 的 **connector 面**：kv_connector_metadata（调度器→worker
# 的不透明计划信封——m6）、partial_tail_offloads（producer 部分尾交接
# ——m15）、new_block_ids_to_zero（清零账——站 6/10 的 _skip_zero 过滤后
# 产物）、finished_req_ids（worker get_finished 的输入）；NewRequestData/
# CachedRequestData 的最小面（ExampleConnector.build_connector_meta 消费）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 1 条 ECConnector 全部字段（ec_connector_metadata/ec_manager_metadata/
#     free_encoder_mm_hashes/scheduled_encoder_inputs 及其 stats）；
#   第 10 条 kv_cache_block_copies（CoW 打包段归 ch15）；
#   spec/structured-output/v2 面字段（ch33/ch05）；
#   anon_repr 脱敏 repr（观测辅助）、prefill_token_ids（v2 runner）。
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import KVConnectorMetadata
    from .request import Request
else:
    KVConnectorMetadata = object


# SOURCE: vllm/v1/core/sched/output.py:L35 NewRequestData
@dataclass
class NewRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L36-L44（字段面：req_id/prompt/
    #   mm/block_ids/num_computed_tokens；sampling/pooling/lora 账位缩）
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    # SOURCE: vllm/v1/core/sched/output.py:L56 from_request
    @classmethod
    def from_request(
        cls,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> "NewRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L61-L68（构造面——sampling/
        #   pooling/lora/embeds 实参随装配面删）
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )


# SOURCE: vllm/v1/core/sched/output.py:L116 CachedRequestData
@dataclass
class CachedRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L117-L128（字段面；new_token_ids
    #   PP 专用/all_token_ids MRV1 专用——两账位删）
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as
    # the request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # SUBTRACTED: anon_repr/__repr__（L132-L156——脱敏观测辅助）。

    # SOURCE: vllm/v1/core/sched/output.py:L158 num_reqs
    @property
    def num_reqs(self) -> int:
        # SOURCE: vllm/v1/core/sched/output.py:L159
        return len(self.req_ids)

    # SOURCE: vllm/v1/core/sched/output.py:L161 _req_id_to_num_output_tokens
    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup."""
        # SOURCE: vllm/v1/core/sched/output.py:L167
        return dict(zip(self.req_ids, self.num_output_tokens))

    # SOURCE: vllm/v1/core/sched/output.py:L170 is_context_phase
    def is_context_phase(self, req_id: str) -> bool:
        # SOURCE: vllm/v1/core/sched/output.py:L171-L172
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    # SOURCE: vllm/v1/core/sched/output.py:L174 make_empty
    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        # SOURCE: vllm/v1/core/sched/output.py:L175-L184
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


# SOURCE: vllm/v1/core/sched/output.py:L193 SchedulerOutput（connector 面）
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    # SOURCE: vllm/v1/core/sched/output.py:L197-L201
    scheduled_new_reqs: list[NewRequestData] = field(default_factory=list)
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    # SOURCE: vllm/v1/core/sched/output.py:L202-L204
    scheduled_cached_reqs: CachedRequestData = field(
        default_factory=CachedRequestData.make_empty
    )

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    # SOURCE: vllm/v1/core/sched/output.py:L205-L207
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    # SOURCE: vllm/v1/core/sched/output.py:L208-L210
    total_num_scheduled_tokens: int = 0
    # SUBTRACTED: scheduled_spec_decode_tokens/scheduled_encoder_inputs
    #   （L211-L216——ch33/EC 面）。
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    # SOURCE: vllm/v1/core/sched/output.py:L217-L219
    num_common_prefix_blocks: list[int] = field(default_factory=list)

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    # SOURCE: vllm/v1/core/sched/output.py:L220-L224
    finished_req_ids: set[str] = field(default_factory=set)
    # SUBTRACTED: free_encoder_mm_hashes/scheduled_encoder_input_stats
    #   （L225-L228——EC 面）。
    # Request IDs that are preempted in this step.
    # SOURCE: vllm/v1/core/sched/output.py:L231-L232
    preempted_req_ids: set[str] | None = None
    # SUBTRACTED: structured-output 两旗标/num_invalid_spec_tokens
    #   （L234-L245——ch05/ch33）。

    # KV Cache Connector metadata.
    # SOURCE: vllm/v1/core/sched/output.py:L246-L247 kv_connector_metadata
    #   ——m6 的信封本体：不透明、由 build_connector_meta 产、worker 侧
    #   bind_connector_metadata 收
    kv_connector_metadata: KVConnectorMetadata | None = None

    # SUBTRACTED: ec_connector_metadata/ec_manager_metadata（L249-L253——
    #   第 1 条 ECConnector 全删）。
    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    # SOURCE: vllm/v1/core/sched/output.py:L254-L258 new_block_ids_to_zero
    #   ——清零账（站 6 的 _skip_zero_block_ids 过滤后产物）
    new_block_ids_to_zero: list[int] | None = None

    # SUBTRACTED: kv_cache_block_copies（L260-L261——第 10 条 CoW 打包归
    #   ch15）。

    # Producer partial-tail offload hand-off for external KV connectors:
    # {request_id: [(group_id, block_id, boundary_tokens), ...]} pointing at
    # the durable boundary block of a producer's last-prompt-boundary partial
    # tail (mamba "align" CoW target). None unless partial hash hits are active.
    # SOURCE: vllm/v1/core/sched/output.py:L263-L268 partial_tail_offloads
    #   ——m15 的过线形态
    partial_tail_offloads: dict[str, list[tuple[int, int, int]]] | None = None

    # SUBTRACTED: num_spec_tokens_to_schedule（L270-L272——ch33）。

    # SOURCE: vllm/v1/core/sched/output.py:L274 make_empty
    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        # SOURCE: vllm/v1/core/sched/output.py:L275-L285
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
        )
