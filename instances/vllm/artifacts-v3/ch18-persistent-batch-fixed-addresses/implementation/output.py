# SOURCE: vllm/v1/core/sched/output.py
# 本章主角文件之一：差量协议三类载体（m01/m11）——NewRequestData（新请求
# 全量）/ CachedRequestData（老请求 diff，含 resumed 语义分叉注释 L118-L121）/
# SchedulerOutput（二分协议头 L193-L205）。除 import 面的 config 占位外全文
# 逐字（协议字段全保留——dossier.delete 未列任何 output.py 删除项）。
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

# SOURCE: vllm/v1/core/sched/output.py:L8 EncoderCacheManagerMetadata 的
#   config 侧 import——HOST 移植无 config 包，落 object 占位（仅作注解默认值
#   None 的类型，行为无涉；真实 vllm/config/ec_manager_config.py，ch03 域）。
EncoderCacheManagerMetadata = object

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from ._host_seams import (
        LoRARequest,
        MultiModalFeatureSpec,
        PoolingParams,
        SamplingParams,
    )
else:
    # SOURCE: vllm/v1/core/sched/output.py:L23-L31 else 分支的对象占位族
    #   （真实逐一对应 vllm.* 内的类；本包内注解-only，行为无涉）
    KVCacheBlockCopy = object  # 真实 vllm/v1/core/kv_cache_utils.py（ch13 域）
    LoRARequest = object  # 真实 vllm/lora/request.py
    MultiModalFeatureSpec = object  # 真实 vllm/multimodal/inputs.py（mm 域）
    PoolingParams = object  # 真实 vllm/pooling_params.py
    SamplingParams = object  # 真实 vllm/sampling_params.py（ch08 域）
    Request = object  # 真实 vllm/v1/request.py（ch10 域）


# SOURCE: vllm/v1/core/sched/output.py:L35 NewRequestData —— 新请求全量
@dataclass
class NewRequestData:
    # SOURCE: vllm/v1/core/sched/output.py:L36-L45
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    # SUBTRACTED: from_request（L50-L69——scheduler 侧构造面，ch10/ch12 域）。

    # SOURCE: vllm/v1/core/sched/output.py:L71-L87 __repr__
    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            f")"
        )

    # Version of __repr__ with the prompt data obfuscated
    # SOURCE: vllm/v1/core/sched/output.py:L90-L112 anon_repr
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            f")"
        )


# SOURCE: vllm/v1/core/sched/output.py:L116 CachedRequestData —— 老请求 diff
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

    # Version of dataclass repr with token IDs obfuscated.
    # SOURCE: vllm/v1/core/sched/output.py:L133-L148 anon_repr
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    # SOURCE: vllm/v1/core/sched/output.py:L150-L151 __repr__
    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    # SOURCE: vllm/v1/core/sched/output.py:L153-L155 num_reqs property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    # SOURCE: vllm/v1/core/sched/output.py:L157-L165 _req_id_to_num_output_tokens
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    # SOURCE: vllm/v1/core/sched/output.py:L167-L169 is_context_phase
    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    # SOURCE: vllm/v1/core/sched/output.py:L171-L181 make_empty
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L185 ScheduledEncoderInputStats
class ScheduledEncoderInputStats:
    """Stats for encoder inputs scheduled in one iteration."""

    num_inputs: int = 0
    output_tokens: int = 0


# SOURCE: vllm/v1/core/sched/output.py:L193 SchedulerOutput —— 差量协议头
#   （协议二分注释原文 L194-L200）
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
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1st images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    scheduled_encoder_input_stats: ScheduledEncoderInputStats | None = None

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    kv_connector_metadata: object | None = None

    # EC Cache Connector metadata
    ec_connector_metadata: object | None = None
    # EC Cache Manager metadata
    ec_manager_metadata: EncoderCacheManagerMetadata | None = None
    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    new_block_ids_to_zero: list[int] | None = None

    # CoW copies to apply after zeroing new blocks and before forward.
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    # Producer partial-tail offload hand-off for external KV connectors:
    # {request_id: [(group_id, block_id, boundary_tokens), ...]} pointing at
    # the durable boundary block of a producer's last-prompt-boundary partial
    # tail (mamba "align" CoW target). None unless partial hash hits are active.
    partial_tail_offloads: dict[str, list[tuple[int, int, int]]] | None = None

    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0

    @classmethod
    # SOURCE: vllm/v1/core/sched/output.py:L271-L283 make_empty
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


# SUBTRACTED: GrammarOutput（L286-L291——结构化输出掩码载体，ch30 域；
#   本章 sample_tokens 的 bitmask 支随 ch30 删除）。
