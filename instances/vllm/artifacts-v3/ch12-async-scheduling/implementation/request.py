# SOURCE: vllm/v1/request.py
# 精简版 Request / RequestStatus —— 本章账本的主角字段群全保留：四个异步计数器
# （num_output_placeholders / num_in_flight_tokens / num_stale_output_tokens /
# drop_stale_output，request.py:L150-L162——占位调度的全部标量账）+ 追赶公式的
# 被减数（num_tokens_with_spec）+ use_structured_output（deferred sampling 的
# 置位条件）+ spec_token_ids（被 AsyncScheduler 整体换成 -1 占位列表）。
# 删除项：mm/lora/streaming/可观测性等（各归邻章；ch30 拥有 structured 的编译
# 流水线——本文件只留 structured_output_request 字段与 use_structured_output
# 谓词，异步编译初始态不设，状态从 WAITING 直入）。
from __future__ import annotations

import enum
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from .engine import FinishReason

if TYPE_CHECKING:
    # SOURCE: vllm/v1/request.py:L27-L29 TYPE_CHECKING 前置声明
    from vllm.lora.request import LoRARequest  # noqa: F401


# SOURCE: vllm/v1/request.py:L348 RequestStatus
class RequestStatus(enum.IntEnum):
    # SOURCE: vllm/v1/request.py:L349-L364 全状态（顺序即语义，不可重排）
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    # SOURCE: vllm/v1/request.py:L366-L367 __str__
    def __str__(self) -> str:
        # SOURCE: vllm/v1/request.py:L367
        return self.name

    # SOURCE: vllm/v1/request.py:L369-L371 is_finished —— 一次整数比较
    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:L370-L371
        return status > RequestStatus.PREEMPTED

    # SOURCE: vllm/v1/request.py:L373-L375 get_finished_reason
    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        # SOURCE: vllm/v1/request.py:L374-L375
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# SOURCE: vllm/v1/request.py:L378-L390 _FINISHED_REASON_MAP
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    # SOURCE: vllm/v1/request.py:L388 流式暂停态的特殊映射（→STOP）
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}


# SOURCE: vllm/sampling_params.py SamplingParams（精简：本章 check_stop 与
# max_tokens 解析用到的字段；采样栈归 ch08）
class SamplingParams:
    # SUBTRACTED: temperature/top_p/logprobs/penalties 等——与异步调度无关。
    # SOURCE: vllm/sampling_params.py SamplingParams.__init__
    def __init__(
        self,
        max_tokens: int = 16,
        min_tokens: int = 0,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        # SOURCE: vllm/sampling_params.py SamplingParams.__init__
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids or []


# SOURCE: vllm/v1/request.py:L59 Request —— 一生的账本
class Request:
    # SOURCE: vllm/v1/request.py:L60-L79 __init__ 签名（保留本章路径用到的参数；
    # structured_output_request 是 deferred sampling 的置位输入）
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        structured_output_request: object | None = None,
        client_index: int = 0,
        arrival_time: float | None = None,
        lora_request: "LoRARequest | None" = None,
        priority: int = 0,
        block_hasher: Callable[["Request"], list[int]] | None = None,
    ) -> None:
        # SUBTRACTED: prompt_embeds/mm_features/cache_salt/trace_headers/
        #   resumable/abort_immediately 等参数与字段（mm/流式/可观测性——
        #   各归邻章）；WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 初始态
        #   （vllm/v1/request.py:L113-L114——异步语法编译流水线归 ch30，
        #   本章状态从 WAITING 直入）。
        # SOURCE: vllm/v1/request.py:L81-L95 基本字段
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = None
        self.lora_request = lora_request
        self.arrival_time = arrival_time if arrival_time is not None else time.time()
        # SOURCE: vllm/v1/request.py:L87-L94 structured_output_request 建档
        self.structured_output_request = structured_output_request

        # SOURCE: vllm/v1/request.py:L97-L99
        self.status = RequestStatus.WAITING
        self.stop_reason: int | str | None = None

        # SOURCE: vllm/v1/request.py:L106-L112 max_tokens 解析（pooling 分支随
        # is_pooling_model 删除——本章恒生成路径）
        assert sampling_params is not None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        # SOURCE: vllm/v1/request.py:L131-L148 prompt 与全量 token 视图
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = (
            len(prompt_token_ids) if prompt_token_ids is not None else 0
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            prompt_token_ids.copy()
            if prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # SOURCE: vllm/v1/request.py:L150-L162 四个异步计数器（占位账本本体）
        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Tokens of output in flight when the request was preempted: delivered
        # on return, but must not mutate the reset counters.
        self.num_stale_output_tokens = 0
        # Drop the stale output instead, for same-step preempt + resume
        # (reset_prefix_cache).
        self.drop_stale_output = False

        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        self.num_in_flight_tokens = 0

        # SUBTRACTED: next_decode_eligible_step（L164-L166——V2+PP+async 步距，
        #   dossier.delete 第 4 条批准：V2 runner 分支已删）、last_sched_seq
        #   （L168-L170——defer_block_free 栅栏）。

        # SOURCE: vllm/v1/request.py:L172-L173
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0

        # SOURCE: vllm/v1/request.py:L187-L188
        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # SOURCE: vllm/v1/request.py:L199-L200
        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # SOURCE: vllm/v1/request.py:L204-L208 block_hashes（前缀缓存伏线）
        self.block_hashes: list[int] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[["Request"], list[int]] | None = block_hasher
        self.update_block_hashes()

        # SUBTRACTED: events/record_event（L98/L314-L325——可观测性埋点）、
        #   kv/ec_transfer_params（L101-L104——connector）、prefill_stats/
        #   num_nans_in_logits/skip_reading_prefix_cache/resumable 等——各归
        #   邻章。

    # SOURCE: vllm/v1/request.py:L249 append_output_token_ids —— 逐 token 收账
    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        # SOURCE: vllm/v1/request.py:L253-L258
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        # SOURCE: vllm/v1/request.py:L260 —— 连带增量算块哈希
        self.update_block_hashes()

    # SOURCE: vllm/v1/request.py:L262 update_block_hashes
    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        # SOURCE: vllm/v1/request.py:L264-L265
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    # SOURCE: vllm/v1/request.py:L267-L269 use_structured_output —— deferred
    # sampling 的置位谓词（AsyncScheduler._update_after_schedule 消费）
    @property
    def use_structured_output(self) -> bool:
        # SOURCE: vllm/v1/request.py:L268-L269
        return self.structured_output_request is not None

    # SOURCE: vllm/v1/request.py:L271-L273 num_tokens —— 账本被减数定义
    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L272-L273
        return len(self._all_token_ids)

    # SOURCE: vllm/v1/request.py:L275-L277 num_tokens_with_spec —— 追赶公式被减数
    @property
    def num_tokens_with_spec(self) -> int:
        # SOURCE: vllm/v1/request.py:L276-L277
        return len(self._all_token_ids) + len(self.spec_token_ids)

    # SOURCE: vllm/v1/request.py:L279-L281 num_output_tokens
    @property
    def num_output_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L280-L281
        return len(self._output_token_ids)

    # SOURCE: vllm/v1/request.py:L182 output_token_ids 只读视图
    @property
    def output_token_ids(self) -> list[int]:
        # SUBTRACTED: ConstantList 只读视图（vllm/v1/utils.py）——裸 list 同名
        #   暴露（观测便利，无必要保只读包装）。
        # SOURCE: vllm/v1/request.py:L182
        return self._output_token_ids

    # SOURCE: vllm/v1/request.py:L183 all_token_ids 只读视图
    @property
    def all_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:L183
        return self._all_token_ids

    # SOURCE: vllm/v1/request.py:L304-L305 is_finished
    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:L305
        return RequestStatus.is_finished(self.status)

    # SOURCE: vllm/v1/request.py:L307-L308 get_finished_reason
    def get_finished_reason(self) -> FinishReason | None:
        # SOURCE: vllm/v1/request.py:L308
        return RequestStatus.get_finished_reason(self.status)
