# SOURCE: vllm/v1/request.py
# 精简版 Request / RequestStatus —— 本章主角之一：请求一生的账本字段群。
# 保留：status 单 IntEnum 全状态、四个生命周期计数器（num_output_placeholders/
# num_stale_output_tokens/drop_stale_output/num_in_flight_tokens——m4 载体）、
# num_computed_tokens/num_preemptions（抢占清零/被抢计数）、block_hashes +
# append_output_token_ids 连带增量块哈希（F2 伏线：前缀恢复的伏线在每次输出
# 时都在续）、get_finished_reason（finish_reason 时序约束的取用面）。
# 删除项全部 dossier.subtraction_plan.delete 批准（mm/lora/structured 细节/
# streaming/可观测性/async 残留——各归邻章）。
from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass
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

    def __str__(self) -> str:
        # SOURCE: vllm/v1/request.py:L366-L367 __str__
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
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
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


# SOURCE: vllm/sampling_params.py:L146 RepetitionDetectionParams
@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens."""

    # SOURCE: vllm/sampling_params.py:L149-L161 三参数
    max_pattern_size: int = 0
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count."""

    min_pattern_size: int = 0
    """Minimum N-gram pattern size to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size."""

    min_count: int = 0
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size."""

    # SUBTRACTED: __post_init__ 校验（L163+）——装配层校验，非运行时语义。


# SOURCE: vllm/sampling_params.py SamplingParams
class SamplingParams:
    # SUBTRACTED: 绝大多数采样字段（temperature/top_p/logprobs/...）——与调度
    #   决策无关；只保留本章 check_stop 五连判与 max_tokens 解析用到的字段。
    def __init__(
        self,
        max_tokens: int = 16,
        min_tokens: int = 0,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
        repetition_detection: RepetitionDetectionParams | None = None,
    ) -> None:
        # SOURCE: vllm/sampling_params.py SamplingParams.__init__
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids or []
        self.repetition_detection = repetition_detection


# SOURCE: vllm/v1/request.py:L59 Request —— 一生的账本
class Request:
    # SOURCE: vllm/v1/request.py:L60-L79 __init__ 签名（保留本章路径用到的参数）
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params=None,
        client_index: int = 0,
        arrival_time: float | None = None,
        lora_request: "LoRARequest | None" = None,
        priority: int = 0,
        block_hasher: Callable[["Request"], list[int]] | None = None,
    ) -> None:
        # SUBTRACTED: prompt_embeds/prompt_is_token_ids/mm_features/cache_salt/
        #   trace_headers/resumable/reasoning_*/abort_immediately 参数与字段
        #   （mm/流式/可观测性，dossier.delete 第 2/5/10 条批准）；structured_
        #   output_request 建档与 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 初始态
        #   （vllm/v1/request.py:L87-L94/L113-L114——第 4 条，归 ch30）。
        # SOURCE: vllm/v1/request.py:L81-L95 基本字段
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # SOURCE: vllm/v1/request.py:L97-L99
        self.status = RequestStatus.WAITING
        self.stop_reason: int | str | None = None

        # SOURCE: vllm/v1/request.py:L106-L129 max_tokens 解析
        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

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

        # SOURCE: vllm/v1/request.py:L150-L162 四个生命周期计数器（m4 载体）
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
        #   第 7 条）/last_sched_seq（L168-L170——defer_block_free 栅栏，第 7 条）。

        # SOURCE: vllm/v1/request.py:L172-L173
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0

        # SOURCE: vllm/v1/request.py:L187-L188
        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # Block-aligned token position of a proven shared prefix worth pinning
        # in the (sparse) prefix cache; 0 means none. Set at admission for
        # hybrid/Mamba models when a shared prefix is detected (Marconi-style).
        # SOURCE: vllm/v1/request.py:L190-L193
        self.shared_prefix_boundary = 0

        # SOURCE: vllm/v1/request.py:L199-L200
        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # SOURCE: vllm/v1/request.py:L204-L208 block_hashes（F2 伏线载体）
        self.block_hashes: list[int] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[int]] | None = block_hasher
        self.update_block_hashes()

        # SUBTRACTED: events 字段与 record_event/take_events（L98/L314-L325——
        #   第 10 条：QUEUED/SCHEDULED/PREEMPTED 埋点，正文讲注释与字段即可）、
        #   kv/ec_transfer_params（L101-L104——connector）、prefill_stats（L202）、
        #   num_nans_in_logits（L195-L197）、skip_reading_prefix_cache（L211——
        #   缓存读取开关归 ch15）、resumable/streaming_queue（L213-L216——第 5 条）、
        #   __lt__（L334-L345——PRIORITY 堆序，第 1 条）。

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
        # SOURCE: vllm/v1/request.py:L182
        # SUBTRACTED: ConstantList 只读视图（vllm/v1/utils.py）——裸 list
        #   同名暴露（dossier.delete 第 10 条：ConstantList 可保留但无必要）。
        return self._output_token_ids

    # SOURCE: vllm/v1/request.py:L183 all_token_ids 只读视图
    @property
    def all_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:L183
        # SUBTRACTED: ConstantList 只读视图——裸 list 同名暴露。
        return self._all_token_ids

    # SOURCE: vllm/v1/request.py:L304-L305 is_finished
    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:L305
        return RequestStatus.is_finished(self.status)

    # SOURCE: vllm/v1/request.py:L307-L308 get_finished_reason
    def get_finished_reason(self) -> FinishReason | None:
        # SOURCE: vllm/v1/request.py:L308
        return RequestStatus.get_finished_reason(self.status)
