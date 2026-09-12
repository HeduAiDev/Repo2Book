# SOURCE: vllm/v1/request.py
# 只做减法的忠实精简版（pin v0.27.1 / 6e448d0ea）：请求对象与状态枚举的
# 语法门切面。保留：structured_output_request 挂载与出生即阻塞（站 2）、
# RequestStatus 全枚举（WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 本章的『门』）、
# from_engine_core_request、append_output_token_ids、use_structured_output/
# is_finished/get_finished_reason、推进侧读的字段面（all_token_ids/
# num_computed_tokens/num_output_placeholders/reasoning 三件套）。
# SUBTRACTED: SPDX 版权头；KV/抢占/prefix 哈希/streaming 会话/pooling 分支/
# mm 面与 prompt_embeds 面（各归 ch13-16/ch38/ch04 域——见各 SUBTRACTED 行）。
import enum
import time
from typing import Any, Callable, Mapping

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList


# SOURCE: vllm/v1/request.py:L59 Request
class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: Any = None,  # HOST SEAM：真实是 torch.Tensor | None
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list | None = None,  # HOST SEAM：真实是 list[MultiModalFeatureSpec] | None
        lora_request: Any = None,  # HOST SEAM：真实是 LoRARequest | None
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        abort_immediately: bool = False,
    ) -> None:
        # SOURCE: vllm/v1/request.py:L81-L95 —— 逐字（structured 挂载在先）
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
            self.structured_output_request.reasoning_parser_kwargs = (
                reasoning_parser_kwargs
            )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # SOURCE: vllm/v1/request.py:L97-L99
        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # SUBTRACTED: vllm/v1/request.py:L101-L104 kv_transfer_params/
        #   ec_transfer_params 两字段（P/D 与 EC 连接器域，ch16）。

        # SOURCE: vllm/v1/request.py:L106-L108 pooling 分支 —— 逐字（池化模型
        #   面，ch04 域；保留以维持 if/elif 控制流形状）
        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        # SOURCE: vllm/v1/request.py:L109-L114 —— 逐字（**门在出生那一刻就
        #   关上**：带约束的生成请求初始 status 直接=
        #   WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR，不是 WAITING）
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            # SUBTRACTED: vllm/v1/request.py:L116-L127 extra_args 的 kv/
            #   ec 传输参数分支（KV 连接器域）——else 位 kv_cache_report_mode
            #   同删。
        else:
            # SOURCE: vllm/v1/request.py:L128-L129 —— 逐字
            raise ValueError("sampling_params and pooling_params can't both be unset")

        # SOURCE: vllm/v1/request.py:L131-L136 prompt 三件套（prompt_is_token_
        #   ids 注释原文；embeds 混合模式归 ch04）
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.prompt_is_token_ids = prompt_is_token_ids
        # SUBTRACTED: L137-L139 _prompt_embeds_per_block_hashes（prefix 哈希
        #   域，ch15）。
        # SOURCE: vllm/v1/request.py:L140-L142 —— HOST SEAM（真实经
        #   length_from_prompt_token_ids_or_embeds 兼容 embeds 面；精简版
        #   无 embeds → len(prompt_token_ids or [])）
        self.num_prompt_tokens = len(prompt_token_ids or [])
        # SOURCE: vllm/v1/request.py:L143-L148 —— 逐字（else 位 embeds 分支
        #   减去：[0] * self.num_prompt_tokens）
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else []
        )

        # SOURCE: vllm/v1/request.py:L150-L154 推进/账面字段
        # Used in async scheduling.
        self.num_output_placeholders = 0
        # SUBTRACTED: L155-L159 num_stale_output_tokens/drop_stale_output
        #   （抢占 stale 输出账，ch11/ch12 域）。
        # SOURCE: vllm/v1/request.py:L160-L166
        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        self.num_in_flight_tokens = 0
        # SUBTRACTED: L168-L172 next_decode_eligible_step/last_sched_seq
        #   （V2+PP 与延迟释放栅栏域）。
        # SOURCE: vllm/v1/request.py:L174-L176
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # SOURCE: vllm/v1/request.py:L178-L179
        self.mm_features = mm_features or []

        # SOURCE: vllm/v1/request.py:L180-L183 —— 逐字（只读视图）
        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # SOURCE: vllm/v1/request.py:L184
        self.trace_headers = trace_headers

        # SOURCE: vllm/v1/request.py:L186-L188
        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False
        # SUBTRACTED: L189-L196 shared_prefix_boundary/num_nans_in_logits
        #   （共享前缀钉住与 NaN 观测面）。
        # SOURCE: vllm/v1/request.py:L197-L198
        self.num_preemptions = 0
        # SUBTRACTED: L200-L203 prefill_stats/block_hashes/_block_hasher/
        #   update_block_hashes() 调用/skip_reading_prefix_cache（统计与
        #   prefix 哈希域）。
        # SOURCE: vllm/v1/request.py:L213-L214
        # Used for streaming
        self.resumable = resumable
        # SUBTRACTED: L215-L216 streaming_queue（流式会话域，ch38——
        #   update_from_output 推进块设 resumable=False 的消费面在收账路径）。
        # SUBTRACTED: L218-L221 abort_immediately 字段（KV 连接器清理域；
        #   构造参数保留原位）。

    # SOURCE: vllm/v1/request.py:L223-L247 from_engine_core_request —— 逐字
    @classmethod
    # SOURCE: vllm/v1/request.py:L223-L247
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
        )

    # SOURCE: vllm/v1/request.py:L249-L257 append_output_token_ids —— 逐字
    #   （update_block_hashes() 调用减去：prefix 哈希域）
    # SOURCE: vllm/v1/request.py:L249-L257
    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        # SUBTRACTED: self.update_block_hashes()（L256——prefix 哈希域，ch15）

    # SUBTRACTED: vllm/v1/request.py:L259-L263 update_block_hashes（prefix
    #   哈希域）。

    # SOURCE: vllm/v1/request.py:L268-L270 use_structured_output —— 逐字
    @property
    # SOURCE: vllm/v1/request.py:L268-L270
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    # SOURCE: vllm/v1/request.py:L272-L274 num_tokens —— 逐字
    @property
    # SOURCE: vllm/v1/request.py:L272-L274
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    # SOURCE: vllm/v1/request.py:L304-L305 is_finished —— 逐字
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    # SOURCE: vllm/v1/request.py:L307-L308 get_finished_reason —— 逐字
    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    # SOURCE: vllm/v1/request.py:L313-L323 record_event/take_events —— 逐字
    #   （收账回执 EngineCoreOutput 的 events 消费面）
    # SOURCE: vllm/v1/request.py:L314-L319
    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    # SOURCE: vllm/v1/request.py:L321-L324
    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    # SUBTRACTED: take_prefill_stats/get_num_encoder_embeds/__lt__ 等文件其余
    #   （统计/mm 面/优先级比较——非本章域）。


# SOURCE: vllm/v1/request.py:L348-L379 RequestStatus —— 逐字（全枚举）
class RequestStatus(enum.IntEnum):
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

    # SOURCE: vllm/v1/request.py:L366-L367（RequestStatus.__str__）
    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:L370-L371 —— 逐字
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        # SOURCE: vllm/v1/request.py:L373-L375 —— 逐字
        return _FINISHED_REASON_MAP.get(status)


# SOURCE: vllm/v1/request.py:L380-L391 _FINISHED_REASON_MAP —— 逐字
# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
