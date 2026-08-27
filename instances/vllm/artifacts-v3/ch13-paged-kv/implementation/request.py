# SOURCE: vllm/v1/request.py
# Request 的**KV 账本消费面**（ch02/ch11 已各建过更全的切面）：allocate_slots
# 读 num_computed_tokens/num_tokens/status/num_in_flight_tokens/num_prompt_
# tokens；构造即带空 block_hashes（data_flow【入场】——增量哈希构造 → ch15）。
# SUBTRACTED: mm/LoRA/pooling/structured-output/streaming/spec 相关字段与
#   方法、ConstantList 只读视图（ch02/11/12/30/33 各章切面）；FinishReason
#   映射表（ch02）。
import enum


# SOURCE: vllm/v1/request.py:L348 RequestStatus
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    # SOURCE: vllm/v1/request.py:L351-L364
    WAITING = enum.auto()
    # SUBTRACTED: WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_
    #   REMOTE_KVS（ch16）/ WAITING_FOR_STREAMING_REQ（L352-L354——各邻章）
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

    # SOURCE: vllm/v1/request.py:L366 __str__
    def __str__(self) -> str:
        # SOURCE: vllm/v1/request.py:L367
        return self.name

    # SOURCE: vllm/v1/request.py:L369 is_finished
    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:L371
        return status > RequestStatus.PREEMPTED


# SOURCE: vllm/v1/request.py:L59 Request（KV 账本消费面）
class Request:
    # SOURCE: vllm/v1/request.py:L60 __init__
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        block_hasher=None,
    ) -> None:
        # SUBTRACTED: EngineCore 装配面（client_index/prompt_embeds/mm/
        #   sampling_params/pooling_params/lora/trace_headers/cache_salt/
        #   priority/resumable 等 L60-L139——ch02 消费面）与 async 占位计数器
        #   num_output_placeholders/num_stale_output_tokens/drop_stale_output
        #   （L151-L157——ch12）、next_decode_eligible_step（L166——V2+PP）、
        #   last_sched_seq（L169-L170——ch12 deferred 栅栏）、spec_token_ids
        #   （L172——ch33）。
        # SOURCE: vllm/v1/request.py:L173 num_computed_tokens
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        # SOURCE: vllm/v1/request.py:L159-L162 num_in_flight_tokens
        self.num_in_flight_tokens = 0
        self.num_computed_tokens = 0
        self.status = RequestStatus.WAITING

        # SOURCE: vllm/v1/request.py:L140 num_prompt_tokens（简化：无 embeds）
        self.num_prompt_tokens = len(prompt_token_ids)
        # SOURCE: vllm/v1/request.py:L143-L148 output/all token 账
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = prompt_token_ids.copy()

        # True if this request is scheduled as a non-final prefill chunk.
        # SOURCE: vllm/v1/request.py:L187-L188 is_prefill_chunk
        self.is_prefill_chunk = False

        # Block-aligned token position of a proven shared prefix worth pinning
        # in the (sparse) prefix cache; 0 means none.
        # SOURCE: vllm/v1/request.py:L190-L193 shared_prefix_boundary
        self.shared_prefix_boundary = 0

        # The number of times this request has been preempted by the scheduler.
        # SOURCE: vllm/v1/request.py:L199-L200 num_preemptions
        self.num_preemptions = 0

        # SUBTRACTED: PrefillStats/mrope/trace 头（L184-L202——观测与多模态）。

        # SOURCE: vllm/v1/request.py:L204 block_hashes——构造即带空账位
        #（增量哈希构造 → ch15；本章恒空）
        self.block_hashes: list = []
        # SUBTRACTED: _block_hasher 装配与 update_block_hashes 调用
        #   （L205-L209——hasher 仅在 enable_prefix_caching 或有 connector 时
        #   装配；哈希链 → ch15）。本章 False 支保留空账位语义。

        # SOURCE: vllm/v1/request.py:L211 skip_reading_prefix_cache（默认 False）
        self.skip_reading_prefix_cache = False

    # SOURCE: vllm/v1/request.py:L249 append_output_token_ids
    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        # SOURCE: vllm/v1/request.py:L253-L258
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

    # SOURCE: vllm/v1/request.py:L271 num_tokens
    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L273
        return len(self._all_token_ids)

    # SOURCE: vllm/v1/request.py:L279 num_output_tokens
    @property
    def num_output_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L281
        return len(self._output_token_ids)

    # SOURCE: vllm/v1/request.py:L304 is_finished
    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:L305
        return RequestStatus.is_finished(self.status)

    # SOURCE: vllm/v1/request.py:L262 update_block_hashes（空 hasher no-op）
    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        # SOURCE: vllm/v1/request.py:L264-L265（_block_hasher 装配面删——
        # 本章 False 支恒 no-op，账位语义保留）
