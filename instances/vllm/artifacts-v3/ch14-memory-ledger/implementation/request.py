# SOURCE: vllm/v1/request.py
# Request 的**两道门消费面**（ch02/ch11/ch13 已各建过更全的切面）：
# allocate_slots 读 num_computed_tokens/num_tokens/status/num_in_flight_
# tokens/num_prompt_tokens；remove_skipped_blocks 的水位基准也挂它们。
# SUBTRACTED: mm/LoRA/pooling/structured-output/streaming/spec 相关字段与
#   方法（ch02/11/12/30/33 各章切面）；block_hashes 哈希账位（→ ch15）；
#   shared_prefix_boundary/PrefillStats（稀疏驻留 → ch15）；FinishReason
#   映射表（ch02）。
import enum


# SOURCE: vllm/v1/request.py:L348 RequestStatus
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    # SOURCE: vllm/v1/request.py:L351-L364
    WAITING = enum.auto()
    # SUBTRACTED: WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_
    #   REMOTE_KVS（ch16）/ WAITING_FOR_STREAMING_REQ（各邻章）
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()

    # SOURCE: vllm/v1/request.py:L369 is_finished
    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:L371
        return status > RequestStatus.PREEMPTED


# SOURCE: vllm/v1/request.py:L59 Request（两道门消费面）
class Request:
    # SOURCE: vllm/v1/request.py:L60 __init__
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
    ) -> None:
        # SUBTRACTED: EngineCore 装配面（sampling_params/mm/priority 等
        #   L60-L139——ch02 消费面）与 async/spec 计数器（ch12/33）。
        # SOURCE: vllm/v1/request.py:L173 request_id / num_computed_tokens
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
