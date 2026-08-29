# SOURCE: vllm/v1/request.py
# Request 的**connector 面 + 哈希面账位**（ch16 消费的请求状态）：
# WAITING_FOR_REMOTE_KVS 状态（request.py:L353——等待态本体）、
# num_computed_tokens 先行记账、num_in_flight_tokens（在途步——窗外回收
# 与栅栏的基准）、drop_stale_output/num_stale_output_tokens（m13 抢占护栏
# 的在途产出账）、last_sched_seq（m12 步序栅栏的 fence）、block_hashes
# 账本（本地前缀缓存的哈希归 Request 持有——ch15 已立、本章只消费）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   streaming/structured-output/events/pooling/prefill_stats 装配面
#     （L32-L127——ch02 各章切面）；
#   prompt_embeds 三件（ch15 同款删法）；
#   next_decode_eligible_step（V2+PP 解码节拍——ch12）、spec_token_ids
#     （ch33）、num_nans_in_logits/trace_headers（观测面）；
#   ConstantList 只读视图包装（GC 优化，不进控制流——直接暴露底层列表）；
#   from_engine_core_request/StreamingUpdate（EngineCore 装配面 → ch02）。
import enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .kv_cache_utils import BlockHash  # noqa: F401


# SOURCE: vllm/v1/request.py:L348 RequestStatus
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    # SOURCE: vllm/v1/request.py:L351-L364
    WAITING = enum.auto()
    # SUBTRACTED: WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR（L352——ch05 结构化
    #   输出面）/ WAITING_FOR_STREAMING_REQ（L354——streaming → ch02）。
    # SOURCE: vllm/v1/request.py:L353 WAITING_FOR_REMOTE_KVS
    WAITING_FOR_REMOTE_KVS = enum.auto()
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


# SOURCE: vllm/v1/request.py:L59 Request（connector 面切面）
class Request:
    # SOURCE: vllm/v1/request.py:L60 __init__
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        mm_features: list | None = None,
        lora_request: Any | None = None,
        cache_salt: str | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        sampling_params: Any | None = None,
        pooling_params: Any | None = None,
        skip_reading_prefix_cache: bool | None = None,
    ) -> None:
        # SUBTRACTED: EngineCore 装配面（client_index/priority/arrival_time/
        #   trace_headers/resumable/reasoning_*/abort_immediately L63-L127
        #   ——ch02 切面）；prompt_embeds 三件；next_decode_eligible_step/
        #   spec_token_ids/num_nans_in_logits/prefill_stats（L167-L176 各邻章）。
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        # SOURCE: vllm/v1/request.py:L97 status
        self.status = RequestStatus.WAITING

        # SOURCE: vllm/v1/request.py:L131 prompt_token_ids（简化：无 embeds）
        self.prompt_token_ids = prompt_token_ids
        # SOURCE: vllm/v1/request.py:L140 num_prompt_tokens
        self.num_prompt_tokens = len(prompt_token_ids)
        # SOURCE: vllm/v1/request.py:L143-L148 output/all token 账
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = prompt_token_ids.copy()

        # Used in async scheduling.
        # SOURCE: vllm/v1/request.py:L150 num_output_placeholders（账位保留）
        self.num_output_placeholders = 0
        # Tokens of output in flight when the request was preempted: delivered
        # on return, but must not mutate the reset counters.
        # SOURCE: vllm/v1/request.py:L152-L153 num_stale_output_tokens
        self.num_stale_output_tokens = 0
        # Drop the stale output instead, for same-step preempt + resume
        # (reset_prefix_cache).
        # SOURCE: vllm/v1/request.py:L155-L156 drop_stale_output
        self.drop_stale_output = False

        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        # SOURCE: vllm/v1/request.py:L158-L162 num_in_flight_tokens
        self.num_in_flight_tokens = 0

        # Seq of the most recent step this request was scheduled in; fences
        # deferred block freeing (see Scheduler._free_request_blocks).
        # SOURCE: vllm/v1/request.py:L170-L171 last_sched_seq
        self.last_sched_seq = 0

        # SOURCE: vllm/v1/request.py:L177 num_computed_tokens
        self.num_computed_tokens = 0
        # SOURCE: vllm/v1/request.py:L174 cache_salt（extra_keys 第三源）
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        # SOURCE: vllm/v1/request.py:L177 mm_features（extra_keys 第一源）
        self.mm_features = mm_features or []

        # True if this request is scheduled as a non-final prefill chunk.
        # SOURCE: vllm/v1/request.py:L187-L188 is_prefill_chunk
        self.is_prefill_chunk = False

        # Block-aligned token position of a proven shared prefix worth pinning
        # in the (sparse) prefix cache; 0 means none. Set at admission for
        # hybrid/Mamba models when a shared prefix is detected (Marconi-style).
        # SOURCE: vllm/v1/request.py:L190-L193 shared_prefix_boundary
        self.shared_prefix_boundary = 0

        # The number of times this request has been preempted by the scheduler.
        # SOURCE: vllm/v1/request.py:L199-L200 num_preemptions
        self.num_preemptions = 0

        # SOURCE: vllm/v1/request.py:L204 block_hashes——哈希账本（本地前缀
        #   缓存面，ch15 已立；本章 get_computed_blocks_for_connector 消费）
        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        # SOURCE: vllm/v1/request.py:L205-L208 _block_hasher（不绑定 self）
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        # SOURCE: vllm/v1/request.py:L209 构造尾即补算（哈希随 Request 出生）
        self.update_block_hashes()

        # SOURCE: vllm/v1/request.py:L211 skip_reading_prefix_cache
        self.skip_reading_prefix_cache = (
            skip_reading_prefix_cache
            if skip_reading_prefix_cache is not None
            else self.get_skip_reading_prefix_cache()
        )

    # SOURCE: vllm/v1/request.py:L249 append_output_token_ids
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

        # SOURCE: vllm/v1/request.py:L260（每次 append 后增量补算——
        #   「哈希随 token 到达」的落点）
        self.update_block_hashes()

    # SOURCE: vllm/v1/request.py:L262 update_block_hashes
    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        # SOURCE: vllm/v1/request.py:L264-L265
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    # SOURCE: vllm/v1/request.py:L179-L183 all_token_ids（ConstantList 包装
    #   删——属性面保留：request_block_hasher 读 all_token_ids 切片）
    @property
    def all_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:L183（ConstantList 只读视图 → 直接暴露）
        return self._all_token_ids

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

    # SOURCE: vllm/v1/request.py:L291 get_skip_reading_prefix_cache
    def get_skip_reading_prefix_cache(self) -> bool:
        # SOURCE: vllm/v1/request.py:L292-L302（sampling/pooling 两支的
        #   谓词语义逐字；extra_args 装配面简化）
        if (
            self.sampling_params is not None
            and getattr(self.sampling_params, "skip_reading_prefix_cache", None)
            is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and getattr(self.pooling_params, "skip_reading_prefix_cache", None)
            is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    # SOURCE: vllm/v1/request.py:L304 is_finished
    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:L305
        return RequestStatus.is_finished(self.status)
