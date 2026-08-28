# SOURCE: vllm/v1/request.py
# Request 的**哈希面**（本章第一主角的宿主）：block_hashes 长在 Request 上
# （构造尾与每次 append token 后 update_block_hashes 增量补算——哈希归
# Request 持有、账本只读不写）；shared_prefix_boundary（Marconi junction 的
# 落点字段：调度器写、cache_blocks/_mamba_block_aligned_split 读——跨模块
# 隐式协议）；skip_reading_prefix_cache（prompt logprobs/pooling 跳读）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 8 条 prompt_embeds（_gen_prompt_embeds_extra_hash_keys 的请求侧账位
#     _prompt_embeds_per_block_hashes L139 与 prompt_embeds/prompt_is_token_
#     ids 字段）；
#   streaming/structured-output/events/pooling 装配面（L32-L56、L87-L127——
#     ch02/ch16 各章切面；get_skip_reading_prefix_cache 的 pooling 支保留
#     谓词语义、参数面简化）；
#   async 占位计数器（num_output_placeholders/num_stale_output_tokens/
#     drop_stale_output L151-L157——ch12）、last_sched_seq/next_decode_
#     eligible_step（ch12 的 deferred 栅栏/V2+PP）、spec_token_ids（ch33）；
#   ConstantList 只读视图包装（L182-L183——GC 优化，不进控制流）；
#   kv_cache_report_mode（kv events 的 full 模式位，第 1 条观测旁路）；
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


# SOURCE: vllm/v1/request.py:L59 Request（哈希面切面）
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
        #   trace_headers/resumable/reasoning_*/abort_immediately L63-L79——
        #   ch02 切面）与 events/kv_transfer_params/structured_output
        #   （L87-L127）；prompt_embeds 三件（L68-L69、L131-L136、L139——
        #   第 8 条）；async/V2/spec 计数器（L150-L172——ch12/ch33）。
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        # SOURCE: vllm/v1/request.py:L97 status
        self.status = RequestStatus.WAITING

        # SOURCE: vllm/v1/request.py:L131 prompt_token_ids（简化：无 embeds）
        self.prompt_token_ids = prompt_token_ids
        # SOURCE: vllm/v1/request.py:L140 num_prompt_tokens（无 embeds 面）
        self.num_prompt_tokens = len(prompt_token_ids)
        # SOURCE: vllm/v1/request.py:L143-L148 output/all token 账
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = prompt_token_ids.copy()

        # SOURCE: vllm/v1/request.py:L173 num_computed_tokens
        self.num_computed_tokens = 0
        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        # SOURCE: vllm/v1/request.py:L159-L162 num_in_flight_tokens（账位保留——
        #   allocate_slots 的窗外回收用它扣在飞步）
        self.num_in_flight_tokens = 0
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

        # SOURCE: vllm/v1/request.py:L204 block_hashes——哈希账本（本章主角）
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

    # SOURCE: vllm/v1/request.py:L179-L183 只读视图（ConstantList 包装删，
    #   属性面保留——request_block_hasher 读 all_token_ids 切片）
    @property
    def all_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:L183（ConstantList 只读视图）
        # SUBTRACTED: ConstantList 防误写包装（L183——GC/只读优化，不进
        #   控制流；这里直接暴露底层列表）
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
