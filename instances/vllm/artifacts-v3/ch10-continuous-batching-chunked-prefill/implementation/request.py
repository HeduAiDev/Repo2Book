# SOURCE: vllm/v1/request.py
# 精简版 Request / RequestStatus / SamplingParams —— 只保留 schedule() 账本
# 触及的标量与计数器：num_computed_tokens / num_tokens / num_tokens_with_spec /
# num_output_placeholders / num_in_flight_tokens / is_prefill_chunk / spec_token_ids
# 及 ⑤ 拍回填用的 append_output_token_ids。与真实 vllm/v1/request.py 同名同语义，
# 删除项全部 dossier.subtraction_plan.delete 批准（mm/lora/structured/streaming/
# 可观测性/前缀哈希细节归邻章）。
from __future__ import annotations

import enum
import time


# SOURCE: vllm/v1/request.py:L348 RequestStatus
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

    def __str__(self) -> str:
        # SOURCE: vllm/v1/request.py:L366-L367 __str__
        return self.name

    # SOURCE: vllm/v1/request.py:L369-L371 is_finished
    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:L370-L371
        return status > RequestStatus.PREEMPTED

    # SUBTRACTED: get_finished_reason + _FINISHED_REASON_MAP（L373-L390）——
    #   FinishReason 属 EngineCore 输出侧（ch7 话头），本章调度账本用不到。
    # 保留全部 WAITING_FOR_* 阻塞态成员：_is_blocked_waiting_status 保留原判
    # （dossier.delete 第 13 条明示「对已删状态无副作用」）。


# SOURCE: vllm/sampling_params.py SamplingParams
class SamplingParams:
    # SUBTRACTED: 绝大多数采样字段（temperature/top_p/logprobs/n_gram/...）——
    #   与调度决策无关；只保留本章路径用到的 max_tokens（Request.max_tokens 来源）。
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


# SOURCE: vllm/v1/request.py:L59 Request
class Request:
    # SOURCE: vllm/v1/request.py:L60-L79 __init__ 签名
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params=None,
        arrival_time: float | None = None,
        priority: int = 0,
    ) -> None:
        # SUBTRACTED: prompt_embeds/prompt_is_token_ids/mm_features/lora_request/
        #   cache_salt/trace_headers/block_hasher/resumable/reasoning_*/abort_
        #   immediately 参数与对应字段（mm/lora/流式/前缀哈希细节，
        #   dossier.delete 批准——分别归 ch6/ch30/ch11/ch15 话头）。
        # SOURCE: vllm/v1/request.py:L81-L95 基本字段
        self.request_id = request_id
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # SOURCE: vllm/v1/request.py:L97
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
        # SUBTRACTED: structured_output_request 建档与 WAITING_FOR_STRUCTURED_
        #   OUTPUT_GRAMMAR 初始态（L87-L94/L113-L114，structured，dossier.delete 批准）。

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

        # SOURCE: vllm/v1/request.py:L150-L173 计数器字段群
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

        # V2+PP+async: Enforces `pp_size` cadence between same-request decode steps
        # so the worker's broadcast slot ring stays consistent.
        self.next_decode_eligible_step = 0

        # Seq of the most recent step this request was scheduled in; fences
        # deferred block freeing (see Scheduler._free_request_blocks).
        self.last_sched_seq = 0

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0

        # SOURCE: vllm/v1/request.py:L187-L200 chunked prefill / 抢占计数
        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # Block-aligned token position of a proven shared prefix worth pinning
        # in the (sparse) prefix cache; 0 means none. Set at admission for
        # hybrid/Mamba models when a shared prefix is detected (Marconi-style).
        self.shared_prefix_boundary = 0

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # SUBTRACTED: mm_features（L177）/ConstantList 只读视图包装（L182-L183，
        #   vllm/v1/utils.py ConstantList——防直改的包装层，本章以裸 list 暴露
        #   同名属性）/block_hashes+update_block_hashes（L204-L209/L262-L265，
        #   前缀缓存链式哈希归 ch15）/prefill_stats（L202，可观测性）/events
        #   record_event（L98，可观测性）/kv_transfer_params（L101-L104，connector）/
        #   streaming_queue+resumable（L213-L216，流式）/__lt__（PRIORITY 堆序，
        #   dossier.delete 批准）。

    # SOURCE: vllm/v1/request.py:L249 append_output_token_ids
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
        # SUBTRACTED: update_block_hashes()（前缀缓存哈希，ch15）。

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
        # SUBTRACTED: ConstantList 只读视图——裸 list 同名暴露。
        return self._output_token_ids

    # SOURCE: vllm/v1/request.py:L183 all_token_ids 只读视图
    @property
    def all_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:L183
        # SUBTRACTED: ConstantList 只读视图——裸 list 同名暴露。
        return self._all_token_ids

    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py Request.is_finished
        return RequestStatus.is_finished(self.status)
