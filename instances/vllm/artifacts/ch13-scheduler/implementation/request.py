# SOURCE: vllm/v1/request.py
# 精简版 Request / RequestStatus / SamplingParams —— 只保留连续批处理 schedule()
# 所触及的标量与方法（num_computed_tokens / num_tokens_with_spec /
# num_output_placeholders / is_prefill_chunk / append_output_token_ids / 状态机）。
# 与真实 vllm/v1/request.py 同名同语义，只删与本章无关的字段（mm/lora/structured/
# block_hashes/events/prefill_stats 等）。
#
# SUBTRACTED: mm_features / lora_request / structured_output_request /
#   block_hashes / _block_hasher / record_event / prefill_stats /
#   prompt_embeds / discard_latest_async_tokens 等字段（原 vllm/v1/request.py:L60-L210）
#   —— 多模态/LoRA/约束解码/可观测性均属 dossier.delete 批准的独立子系统，
#   纯文本采样请求不触发，删后状态机与 token 计数完整自洽。
from __future__ import annotations

import enum


# SOURCE: vllm/v1/request.py (RequestStatus, 见 vllm/v1/request.py 末尾枚举定义)
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    # SUBTRACTED: WAITING_FOR_REMOTE_KVS / WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR /
    #   WAITING_FOR_STREAMING_REQ（KVConnector/约束解码/流式输入阻塞态，dossier.delete 批准）
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:RequestStatus.is_finished
        return status >= RequestStatus.FINISHED_STOPPED


# SOURCE: vllm/sampling_params.py:SamplingParams
class SamplingParams:
    # SUBTRACTED: 绝大多数采样字段（temperature/top_p/logprobs/...）—— 与连续批处理
    #   调度决策无关；这里只保留 check_stop 用到的停止判据字段。
    def __init__(
        self,
        max_tokens: int = 16,
        min_tokens: int = 0,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        # SOURCE: vllm/sampling_params.py:SamplingParams.__init__
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids or []
        self.repetition_detection = None  # SUBTRACTED: 重复检测细节


# SOURCE: vllm/v1/request.py:Request
class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        arrival_time: float = 0.0,
        priority: int = 0,
        client_index: int = 0,
    ) -> None:
        # SOURCE: vllm/v1/request.py:L96
        self.request_id = request_id
        self.status = RequestStatus.WAITING
        self.arrival_time = arrival_time
        self.priority = priority
        self.client_index = client_index

        self.prompt_token_ids = list(prompt_token_ids)
        # SOURCE: vllm/v1/request.py:L129
        self.num_prompt_tokens = len(self.prompt_token_ids)
        # SOURCE: vllm/v1/request.py:L136 —— _all_token_ids = prompt + output（按位置展开）
        self._all_token_ids: list[int] = list(self.prompt_token_ids)
        self._output_token_ids: list[int] = []

        self.sampling_params = sampling_params
        self.pooling_params = None  # SUBTRACTED: 池化请求（另章）

        # SOURCE: vllm/v1/request.py:L140 —— AsyncScheduler 占位计数
        self.num_output_placeholders = 0
        # SOURCE: vllm/v1/request.py:L144
        self.spec_token_ids: list[int] = []
        # SOURCE: vllm/v1/request.py:L145
        self.num_computed_tokens = 0
        # SOURCE: vllm/v1/request.py:L160
        self.is_prefill_chunk = False

        self.num_preemptions = 0
        self.stop_reason: int | str | None = None
        # SUBTRACTED: has_encoder_inputs 恒 False（纯文本，无 mm_features）
        self.has_encoder_inputs = False

    @property
    def max_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:Request.max_tokens
        return self.sampling_params.max_tokens

    @property
    def output_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:Request.output_token_ids
        return self._output_token_ids

    @property
    def all_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:Request.all_token_ids
        return self._all_token_ids

    # SOURCE: vllm/v1/request.py:L211
    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
        # SUBTRACTED: update_block_hashes()（前缀缓存哈希，KV cache 章）

    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L234
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        # SOURCE: vllm/v1/request.py:L238
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L242
        return len(self._output_token_ids)

    # SUBTRACTED: use_structured_output 恒 False（约束解码，dossier.delete 批准）
    use_structured_output = False
    # SUBTRACTED: lora_request / mm_features / prompt_embeds / prefill_stats 等

    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:Request.is_finished
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> "RequestStatus | None":
        # SOURCE: vllm/v1/request.py:Request.get_finished_reason
        return self.status if self.is_finished() else None

    def __repr__(self) -> str:
        # SOURCE: vllm/v1/request.py:Request.__repr__
        return (
            f"Request(id={self.request_id}, status={self.status.name}, "
            f"num_computed={self.num_computed_tokens}, "
            f"num_tokens_with_spec={self.num_tokens_with_spec}, "
            f"placeholders={self.num_output_placeholders})"
        )


# SOURCE: vllm/v1/core/sched/utils.py:L94 check_stop
def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params
    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # SUBTRACTED: repetition_detection 分支（重复检测，非调度骨架）
    return False
