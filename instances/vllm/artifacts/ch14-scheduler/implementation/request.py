# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
import enum


# SOURCE: vllm/sampling_params.py（FinishReason 枚举，本章只用到这些成员）
class FinishReason(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


# SOURCE: vllm/v1/request.py:L310 (class RequestStatus)
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

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:L332
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> "FinishReason | None":
        # SOURCE: vllm/v1/request.py:L335
        return _FINISHED_REASON_MAP.get(status)


# SOURCE: vllm/v1/request.py:L339 (_FINISHED_REASON_MAP)
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}


# SOURCE: vllm/sampling_params.py (SamplingParams — 本章只用到停止相关字段)
class SamplingParams:
    def __init__(self, max_tokens=100, min_tokens=0, eos_token_id=None,
                 stop_token_ids=None):
        # SOURCE: vllm/sampling_params.py (SamplingParams 字段)
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids or []
        # SUBTRACTED: repetition_detection —— 重复检测参数；check_stop 里的重复
        # 分支属停止判定细节，精简版恒为 None 不触发，停止主线（EOS/stop_token/
        # length）完整。原 vllm/sampling_params.py RepetitionDetectionParams。
        self.repetition_detection = None


# SOURCE: vllm/v1/request.py:L40 (class Request) —— 只保留本章用到的状态字段
class Request:
    # SOURCE: vllm/v1/request.py:L40 (Request.__init__ — 只保留本章状态字段)
    def __init__(self, request_id, prompt_token_ids, max_tokens=100,
                 min_tokens=0, eos_token_id=None, stop_token_ids=None):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens, min_tokens=min_tokens,
            eos_token_id=eos_token_id, stop_token_ids=stop_token_ids,
        )
        # SOURCE: vllm/v1/request.py:L98
        self.stop_reason = None
        # SOURCE: vllm/v1/request.py:L109
        self.max_tokens = max_tokens
        self.status = RequestStatus.WAITING

        self._all_token_ids = list(prompt_token_ids)
        self._output_token_ids: list[int] = []

        # SOURCE: vllm/v1/request.py:L140
        self.num_output_placeholders = 0
        # SOURCE: vllm/v1/request.py:L144
        self.spec_token_ids: list[int] = []
        # SOURCE: vllm/v1/request.py:L145
        self.num_computed_tokens = 0
        # SOURCE: vllm/v1/request.py:L167
        self.num_preemptions = 0
        # SOURCE: vllm/v1/request.py:L181 (resumable — streaming-input 会话)
        # SUBTRACTED: streaming_queue / 会话续接状态 —— streaming-input 多轮是
        # 高级特性；精简版恒 resumable=False，停止即真完成。原 vllm/v1/request.py。
        self.resumable = False

        # AsyncScheduler 用：强制抢占下丢弃在途异步 token
        # SOURCE: vllm/v1/request.py (discard_latest_async_tokens)
        self.discard_latest_async_tokens = False

        # SUBTRACTED: pooling_params / structured_output_request / mm_features /
        # lora_request / client_index / trace_headers / block_hashes 等 —— 多模态/
        # 约束解码/LoRA/pooling/编码器输入与抢占·回流主线正交。原 vllm/v1/request.py:L40+。
        self.pooling_params = None

    # SOURCE: vllm/v1/request.py:L211 (append_output_token_ids)
    def append_output_token_ids(self, token_ids) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
        # SUBTRACTED: update_block_hashes() —— 前缀缓存块哈希更新，属 KV 缓存子系统，
        # 与停止检测正交。原 vllm/v1/request.py:L222。

    @property
    def output_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py (output_token_ids 视图)
        return self._output_token_ids

    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L234
        return len(self._all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L242
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:L266
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> "FinishReason | None":
        # SOURCE: vllm/v1/request.py:L269
        return RequestStatus.get_finished_reason(self.status)
