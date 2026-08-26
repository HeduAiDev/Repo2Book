# SOURCE: vllm/v1/engine/__init__.py
# EngineCore 侧输出三件套：FinishReason（终态→对外原因的映射目标）、
# EngineCoreOutput（单请求一拍输出——dossier.delete 第 10 条批准精简为
# request_id/new_token_ids/finish_reason/stop_reason 四字段）、EngineCoreOutputs
# （按 client_index 分桶的整拍打包）。真实位于 vllm/v1/engine/__init__.py
# （msgspec.Struct 序列化——跨进程序列化归 ch5 话头），精简版换 dataclass
# 承载同名字段；完整字段版已在 ch9/Part II 立过。
from __future__ import annotations

import enum
from dataclasses import dataclass, field


# SOURCE: vllm/v1/engine/__init__.py:L43 FinishReason
class FinishReason(enum.IntEnum):
    # SOURCE: vllm/v1/engine/__init__.py:L44-L56 docstring
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """

    # SOURCE: vllm/v1/engine/__init__.py:L58-L62
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    # SUBTRACTED: __str__ → FINISH_REASON_STRINGS（L64-L65，前端展示层）。


# SOURCE: vllm/v1/engine/__init__.py:L184 EngineCoreOutput
@dataclass
class EngineCoreOutput:
    # SUBTRACTED: msgspec.Struct 装配（array_like/omit_defaults/gc）与
    #   new_logprobs/new_prompt_logprobs_tensors/pooling_output/events/
    #   kv_transfer_params/ec_transfer_params/trace_headers/prefill_stats/
    #   routed_experts/num_nans_in_logits 字段（L193-L211）——dossier.delete
    #   第 1/2/10 条批准（connector/encoder/观测统计）；完整字段归 Part II/ch9。
    # SOURCE: vllm/v1/engine/__init__.py:L190-L191
    request_id: str
    new_token_ids: list[int]
    # SOURCE: vllm/v1/engine/__init__.py:L198-L199
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None

    @property
    def finished(self) -> bool:
        # SOURCE: vllm/v1/engine/__init__.py:L213-L215
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L230 EngineCoreOutputs
@dataclass
class EngineCoreOutputs:
    # SUBTRACTED: engine_index/scheduler_stats/timestamp/utility_output/
    #   finished_requests/wave_complete/start_wave（L239-L258）——DP 波次与
    #   finished_req_ids_dict 分桶（V2 细节）均 dossier.delete 第 10/11 条批准。
    # SOURCE: vllm/v1/engine/__init__.py:L241-L242
    # [num_reqs]
    outputs: list[EngineCoreOutput] = field(default_factory=list)
