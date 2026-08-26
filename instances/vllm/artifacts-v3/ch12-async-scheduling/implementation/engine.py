# SOURCE: vllm/v1/engine/__init__.py
# 输出三件套：FinishReason / EngineCoreOutput / EngineCoreOutputs（完整字段版
# 已在 ch9/Part II 立过，此处四字段版）。EngineCore 本体在 core.py（对应真实
# vllm/v1/engine/core.py 的模块划分）。
from __future__ import annotations

import enum
from dataclasses import dataclass, field


# SOURCE: vllm/v1/engine/__init__.py:L43 FinishReason
class FinishReason(enum.IntEnum):
    # SOURCE: vllm/v1/engine/__init__.py:L44-L62 docstring（精简）
    """Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.
    """

    # SOURCE: vllm/v1/engine/__init__.py:L58-L62
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


# SOURCE: vllm/v1/engine/__init__.py:L184 EngineCoreOutput
@dataclass
class EngineCoreOutput:
    # SUBTRACTED: msgspec.Struct 装配与 logprobs/pooling_output/events/
    #   kv_transfer_params/routed_experts 等字段——观测/connector 面，
    #   完整版归 ch9；四字段承载本章可观察行为。
    # SOURCE: vllm/v1/engine/__init__.py:L190-L191
    request_id: str
    new_token_ids: list[int]
    # SOURCE: vllm/v1/engine/__init__.py:L198-L199
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None

    # SOURCE: vllm/v1/engine/__init__.py:L213-L215 finished
    @property
    def finished(self) -> bool:
        # SOURCE: vllm/v1/engine/__init__.py:L214-L215
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L230 EngineCoreOutputs
@dataclass
class EngineCoreOutputs:
    # SUBTRACTED: scheduler_stats/timestamp/finished_requests 等（观测/DP 面）。
    # SOURCE: vllm/v1/engine/__init__.py:L241-L242 [num_reqs]
    outputs: list[EngineCoreOutput] = field(default_factory=list)

# SUBTRACTED: EngineCore 本体随装配切面移至同目录 core.py（对应真实
# vllm/v1/engine/core.py 的模块划分——engine/__init__ 只持输出件）。
