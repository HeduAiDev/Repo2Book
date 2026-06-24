"""协议与数据模型 —— EngineClient 协议 + OpenAI 响应模型（精简版，只做减法）。

汇集本章主线触及的几类对象，与真实 vLLM 同名、同字段语义：
  * EngineClient        ← vllm/engine/protocol.py：handler 依赖的引擎协议（generate/errored/shutdown...）。
  * 响应模型            ← vllm/entrypoints/openai/chat_completion/protocol.py 等：
                          ChatCompletionResponse / ChatCompletionStreamResponse /
                          DeltaMessage / ChatMessage / UsageInfo / ErrorResponse。
  * RequestOutput       ← vllm/outputs.py：engine_client.generate 异步生成器逐个产出的对象。
  * SamplingParams      ← vllm/sampling_params.py。

真实模型是 pydantic BaseModel；精简版用 dataclass + model_dump/model_dump_json，
保留 vLLM 序列化 SSE/JSON 时实际调用的 model_dump(_json) 接口语义。
# SUBTRACTED: pydantic.BaseModel 校验/序列化引擎，用 dataclasses + json 复刻其 model_dump(_json)。
#   原 vllm/entrypoints/openai/chat_completion/protocol.py 各 class(... BaseModel)。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from typing import Any


# --- 异常（vllm/v1/engine/exceptions.py & vllm/engine/protocol.py） ---

class EngineGenerateError(Exception):
    # SOURCE: vllm/v1/engine/exceptions.py:EngineGenerateError
    pass


class EngineDeadError(Exception):
    # SOURCE: vllm/v1/engine/exceptions.py:EngineDeadError
    pass


class GenerationError(Exception):
    # SOURCE: vllm/v1/engine/exceptions.py:GenerationError —— finish_reason=='error' 时抛出
    pass


# --- 引擎产出对象（vllm/outputs.py） ---

@dataclass
class CompletionOutput:
    # SOURCE: vllm/outputs.py:CompletionOutput —— 单个候选输出（增量语义下逐步累积）
    index: int = 0
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    stop_reason: str | None = None


@dataclass
class RequestOutput:
    # SOURCE: vllm/outputs.py:RequestOutput —— generate 异步生成器逐个 yield 的对象
    request_id: str = ""
    prompt_token_ids: list[int] | None = None
    outputs: list[CompletionOutput] = field(default_factory=list)
    num_cached_tokens: int = 0
    finished: bool = False
    # SUBTRACTED: prompt_logprobs / kv_transfer_params 等字段 —— logprobs 装配在 ch10，
    #   PD 在 ch29/30；本章不解读其计算。原 vllm/outputs.py:RequestOutput。


# --- 采样参数（vllm/sampling_params.py） ---

class RequestOutputKind:
    # SOURCE: vllm/sampling_params.py:RequestOutputKind
    CUMULATIVE = 0   # 流式：每步推增量
    FINAL_ONLY = 2   # 非流式：只在末尾给最终结果


@dataclass
class SamplingParams:
    # SOURCE: vllm/sampling_params.py:SamplingParams
    max_tokens: int | None = 16
    output_kind: int = RequestOutputKind.CUMULATIVE
    n: int = 1


# --- OpenAI 响应模型（vllm/entrypoints/openai/chat_completion/protocol.py 等） ---

class _Model:
    """pydantic BaseModel 的极小替身：提供 model_dump / model_dump_json。"""

    # SOURCE: pydantic.BaseModel（dataclass 替身）—— stub
    def model_dump(self, exclude_unset: bool = False, exclude_none: bool = False) -> dict:
        # SOURCE: pydantic.BaseModel.model_dump —— stub
        def keep(v):
            return not (exclude_none and v is None)
        return {k: v for k, v in asdict(self).items() if keep(v)}

    def model_dump_json(self, exclude_unset: bool = False,
                        exclude_none: bool = False) -> str:
        # SOURCE: pydantic.BaseModel.model_dump_json —— stub
        return json.dumps(self.model_dump(exclude_unset, exclude_none))


@dataclass
class UsageInfo(_Model):
    # SOURCE: vllm/entrypoints/openai/protocol.py:UsageInfo
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class DeltaMessage(_Model):
    # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:DeltaMessage
    #   —— 流式增量载体：首块带 role，后续块带 content/tool_calls 增量
    role: str | None = None
    content: str | None = None
    tool_calls: list[Any] = field(default_factory=list)


@dataclass
class ChatMessage(_Model):
    # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:ChatMessage
    #   —— 非流式整条消息
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[Any] = field(default_factory=list)


@dataclass
class ChatCompletionResponseStreamChoice(_Model):
    # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:ChatCompletionResponseStreamChoice
    index: int = 0
    delta: DeltaMessage | None = None
    finish_reason: str | None = None
    logprobs: Any = None


@dataclass
class ChatCompletionResponseChoice(_Model):
    # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:ChatCompletionResponseChoice
    index: int = 0
    message: ChatMessage | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    logprobs: Any = None


@dataclass
class ChatCompletionStreamResponse(_Model):
    # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:ChatCompletionStreamResponse
    #   —— 流式 chunk（object='chat.completion.chunk'）
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionResponseStreamChoice] = field(default_factory=list)
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


@dataclass
class ChatCompletionResponse(_Model):
    # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:ChatCompletionResponse
    #   —— 非流式一次性 JSON 响应（object='chat.completion'）
    id: str = ""
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionResponseChoice] = field(default_factory=list)
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


@dataclass
class _ErrorBody:
    # SOURCE: vllm/entrypoints/openai/engine/protocol.py:ErrorInfo
    message: str = ""
    type: str = "BadRequestError"
    param: str | None = None
    code: int = HTTPStatus.BAD_REQUEST.value


@dataclass
class ErrorResponse(_Model):
    # SOURCE: vllm/entrypoints/openai/engine/protocol.py:ErrorResponse
    error: _ErrorBody = field(default_factory=_ErrorBody)

    def model_dump(self, exclude_unset: bool = False, exclude_none: bool = False) -> dict:
        # SOURCE: ErrorResponse.model_dump —— 嵌套 error 体
        return {"error": asdict(self.error)}


# --- 引擎协议（vllm/engine/protocol.py） ---

class EngineClient(ABC):
    """handler 依赖的引擎协议。AsyncLLM（ch04）是其 v1 实现。

    本章只用到：generate(异步生成器) / errored / dead_error / is_running /
    renderer / model_config / vllm_config / shutdown / get_supported_tasks / do_log_stats。
    """

    # SOURCE: vllm/engine/protocol.py:EngineClient
    model_config: Any
    renderer: Any
    vllm_config: Any

    @property
    @abstractmethod
    def errored(self) -> bool:
        # SOURCE: vllm/engine/protocol.py:EngineClient.errored
        ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException:
        # SOURCE: vllm/engine/protocol.py:EngineClient.dead_error
        ...

    @property
    def is_running(self) -> bool:
        # SOURCE: vllm/engine/protocol.py:EngineClient.is_running
        return not self.errored

    @abstractmethod
    def generate(
        self,
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        lora_request: Any = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        # SOURCE: vllm/engine/protocol.py:EngineClient.generate
        """Generate outputs for a request."""
        ...

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        # SOURCE: vllm/engine/protocol.py / vllm/v1/engine/async_llm.py:shutdown
        ...

    async def get_supported_tasks(self):
        # SOURCE: vllm/engine/protocol.py:EngineClient.get_supported_tasks
        return ("generate",)

    async def do_log_stats(self) -> None:
        # SOURCE: vllm/engine/protocol.py:EngineClient.do_log_stats
        ...
