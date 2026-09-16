# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/parser/metrics.py —— 逐字承载（108 行全量，无删改）：
# record_tool_parser_invocation 是 DelegatingParser 双路提取的观测埋点
# （未 init 时恒 no-op——精简环境不注册计数器，控制流不变）。
"""Prometheus metrics for the parsers."""

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import cast

from prometheus_client import REGISTRY, Counter

_model_name: str | None = None

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"
_tool_call_parser_invocations: Counter | None = None


# SOURCE: vllm/parser/metrics.py:L19-L22
class ToolCallOutcome(Enum):
    TOOL_CALL = "tool_call"
    NO_TOOL_CALL = "no_tool_call"


# SOURCE: vllm/parser/metrics.py:L24-L27
class RequestType(Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    OTHER = "other"


# SOURCE: vllm/parser/metrics.py:L30-L62
def init_parser_metrics(*, model_name: str) -> None:
    """Lazily register parser metrics and cache the shared model label."""
    global _model_name
    _model_name = model_name

    global _tool_call_parser_invocations
    try:
        _tool_call_parser_invocations = Counter(
            name=_TOOL_CALL_PARSER_INVOCATIONS_TOTAL,
            documentation=(
                "Total number of ToolParser invocations. "
                "Non-streaming increments once per choice; "
                "streaming increments once per delta."
            ),
            labelnames=["model_name", "mode", "outcome", "request_type"],
        )
    except ValueError:
        _tool_call_parser_invocations = cast(
            Counter,
            REGISTRY._names_to_collectors[_TOOL_CALL_PARSER_INVOCATIONS_TOTAL],
        )

    for mode, outcome, request_type in product(
        ("streaming", "non_streaming"),
        ToolCallOutcome,
        RequestType,
    ):
        _tool_call_parser_invocations.labels(
            model_name=_model_name,
            mode=mode,
            outcome=outcome.value,
            request_type=request_type.value,
        )


# SOURCE: vllm/parser/metrics.py:L65-L108（ResponsesRequest 的 match 位随
# delete[13] 删 Responses API 后不可达，保留 case 臂按真实原样——HOST
# SEAM 注：responses 分支在精简环境永不命中）
def record_tool_parser_invocation(
    *,
    is_tool_called: bool | Exception,
    is_streaming: bool,
    request: object,
) -> None:
    """Increment the tool-call parser invocation counter when registered.
    Currently parser failures are treated as no tool calls.

    TODO: To accurately track parser failures, add a new ToolCallOutcome and
    more importantly, ensure exceptions are propagated out of the ToolParsers
    instead of being caught internally. This would require going through
    ToolParser implementation on a case-by-case basis.
    """
    if _tool_call_parser_invocations is None:
        return

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    match request:
        case ChatCompletionRequest():
            request_type = RequestType.CHAT_COMPLETIONS
        case _:
            request_type = RequestType.OTHER

    match is_tool_called:
        case bool():
            outcome = (
                ToolCallOutcome.TOOL_CALL
                if is_tool_called
                else ToolCallOutcome.NO_TOOL_CALL
            )
        case _:
            outcome = ToolCallOutcome.NO_TOOL_CALL

    _tool_call_parser_invocations.labels(
        model_name=_model_name,
        mode="streaming" if is_streaming else "non_streaming",
        outcome=outcome.value,
        request_type=request_type.value,
    ).inc()
