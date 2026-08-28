# SOURCE: vllm/v1/sample/logits_processor/__init__.py
# 本章消费面：BatchUpdateBuilder / LogitsProcessors 的 re-export（真实在
# L34-L35 从 state 与 interface re-export）+ build_logitsprocs（runner 装配
# InputBatch 时的调用点 gpu_model_runner.py:L716-L722）。
# SUBTRACTED: BUILTIN_LOGITS_PROCESSORS 装配链（MinTokens/LogitBias/MinP——
#   ch30 域）、_load_custom_logitsprocs 插件装载、validate_logits_processors_
#   parameters、AdapterLogitsProcessor。
from __future__ import annotations

from .interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from .state import BatchUpdateBuilder, LogitsProcessors

__all__ = [
    "BatchUpdate",
    "BatchUpdateBuilder",
    "LogitsProcessor",
    "LogitsProcessors",
    "MoveDirectionality",
    "build_logitsprocs",
]


# SOURCE: vllm/v1/sample/logits_processor/__init__.py:L185-L215 build_logitsprocs
#   —— 签名逐字；函数体压缩为空集支：pooling 模型返回空 LogitsProcessors()
#   是真实代码 L192-L199 的原生返回值；spec/builtin 链（L201-L215）属 ch30 域，
#   本章精简配置（无 pooling、无 spec、无自定义 processor）下真实返回值同为
#   只含 BUILTIN 处理器的集合——BUILTIN 的构造与行为归 ch30，此处以空集承载
#   同一调用面。
def build_logitsprocs(
    vllm_config: object,
    device: object,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs=(),
) -> LogitsProcessors:
    # SOURCE: vllm/v1/sample/logits_processor/__init__.py:L192-L199（pooling
    #   短路支逐字语义）
    if is_pooling_model:
        return LogitsProcessors()
    # SUBTRACTED: speculative 拒绝支（L201-L207）与 BUILTIN+custom 装配链
    #   （L209-L215——MinTokens/LogitBias/MinP 处理器，ch30 域）。
    return LogitsProcessors()
