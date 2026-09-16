# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/engine/exceptions.py —— 逐字承载（21 行全量，无删改）：
# WC3 错误层级（PR #49665）的引擎侧两异常——EngineGenerateError 请求级可
# 恢复 / EngineDeadError 全局不可恢复。
from vllm.exceptions import VLLMServerError


# SOURCE: vllm/v1/engine/exceptions.py:L6-L10
class EngineGenerateError(VLLMServerError):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


# SOURCE: vllm/v1/engine/exceptions.py:L12-L21
class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
