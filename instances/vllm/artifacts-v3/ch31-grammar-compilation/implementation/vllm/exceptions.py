# SOURCE: vllm/exceptions.py
# HOST SEAM：本章消费面只有 VLLMValidationError（sampling_params 校验抛出）。
# 真实文件另有完整的 vLLM 异常树（VLLMError→VLLMClientError/VLLMServerError→
# 数十个具体异常）——与结构化输出校验主控制流正交，只镜像本章用到的骨架。
# SUBTRACTED: SPDX 版权头；VLLMError/VLLMClientError 之外的异常族。
from typing import Any


# SOURCE: vllm/exceptions.py:L10-L13 VLLMError 基类骨架
class VLLMError(Exception):
    """Base class for errors raised by vLLM."""


# SOURCE: vllm/exceptions.py:L17-L21 VLLMClientError（4xx 侧基类）
class VLLMClientError(VLLMError):
    """Base class for errors caused by the client request (4xx)."""


# SOURCE: vllm/exceptions.py:L26-L52 VLLMValidationError —— 逐字
class VLLMValidationError(VLLMClientError):
    """vLLM-specific validation error for request validation failures.

    Args:
        message: The error message describing the validation failure.
        parameter: Optional parameter name that failed validation.
        value: Optional value that was rejected during validation.
    """

    # SOURCE: vllm/exceptions.py:L36-L44（VLLMValidationError.__init__）
    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value

    # SOURCE: vllm/exceptions.py:L45-L52（VLLMValidationError.__str__）
    def __str__(self):
        base = super().__str__()
        extras = []
        if self.parameter is not None:
            extras.append(f"parameter={self.parameter}")
        if self.value is not None:
            extras.append(f"value={self.value}")
        return f"{base} ({', '.join(extras)})" if extras else base
