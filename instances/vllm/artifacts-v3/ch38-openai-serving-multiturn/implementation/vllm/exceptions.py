# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/exceptions.py —— 逐字承载（118 行全量，无删改）：错误分层
# 树是本章 WC3「错误层级（PR #49665）：请求级 VLLMClientError 不上抛为引擎
# 死；EngineDeadError/EngineGenerateError 全局」的载体。
from typing import Any


# SOURCE: vllm/exceptions.py:L8
class VLLMError(Exception):
    """Base class for all vLLM-specific errors.

    Subclasses are split into `VLLMClientError` (caused by the request, mapped
    to 4xx) and `VLLMServerError` (caused by the server, mapped to 5xx).
    Dispatching on this hierarchy lets the entrypoints decide the HTTP status
    without relying on raw Python exception types such as `ValueError`.
    """


# SOURCE: vllm/exceptions.py:L19
class VLLMClientError(VLLMError):
    """Base class for errors caused by the client request (4xx)."""


# SOURCE: vllm/exceptions.py:L24
class VLLMServerError(VLLMError):
    """Base class for errors caused by the server (5xx)."""


# SOURCE: vllm/exceptions.py:L29-L61
class VLLMValidationError(VLLMClientError):
    """vLLM-specific validation error for request validation failures.

    Args:
        message: The message describing the validation failure.
        parameter: Optional parameter name that failed validation.
        value: Optional value that was rejected during validation.
    """

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

    def __str__(self):
        base = super().__str__()
        extras = []
        if self.parameter is not None:
            extras.append(f"parameter={self.parameter}")
        if self.value is not None:
            extras.append(f"value={self.value}")
        return f"{base} ({', '.join(extras)})" if extras else base


# SOURCE: vllm/exceptions.py:L64-L66
class VLLMNotFoundError(VLLMClientError):
    """vLLM-specific NotFoundError"""

    pass


# SOURCE: vllm/exceptions.py:L69-L88
class LoRAAdapterNotFoundError(VLLMNotFoundError):
    """Exception raised when a LoRA adapter is not found.

    This exception is thrown when a requested LoRA adapter does not exist
    in the system.
    """

    message: str

    def __init__(
        self,
        lora_name: str,
        lora_path: str,
    ) -> None:
        message = f"Loading lora {lora_name} failed: No adapter found for {lora_path}"
        self.message = message

    def __str__(self):
        return self.message


# SOURCE: vllm/exceptions.py:L91-L117
class VLLMUnprocessableEntityError(VLLMClientError):
    """vLLM-specific error for unprocessable entity requests.

    This exception is raised when the request content is invalid or cannot be
    processed, such as when an image URL points to a non-existent or inaccessible
    resource (404, 403, DNS failure, etc.).
    """

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

    def __str__(self):
        base = super().__str__()
        extras = []
        if self.parameter is not None:
            extras.append(f"parameter={self.parameter}")
        if self.value is not None:
            extras.append(f"value={self.value}")
        return f"{base} ({', '.join(extras)})" if extras else base


# SUBTRACTED: vllm/exceptions.py:L118 之后的平台特定异常族（NPU/HPU 等
# accelerator 检查）——本章服务面不消费。
