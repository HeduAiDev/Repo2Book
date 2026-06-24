"""OpenAIServing 基类 —— 所有 OpenAI handler 的公共底座（精简版，只做减法）。

与真实 vllm/entrypoints/openai/engine/serving.py 同名、同结构、同控制流。本章主线保留：
  * __init__：持有 engine_client / models / renderer / input_processor + 算一次 system_fingerprint。
  * _check_model：模型名/LoRA 校验，404。
  * _base_request_id：X-Request-Id 头优先，否则 random_uuid。
  * create_error_response / create_streaming_error_response：统一错误工厂。
  * _raise_if_error：finish_reason=='error' → GenerationError。
  * _maybe_get_adapters / _is_model_supported / _get_data_parallel_rank。

把所有 # SUBTRACTED 删回去 ≈ 真实基类在 chat 主线上的样子。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from messages import ErrorResponse, GenerationError, _ErrorBody

# SOURCE: vllm/entrypoints/serve/render/_renderer 引入的占位 —— Request 由 _framework 提供
from _framework import Request


# SUBTRACTED: get_system_fingerprint 真实实现读 vllm_config 算 hash；精简版给确定占位。
#   原 vllm/entrypoints/openai/fingerprint.py:get_system_fingerprint。
def get_system_fingerprint(vllm_config: Any) -> str:
    # SOURCE: vllm/entrypoints/openai/fingerprint.py:get_system_fingerprint —— stub
    return "fp_companion"


def random_uuid() -> str:
    # SOURCE: vllm/utils/__init__.py:random_uuid
    return str(uuid.uuid4())


def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> ErrorResponse:
    # SOURCE: vllm/entrypoints/openai/engine/protocol.py:create_error_response（模块级）
    return ErrorResponse(
        error=_ErrorBody(
            message=str(message),
            type=err_type,
            param=param,
            code=status_code.value,
        )
    )


class OpenAIServingModels:
    """模型注册表的极小替身：base 模型名 + lora_requests 映射。

    # SUBTRACTED: 真实 OpenAIServingModels 还管 init_static_loras / 运行时 LoRA 加载 /
    #   resolve_lora 等（与 chat 主线正交）。原 vllm/entrypoints/openai/models/serving.py。
    """

    # SOURCE: vllm/entrypoints/openai/models/serving.py:OpenAIServingModels —— 减法版
    def __init__(self, base_model_names: list[str] | None = None,
                 lora_requests: dict[str, Any] | None = None):
        # SOURCE: OpenAIServingModels.__init__ —— 减法版
        self.base_model_names = base_model_names or ["companion-model"]
        self.lora_requests = lora_requests or {}

    def is_base_model(self, model_name: str | None) -> bool:
        # SOURCE: vllm/entrypoints/openai/models/serving.py:is_base_model
        return model_name in self.base_model_names

    def model_name(self, lora_request: Any = None) -> str:
        # SOURCE: vllm/entrypoints/openai/models/serving.py:model_name
        return self.base_model_names[0]


class OpenAIServing:
    """所有 OpenAI handler 的基类。"""

    # SOURCE: vllm/entrypoints/openai/engine/serving.py:L135 OpenAIServing
    def __init__(
        self,
        engine_client,
        models: OpenAIServingModels,
        *,
        request_logger=None,
        return_tokens_as_token_ids: bool = False,
    ):
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L140 OpenAIServing.__init__
        self.engine_client = engine_client
        self.models = models
        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids

        self.model_config = engine_client.model_config
        self.renderer = engine_client.renderer
        self.input_processor = getattr(engine_client, "input_processor", None)

        # Computed once at startup; stamped on non-streaming responses.
        # Streaming chunks deliberately omit it to avoid per-chunk overhead.
        try:
            self.system_fingerprint: str | None = get_system_fingerprint(
                engine_client.vllm_config
            )
        except Exception:
            # Never fail server startup over the fingerprint.
            self.system_fingerprint = None

    # SUBTRACTED: beam_search 方法（自成一套多 beam 调度，dossier delete 批准）。
    #   原 vllm/entrypoints/openai/engine/serving.py:L173-L370。

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L372 create_error_response
        return create_error_response(message, err_type, status_code, param)

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L381 create_streaming_error_response
        json_str = json.dumps(
            self.create_error_response(
                message=message, err_type=err_type,
                status_code=status_code, param=param,
            ).model_dump()
        )
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L398 _raise_if_error
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            raise GenerationError("Internal server error")

    async def _check_model(self, request) -> ErrorResponse | None:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L417 _check_model
        error_response = None

        if self._is_model_supported(request.model):
            return None
        if request.model in self.models.lora_requests:
            return None
        # SUBTRACTED: 运行时 LoRA 加载分支（VLLM_ALLOW_RUNTIME_LORA_UPDATING），dossier
        #   标注非主线。原 vllm/entrypoints/openai/engine/serving.py:L425-L443。

        return error_response or self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    def _maybe_get_adapters(self, request, supports_default_mm_loras: bool = False):
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L470 _maybe_get_adapters
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]
        # SUBTRACTED: default_mm_loras 多模态默认 LoRA 匹配分支。
        #   原 vllm/entrypoints/openai/engine/serving.py:L478-L483。
        if self._is_model_supported(request.model):
            return None
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L592 _base_request_id
        """Pulls the request id to use from a header, if provided"""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L605 _get_data_parallel_rank
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None
        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None
        try:
            return int(rank_str)
        except ValueError:
            return None

    def _is_model_supported(self, model_name: str | None) -> bool:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:L755 _is_model_supported
        if not model_name:
            return True
        # SUBTRACTED: VLLM_SKIP_MODEL_NAME_VALIDATION 短路分支（部署开关）。
        return self.models.is_base_model(model_name)

    def get_chat_request_role(self, request) -> str:
        # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:get_chat_request_role
        #   —— 简化：assistant（真实版会按 add_generation_prompt 决定）
        return "assistant"

    def _log_inputs(self, *args, **kwargs) -> None:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:_log_inputs
        # SUBTRACTED: 请求日志细节（RequestLogger）非主线。
        pass

    async def _get_trace_headers(self, headers) -> Mapping[str, str] | None:
        # SOURCE: vllm/entrypoints/openai/engine/serving.py:_get_trace_headers
        # SUBTRACTED: OTel trace context 提取非主线。
        return None
