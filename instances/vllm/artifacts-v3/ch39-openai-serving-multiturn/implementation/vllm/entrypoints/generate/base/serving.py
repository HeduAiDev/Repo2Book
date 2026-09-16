# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/generate/base/serving.py —— 忠实承载：
# GenerateBaseServing（OpenAIServingChat 的基类）与 _raise_if_error /
# create_streaming_error_response / _convert_generation_error_to_streaming_
# response / _get_data_parallel_rank / _get_decoded_token / clamp_prompt_
# logprobs 逐字。删：BeamSearchOnlineMixin 基类（delete[0]）、build_per_
# request_timing_metrics 与 metrics 消费（delete[3]）、_with_kv_transfer_
# rejection_cleanup（delete[11]）、tracing 的 is_tracing_enabled 真链（退化）。
import json
import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import ClassVar, Generic, TypeVar

from fastapi import Request
from pydantic import ConfigDict
from starlette.datastructures import Headers

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)

# SUBTRACTED: vllm/entrypoints/generate/base/serving.py:L15 BeamSearchOnlineMixin
# 导入——delete[0]（beam_search 全家：GenerateBaseServing 的 Mixin 基类与
# serving.py 的 self.beam_search 调用点一并删）。
# SUBTRACTED: vllm/entrypoints/generate/base/serving.py:L17-L18 CompletionRequest
# /ResponsesRequest 导入——completion 面（m18 同构省略）与 delete[13]。
# SUBTRACTED: vllm/entrypoints/generate/base/serving.py:L38 RequestStateStats
# 导入与 build_per_request_timing_metrics（L46-L98）——delete[3] per_request
# metrics。

logger = init_logger(__name__)

RequestT = TypeVar("RequestT", bound=AnyRequest)
_T = TypeVar("_T")


# SUBTRACTED: vllm/entrypoints/generate/base/serving.py:L46-L98
# build_per_request_timing_metrics——delete[3]（计时指标）。


# SOURCE: vllm/entrypoints/generate/base/serving.py:L101-L110 —— ServeContext 逐字
@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_inputs: list[EngineInput] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


# SOURCE: vllm/entrypoints/generate/base/serving.py:L113-L151 —— GenerateBaseServing
# （基类面：BeamSearchOnlineMixin 删——delete[0]）
class GenerateBaseServing(BaseServing):
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID.
    """

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L118-L133
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(
            models=models,
            model_config=engine_client.model_config,
            request_logger=request_logger,
        )

        self.engine_client = engine_client
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        vllm_config = getattr(engine_client, "vllm_config", None)
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        self.has_kv_connector = kv_transfer_config is not None

        # Computed once at startup (cached by ``vllm_config`` identity) and
        # stamped on non-streaming responses. Streaming chunks deliberately
        # omit it to avoid per-chunk overhead.
        from vllm.entrypoints.serve.utils.fingerprint import get_system_fingerprint

        try:
            self.system_fingerprint: str | None = get_system_fingerprint(
                engine_client.vllm_config
            )
        except Exception:
            # Never fail server startup over the fingerprint.
            self.system_fingerprint = None

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L153-L168 —— create_streaming_error_response 逐字
    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L170-L177 —— _raise_if_error 逐字
    # （must_keep：流中 finish_reason=='error' 抛 GenerationError）
    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            logger.error(
                "Request %s failed with an internal error during generation",
                request_id,
            )
            raise GenerationError("Internal server error")

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L179-L187 —— _convert_generation_error_to_streaming_response 逐字
    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        """Convert GenerationError to streaming error response."""
        return self.create_streaming_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L189-L201 —— _get_trace_headers
    # （HOST SEAM：真实经 engine_client.is_tracing_enabled 查询引擎 OTel 面；
    # 精简环境引擎恒未启用 tracing——与真实关闭路径行为一致）
    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L203-L216 —— _get_data_parallel_rank 逐字
    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
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

    # SUBTRACTED: vllm/entrypoints/generate/base/serving.py:L218-L250
    # _with_kv_transfer_rejection_cleanup——delete[11]（P/D 分离 KV-transfer
    # 拒绝清理；无 kv_transfer_params 的普通请求直接 return await awaitable）。

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L252-L270 —— _get_decoded_token 逐字
    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return format_token_id_placeholder(token_id)

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])


# SOURCE: vllm/entrypoints/generate/base/serving.py:L273-L274 —— format_token_id_placeholder 逐字
def format_token_id_placeholder(token_id: int) -> str:
    return f"token_id:{token_id}"


# SOURCE: vllm/entrypoints/generate/base/serving.py:L277-L302 —— resolve_token_id_placeholder
# 逐字（tokenizer.convert_ids_to_tokens 消费面）
def resolve_token_id_placeholder(
    token: str, tokenizer: TokenizerLike
) -> tuple[str, list[int] | None]:
    """Decode a 'token_id:N' placeholder back to a token string and UTF-8 bytes.

    Returns (token, None) unchanged if token is not a placeholder.
    This is the inverse of format_token_id_placeholder / _get_decoded_token
    when return_as_token_id=True.
    """
    suffix = token.removeprefix("token_id:")
    if suffix == token:
        return token, None
    try:
        token_id = int(suffix)
    except ValueError:
        return token, None
    token_repr = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token_repr is None:
        logger.warning_once(
            "resolve_token_id_placeholder: token_id %d has no vocab entry; "
            "substituting empty string",
            token_id,
        )
        return "", None
    token_str = tokenizer.convert_tokens_to_string([token_repr])
    return token_str, list(token_str.encode("utf-8", errors="replace"))


# SOURCE: vllm/entrypoints/generate/base/serving.py:L305-L317 —— clamp_prompt_logprobs 逐字
def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
