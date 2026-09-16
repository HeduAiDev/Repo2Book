# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/openai/chat_completion/api_router.py —— 忠实承载
# （站 1，m2）：POST /v1/chat/completions 路由 + @with_cancellation + 三分支
# 返回逐字；batch 端点（create_batch_chat_completion，dossier「批量面，略」）
# 与 @load_aware_call / metrics_header（orca 负载追踪，elide「与主线正交，
# 可整段省」）删。


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# SUBTRACTED: vllm/entrypoints/openai/chat_completion/api_router.py:L10
# OpenAIServingChatBatch 导入与 L12-L13 BatchChatCompletionRequest——
# dossier：batch 端点「批量面，略」。
# SUBTRACTED: vllm/entrypoints/openai/chat_completion/api_router.py:L19-L23
# load_aware_call 导入与 L23 orca_metrics 导入——elide：orca 负载追踪侧路。

router = APIRouter()
# SUBTRACTED: vllm/entrypoints/openai/chat_completion/api_router.py:L29
# ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL——orca 负载回执头（elide）。


# SOURCE: vllm/entrypoints/openai/chat_completion/api_router.py:L32-L33 —— chat 位
def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


# SUBTRACTED: vllm/entrypoints/openai/chat_completion/api_router.py:L36-L37
# batch_chat——batch 面（略）。


# SOURCE: vllm/entrypoints/openai/chat_completion/api_router.py:L40-L74 —— 路由与
# handler 逐字（must_keep：站 1——@with_cancellation 断连竞速自第一行起生效；
# 返回值三分支 ErrorResponse→JSONResponse(错误码)/ChatCompletionResponse→
# JSONResponse/异步生成器→StreamingResponse(text/event-stream)）
@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    # SUBTRACTED: vllm/entrypoints/openai/chat_completion/api_router.py:L54-L56
    # metrics_header_format 读取——orca 负载回执头（elide）。
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        # SUBTRACTED: vllm/entrypoints/openai/chat_completion/api_router.py:L71
        # headers=metrics_header(metrics_header_format)——orca（elide）。
        return JSONResponse(
            content=generator.model_dump(),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


# SUBTRACTED: vllm/entrypoints/openai/chat_completion/api_router.py:L77-L102
# create_batch_chat_completion（/v1/chat/completions/batch 批量面）——略。


# SOURCE: vllm/entrypoints/openai/chat_completion/api_router.py:L105-L106
def attach_router(app: FastAPI):
    app.include_router(router)
