"""chat/completions API router + 请求级取消装饰器（精简版，只做减法）。

聚合两处真实源码：
  * vllm/entrypoints/openai/chat_completion/api_router.py:L40-L74
    —— POST /v1/chat/completions handler：取 app.state.openai_serving_chat → create_chat_completion
       → 按返回类型分流 ErrorResponse/ChatCompletionResponse → JSONResponse，否则 StreamingResponse(SSE)。
  * vllm/entrypoints/utils.py:L41-L98
    —— with_cancellation / listen_for_disconnect：双任务竞速监听客户端断连，先完成者取消另一个。

这是 SSE vs JSON 分流的落点：流式走 StreamingResponse(media_type='text/event-stream')。
"""

from __future__ import annotations

import asyncio
import functools

from _framework import (
    APIRouter,
    Depends,
    JSONResponse,
    Request,
    StreamingResponse,
)
from messages import ChatCompletionResponse, ErrorResponse


async def listen_for_disconnect(request: Request) -> None:
    # SOURCE: vllm/entrypoints/utils.py:L41 listen_for_disconnect
    """Returns if a disconnect message is received"""
    # SUBTRACTED: 真实版从 request.receive() 读 ASGI 消息找 http.disconnect；精简版无真
    #   ASGI 通道，用 is_disconnected() 轮询替身保持『监听断连』语义。原 utils.py:L43-L53。
    while True:
        if await request.is_disconnected():
            break
        await asyncio.sleep(0)
        return  # stub：替身永不断连，立即返回以免空转（语义：未断连）


def with_cancellation(handler_func):
    # SOURCE: vllm/entrypoints/utils.py:L56 with_cancellation
    """Decorator that allows a route handler to be cancelled by client
    disconnections (races handler against a disconnect-listener; first
    to finish cancels the other). Follows the starlette.StreamingResponse
    pattern; does NOT use request.is_disconnected which is unreliable under
    middleware."""

    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        # SOURCE: with_cancellation.wrapper
        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait(
            [handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def validate_json_request(raw_request: Request) -> None:
    # SOURCE: vllm/entrypoints/openai/utils.py:validate_json_request
    # SUBTRACTED: 真实版校验 Content-Type: application/json 并 415；精简版无真 header 检查，占位放行。
    return None


router = APIRouter()  # SOURCE: vllm/entrypoints/openai/chat_completion/api_router.py:L28


def chat(request: Request):
    # SOURCE: vllm/entrypoints/openai/chat_completion/api_router.py:L32 chat
    return request.app.state.openai_serving_chat


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
)
@with_cancellation
# SUBTRACTED: @load_aware_call 装饰器（负载追踪侧路，dossier delete 批准）。
async def create_chat_completion(request, raw_request: Request):
    # SOURCE: vllm/entrypoints/openai/chat_completion/api_router.py:L53 create_chat_completion
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        # SUBTRACTED: metrics_header(orca 负载头) 透传（dossier delete 批准）。
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")
