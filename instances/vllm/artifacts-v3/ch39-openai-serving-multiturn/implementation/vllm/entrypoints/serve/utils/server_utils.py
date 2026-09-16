# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/serve/utils/server_utils.py —— 忠实承载（消费面）：
# build_app 的四层异常处理器 + AuthenticationMiddleware + lifespan +
# get_uvicorn_log_config 逐字；XRequestIdMiddleware / log_response /
# SSEDecoder 观测族删（delete[8] 中间件家族——XRequestId/log_response 名列
# 删除项）。validation_exception_handler 的 pydantic-core 内部词汇清洗段
# 保留主干。
import asyncio
import hashlib
import json
import secrets
from argparse import Namespace
from collections.abc import Awaitable
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import terminate_if_errored
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    GenerationError,
)
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMError
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)

# SUBTRACTED: vllm/entrypoints/serve/utils/server_utils.py:L96-L125
# XRequestIdMiddleware——delete[8] 中间件家族明列删除项。

# SOURCE: vllm/entrypoints/serve/utils/server_utils.py —— GUARDED_PREFIX
# （真实在文件头定义：受鉴权保护的路由前缀族）
GUARDED_PREFIX = (
    "/v1",
    "/rerank",
    "/v2",
    "/pooling",
    "/classify",
    "/embed",
    "/score",
    "/tokenize",
    "/detokenize",
    "/validate",
)

# SUBTRACTED: vllm/entrypoints/serve/utils/server_utils.py:L203-L326
# SSEDecoder/_log_streaming_response/_log_non_streaming_response/
# log_response——delete[8] 观测中间键家族明列删除项。


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L127-L138 —— load_log_config 逐字
def load_log_config(log_config_file: str | None) -> dict | None:
    if not log_config_file:
        return None
    try:
        with open(log_config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(
            "Failed to load log config from file %s: error %s", log_config_file, e
        )
        return None


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L140-L172 —— HOST SEAM：
# get_uvicorn_log_config 主干（disable_access_log_for_endpoints 的
# create_uvicorn_log_config 支路删——logging_utils 观测面，None 回落）
def get_uvicorn_log_config(args: Namespace) -> dict | None:
    """
    Get the uvicorn log config based on the provided arguments.

    Priority:
    1. If log_config_file is specified, use it
    2. If disable_access_log_for_endpoints is specified, create a config with
       the access log filter
    3. Otherwise, return None (use uvicorn defaults)
    """
    # First, try to load from file if specified
    log_config = load_log_config(getattr(args, "log_config_file", None))
    if log_config is not None:
        return log_config

    # SUBTRACTED: vllm/entrypoints/serve/utils/server_utils.py:L154-L169
    # disable_access_log_for_endpoints 的 uvicorn access-log 过滤配置
    # （vllm/logging_utils 域）。

    return None


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L45-L93 —— AuthenticationMiddleware 逐字
class AuthenticationMiddleware:
    """
    Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health).
    """

    # SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L57-L59
    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    # SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L61-L76
    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    # SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L78-L93
    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L328-L336 —— vllm_error_handler 逐字
async def vllm_error_handler(req: Request, exc: VLLMError):
    """Dispatch a vLLM-specific error to the appropriate handler."""
    if isinstance(exc, (EngineGenerateError, EngineDeadError)):
        return await engine_error_handler(req, exc)
    elif isinstance(exc, GenerationError):
        return await generation_error_handler(req, exc)
    else:
        return await exception_handler(req, exc)


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L338-L385 —— engine_error_handler
# 逐字（docstring 是 WC3 的正文素材：流式 200 已发 + watchdog 兜底）
async def engine_error_handler(
    req: Request,
    exc: EngineDeadError | EngineGenerateError,
):
    """
    VLLM V1 AsyncLLM catches exceptions and returns
    only two types: EngineGenerateError and EngineDeadError.

    EngineGenerateError is raised by the per request generate()
    method. This error could be request specific (and therefore
    recoverable - e.g. if there is an error in input processing).

    EngineDeadError is raised by the background output_handler
    method. This error is global and therefore not recoverable.

    We register these @app.exception_handlers to return nice
    responses to the end user if they occur and shut down if needed.
    See https://fastapi.tiangolo.com/tutorial/handling-errors/
    for more details on how exception handlers work.

    If an exception is encountered in a StreamingResponse
    generator, the exception is not raised, since we already sent
    a 200 status. Rather, we send an error message as the next chunk.
    Since the exception is not raised, this means that the server
    will not automatically shut down. Instead, we use the watchdog
    background task for check for errored state.
    """

    if req.app.state.args.log_error_stack:
        logger.exception(
            "Engine Exception caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    terminate_if_errored(
        server=req.app.state.server,
        engine=req.app.state.engine_client,
    )
    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L381-L391 —— generation_error_handler 逐字
async def generation_error_handler(req: Request, exc: GenerationError):
    """Handle GenerationError without logging stack traces.

    GenerationError is a known, expected error (e.g. KV cache load failure)
    that should be returned to the client as a 500 response without polluting
    server logs with stack traces.
    """
    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L392-L404 —— exception_handler 逐字
async def exception_handler(req: Request, exc: Exception):
    if req.app.state.args.log_error_stack:
        logger.error(
            "Exception caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L405-L422 —— http_exception_handler 逐字
async def http_exception_handler(req: Request, exc):
    if req.app.state.args.log_error_stack:
        logger.exception(
            "HTTPException caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )
    err = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(exc.detail),
            type=HTTPStatus(exc.status_code).phrase,
            code=exc.status_code,
        )
    )
    return JSONResponse(err.model_dump(), status_code=exc.status_code)


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L498-L546 —— HOST SEAM：
# validation_exception_handler 主干（pydantic 错误 → 400 JSONResponse；
# _BRACKETED_INTERNAL_RE 内部词汇清洗段按观测需要删）
async def validation_exception_handler(req: Request, exc: RequestValidationError):
    err = ErrorResponse(
        error=ErrorInfo(
            message=str(exc.errors()),
            type="BadRequestError",
            code=400,
        )
    )
    return JSONResponse(err.model_dump(), status_code=400)


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L548-L580 —— lifespan
# （HOST SEAM：freeze_gc_heap 与 transcription/translation 关停位删——
# GC 优化与语音面；log_stats 后台任务主干保留）
_running_tasks: set[asyncio.Task] = set()


# SOURCE: vllm/entrypoints/serve/utils/server_utils.py:L548
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # SUBTRACTED: vllm/entrypoints/serve/utils/server_utils.py:L572-L573
        # freeze_gc_heap()——GC 优化（vllm/gc_utils 域）。
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
            # SUBTRACTED: L577-L586 transcription/translation serving 关停
            # 位——语音面（m18 平行面）。
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


# SUBTRACTED: vllm/entrypoints/serve/utils/server_utils.py:L476-L496
# _is_internal_loc_segment/clean_loc_for_param——pydantic 错误 loc 清洗。
