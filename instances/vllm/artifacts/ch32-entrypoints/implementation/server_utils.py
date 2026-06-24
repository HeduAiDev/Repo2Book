"""FastAPI lifespan + 纯 ASGI 中间件（精简版，只做减法）。

与真实 vllm/entrypoints/openai/server_utils.py 同名、同结构、同控制流。本章主线保留：
  * lifespan：启动期拉起周期性 do_log_stats 后台任务 + freeze_gc_heap，yield 服务，关停 cancel。
  * AuthenticationMiddleware：Bearer token 鉴权（仅 /v1 路径、跳过 OPTIONS，sha256+compare_digest 防时序侧信道）。
  * XRequestIdMiddleware：X-Request-Id 注入响应头。
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import uuid
from contextlib import asynccontextmanager

import envs
from _framework import FastAPI


def freeze_gc_heap() -> None:
    # SOURCE: vllm/utils/gc_utils.py:freeze_gc_heap
    # SUBTRACTED: 真实版调 gc.freeze() 把启动堆标为静态以减少老年代 GC 停顿；精简版占位
    #   （不引入 GC 调优副作用，语义保留：『冻结启动堆』被调用一次）。
    import gc
    gc.collect()


_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # SOURCE: vllm/entrypoints/openai/server_utils.py:L446 lifespan
    try:
        if app.state.log_stats:
            engine_client = app.state.engine_client

            async def _force_log():
                # SOURCE: lifespan._force_log
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


class AuthenticationMiddleware:
    # SOURCE: vllm/entrypoints/openai/server_utils.py:L38 AuthenticationMiddleware
    """
    Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Skipped when: HTTP method is OPTIONS, or path doesn't start with /v1.
    """

    def __init__(self, app, tokens: list[str]) -> None:
        # SOURCE: AuthenticationMiddleware.__init__
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    def verify_token(self, headers) -> bool:
        # SOURCE: AuthenticationMiddleware.verify_token
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

    def __call__(self, scope, receive, send):
        # SOURCE: AuthenticationMiddleware.__call__
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            return self.app(scope, receive, send)
        # SUBTRACTED: starlette URL/Headers 解析（root_path 剥离）；精简版直接读 scope。
        root_path = scope.get("root_path", "")
        url_path = scope.get("path", "").removeprefix(root_path)
        headers = scope.get("headers_obj", {})
        if url_path.startswith("/v1") and not self.verify_token(headers):
            from _framework import JSONResponse
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response
        return self.app(scope, receive, send)


class XRequestIdMiddleware:
    # SOURCE: vllm/entrypoints/openai/server_utils.py:L89 XRequestIdMiddleware
    """Middleware that sets the X-Request-Id response header to the incoming
    request id, or a random uuid4 hex if not present."""

    def __init__(self, app) -> None:
        # SOURCE: XRequestIdMiddleware.__init__
        self.app = app

    def __call__(self, scope, receive, send):
        # SOURCE: XRequestIdMiddleware.__call__
        if scope["type"] not in ("http", "websocket"):
            return self.app(scope, receive, send)

        request_headers = scope.get("headers_obj", {})

        async def send_with_request_id(message):
            # SOURCE: XRequestIdMiddleware.__call__.send_with_request_id
            if message["type"] == "http.response.start":
                request_id = request_headers.get("X-Request-Id", uuid.uuid4().hex)
                message.setdefault("headers", []).append(
                    ("X-Request-Id", request_id)
                )
            await send(message)

        return self.app(scope, receive, send_with_request_id)
