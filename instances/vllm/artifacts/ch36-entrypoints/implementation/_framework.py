"""Minimal framework stubs —— FastAPI / uvicorn / Starlette 的极小替身。

# SUBTRACTED: 真实代码 import fastapi.FastAPI / uvicorn.Server / starlette 的
#   Request/Headers/JSONResponse/StreamingResponse/CORSMiddleware。这些是第三方
#   HTTP 框架，与本章主线（请求如何穿过 vLLM 自己的代码到 AsyncLLM）正交。这里给出
#   只够跑通 vLLM 控制流的极小替身：保留 app.state / 路由注册 / 中间件 / exception_handler /
#   StreamingResponse(media_type) / server.should_exit 等被 vLLM 直接读写的接口语义，
#   行为上不做真正的 socket IO。把这些替身换成真 fastapi/uvicorn ≈ 真实运行。
#   原依赖见 vllm/entrypoints/openai/api_server.py 顶部 import 与
#   vllm/entrypoints/launcher.py:L10-L11。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any


class _State:
    """app.state 的替身：一个允许任意属性赋值的命名空间。"""

    # SOURCE: starlette.datastructures.State —— stub（FastAPI app.state 的语义）
    def __delattr__(self, name: str) -> None:
        self.__dict__.pop(name, None)


class Request:
    """starlette.Request 的极小替身：只保留 vLLM 读到的 headers / app / state。"""

    # SOURCE: starlette.requests.Request —— stub
    def __init__(self, app: "FastAPI" | None = None, headers: dict | None = None):
        # SOURCE: starlette.requests.Request.__init__ —— stub
        self.app = app
        self.headers = headers or {}
        self.state = _State()

    async def is_disconnected(self) -> bool:
        # SOURCE: starlette.requests.Request.is_disconnected —— stub
        return False


class JSONResponse:
    """starlette.responses.JSONResponse 替身。"""

    # SOURCE: starlette.responses.JSONResponse —— stub
    def __init__(self, content: Any, status_code: int = HTTPStatus.OK.value,
                 headers: dict | None = None):
        # SOURCE: starlette.responses.JSONResponse.__init__ —— stub
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}


class StreamingResponse:
    """starlette.responses.StreamingResponse 替身。

    vLLM 用它把 SSE 异步生成器包成 text/event-stream 出口；本章主线的关键区分点。
    """

    # SOURCE: starlette.responses.StreamingResponse —— stub
    def __init__(self, content, media_type: str = "text/plain"):
        # SOURCE: starlette.responses.StreamingResponse.__init__ —— stub
        self.body_iterator = content
        self.media_type = media_type


class FastAPI:
    """fastapi.FastAPI 的极小替身：保留 state / 路由表 / 中间件 / exception_handler。"""

    # SOURCE: fastapi.FastAPI —— stub
    def __init__(self, lifespan=None, **kwargs):
        # SOURCE: fastapi.FastAPI.__init__ —— stub
        self.lifespan = lifespan
        self.state = _State()
        self.routes: dict[str, Callable] = {}
        self.middleware_stack: list = []
        self.exception_handlers: dict = {}
        self.root_path = ""

    def include_router(self, router: "APIRouter") -> None:
        # SOURCE: fastapi.FastAPI.include_router —— stub
        self.routes.update(router.routes)

    def add_middleware(self, cls, **kwargs) -> None:
        # SOURCE: fastapi.FastAPI.add_middleware —— stub
        self.middleware_stack.append((cls, kwargs))

    def exception_handler(self, exc):
        # SOURCE: fastapi.FastAPI.exception_handler —— stub（返回登记用装饰器）
        def deco(handler):
            self.exception_handlers[exc] = handler
            return handler
        return deco


class APIRouter:
    """fastapi.APIRouter 替身：post 装饰器把 path→handler 记进 routes。"""

    # SOURCE: fastapi.APIRouter —— stub
    def __init__(self):
        # SOURCE: fastapi.APIRouter.__init__ —— stub
        self.routes: dict[str, Callable] = {}

    def post(self, path: str, **kwargs):
        # SOURCE: fastapi.APIRouter.post —— stub
        def deco(fn):
            self.routes[path] = fn
            return fn
        return deco


def Depends(dependency: Callable):  # noqa: N802
    # SOURCE: fastapi.Depends —— stub（本章不解析依赖图，仅占位）
    return dependency


class HTTPException(Exception):
    # SOURCE: fastapi.HTTPException —— stub
    def __init__(self, status_code: int, detail: str = ""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class CORSMiddleware:
    # SOURCE: starlette.middleware.cors.CORSMiddleware —— stub（仅作 add_middleware 占位）
    pass
