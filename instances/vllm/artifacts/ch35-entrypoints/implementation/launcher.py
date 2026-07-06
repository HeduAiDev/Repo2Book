"""uvicorn 启动 + 优雅关停 + watchdog（精简版，只做减法）。

与真实 vllm/entrypoints/launcher.py 同名、同结构、同控制流。本章主线保留：
  * serve_http：建 uvicorn.Config/Server → watchdog_task + server_task + handle_shutdown
                → SIGINT/SIGTERM → shutdown_event → engine.shutdown(run_in_executor) → server.should_exit。
  * watchdog_loop / terminate_if_errored：每 5s 检 engine.errored，兜底触发优雅退出
                （修复 StreamingResponse 生成器内异常不会自动关停的问题）。

# SUBTRACTED: import uvicorn / fastapi —— 用 _framework + _uvicorn 替身（第三方框架，正交于
#   vLLM 自己的关停编排）。把替身换成真 uvicorn ≈ 真实运行。原 launcher.py:L10-L11。
"""

from __future__ import annotations

import asyncio
import signal
from functools import partial
from typing import Any

import envs
from messages import EngineClient
from _framework import FastAPI


class _UvicornServer:
    """uvicorn.Server 的极小替身：保留 serve / should_exit / shutdown。"""

    # SOURCE: uvicorn.Server —— stub
    def __init__(self, config=None):
        # SOURCE: uvicorn.Server.__init__ —— stub
        self.config = config
        self.should_exit = False
        self._served = asyncio.Event()

    async def serve(self, sockets=None):
        # SOURCE: uvicorn.Server.serve —— stub：阻塞直到 should_exit 被置位
        while not self.should_exit:
            await asyncio.sleep(0.001)

    async def shutdown(self):
        # SOURCE: uvicorn.Server.shutdown —— stub
        self.should_exit = True


class _UvicornConfig:
    # SOURCE: uvicorn.Config —— stub
    def __init__(self, app, **kwargs):
        # SOURCE: uvicorn.Config.__init__ —— stub
        self.app = app
        self.kwargs = kwargs

    def load(self):
        # SOURCE: uvicorn.Config.load —— stub
        pass


async def serve_http(
    app: FastAPI,
    sock: Any | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    # SOURCE: vllm/entrypoints/launcher.py:L26 serve_http
    """Start a FastAPI app using Uvicorn, with graceful shutdown + watchdog."""
    # SUBTRACTED: 打印路由列表、h11 header 限制取默认（launcher.py:L37-L74 头部）。

    config = _UvicornConfig(app, **uvicorn_kwargs)
    config.load()
    server = _UvicornServer(config)
    app.state.server = server

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))
    # SUBTRACTED: SSLCertRefresher 分支（部署期可选项，dossier delete 批准）。

    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        # SOURCE: serve_http.signal_handler
        shutdown_event.set()

    async def dummy_shutdown() -> None:
        # SOURCE: serve_http.dummy_shutdown
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async def handle_shutdown() -> None:
        # SOURCE: vllm/entrypoints/launcher.py:L106 serve_http.handle_shutdown
        await shutdown_event.wait()

        engine_client = app.state.engine_client
        timeout = engine_client.vllm_config.shutdown_timeout

        # engine.shutdown 是同步阻塞调用，丢进 executor 避免阻塞事件循环。
        await loop.run_in_executor(
            None, partial(engine_client.shutdown, timeout=timeout)
        )

        server.should_exit = True
        server_task.cancel()
        watchdog_task.cancel()

    shutdown_task = loop.create_task(handle_shutdown())

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        # SUBTRACTED: find_process_using_port 占用诊断日志（launcher.py:L128-L136）。
        return server.shutdown()
    finally:
        shutdown_task.cancel()
        watchdog_task.cancel()


async def watchdog_loop(server: "_UvicornServer", engine: EngineClient):
    # SOURCE: vllm/entrypoints/launcher.py:L144 watchdog_loop
    """
    # Watchdog task that runs in the background, checking
    # for error state in the engine. Needed to trigger shutdown
    # if an exception arises is StreamingResponse() generator.
    """
    VLLM_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)


def terminate_if_errored(server: "_UvicornServer", engine: EngineClient):
    # SOURCE: vllm/entrypoints/launcher.py:L156 terminate_if_errored
    """
    See discussions here on shutting down a uvicorn server
    https://github.com/encode/uvicorn/discussions/1103
    In this case we cannot await the server shutdown here
    because handler must first return to close the connection
    for this request.
    """
    engine_errored = engine.errored and not engine.is_running
    if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
        server.should_exit = True
