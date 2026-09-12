# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/launcher.py —— 忠实承载（站 0，m1/m8）：
# serve_http（uvicorn 启动 + server_task/watchdog_task 双任务 + 内嵌
# handle_shutdown 的 drain/abort 两模式）与 watchdog_loop/
# terminate_if_errored（WC3 第三道防线）逐字。删：h11 头限制与 ssl 家族
# （delete[10]）、SSLCertRefresher、端口占用排查块（elide「非主线」）。
import asyncio
import signal
import socket
from functools import partial
from typing import Any

import uvicorn
from fastapi import FastAPI

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

# SUBTRACTED: vllm/entrypoints/launcher.py:L15-L19 H11 默认限制常量与
# SSLCertRefresher 导入——delete[10]（h11 限制 + ssl 家族）。
# SUBTRACTED: vllm/entrypoints/launcher.py:L21 find_process_using_port 导入
# ——elide（端口占用排查非主线）。

logger = init_logger(__name__)


# SOURCE: vllm/entrypoints/launcher.py:L26-L166 —— serve_http 逐字（must_keep：
# 优雅关停中枢；h11 头限制弹出段删——delete[10]）
async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    """
    Start a FastAPI app using Uvicorn, with support for custom Uvicorn config
    options.  Supports http header limits via h11_max_incomplete_event_size and
    h11_max_header_count.
    """
    logger.info("Available routes are:")
    # post endpoints
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    # other endpoints
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if endpoint is None or path is None or methods is not None:
            continue

        logger.info("Route: %s, Endpoint: %s", path, endpoint.__name__)

    # SUBTRACTED: vllm/entrypoints/launcher.py:L59-L74 h11 头限制两参的弹出
    # 与 config 装配——delete[10]。

    config = uvicorn.Config(app, **uvicorn_kwargs)
    # SUBTRACTED: vllm/entrypoints/launcher.py:L73-L74 config 的 h11 两限制
    # 回填——delete[10]。
    config.load()
    server = uvicorn.Server(config)
    app.state.server = server

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    # SUBTRACTED: vllm/entrypoints/launcher.py:L84-L93 SSLCertRefresher 装配
    # ——delete[10]（ssl 家族）。

    # make sure uvicorn server signal handler has been registered.
    # Otherwise the app signal handler below will be overwritten in server.serve.
    # server.started is set to True after server complete server.startup.
    # We need to constraint the time sequence.
    logger.info("API server: waiting for HTTP server to start")
    while not server.started and not server_task.done():
        await asyncio.sleep(0.1)
    if server_task.done():
        # Propagate startup failure (e.g. port bind error) instead of hanging.
        await server_task
    logger.info("API server: HTTP server started")

    shutdown_event = asyncio.Event()

    # SOURCE: vllm/entrypoints/launcher.py:L109-L113
    def signal_handler() -> None:
        if shutdown_event.is_set():
            return
        logger.info_once("[shutdown] API server: shutdown triggered")
        shutdown_event.set()

    # SOURCE: vllm/entrypoints/launcher.py:L115-L116
    async def dummy_shutdown() -> None:
        pass

    # SOURCE: vllm/entrypoints/launcher.py:L118-L119
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    # SOURCE: vllm/entrypoints/launcher.py:L121-L146 —— handle_shutdown 逐字
    # （must_keep：SIGINT/SIGTERM → engine.shutdown(timeout)（drain/abort 两
    # 模式：timeout==0 为 abort）→ server.should_exit + 双任务 cancel）
    async def handle_shutdown() -> None:
        await shutdown_event.wait()

        engine_client = app.state.engine_client
        timeout = engine_client.vllm_config.shutdown_timeout
        mode = "abort" if timeout == 0 else "drain"

        logger.info(
            "[shutdown] API server: stopping engine client mode=%s timeout=%ss",
            mode,
            timeout,
        )

        await loop.run_in_executor(
            None, partial(engine_client.shutdown, timeout=timeout)
        )
        logger.info_once("[shutdown] API server: engine client stopped")

        server.should_exit = True
        logger.info_once("[shutdown] API server: signalling HTTP server shutdown")
        server_task.cancel()
        watchdog_task.cancel()
        # SUBTRACTED: vllm/entrypoints/launcher.py:L143-L144
        # ssl_cert_refresher.stop()——delete[10]。

    # SOURCE: vllm/entrypoints/launcher.py:L146
    shutdown_task = loop.create_task(handle_shutdown())

    # SOURCE: vllm/entrypoints/launcher.py:L148-L165（端口占用排查块删——elide）
    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        # SUBTRACTED: vllm/entrypoints/launcher.py:L152-L160
        # find_process_using_port 的端口占用排查——elide（非主线）。
        logger.info_once("[shutdown] API server: shutting down FastAPI HTTP server")
        return server.shutdown()
    finally:
        shutdown_task.cancel()
        watchdog_task.cancel()


# SOURCE: vllm/entrypoints/launcher.py:L168-L177 —— watchdog_loop 逐字
# （must_keep：每 5s 检引擎死——WC3 第三道防线，docstring 点名
# StreamingResponse 生成器吞异常场景）
async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    """
    # Watchdog task that runs in the background, checking
    # for error state in the engine. Needed to trigger shutdown
    # if an exception arises is StreamingResponse() generator.
    """
    VLLM_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)


# SOURCE: vllm/entrypoints/launcher.py:L180-L190 —— terminate_if_errored 逐字
# （must_keep：errored→should_exit 判定；VLLM_KEEP_ALIVE_ON_ENGINE_DEATH 可关）
def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
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
