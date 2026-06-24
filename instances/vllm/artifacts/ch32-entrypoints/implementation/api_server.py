"""API server 顶层编排（精简版，只做减法）。

与真实 vllm/entrypoints/openai/api_server.py 同名、同结构、同控制流。本章主线保留：
  * build_async_engine_client / build_async_engine_client_from_engine_args
        —— EngineClient 生命周期边界：AsyncLLM.from_vllm_config 起引擎（承接 ch04），finally shutdown。
  * run_server_worker —— 『async with 引擎上下文 → build_and_serve → await shutdown_task』顶层编排。
  * setup_server —— 先绑 socket、装 SIGTERM handler（先于引擎绑端口避免端口竞争）。
  * build_and_serve —— get_supported_tasks → build_app → init_app_state → serve_http。
  * build_app —— FastAPI(lifespan=lifespan) + 注册 chat 路由 + CORS/鉴权/X-Request-Id 中间件 + exception_handler。
  * init_app_state —— 造 OpenAIServingModels/Render/Chat 并挂到 app.state。

把所有 # SUBTRACTED 删回去 ≈ 真实顶层在 generate 任务、单进程 TCP 无 SSL 下的启动主干。
"""

from __future__ import annotations

import asyncio
import signal
import socket
from contextlib import asynccontextmanager
from typing import Any

import envs
from _framework import CORSMiddleware, FastAPI, HTTPException
from server_utils import (
    AuthenticationMiddleware,
    XRequestIdMiddleware,
    lifespan,
)
from engine_serving import OpenAIServingModels
from render_serving import OpenAIServingRender
from chat_serving import OpenAIServingChat
import api_router


# --- EngineClient 生命周期（承接 ch04 AsyncLLM 三段式） ---

@asynccontextmanager
async def build_async_engine_client(
    args,
    *,
    usage_context: str = "OPENAI_API_SERVER",
    client_config: dict[str, Any] | None = None,
):
    # SOURCE: vllm/entrypoints/openai/api_server.py:L77 build_async_engine_client
    # SUBTRACTED: forkserver 预导入分支（VLLM_WORKER_MULTIPROC_METHOD==forkserver），
    #   dossier delete 批准。原 api_server.py:L84-L91。

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)
    if client_config:
        engine_args._api_process_count = client_config.get("client_count", 1)
        engine_args._api_process_rank = client_config.get("client_index", 0)

    async with build_async_engine_client_from_engine_args(
        engine_args,
        usage_context=usage_context,
        client_config=client_config,
    ) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args,
    *,
    usage_context: str = "OPENAI_API_SERVER",
    client_config: dict[str, Any] | None = None,
):
    # SOURCE: vllm/entrypoints/openai/api_server.py:L108 build_async_engine_client_from_engine_args
    # Create the EngineConfig (determines if we can use V1).
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    # 真实代码在此惰性 import AsyncLLM（ch04）。
    async_llm = None

    # Don't mutate the input client_config
    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown()


async def run_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    # SOURCE: vllm/entrypoints/openai/api_server.py:L681 run_server_worker
    """Run a single API server worker."""
    # SUBTRACTED: tool/reasoning parser plugin 导入（部署期可选，dossier delete 批准）。

    async with build_async_engine_client(
        args,
        client_config=client_config,
    ) as engine_client:
        shutdown_task = await build_and_serve(
            engine_client, listen_address, sock, args, **uvicorn_kwargs
        )
    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


# --- 启动序：先绑 socket（先于引擎，避端口竞争） ---

def setup_server(args):
    # SOURCE: vllm/entrypoints/openai/api_server.py:L533 setup_server
    """Validate API server args, set up signal handler, create socket
    ready to serve."""
    # SUBTRACTED: log_version_and_model / log_non_default_args / parser plugin 导入（仅日志/插件）。

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    # SUBTRACTED: uds(unix socket) 分支（dossier delete 批准）；只保留 TCP。
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # SUBTRACTED: set_ulimit()（提高 fd 上限，部署细节）。

    def signal_handler(*_) -> None:
        # SOURCE: setup_server.signal_handler
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    # SUBTRACTED: ipv6/ssl 监听地址拼接分支；主线 http://host:port。
    addr, port = sock_addr
    listen_address = f"http://{addr or '0.0.0.0'}:{port}"
    return listen_address, sock


async def build_and_serve(
    engine_client,
    listen_address: str,
    sock: socket.socket,
    args,
    **uvicorn_kwargs,
) -> asyncio.Task:
    # SOURCE: vllm/entrypoints/openai/api_server.py:L578 build_and_serve
    """Build FastAPI app, initialize state, and start serving.

    Returns the shutdown task for the caller to await.
    """
    # SUBTRACTED: get_uvicorn_log_config 透传（日志配置，非主线）。

    supported_tasks = await engine_client.get_supported_tasks()
    model_config = engine_client.model_config

    app = build_app(args, supported_tasks, model_config)
    await init_app_state(engine_client, app.state, args, supported_tasks)

    from launcher import serve_http
    return await serve_http(
        app,
        sock=sock,
        host=args.host,
        port=args.port,
        # SUBTRACTED: 一大串 ssl_* / h11_* / log_level / access_log kwargs 透传（部署期）。
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        **uvicorn_kwargs,
    )


# --- FastAPI app 装配 ---

def build_app(args, supported_tasks=None, model_config=None) -> FastAPI:
    # SOURCE: vllm/entrypoints/openai/api_server.py:L157 build_app
    # SUBTRACTED: disable_fastapi_docs / enable_offline_docs 分支；主线 FastAPI(lifespan=lifespan)。
    app = FastAPI(lifespan=lifespan)
    app.state.args = args

    # SUBTRACTED: 除 chat 外的条件路由注册（models/sagemaker/pooling/transcription/realtime/
    #   disagg/rlhf/elastic_ep/generative_scoring/render router 等），dossier delete 批准。
    #   只保留 generate 任务的 chat 路由。
    if supported_tasks is None or "generate" in supported_tasks:
        app.include_router(api_router.router)

    app.root_path = args.root_path
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
    )

    app.exception_handler(HTTPException)(http_exception_handler)
    # SUBTRACTED: 另 5 个 exception_handler（RequestValidationError / EngineGenerateError /
    #   EngineDeadError / GenerationError / Exception）登记同样模式，保留一个作示例。

    # Ensure --api-key option from CLI takes precedence over VLLM_API_KEY
    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    if args.enable_request_id_headers:
        app.add_middleware(XRequestIdMiddleware)

    # SUBTRACTED: ScalingMiddleware / realtime metrics / debug log / 用户自定义 middleware /
    #   sagemaker bootstrap（dossier delete 批准）。
    return app


async def init_app_state(engine_client, state, args, supported_tasks=None) -> None:
    # SOURCE: vllm/entrypoints/openai/api_server.py:L317 init_app_state
    vllm_config = engine_client.vllm_config

    # SUBTRACTED: served_model_name / request_logger / base_model_paths / lora_modules 合并
    #   的细节构造（dossier 标注点到即止）。
    served_model_names = args.served_model_name or [args.model]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    state.args = args

    state.openai_serving_models = OpenAIServingModels(
        base_model_names=served_model_names,
    )

    state.openai_serving_render = OpenAIServingRender(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        chat_template=args.chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
    )

    # SUBTRACTED: OpenAIServingTokenization 与 transcription/realtime/pooling 服务对象
    #   实例化（与 chat 主线正交，dossier delete 批准）。

    if supported_tasks is None or "generate" in supported_tasks:
        # 真实代码经 generate.api_router.init_generate_state 造 OpenAIServingChat/Completion；
        # 精简版直接在此造 chat handler 并挂到 state（同一归宿）。
        state.openai_serving_chat = OpenAIServingChat(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
        )


# --- 以下为本章不展开但被主线调用的轻量协作对象（非 vLLM 杜撰，均为真实符号的减法替身） ---

class AsyncEngineArgs:
    # SOURCE: vllm/engine/arg_utils.py:AsyncEngineArgs（减法版：只留 from_cli_args + create_engine_config 接缝）
    def __init__(self, args):
        # SOURCE: AsyncEngineArgs.__init__ —— 减法版
        self.args = args
        self._api_process_count = 1
        self._api_process_rank = 0

    @classmethod
    def from_cli_args(cls, args) -> "AsyncEngineArgs":
        # SOURCE: vllm/engine/arg_utils.py:AsyncEngineArgs.from_cli_args
        return cls(args)

    def create_engine_config(self, usage_context=None):
        # SOURCE: vllm/engine/arg_utils.py:create_engine_config
        # SUBTRACTED: 真实版从 CLI 拼出完整 VllmConfig；精简版透传 args 上挂的 vllm_config。
        return self.args.vllm_config


class AsyncLLM:
    """AsyncLLM 起引擎入口的减法接缝（真正实现是 ch04）。"""

    # SOURCE: vllm/v1/engine/async_llm.py:AsyncLLM（本章只用 from_vllm_config / reset_mm_cache / shutdown 接缝）
    @classmethod
    def from_vllm_config(cls, vllm_config, **kwargs):
        # SOURCE: vllm/v1/engine/async_llm.py:AsyncLLM.from_vllm_config
        # SUBTRACTED: 真实版起 EngineCore + output_handler 背景任务（ch04 三段式）；
        #   本章是消费侧，from_vllm_config 的内部在 ch04 展开，这里返回已注入的引擎实例。
        return getattr(vllm_config, "_engine_client", None) or cls()

    async def reset_mm_cache(self) -> None:
        # SOURCE: vllm/v1/engine/async_llm.py:AsyncLLM.reset_mm_cache
        pass

    def shutdown(self, timeout=None) -> None:
        # SOURCE: vllm/v1/engine/async_llm.py:AsyncLLM.shutdown
        pass


def validate_api_server_args(args) -> None:
    # SOURCE: vllm/entrypoints/openai/cli_args.py:validate_api_server_args
    # SUBTRACTED: 各参数互斥/取值校验细节，占位放行。
    return None


def create_server_socket(addr) -> socket.socket:
    # SOURCE: vllm/utils/network_utils.py:create_server_socket
    family = socket.AF_INET
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)
    return sock


def http_exception_handler(request, exc):
    # SOURCE: vllm/entrypoints/openai/server_utils.py:http_exception_handler
    from _framework import JSONResponse
    return JSONResponse({"error": str(getattr(exc, "detail", exc))},
                        status_code=getattr(exc, "status_code", 500))
