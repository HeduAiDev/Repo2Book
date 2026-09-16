# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/openai/api_server.py —— 忠实承载（m1 启动链）：
# build_async_engine_client(_from_engine_args) / setup_server（先绑 socket，
# 先于引擎——避 ray 端口竞态 #8204）/ build_app（FastAPI 装配中心 + 四层
# 异常处理器）/ init_app_state（serving 对象挂 state）/ build_and_serve /
# run_server(_worker) 逐字。删：chat/completion 之外的条件路由与中间件家族
# （delete[8]）、非 generate 分支与 OnlineDerenderer/ServingTokenization
# （delete[9]）、ssl/uds/ipv6/set_ulimit/h11（delete[10]）、render-only 面、
# __main__ CLI 块（uvloop/FlexibleArgumentParser——ch3 参数面）。
import asyncio
import multiprocessing
import multiprocessing.forkserver as forkserver
import os
import signal
import socket
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import State

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.api_utils import (
    log_non_default_args,
    log_version_and_model,
    process_lora_modules,
)
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.server_utils import (
    exception_handler,
    get_uvicorn_log_config,
    http_exception_handler,
    lifespan,
    validation_exception_handler,
    vllm_error_handler,
)
from vllm.exceptions import VLLMError
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tasks import SupportedTask
from vllm.tool_parsers import ToolParserManager
from vllm.tracing import instrument
from vllm.usage.usage_lib import UsageContext
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import decorate_logs
from vllm.version import __version__ as VLLM_VERSION

# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L4-L12 importlib/inspect/
# tempfile/warnings/uvloop 导入——uvloop 属 __main__ CLI 面（删），其余为
# 中间件动态导入（delete[8]）。
# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L30-L31 make_arg_parser/
# validate_parsed_serve_args——CLI 参数面（ch3 域）。
# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L33 ScalingMiddleware 导入
# ——delete[8] 中间件家族。
# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L34-L35 sagemaker/
# tokenize serving 导入——delete[8]/[9]。
# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L43-L51 log_response 导入
# ——delete[8]。
# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L55-L62 OnlineDerenderer/
# FlexibleArgumentParser/set_uliment 等导入——delete[9]/[10]/CLI 面。
# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L66 prometheus_multiproc_dir
# ——观测域。

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger("vllm.entrypoints.openai.api_server")

_FALLBACK_SUPPORTED_TASKS: tuple[SupportedTask, ...] = ("generate",)

# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L74-L106
# _attach_endpoint_plugins / _init_endpoint_plugins_state——delete[8]
# （endpoint 插件面）。


# SOURCE: vllm/entrypoints/openai/api_server.py:L109-L137 —— build_async_engine_client 逐字
# （m1：AsyncEngineArgs → from_vllm_config + async with 生命周期边界；client_
# count/client_index 为 ch34 多 API server 携带位）
@asynccontextmanager
async def build_async_engine_client(
    args: Namespace,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

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


# SOURCE: vllm/entrypoints/openai/api_server.py:L140-L187 —— build_async_engine_client_from_engine_args 逐字
# （must_keep：AsyncLLM.from_vllm_config——ch4 在线面入口，黑盒回指；reset_
# mm_cache 与 finally shutdown 保留）
@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig (determines if we can use V1).
    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L156
    # engine_args.create_engine_config 的完整校验链——HOST SEAM（ch3 域：
    # 精简树 ModelConfig 由注入方直接构造）。
    vllm_config = getattr(engine_args, "_vllm_config", None) or _make_vllm_config(
        engine_args
    )

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None

    # Don't mutate the input client_config
    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L181
        # await async_llm.reset_mm_cache()——多模态缓存预热（ch6 域；精简
        # 引擎 seam 无该面，构造即绪）。

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown(timeout=vllm_config.shutdown_timeout)


# SOURCE: vllm/entrypoints/openai/api_server.py:L156 —— HOST SEAM：
# create_engine_config 退化位（真实：AsyncEngineArgs.create_engine_config
# 全链校验产 VllmConfig——ch3 域；精简树按注入的 _vllm_config 直取）
def _make_vllm_config(engine_args: AsyncEngineArgs):
    from vllm.config import VllmConfig

    vllm_config = getattr(engine_args, "_vllm_config", None)
    if vllm_config is not None:
        return vllm_config
    return VllmConfig()


# SOURCE: vllm/entrypoints/openai/api_server.py:L189-L352 —— build_app 逐字
# （must_keep：FastAPI 装配中心；条件路由族与中间件家族删——delete[8]，
# 保留 CORS + 四层异常处理器 + api-key 鉴权）
def build_app(
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
    model_config: ModelConfig | None = None,
) -> FastAPI:
    if supported_tasks is None:
        logger.warning(
            "The 'supported_tasks' parameter was not provided to "
            "build_app and will be required in a future version. "
            "Defaulting to ('generate',).",
        )
        supported_tasks = _FALLBACK_SUPPORTED_TASKS

    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    elif args.enable_offline_docs:
        app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.state.args = args

    from vllm.entrypoints.serve import register_vllm_serve_api_routers

    register_vllm_serve_api_routers(app)

    from vllm.entrypoints.openai.models.api_router import (
        attach_router as register_models_api_router,
    )

    register_models_api_router(app)

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L224-L276 条件路由族
    # ——delete[8]：sagemaker、dev 模式、elastic_ep、scale_out、
    # transcription/realtime、pooling、fault_tolerance、endpoint 插件
    # （chat/completion 之外全部条件分支与 generate 面内非 chat 路由）。

    if "generate" in supported_tasks:
        from vllm.entrypoints.generate.api_router import (
            register_generate_api_routers,
        )

        register_generate_api_routers(app)

        # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L242-L246
        # elastic_ep_attach_router——delete[8]。

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L248-L276 其余条件
    # 分支与 _attach_endpoint_plugins——delete[8]（见上）。

    app.root_path = args.root_path
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # Exception handlers are registered in four layers:
    #   1. framework errors raised by FastAPI/Starlette
    #   2. vLLM-specific errors dispatched via a single ``VLLMError`` handler
    #   3. fallback handlers for raw exceptions not yet migrated to ``VLLMError``
    #   4. the raw ``Exception`` handler as a safety net
    # Registering specific exception types (rather than only ``Exception``)
    # ensures they are handled by ``ExceptionMiddleware`` (inside the Prometheus
    # middleware) rather than ``ServerErrorMiddleware`` (outside it), so their
    # status codes are recorded correctly.
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)

    app.exception_handler(VLLMError)(vllm_error_handler)

    # TODO(zqzten): remove these fallback handlers after migration to VLLMError
    app.exception_handler(ValueError)(exception_handler)
    app.exception_handler(TypeError)(exception_handler)
    app.exception_handler(OverflowError)(exception_handler)
    app.exception_handler(NotImplementedError)(exception_handler)

    app.exception_handler(Exception)(exception_handler)

    # Ensure --api-key option from CLI takes precedence over VLLM_API_KEY
    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware

        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L315-L349 中间件家族
    # ——delete[8]：XRequestIdMiddleware、ScalingMiddleware、websocket metrics、
    # log_response、args.middleware 动态导入。

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L351
    # sagemaker_standards_bootstrap(app)——delete[8]（sagemaker）。
    return app


# SOURCE: vllm/entrypoints/openai/api_server.py:L355-L482 —— init_app_state 逐字
# （must_keep：serving 对象挂 state 的连接点；非 generate 分支与
# OnlineDerenderer/ServingTokenization 删——delete[9]）
async def init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
) -> None:
    vllm_config = engine_client.vllm_config

    if args.tool_call_parser is not None:
        from vllm.parser.metrics import init_parser_metrics

        init_parser_metrics(
            model_name=cast(str, vllm_config.model_config.served_model_name)
        )

    if supported_tasks is None:
        logger.warning(
            "The 'supported_tasks' parameter was not provided to "
            "init_app_state and will be required in a future version. "
            "Please pass 'supported_tasks' explicitly.",
        )
        supported_tasks = _FALLBACK_SUPPORTED_TASKS

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    state.args = args
    resolved_chat_template = load_chat_template(args.chat_template)

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()

    state.online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )
    state.online_renderer.warmup()

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L431-L454
    # OnlineDerenderer / ServingTokenization 实例化——delete[9]（render 与
    # tokenize 面）。

    if "generate" in supported_tasks:
        from vllm.entrypoints.generate.api_router import init_generate_state

        await init_generate_state(
            engine_client, state, args, request_logger, supported_tasks
        )

        # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L463-L465
        # init_scale_out_state——delete[8]。

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L467-L479
    # transcription/realtime/pooling 分支与 _init_endpoint_plugins_state
    # ——delete[8]。

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L484-L594 其余（init_
# render_app_state 等 render-only 面函数）——render-only 服务器面非本章主线。

# SOURCE: vllm/entrypoints/openai/api_server.py:L578-L594 —— create_server_socket 逐字
def create_server_socket(
    addr: tuple[str, int],
    *,
    reuse_port: bool,
) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if reuse_port:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L596-L599
# create_server_unix_socket——delete[10]（uds 分支）。


# SOURCE: vllm/entrypoints/openai/api_server.py:L602-L617 —— validate_api_server_args 逐字
def validate_api_server_args(args):
    valid_tool_parses = ToolParserManager.list_registered()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parsers = ReasoningParserManager.list_registered()
    if (
        reasoning_parser := args.structured_outputs_config.reasoning_parser
    ) and reasoning_parser not in valid_reasoning_parsers:
        raise KeyError(
            f"invalid reasoning parser: {reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parsers)} }})"
        )


# SOURCE: vllm/entrypoints/openai/api_server.py:L620-L655 —— setup_server 逐字
# （must_keep：先绑端口先于引擎的启动序——注释原话 'workaround to make sure
# that we bind the port before the engine is set up. This avoids race
# conditions with ray.'；uds/set_ulimit/ssl 段删——delete[10]）
@instrument(span_name="API server setup")
def setup_server(args, *, reuse_port: bool):
    """Validate API server args and create the server socket."""

    log_version_and_model(logger, VLLM_VERSION, args.model)
    log_non_default_args(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L638-L639 args.uds 的
    # unix socket 分支——delete[10]。
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr, reuse_port=reuse_port)

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L644-L646 set_ulimit()
    # ——delete[10]（uvicorn 高并发丢请求 workaround，部署可选项）。

    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L648-L649 args.uds 的
    # listen_address 分支——delete[10]。
    addr, port = sock_addr
    # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L652 args.ssl_keyfile/
    # ssl_certfile 的 is_ssl 判定——delete[10]。
    host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
    listen_address = f"http://{host_part}:{port}"
    return listen_address, sock


# SOURCE: vllm/entrypoints/openai/api_server.py:L658-L703 —— build_and_serve 逐字
# （must_keep：build_app → init_app_state → serve_http 的组装点；ssl/h11
# kwargs 删——delete[10]）
async def build_and_serve(
    engine_client: EngineClient,
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    **uvicorn_kwargs,
) -> asyncio.Task:
    """Build FastAPI app, initialize state, and start serving.

    Returns the shutdown task for the caller to await.
    """

    # Get uvicorn log config (from file or with endpoint filter)
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    supported_tasks = await engine_client.get_supported_tasks()
    model_config = engine_client.model_config

    logger.info("Supported tasks: %s", supported_tasks)
    app = build_app(args, supported_tasks, model_config)
    await init_app_state(engine_client, app.state, args, supported_tasks)

    logger.info("Starting vLLM server on %s", listen_address)

    return await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # NOTE: When the 'disable_uvicorn_access_log' value is True,
        # no access log will be output.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        # SUBTRACTED: vllm/entrypoints/openai/api_server.py:L695-L701 ssl_*
        # 五参与 h11_* 两参——delete[10]。
        **uvicorn_kwargs,
    )


# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L706-L748
# build_and_serve_renderer——render-only CPU 服务器面（非本章主线）。


# SOURCE: vllm/entrypoints/openai/api_server.py:L751-L764 —— run_server 逐字
# （must_keep：decorate_logs + SIGTERM 临时 handler + setup_server）
async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""

    decorate_logs("APIServer", skip_if_decorated=True)

    # Interrupt initialization if SIGTERM arrives before uvicorn installs its
    # own signal handlers. Once uvicorn is running it replaces this.
    def _interrupt_init(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)

    listen_address, sock = setup_server(args, reuse_port=False)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


# SOURCE: vllm/entrypoints/openai/api_server.py:L767-L789 —— run_server_worker 逐字
# （must_keep：async with 引擎上下文 → build_and_serve → NB 注释「先退引擎
# 上下文再等 HTTP 关停」→ sock.close）
async def run_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

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


# SUBTRACTED: vllm/entrypoints/openai/api_server.py:L792-L804 __main__ CLI 块
# （cli_env_setup + FlexibleArgumentParser + uvloop.run）——CLI 参数面归
# ch3；uvloop 装配属 CLI 启动器。
