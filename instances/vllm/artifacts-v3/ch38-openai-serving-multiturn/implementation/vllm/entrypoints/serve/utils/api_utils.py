# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/serve/utils/api_utils.py —— 忠实承载（本章 m2/
# m3/m7 素材源）：listen_for_disconnect + with_cancellation 断连竞速双任务
# 逐字（F5 的 HTTP 端——load 计数递减侧路删）、get_max_tokens 四方取 min
# 逐字、should_include_usage/validate_json_request/process_lora_modules/
# sanitize_message/log_version_and_model 逐字；load_aware_call 负载追踪
# 装饰器族删（dossier elide：--enable-server-load-tracking 的 orca 侧路，
# 与主线正交）。
import asyncio
import dataclasses
import functools
import os
from argparse import Namespace
from logging import Logger
from string import Template
from typing import Any

import regex as re
from fastapi import Request
from fastapi.exceptions import RequestValidationError

from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm.logger import current_formatter_type, init_logger
from vllm.platforms import current_platform

# SUBTRACTED: vllm/entrypoints/serve/utils/api_utils.py:L16-L17
# BackgroundTask/BackgroundTasks 导入——load_aware_call 族删除项的连带。
# SUBTRACTED: vllm/entrypoints/serve/utils/api_utils.py:L23 orca_metrics
# 导入——同上。

logger = init_logger(__name__)

# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L29-L34
VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L37-L49 —— listen_for_disconnect
# 逐字（must_keep：竞速另一半；server_load_metrics 计数递减侧路删——
# dossier elide L42-48「负载追踪侧路，删」）
async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            # SUBTRACTED: vllm/entrypoints/serve/utils/api_utils.py:L42-L48
            # enable_server_load_tracking 的 server_load_metrics 计数递减
            # ——orca 负载追踪侧路（elide 批准）。
            break


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L52-L94 —— with_cancellation 逐字
# （must_keep：F5 核心——handler 任务与断连监听任务 asyncio.wait
# FIRST_COMPLETED 竞速；docstring 三要点原话）
def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
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


# SUBTRACTED: vllm/entrypoints/serve/utils/api_utils.py:L97-L146
# decrement_server_load / load_aware_call——--enable-server-load-tracking 的
# orca 负载回执装饰器族（dossier elide「与主线正交，可整段省」）；api_router
# 的 @load_aware_call 与 metrics_header 连带删。


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L149-L167 —— cli_env_setup 逐字
def cli_env_setup():
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using vLLM as a library. `spawn` will cause
    # unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cpu-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L170-L206 —— get_max_tokens 逐字
# （must_keep：四方取 min 的输出预算——max_model_len−输入长 / 请求值 /
# server 默认 / 平台上限）
def get_max_tokens(
    max_model_len: int,
    max_tokens: int | None,
    input_length: int,
    default_sampling_params: dict,
    override_max_tokens: int | None = None,
    truncate_prompt_tokens: int | None = None,
) -> int:
    if truncate_prompt_tokens is not None:
        limit = truncate_prompt_tokens
        input_length = min(
            input_length,
            max_model_len if limit == -1 else limit,
        )
    if max_model_len < input_length:
        raise ValueError(
            f"Input length ({input_length}) exceeds model's maximum "
            f"context length ({max_model_len})."
        )
    model_max_tokens = max_model_len - input_length
    platform_max_tokens = current_platform.get_max_output_tokens(input_length)
    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )

    return min(
        val
        for val in (
            model_max_tokens,
            fallback_max_tokens,
            override_max_tokens,
            platform_max_tokens,
        )
        if val is not None
    )


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L209-L236 —— get_non_default_args
# （HOST SEAM：Namespace 分支的 make_arg_parser CLI 解析链删——vllm/
# entrypoints/openai/cli_args 参数面归 ch3；EngineArgs 分支逐字）
def get_non_default_args(args: Namespace | EngineArgs) -> dict[str, Any]:
    non_default_args = {}

    # SUBTRACTED: vllm/entrypoints/serve/utils/api_utils.py:L214-L219
    # Namespace 分支（make_arg_parser 全参数解析）——CLI 参数面（ch3 域）。

    # Handle EngineArgs instance
    if isinstance(args, EngineArgs):
        default_args = EngineArgs(model=args.model)  # Create default instance
        for field in dataclasses.fields(args):
            current_val = getattr(args, field.name)
            default_val = getattr(default_args, field.name)
            if current_val != default_val:
                non_default_args[field.name] = current_val
        if default_args.model != EngineArgs.model:
            non_default_args["model"] = default_args.model
    else:
        raise TypeError(
            "Unsupported argument type. Must be Namespace or EngineArgs instance."
        )

    return non_default_args


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L239-L255 —— _jsonify_arg_value 逐字
def _jsonify_arg_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            key: _jsonify_arg_value(val)
            for key, val in dataclasses.asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _jsonify_arg_value(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonify_arg_value(item) for item in value]
    if (model_dump := getattr(value, "model_dump", None)) is not None:
        return _jsonify_arg_value(model_dump(mode="json"))
    if (to_dict := getattr(value, "dict", None)) is not None:
        return _jsonify_arg_value(to_dict())
    return repr(value)


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L258-L268 —— jsonify_non_default_args 逐字
def jsonify_non_default_args(
    args: Namespace | EngineArgs,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    non_default_args = get_non_default_args(args)
    if exclude is not None:
        for key in exclude:
            non_default_args.pop(key, None)

    return {key: _jsonify_arg_value(value) for key, value in non_default_args.items()}


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L271-L273 —— log_non_default_args 逐字
def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L276-L288 —— should_include_usage 逐字
# （stream_options.include_usage 的两值解析）
def should_include_usage(
    stream_options: StreamOptions | None, enable_force_include_usage: bool
) -> tuple[bool, bool]:
    if enable_force_include_usage:
        return True, True
    if stream_options:
        include_usage = bool(stream_options.include_usage)
        include_continuous_usage = include_usage and bool(
            stream_options.continuous_usage_stats
        )
    else:
        include_usage, include_continuous_usage = False, False
    return include_usage, include_continuous_usage


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L291-L309 —— process_lora_modules 逐字
def process_lora_modules(
    args_lora_modules: list[LoRAModulePath], default_mm_loras: dict[str, str] | None
) -> list[LoRAModulePath]:
    from vllm.entrypoints.openai.models.serving import LoRAModulePath

    lora_modules = args_lora_modules
    if default_mm_loras:
        default_mm_lora_paths = [
            LoRAModulePath(
                name=modality,
                path=lora_path,
            )
            for modality, lora_path in default_mm_loras.items()
        ]
        if args_lora_modules is None:
            lora_modules = default_mm_lora_paths
        else:
            lora_modules += default_mm_lora_paths
    return lora_modules


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L312-L320 —— sanitize_message 逐字
def sanitize_message(message: str) -> str:
    """Strip memory addresses, tracebacks, and file paths from error messages."""
    message = re.sub(r" at 0x[0-9a-f]+>", ">", message)
    message = re.sub(r'\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?', "", message)
    message = re.sub(
        r"/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+", "<path>", message
    )
    message = re.sub(r"(?:/[\w\-]+)+/[\w\-]+\.\w+", "<path>", message)
    return message.strip()


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L323-L345 —— log_version_and_model
# 逐字（vLLM logo 模板）
def log_version_and_model(lgr: Logger, version: str, model_name: str) -> None:
    if envs.VLLM_DISABLE_LOG_LOGO or (formatter := current_formatter_type(lgr)) is None:
        message = "vLLM server version %s, serving model %s"
    else:
        logo_template = Template(
            "\n       ${w}█     █     █▄   ▄█${r}\n"
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}  version ${w}%s${r}\n"
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}  model   ${w}%s${r}\n"
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}\n"
        )
        colors = {
            "w": "\033[1m",  # bold, default foreground
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
        }
        if formatter != "color":
            # monochrome logo (no ansi escape codes)
            colors = dict.fromkeys(colors, "")

        message = logo_template.substitute(colors)

    lgr.info(message, version, model_name)


# SOURCE: vllm/entrypoints/serve/utils/api_utils.py:L348-L354 —— validate_json_request 逐字
async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=["Unsupported Media Type: Only 'application/json' is allowed"]
        )
