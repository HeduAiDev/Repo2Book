# SOURCE: vllm/v1/sample/logits_processor/__init__.py
# ch29 构造入口：build_logitsprocs（BUILTIN 三件套 + spec 分支的互斥警告
# 前指 ch32/33）与插件/FQCN 加载链（_load_logitsprocs_plugins →
# _load_logitsprocs_by_fqcns → _load_custom_logitsprocs——必经调用链禁删）。
# SUBTRACTED：delete[6] 插件/自定义加载面之外的 API——
#   cached_load_custom_logitsprocs（L221-L222）、
#   validate_logits_processors_parameters（L224-L237）、
#   AdapterLogitsProcessor（L240-L347，OOT 请求级处理器包装面）；以及
#   __all__ 的四个已删名（AdapterLogitsProcessor/BatchUpdate/
#   BatchUpdateBuilder/MoveDirectionality）。连带 import 修剪：L21-L31 里
#   process_dict_updates/BatchUpdate/MoveDirectionality/BatchUpdateBuilder
#   （计划明列）与只被已删符号消费的 abstractmethod/inspect/lru_cache/
#   partial/VLLMValidationError/RequestLogitsProcessor/SamplingParams。
#   build_logitsprocs 对空 custom 列表行为不变（BUILTIN 三件套照常构造）。
import importlib
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import guard_cuda_initialization
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.logits_processor.state import LogitsProcessors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Error message when the user tries to initialize vLLM with a pooling model
# and custom logitsproces
STR_POOLING_REJECTS_LOGITSPROCS = (
    "Pooling models do not support custom logits processors."
)

# Error message when the user tries to initialize vLLM with a speculative
# decoding enabled and custom logitsproces
STR_SPEC_DEC_REJECTS_LOGITSPROCS = (
    "Custom logits processors are not supported when speculative decoding is enabled."
)

LOGITSPROCS_GROUP = "vllm.logits_processors"

BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]


# SOURCE: vllm/v1/sample/logits_processor/__init__.py:L57-L84
#   _load_logitsprocs_plugins —— 逐字（entry_points 插件加载）
def _load_logitsprocs_plugins() -> list[type[LogitsProcessor]]:
    """Load all installed logit processor plugins"""

    from importlib.metadata import entry_points

    installed_logitsprocs_plugins = entry_points(group=LOGITSPROCS_GROUP)
    if len(installed_logitsprocs_plugins) == 0:
        logger.debug("No logitsprocs plugins installed (group %s).", LOGITSPROCS_GROUP)
        return []

    # Load logitsprocs plugins
    logger.debug("Loading installed logitsprocs plugins (group %s):", LOGITSPROCS_GROUP)
    classes: list[type[LogitsProcessor]] = []
    for entrypoint in installed_logitsprocs_plugins:
        try:
            logger.debug(
                "- Loading logitproc plugin entrypoint=%s target=%s",
                entrypoint.name,
                entrypoint.value,
            )
            with guard_cuda_initialization():
                classes.append(entrypoint.load())
        except Exception as e:
            logger.error("Failed to load LogitsProcessor plugin %s: %s", entrypoint, e)
            raise RuntimeError(
                f"Failed to load LogitsProcessor plugin {entrypoint}"
            ) from e
    return classes


# SOURCE: vllm/v1/sample/logits_processor/__init__.py:L87-L156
#   _load_logitsprocs_by_fqcns —— 逐字（FQCN <module>:<type> 加载）
def _load_logitsprocs_by_fqcns(
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
) -> list[type[LogitsProcessor]]:
    """Load logit processor types, identifying them by fully-qualified class
    names (FQCNs).

    Effectively, a mixed list of logitproc types and FQCN strings is converted
    into a list of entirely logitproc types, by loading from the FQCNs.

    FQCN syntax is <module>:<type> i.e. x.y.z:CustomLogitProc

    Already-loaded logitproc types must be subclasses of LogitsProcessor

    Args:
      logits_processors: Potentially mixed list of logitsprocs types and FQCN
                         strings for logitproc types

    Returns:
        List of logitproc types

    """

    if not logits_processors:
        return []

    logger.debug(
        "%s additional custom logits processors specified, checking whether "
        "they need to be loaded.",
        len(logits_processors),
    )

    classes: list[type[LogitsProcessor]] = []
    for ldx, logitproc in enumerate(logits_processors):
        if isinstance(logitproc, type):
            logger.debug(" - Already-loaded logit processor: %s", logitproc.__name__)
            if not issubclass(logitproc, LogitsProcessor):
                raise ValueError(
                    f"{logitproc.__name__} is not a subclass of LogitsProcessor"
                )
            classes.append(logitproc)
            continue

        logger.debug("- Loading logits processor %s", logitproc)
        module_path, qualname = logitproc.split(":")

        try:
            # Load module
            with guard_cuda_initialization():
                module = importlib.import_module(module_path)
        except Exception as e:
            logger.error(
                "Failed to load %sth LogitsProcessor plugin %s: %s",
                ldx,
                logitproc,
                e,
            )
            raise RuntimeError(
                f"Failed to load {ldx}th LogitsProcessor plugin {logitproc}"
            ) from e

        # Walk down dotted name to get logitproc class
        obj = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if not isinstance(obj, type):
            raise ValueError("Loaded logit processor must be a type.")
        if not issubclass(obj, LogitsProcessor):
            raise ValueError(f"{obj.__name__} must be a subclass of LogitsProcessor")
        classes.append(obj)

    return classes


# SOURCE: vllm/v1/sample/logits_processor/__init__.py:L159-L182
#   _load_custom_logitsprocs —— 逐字
def _load_custom_logitsprocs(
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
) -> list[type[LogitsProcessor]]:
    """Load all custom logits processors.

    * First load all installed logitproc plugins
    * Second load custom logitsprocs pass by the user at initialization time

    Args:
        logits_processors: potentially mixed list of logitproc types and
                           logitproc type fully-qualified names (FQCNs)
                           which need to be loaded

    Returns:
        A list of all loaded logitproc types
    """
    from vllm.platforms import current_platform

    if current_platform.is_tpu():
        # No logitsprocs specified by caller
        # TODO(andy) - vLLM V1 on TPU does not support custom logitsprocs
        return []

    return _load_logitsprocs_plugins() + _load_logitsprocs_by_fqcns(logits_processors)


# SOURCE: vllm/v1/sample/logits_processor/__init__.py:L185-L218 build_logitsprocs
#   —— 逐字（BUILTIN 三件套构造入口；spec 分支的互斥警告在此）
def build_logitsprocs(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors:
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug(
            "Skipping logits processor loading because pooling models"
            " do not support logits processors."
        )
        return LogitsProcessors()

    # Check if speculative decoding is enabled.
    if vllm_config.speculative_config:
        if custom_logitsprocs:
            raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
        logger.warning(
            "min_p and logit_bias parameters won't work with speculative decoding."
        )
        return LogitsProcessors(
            [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
        )

    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory)
        for ctor in itertools.chain(
            BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes
        )
    )


# SUBTRACTED: vllm/v1/sample/logits_processor/__init__.py:L221-L222
#   cached_load_custom_logitsprocs —— delete[6]（lru_cache 包装，只被已删的
#   validate_logits_processors_parameters 消费）。
# SUBTRACTED: vllm/v1/sample/logits_processor/__init__.py:L224-L237
#   validate_logits_processors_parameters —— delete[6]（请求级参数校验面，
#   调用方在 engine 边界，不在本章范围）。
# SUBTRACTED: vllm/v1/sample/logits_processor/__init__.py:L240-L347
#   AdapterLogitsProcessor —— delete[6]（OOT 请求级处理器包装面：req_info
#   partial 状态 + 逐请求 python 循环 apply）。


__all__ = [
    "LogitsProcessor",
    "LogitBiasLogitsProcessor",
    "MinPLogitsProcessor",
    "MinTokensLogitsProcessor",
    "LogitsProcessors",
    "build_logitsprocs",
    "STR_POOLING_REJECTS_LOGITSPROCS",
    "LOGITSPROCS_GROUP",
]
# SUBTRACTED: vllm/v1/sample/logits_processor/__init__.py:L354-L362 __all__
#   四名移除（计划明列）：AdapterLogitsProcessor / BatchUpdate /
#   BatchUpdateBuilder / MoveDirectionality —— 均为已删符号（星号导入防
#   AttributeError 的清理）。
