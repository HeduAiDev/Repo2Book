# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 日志器：init_logger + `*_once` 方法补丁。

# SOURCE: vllm/logger.py:L74-L152（_print_*_once / _VllmLogger / _METHODS_TO_PATCH）
# SUBTRACTED: 根 logger 配置（环境变量/uvicorn/每进程文件重定向，L160-L240）——
#   本章的观测面只有调试输出，不涉日志基建。
"""

from functools import lru_cache
from typing import Any, Hashable

from vllm import envs

# SOURCE: vllm/logger.py:L74-L90
# SUBTRACTED: stacklevel=3（源码为打印原始调用方行号；本章单进程可读性不需要）


# SOURCE: vllm/logger.py:L76-L79
@lru_cache
def _print_debug_once(logger: Any, msg: str, *args: Hashable) -> None:
    # SOURCE: vllm/logger.py:L76-L79
    logger.debug(msg, *args)


# SOURCE: vllm/logger.py:L82-L85
@lru_cache
def _print_info_once(logger: Any, msg: str, *args: Hashable) -> None:
    # SOURCE: vllm/logger.py:L82-L85
    logger.info(msg, *args)


# SOURCE: vllm/logger.py:L88-L91
@lru_cache
def _print_warning_once(logger: Any, msg: str, *args: Hashable) -> None:
    # SOURCE: vllm/logger.py:L88-L91
    logger.warning(msg, *args)


# SOURCE: vllm/logger.py:L109-L145
# SUBTRACTED: scope="global"/"process" 的跨 rank 去重分支（需要分布式 rank 状态）——
#   本章单实例视角，只保留 "local"（同进程同消息只打一次）。
# SOURCE: vllm/logger.py:L109-L152
class _VllmLogger:
    # SOURCE: vllm/logger.py:L109-L152
    """仅为提供类型信息；方法直接打到 logging.Logger 实例上。"""

    # SOURCE: vllm/logger.py:L118-L126
    def debug_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L118-L126
        _print_debug_once(self, msg, *args)  # type: ignore[arg-type]

    # SOURCE: vllm/logger.py:L127-L135
    def info_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L127-L135
        _print_info_once(self, msg, *args)  # type: ignore[arg-type]

    # SOURCE: vllm/logger.py:L136-L145
    def warning_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L136-L145
        _print_warning_once(self, msg, *args)  # type: ignore[arg-type]


_METHODS_TO_PATCH = {
    "debug_once": _VllmLogger.debug_once,
    "info_once": _VllmLogger.info_once,
    "warning_once": _VllmLogger.warning_once,
}

import logging  # noqa: E402

for _name, _fn in _METHODS_TO_PATCH.items():
    setattr(logging.Logger, _name, _fn)


# SOURCE: vllm/logger.py:L204-L240
def init_logger(name: str) -> logging.Logger:
    """同源码签名：取（或建）一个 vllm 命名空间下的 logger。"""
    _configure_vllm_root_logger()
    return logging.getLogger(name)


# SOURCE: vllm/logger.py:L156-L200
# SUBTRACTED: dictConfig 全量配置（handler/formatter/环境变量）——只留一行默认 handler，
#   保证 `logger.setLevel(logging.DEBUG)` 之类的测试可控，且不污染宿主日志。
# SOURCE: vllm/logger.py:L156-L200
def _configure_vllm_root_logger() -> None:
    # SOURCE: vllm/logger.py:L156-L200
    logging_config: dict[str, dict[str, Any] | Any] = {}
    if not logging_config:
        return  # 本章不配置（宿主/测试自行决定 handler）


__all__ = ["init_logger"]
