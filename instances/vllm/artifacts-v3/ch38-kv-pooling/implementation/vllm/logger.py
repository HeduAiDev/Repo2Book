# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 日志器：init_logger + `*_once` 方法补丁。

# SOURCE: vllm/logger.py:L74-L152（_print_*_once / _VllmLogger / _METHODS_TO_PATCH）
# SUBTRACTED: 根 logger 配置（环境变量/uvicorn/每进程文件重定向，L160-L240）——
#   本章的观测面只有调试输出，不涉日志基建（与 ch37 同款删法）。
"""

from functools import lru_cache
from typing import Any, Hashable

# SOURCE: vllm/logger.py:L76-L90
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


# SOURCE: vllm/logger.py:L109-L152
# SUBTRACTED: scope="global"/"process" 的跨 rank 去重分支——本章单实例视角，
#   只保留 "local"（同进程同消息只打一次）。
# SOURCE: vllm/logger.py:L109-L152
class _VllmLogger:
    # SOURCE: vllm/logger.py:L109-L152
    """仅为提供类型信息；方法直接打到 logging.Logger 实例上。"""

    # SOURCE: vllm/logger.py:L118-L126
    def debug_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L118-L126
        _print_debug_once(self, msg, *args)

    # SOURCE: vllm/logger.py:L128-L136
    def info_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L128-L136
        _print_info_once(self, msg, *args)

    # SOURCE: vllm/logger.py:L138-L146
    def warning_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L138-L146
        _print_warning_once(self, msg, *args)

    # SOURCE: vllm/logger.py:L148-L152（error_once/exception_once 同族 *_once）
    def error_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L148-L152
        _print_warning_once(self, msg, *args)

    # SOURCE: vllm/logger.py:L148-L152
    def exception_once(self, msg: str, *args: Hashable, scope: str = "local") -> None:
        # SOURCE: vllm/logger.py:L148-L152
        _print_warning_once(self, msg, *args)


# SOURCE: vllm/logger.py:L148-L152
_METHODS_TO_PATCH = ["debug_once", "info_once", "warning_once", "error_once", "exception_once"]


# SOURCE: vllm/logger.py:L154-L190（init_logger）
# SUBTRACTED: logger 配置与格式化——直接取标准 logging.getLogger。
import logging


# SOURCE: vllm/logger.py:L154-L190
def init_logger(name: str) -> logging.Logger:
    # SOURCE: vllm/logger.py:L154-L190
    from types import MethodType

    logger = logging.getLogger(name)
    for method in _METHODS_TO_PATCH:
        if not hasattr(logger, method):
            fn = getattr(_VllmLogger, method)
            setattr(logger, method, MethodType(fn, logger))
    return logger
