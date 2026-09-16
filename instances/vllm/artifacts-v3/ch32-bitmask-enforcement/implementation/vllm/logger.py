# SOURCE: vllm/logger.py
# HOST SEAM：本章消费面两件——init_logger（各 spine 文件的 logger 工厂）与
# warning_once（use_v2_model_runner 的回退告警）。机制逐字：_VllmLogger 的
# once 系方法在 init_logger 里 setattr 到标准 logger 实例上，once 语义由
# _print_warning_once 上的 @lru_cache 兜底。真实文件另有 LogScope 分布式
# 判据与 root logger 配置（L1-L100/L224 起——ch9 启动域），HOST 侧不展开。
from __future__ import annotations

import logging
from functools import lru_cache
from types import MethodType
from typing import Any, cast


# SOURCE: vllm/logger.py:L88-L92 _print_warning_once —— 逐字（once 语义本体）
@lru_cache
# SOURCE: vllm/logger.py:L88-L92 _print_warning_once —— 逐字（lru_cache once 语义本体）
def _print_warning_once(logger: logging.Logger, msg: str, *args: Any) -> None:
    # Set the stacklevel to 3 to print the original caller's line info
    logger.warning(msg, *args, stacklevel=3)


# SOURCE: vllm/logger.py:L105-L146 _VllmLogger —— 消费切片（仅 warning_once；
#   debug_once/info_once 同构，删）
# SOURCE: vllm/logger.py:L105-L146 _VllmLogger —— 消费切片（HOST SEAM）
class _VllmLogger(logging.Logger):
    # SOURCE: vllm/logger.py:L136-L145 warning_once —— 逐字（LogScope 判据删——ch9）
    def warning_once(self, msg: str, *args: Any, scope: str = "local") -> None:
        """
        As [`warning`][logging.Logger.warning], but subsequent calls with
        the same message are silently dropped.
        """
        # SUBTRACTED: _should_log_with_scope 的 LogScope 分布式判据（ch9）——
        #   "local" scope 在单进程演示部署下恒 True
        _print_warning_once(self, msg, *args)


# Pre-defined methods mapping to avoid repeated dictionary creation
# SOURCE: vllm/logger.py:L149-L153 _METHODS_TO_PATCH —— 切面
_METHODS_TO_PATCH = {
    "warning_once": _VllmLogger.warning_once,
}


# SOURCE: vllm/logger.py:L203-L214 init_logger —— 逐字
def init_logger(name: str) -> "_VllmLogger":
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root vllm logger has
    already been configured."""

    logger = logging.getLogger(name)

    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_VllmLogger, logger)
