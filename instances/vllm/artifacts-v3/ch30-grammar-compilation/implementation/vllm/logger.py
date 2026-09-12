# SOURCE: vllm/logger.py
# HOST SEAM：本章消费面只有 init_logger(name)（backend_xgrammar/backend_
# guidance/structured_output/__init__ 的模块级 logger = init_logger(__name__)；
# logger 调用本身已按 subtraction_plan.delete[7] 删除，init_logger 只剩构造面）。
# 真实 init_logger（vllm/logger.py:L204-L212）还会给 logger 打 patch
# （_METHODS_TO_PATCH 的 MethodType 绑定）——与本章控制流无关，退化用
# logging.getLogger。
# SUBTRACTED: SPDX 版权头；_VllmLogger/TraceLogger/配置面全链。
import logging


# SOURCE: vllm/logger.py:L204-L207 init_logger —— HOST SEAM（docstring 原话）
def init_logger(name: str) -> logging.Logger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root vllm logger has
    already been configured."""
    return logging.getLogger(name)
