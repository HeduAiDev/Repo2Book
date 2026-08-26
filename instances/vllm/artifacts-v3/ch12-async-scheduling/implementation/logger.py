# SOURCE: vllm/logger.py
# 日志 HOST SEAM：真实 vllm/logger.py 装配 vLLM 的格式化/告警体系（数百行），
# 本章只消费 logger.debug / logger.warning_once 两个面（config/vllm.py 仲裁的
# 五个降级分支各打一行）。HOST SEAM：标准 logging + once 去重包装，接口面与
# 真实 init_logger 一致。
from __future__ import annotations

import logging

_seen: set[tuple[str, str]] = set()


# SOURCE: vllm/logger.py init_logger（HOST SEAM：标准 logging 替身）
def init_logger(name: str) -> logging.Logger:
    # SOURCE: vllm/logger.py init_logger
    log = logging.getLogger(name)
    log.addHandler(logging.NullHandler())

    # SOURCE: vllm/logger.py once-messaging wrapper（warning_once/info_once 去重）
    def _once(level: str, msg: str, *args):
        key = (level, msg)
        if key not in _seen:
            _seen.add(key)
            getattr(log, level)(msg, *args)

    log.warning_once = lambda msg, *args: _once("warning", msg, *args)
    log.info_once = lambda msg, *args: _once("info", msg, *args)
    log.debug_once = lambda msg, *args: _once("debug", msg, *args)
    return log
