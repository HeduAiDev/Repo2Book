# SOURCE: vllm/logger.py —— HOST SEAM：init_logger + *_once 族（ch09/ch17 同款）。
# 真实文件接 torch/os 环境配 logging；本 seam 只保留被精简版触碰的接口面。

from __future__ import annotations

import logging

_seen: set = set()


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
def init_logger(name: str):
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())

    # SOURCE: vllm/logger.py once-messaging wrapper (info_once/warning_once/debug_once)
    class _Once:  # HOST SEAM
        # SOURCE: vllm/logger.py once-wrapper construction
        def __init__(self, fn):
            self._fn = fn

        # SOURCE: vllm/logger.py once-wrapper call
        def __call__(self, msg, *args, **kwargs):
            key = (self._fn.__name__, msg)
            if key not in _seen:
                _seen.add(key)
                self._fn(msg, *args, **kwargs)

    log.info_once = _Once(log.info)
    log.warning_once = _Once(log.warning)
    log.debug_once = _Once(log.debug)
    return log
