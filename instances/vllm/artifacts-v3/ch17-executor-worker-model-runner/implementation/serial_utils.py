# Subtract-only companion for v3 ch17 — vllm/v1/serial_utils.py (pin v0.27.1
# / 6e448d0ea)。本章只取 run_method（uni 直调的派发底层，m3）：str→getattr /
# bytes→cloudpickle / callable→直接调——与 mp 侧 worker_busy_loop 的三分支
# 同构，『一个抽象两种拓扑』的最小证据。
#
# SUBTRACTED: vllm/v1/serial_utils.py 的其余部分（MsgpackEncoder/Decoder
#   多帧零拷贝与张量/OOB 面、PydanticMsgspecMixin——ch05/ch09 的精简版持有；
#   本章无对应删除项，属邻章产物不入本档）。

from __future__ import annotations

from functools import partial
from typing import Any

import cloudpickle


# SOURCE: vllm/v1/serial_utils.py:L486-L514 run_method — 三分支派发
def run_method(
    obj: Any,
    method: str | bytes | Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)
