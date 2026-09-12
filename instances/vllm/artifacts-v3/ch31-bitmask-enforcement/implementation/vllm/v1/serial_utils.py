# SOURCE: vllm/v1/serial_utils.py
# HOST SEAM：本章消费面一件——run_method（UniProcExecutor.collective_rpc 的
# 方法分发：字符串方法名 → getattr 直调；cloudpickle 字节/可调用分支删——
# 单进程同构转发用不到）。
from __future__ import annotations

from functools import partial
from typing import Any, Callable


# SOURCE: vllm/v1/serial_utils.py:L486-L513 run_method —— 消费切片承载
def run_method(
    obj: Any,
    method: str | bytes | Callable,
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
    # SUBTRACTED: isinstance(method, bytes) 的 cloudpickle 分支（多进程 RPC 面）
    if isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)
