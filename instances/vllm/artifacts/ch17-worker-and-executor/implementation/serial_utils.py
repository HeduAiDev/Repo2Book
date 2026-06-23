# 只做减法的忠实精简版 —— 镜像两个被执行器/worker 复用的工具：
#   run_method            ← vllm/v1/serial_utils.py:L486-L510
#   resolve_obj_by_qualname ← vllm/utils/import_utils.py（最小镜像）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: serial_utils.py 中 MsgpackEncoder/MsgpackDecoder/PydanticMsgspecMixin 等数百行
#   序列化机制（vllm/v1/serial_utils.py 其余部分）—— 与本章控制平面派发无关，只取 run_method。

import importlib
from collections.abc import Callable
from functools import partial
from typing import Any

import cloudpickle


# SOURCE: vllm/utils/import_utils.py  resolve_obj_by_qualname
def resolve_obj_by_qualname(qualname: str) -> Any:
    """Resolve an object by its fully qualified name."""
    # SUBTRACTED: 真实实现含对 None/空串的兜底与缓存；这里保留 rsplit('.',1)→import_module→getattr
    #   这条核心解析路径——『字符串类名 → 类对象』正是 init_worker 延迟实例化的关键能力。
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# SOURCE: vllm/v1/serial_utils.py:L486-L510
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
