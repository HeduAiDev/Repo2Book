"""Subtract-only companion — qualname 字符串 → 真实类对象 的解析点。

规范源码：vllm/utils/import_utils.py
这是‘推迟的 import 真正发生’的落点：把 "vllm_ascend.platform.NPUPlatform"
这种字符串切成 (模块, 类名)，import 该模块、取出类对象。
"""
import importlib
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L104-L110
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
