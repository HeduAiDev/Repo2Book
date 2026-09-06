# Subtract-only companion for v3 ch21 — vllm/utils/import_utils.py
# (pin v0.27.1 / 6e448d0ea). 本章消费面：resolve_obj_by_qualname——
# 「字符串→类」的懒加载原语（registry.get_class 与 selector._cached_get_
# attn_backend 都经它把类路径 import 成类对象，用到哪个才 import 哪个）。
from __future__ import annotations

import importlib
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L104-L110 resolve_obj_by_qualname ——（逐字）
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# SUBTRACTED: vllm/utils/import_utils.py 其余（import_pynvml / try_import_*
#   / _PlaceholderBase 占位模块族等）——平台/依赖探测域（ch17），本章零调用。
