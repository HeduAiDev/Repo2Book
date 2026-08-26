# Subtract-only companion for v3 ch17 — vllm/utils/import_utils.py (pin
# v0.27.1 / 6e448d0ea)。本章只取 resolve_obj_by_qualname（延迟初始化的机制
# 件：worker_cls 字符串 → 真类）。
#
# SUBTRACTED: import_utils.py 的其余部分（get_vllm_optional_dependencies 等——
#   依赖清单工具面，本章不消费）。
#
# HOST SEAM: 真实 vllm 包缺席时，"vllm.…"-前缀的 qualname 经包 __init__ 预置
# 的 sys.modules 别名解析到本精简包（见 _host_seams.install_vllm_module_aliases）；
# 本函数本体逐字保留。

from __future__ import annotations

import importlib
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L104-L110 resolve_obj_by_qualname — 逐字
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
