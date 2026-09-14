# SOURCE: vllm/utils/import_utils.py
# HOST SEAM（ch25 同款）：resolve_obj_by_qualname（selector 显式后端支的
# 类解析位）+ has_cutedsl 恒 False（DCP CuteDSL 路径已按 delete[0] 删——
# 此处为 compress/dequant 分派的 cutedsl 探针消费位，host 恒走 triton 支）。
from __future__ import annotations

import importlib


# SOURCE: vllm/utils/import_utils.py resolve_obj_by_qualname —— HOST SEAM
#   （真实为分层 import；host 单进程等价直 import）
def resolve_obj_by_qualname(qualname: str):
    # SOURCE: vllm/utils/import_utils.py resolve_obj_by_qualname
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# SOURCE: vllm/utils/import_utils.py has_cutedsl —— HOST SEAM：恒 False
def has_cutedsl() -> bool:
    # SOURCE: vllm/utils/import_utils.py has_cutedsl —— HOST SEAM
    return False
