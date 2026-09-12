# SOURCE: vllm/utils/import_utils.py
# ch25 消费面：resolve_obj_by_qualname（MLAPrefillBackendEnum.get_class 与
# AttentionBackendEnum.get_class 的懒加载原语）——逐字。
from __future__ import annotations

import importlib


# SOURCE: vllm/utils/import_utils.py:L104-L109 resolve_obj_by_qualname —— 逐字
def resolve_obj_by_qualname(qualname: str) -> type:
    # SOURCE: vllm/utils/import_utils.py:L104-L109 resolve_obj_by_qualname
    """Resolve a fully qualified name of a class into the class object."""
    module_name, class_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
