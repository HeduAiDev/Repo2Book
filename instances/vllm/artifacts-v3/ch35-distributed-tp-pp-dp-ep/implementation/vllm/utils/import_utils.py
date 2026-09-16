# SOURCE: vllm/utils/import_utils.py —— resolve_obj_by_qualname 逐字（L104-L110）；
# 其余（可选依赖探测族）属各消费章域，不携带。

from __future__ import annotations

import importlib
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L104-L110 resolve_obj_by_qualname
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
