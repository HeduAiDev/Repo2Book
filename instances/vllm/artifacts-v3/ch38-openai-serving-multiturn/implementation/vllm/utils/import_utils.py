# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/utils/import_utils.py —— HOST SEAM（最小承载）：本章消费面
# 只有 import_from_path（parser 插件加载，ToolParserManager/
# ReasoningParserManager.import_*_parser 消费）与 resolve_obj_by_qualname
# （get_logits_processors 的自定义 logits processor 解析）。
import importlib
import importlib.util
import os
import sys
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L85-L99 —— import_from_path 逐字
def import_from_path(module_name: str, file_path: str | os.PathLike):
    """
    Import a Python file according to its file path.

    Based on the official recipe:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named {module_name!r}")

    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# SOURCE: vllm/utils/import_utils.py:L104-L109 —— resolve_obj_by_qualname 逐字
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# SUBTRACTED: vllm/utils/import_utils.py 其余（get_vllm_optional_dependencies/
# _PlaceholderBase/动态库探测）——归各自域。
