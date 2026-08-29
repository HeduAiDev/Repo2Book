# Subtract-only companion for v3 ch19 — vllm/utils/import_utils.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions, each marked `# SUBTRACTED:`.
from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L104-L110 resolve_obj_by_qualname
def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
