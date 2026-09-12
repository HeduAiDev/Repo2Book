# SOURCE: vllm/utils/import_utils.py
# HOST SEAM：本章消费面一件——LazyLoader（structured_output/utils.py 对
# xgrammar 的惰性导入：host 无 xgrammar 时 import 期不炸、首次属性访问才
# 触发）。逐字承载（含 _load 的父命名空间回写与 sys.modules 记忆）。
from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


# SOURCE: vllm/utils/import_utils.py class LazyLoader —— 逐字
class LazyLoader(ModuleType):
    """
    `LazyLoader` module borrowed from [TensorFlow]
    (https://github.com/tensorflow/tensorflow/blob/main/tensorflow/python/util/lazy_loader.py)
    with an addition of "module caching".

    Lazily import a module, mainly to avoid pulling in large dependencies.
    Modules such as `xgrammar` might do additional side effects, so we
    only want to use this when it is needed, delaying all eager effects.
    """

    # SOURCE: vllm/utils/import_utils.py LazyLoader.__init__ —— 逐字
    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, Any],
        name: str,
    ):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._module: ModuleType | None = None

        super().__init__(str(name))

    # SOURCE: vllm/utils/import_utils.py LazyLoader._load —— 逐字
    def _load(self) -> ModuleType:
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module
            # The additional add to sys.modules
            # ensures library is actually loaded.
            sys.modules[self._local_name] = module
        except ModuleNotFoundError as err:
            raise err from None

        # Update this object's dict so that if someone keeps a
        # reference to the LazyLoader, lookups are efficient
        # (__getattr__ is only called on lookups that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: Any) -> Any:
        # SOURCE: vllm/utils/import_utils.py LazyLoader.__getattr__ —— 逐字
        if self.__name__ not in sys.modules:
            self._load()
        return getattr(sys.modules[self.__name__], item)
