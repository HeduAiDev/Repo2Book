# SOURCE: vllm/utils/import_utils.py
# 只做减法的忠实精简版：本章消费面只有 LazyLoader（backend_xgrammar/
# backend_guidance 对 xgrammar/llguidance 的惰性导入）。
# SUBTRACTED: SPDX 版权头；文件其余（_has_module 可选依赖探测族等）。
import importlib
import sys
from types import ModuleType
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L341-L390 LazyLoader —— 逐字
class LazyLoader(ModuleType):
    """
    `LazyLoader` module borrowed from [Tensorflow]
    (https://github.com/tensorflow/tensorflow/blob/main/tensorflow/python/util/lazy_loader.py)
    with an addition of "module caching".

    Lazily import a module, mainly to avoid pulling in large dependencies.
    Modules such as `xgrammar` might do additional side effects, so we
    only want to use this when it is needed, delaying all eager effects.
    """

    # SOURCE: vllm/utils/import_utils.py:L352-L360（LazyLoader.__init__）
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

    # SOURCE: vllm/utils/import_utils.py:L364-L379
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

    # SOURCE: vllm/utils/import_utils.py:L381-L384
    def __getattr__(self, item: Any) -> Any:
        if self._module is None:
            self._module = self._load()
        return getattr(self._module, item)

    # SOURCE: vllm/utils/import_utils.py:L386-L390
    def __dir__(self) -> list[str]:
        if self._module is None:
            self._module = self._load()
        return dir(self._module)
