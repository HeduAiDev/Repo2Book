# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""函数签名小工具（本章消费面：supports_kw——out-of-tree connector 构造签名门）。

# SOURCE: vllm/utils/func_utils.py:L60-L125（supports_kw）
# SUBTRACTED: get_allowed_kwarg_only_overrides 等其余工具。
"""

import inspect
from collections.abc import Callable
from functools import lru_cache
from typing import Any


# SOURCE: vllm/utils/func_utils.py:L60-L99（_supports_kw，lru_cache 包一层）
@lru_cache(maxsize=None)
def _supports_kw(
    callable: Callable[..., object],
    kw_name: str,
    requires_kw_only: bool = False,
    allow_var_kwargs: bool = True,
) -> bool:
    # SOURCE: vllm/utils/func_utils.py:L60-L99
    """Check if a keyword is a valid kwarg for a callable; if requires_kw_only
    disallows kwargs names that can also be positional arguments.
    """
    sig = inspect.signature(callable)
    for p in sig.parameters.values():
        if p.name == kw_name:
            return p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        if p.kind == inspect.Parameter.VAR_KEYWORD and allow_var_kwargs:
            return True
    return False


# SOURCE: vllm/utils/func_utils.py:L100-L116
def supports_kw(
    callable: Callable[..., object],
    kw_name: str,
    *,
    requires_kw_only: bool = False,
    allow_var_kwargs: bool = True,
) -> bool:
    # SOURCE: vllm/utils/func_utils.py:L100-L116
    """Check if a keyword is a valid kwarg for a callable."""
    # Unwrap bound methods so that the lru_cache key is the underlying
    # function, not the instance.
    if hasattr(callable, "__func__"):
        callable = callable.__func__
    return _supports_kw(
        callable,
        kw_name,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )


__all__ = ["supports_kw"]
