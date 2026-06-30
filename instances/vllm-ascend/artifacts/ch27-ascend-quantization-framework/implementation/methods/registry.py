#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/methods/registry.py

整章「注册表范式」的核心数据结构：一个 dict + 一个装饰器 + 一个查表函数。
register_scheme 把 (quant_type, layer_type)->SchemeClass 登记进 _SCHEME_REGISTRY，
get_scheme_class 据此查表选 scheme；重复注册立即 ValueError 防冲突。
"""

from typing import Any

# SOURCE: vllm_ascend/quantization/methods/registry.py:L20
# Registry: maps (quant_type, layer_type) -> SchemeClass
_SCHEME_REGISTRY: dict[tuple[str, str], type[Any]] = {}


# SOURCE: vllm_ascend/quantization/methods/registry.py:L24
def register_scheme(quant_type: str, layer_type: str):
    """Decorator to register a quantization scheme.

    Example:
        @register_scheme("W8A8_DYNAMIC", "linear")
        class W8A8DynamicLinearScheme(AscendLinearScheme):
            ...
    """

    # SOURCE: vllm_ascend/quantization/methods/registry.py:L40
    def decorator(cls: type[Any]) -> type[Any]:
        key = (quant_type, layer_type)
        if key in _SCHEME_REGISTRY:
            raise ValueError(
                f"Scheme already registered for {quant_type}/{layer_type}: {_SCHEME_REGISTRY[key].__name__}"
            )
        _SCHEME_REGISTRY[key] = cls
        return cls

    return decorator


# SOURCE: vllm_ascend/quantization/methods/registry.py:L52
def get_scheme_class(quant_type: str, layer_type: str) -> type[Any] | None:
    """Get scheme class for given quant_type and layer_type."""
    return _SCHEME_REGISTRY.get((quant_type, layer_type))
