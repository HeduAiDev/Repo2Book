#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/methods/__init__.py

全表装载点：import 各 scheme 模块即触发它们的 @register_scheme 装饰器执行（注册表被填满的时机）。
is_mx_quant_type 标出哪些 scheme 是 MXFP microscaling 类型——MXFP 是 NPU 硬特化。
"""

from typing import Any

# Import base classes
from .base import AscendAttentionScheme, AscendLinearScheme, AscendMoEScheme, QuantType

# Import registry functions
from .registry import get_scheme_class, register_scheme

# Import scheme classes —— import 即触发 @register_scheme 落表。
# SUBTRACTED: 十余个 scheme 模块的 import（fp8 / w4a4_* / w4a8_* / w8a8_mxfp8 / w8a16 ... ，原 __init__.py:L36-L52）。
#   本章只走通 W8A8_DYNAMIC 一条全链，故只保留 w8a8_dynamic 这一个代表性 scheme 模块。
from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod, AscendW8A8DynamicLinearMethod


# SOURCE: vllm_ascend/quantization/methods/__init__.py:L55
def is_mx_quant_type(instance: Any) -> bool:
    """Checks if the quantization method is a microscaling (MX) type."""
    # SUBTRACTED: 原 6-元组 MX scheme 类（AscendW8A8MXFP8DynamicLinearMethod / AscendW4A4MXFP4* /
    #   AscendW4A8MXFP* 等，原 __init__.py:L57-L64）——本章不复现 MX scheme，置空元组（恒 False）。
    MX_QUANT_TYPES: tuple = ()
    return isinstance(instance, MX_QUANT_TYPES)


__all__ = [
    "AscendAttentionScheme",
    "AscendLinearScheme",
    "AscendMoEScheme",
    "QuantType",
    "register_scheme",
    "get_scheme_class",
    "is_mx_quant_type",
    "AscendW8A8DynamicLinearMethod",
    "AscendW8A8DynamicFusedMoEMethod",
]
