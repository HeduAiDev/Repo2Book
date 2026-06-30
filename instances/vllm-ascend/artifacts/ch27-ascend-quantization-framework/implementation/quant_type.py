#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/quant_type.py

刻意做轻、side-effect free 的共享枚举：核心运行模块可只 import QuantType，
不触发整个量化包的重初始化。整章「按 quant_type 选 scheme」用的就是它。
"""

from enum import Enum


# SOURCE: vllm_ascend/quantization/quant_type.py:L26
class QuantType(Enum):
    """Quantization type enum for MoE schemes."""

    NONE = 0
    W8A8 = 1
    W4A8 = 2
    MXFP8 = 3
    W4A16 = 4
    MXFP4 = 5
    W4A8MXFP = 6
