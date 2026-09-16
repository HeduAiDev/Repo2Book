# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章用到的数学小工具（SWA 窗口块数换算）。

# SOURCE: vllm/utils/math_utils.py:L1-L40（cdiv 定义段）
# SUBTRACTED: 其余工具（next_power_of_2/floor_div 等）——本章不消费。
"""


# SOURCE: vllm/utils/math_utils.py:L10-L15
def cdiv(a: int, b: int) -> int:
    """Ceiling division: ceil(a / b)."""
    return -(a // -b)


__all__ = ["cdiv"]
