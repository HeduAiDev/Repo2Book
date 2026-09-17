# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章用到的数学小工具（cdiv / round_up / round_down）。

# SOURCE: vllm/utils/math_utils.py:L1-L40
# SUBTRACTED: next_power_of_2 / floor_div 等其余工具——本章不消费（Triton 路径
#   已按减法计划删除）。
"""


# SOURCE: vllm/utils/math_utils.py:L10-L15
def cdiv(a: int, b: int) -> int:
    """Ceiling division: ceil(a / b)."""
    return -(a // -b)


# SOURCE: vllm/utils/math_utils.py:round_up 条目
def round_up(value: int, divisor: int) -> int:
    """Round up to the nearest multiple of divisor."""
    return ((value + divisor - 1) // divisor) * divisor


# SOURCE: vllm/utils/math_utils.py:round_down 条目
def round_down(value: int, divisor: int) -> int:
    """Round down to the nearest multiple of divisor."""
    return (value // divisor) * divisor


__all__ = ["cdiv", "round_up", "round_down"]
