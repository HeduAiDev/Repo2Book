# SOURCE: vllm/utils/math_utils.py
# 本章消费面：cdiv——定账的全部换算算术（块数 cdiv、准入上限 cdiv、
# 并发核算 cdiv）。
# SUBTRACTED: 其余数学工具（next_power_of_2/largest_power_of_2_divisor 等）
#   ——本章链路不用（清零段的 largest_power_of_2_divisor 归 ch13 精简版）。


# SOURCE: vllm/utils/math_utils.py:L10 cdiv
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    # SOURCE: vllm/utils/math_utils.py:L12
    return -(a // -b)
