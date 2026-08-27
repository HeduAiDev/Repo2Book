# SOURCE: vllm/utils/math_utils.py
# 本章消费面：cdiv（块数换算主算术）/ largest_power_of_2_divisor
# （KVBlockZeroer 的 blk_size 取小）。
# SUBTRACTED: 其余数学工具（next_power_of_2/等）——本章链路不用。


# SOURCE: vllm/utils/math_utils.py:L10 cdiv
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    # SOURCE: vllm/utils/math_utils.py:L12
    return -(a // -b)


# SOURCE: vllm/utils/math_utils.py:L30 largest_power_of_2_divisor
def largest_power_of_2_divisor(n: int) -> int:
    """Return the largest power-of-2 that divides *n* (isolate lowest set bit)."""
    # SOURCE: vllm/utils/math_utils.py:L32
    return n & (-n)
