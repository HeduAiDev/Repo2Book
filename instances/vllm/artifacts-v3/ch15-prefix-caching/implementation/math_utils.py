# SOURCE: vllm/utils/math_utils.py
# 本章只消费 cdiv（块数 = ⌈tokens/block_size⌉，贯穿 find/分配/裁剪）。
# SUBTRACTED: 该模块其余函数（next_power_of_2/round_up/down/
#   largest_power_of_2_divisor 等——ch13 已建全量切面，本章哈希面不用）。
# SOURCE: vllm/utils/math_utils.py:L10 cdiv
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    # SOURCE: vllm/utils/math_utils.py:L12
    return -(a // -b)
