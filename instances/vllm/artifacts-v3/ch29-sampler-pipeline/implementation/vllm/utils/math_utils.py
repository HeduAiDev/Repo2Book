# SOURCE: vllm/utils/math_utils.py
# HOST SEAM：本章消费面一件——next_power_of_2（Triton wrapper 的 buffer
# 尺寸取整，topk_topp_triton.py:L922）。逐字。
from __future__ import annotations


# SOURCE: vllm/utils/math_utils.py:L15-L17 next_power_of_2 —— 逐字
def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)"""
    return 1 if n < 1 else 1 << (n - 1).bit_length()
