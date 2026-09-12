# SOURCE: vllm/utils/math_utils.py
# ch25 消费面：cdiv / round_up / round_down（chunked prefill 定容与块对齐）。
from __future__ import annotations


# SOURCE: vllm/utils/math_utils.py:L13-L15 cdiv —— 逐字
def cdiv(a: int, b: int) -> int:
    # SOURCE: vllm/utils/math_utils.py:L13-L15 cdiv
    return -(a // -b)


# SOURCE: vllm/utils/math_utils.py round_up —— 逐字
def round_up(value: int, divisor: int) -> int:
    # SOURCE: vllm/utils/math_utils.py round_up
    return value + (-value % divisor)


# SOURCE: vllm/utils/math_utils.py round_down —— 逐字
def round_down(value: int, divisor: int) -> int:
    # SOURCE: vllm/utils/math_utils.py round_down
    return value // divisor * divisor
