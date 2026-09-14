# SOURCE: vllm/utils/math_utils.py
# HOST SEAM：本章消费的小数学件（真实定义处逐字）。
from __future__ import annotations


# SOURCE: vllm/utils/math_utils.py cdiv —— 逐字
def cdiv(a: int, b: int) -> int:
    # SOURCE: vllm/utils/math_utils.py cdiv
    return -(-a // b)


# SOURCE: vllm/utils/math_utils.py round_up —— 逐字
def round_up(value: int, k: int) -> int:
    # SOURCE: vllm/utils/math_utils.py round_up
    return (value + k - 1) // k * k


# SOURCE: vllm/utils/math_utils.py next_power_of_2? 无——next_power_of_2 在
#   triton（见 vllm/triton_utils.py HOST SEAM）；此处只承载 cdiv/round_up。
