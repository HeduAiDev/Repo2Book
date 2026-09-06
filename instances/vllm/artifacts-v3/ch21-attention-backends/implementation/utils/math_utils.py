# Subtract-only companion for v3 ch21 — vllm/utils/math_utils.py
# (pin v0.27.1 / 6e448d0ea). 本章消费面：cdiv（select_common_block_size 的
# 分块算术与 spec 的 max_num_blocks_per_req）与 round_up（FA3 scheduler_
# metadata 预分配的 4 对齐）。
from __future__ import annotations

import math


# SOURCE: vllm/utils/math_utils.py cdiv —— ceiling division
def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# SUBTRACTED: vllm/utils/math_utils.py 其余（is_power_of_2 / next_power_of_2 /
#   safe_int / precise_math 等）——本章零调用，ch03/ch13 域。


# SOURCE: vllm/utils/math_utils.py round_up —— 向上对齐到 multiple
def round_up(x: int, multiple: int) -> int:
    return int(math.ceil(x / multiple)) * multiple
