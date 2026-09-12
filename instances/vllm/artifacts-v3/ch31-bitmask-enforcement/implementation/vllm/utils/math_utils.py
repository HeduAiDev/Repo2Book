# SOURCE: vllm/utils/math_utils.py
# HOST SEAM：本章消费面一件——cdiv（V2 StructuredOutputsWorker 的掩码缓冲
# 列数 ceil(V/32) 与 Triton grid 第二维）。逐字。
from __future__ import annotations


# SOURCE: vllm/utils/math_utils.py:L10-L12 cdiv —— 逐字
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)
