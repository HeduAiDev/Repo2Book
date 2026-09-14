# SOURCE: vllm/utils/flashinfer.py
# HOST SEAM：has_flashinfer 恒 False（ch25 同款——host 无 flashinfer 包，
# 与真实未安装平台同型；SparseMLACommonImpl 的 concat 探测因此走直拷贝臂）。
from __future__ import annotations


# SOURCE: vllm/utils/flashinfer.py has_flashinfer —— HOST SEAM
def has_flashinfer() -> bool:
    # SOURCE: vllm/utils/flashinfer.py —— HOST SEAM
    return False
