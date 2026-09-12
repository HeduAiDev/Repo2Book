# SOURCE: vllm/utils/flashinfer.py
# HOST SEAM：has_flashinfer 探针的 host 退化——真实按 vllm_flashinfer 包
# 与算力代探测（utils/flashinfer.py 探测族）；host 无该包，恒 False
#（MLACommonImpl._use_flashinfer_concat_mla_k 因此走直拷贝 fallback 支——
# 真实「无 flashinfer」同型路径）。
from __future__ import annotations


# SOURCE: vllm/utils/flashinfer.py has_flashinfer —— HOST SEAM 恒 False
def has_flashinfer() -> bool:
    # SOURCE: vllm/utils/flashinfer.py has_flashinfer —— HOST SEAM
    return False
