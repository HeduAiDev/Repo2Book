# SOURCE: vllm/triton_utils/__init__.py
# HOST SEAM：Triton 可用性承载（ch29 同款 seam）。本章消费面 =
# `from vllm.triton_utils import tl, triton`（gpu/structured_outputs.py:L6 的
# kernel 语言面）与 HAS_TRITON（config/vllm.py use_v2_model_runner 的
# 『Triton 缺失回退 V1』判据）。真实包按 find_spec + 活跃驱动数裁决并给出
# Placeholder 兜底；HOST SEAM 保留 find_spec 探测 + 占位符结构。
from __future__ import annotations

from importlib.util import find_spec

# SOURCE: vllm/triton_utils/importing.py HAS_TRITON 探测 —— 逐字
#   （find_spec 双探测；真实另有活跃驱动数校验，无驱动需求时结论一致）
HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)

if HAS_TRITON:
    # SOURCE: vllm/triton_utils/__init__.py 真实 triton 导入位 —— 逐字
    import triton
    import triton.language as tl
else:
    # SOURCE: vllm/triton_utils/__init__.py Placeholder 兜底位 —— HOST SEAM
    #   最小占位（真实 TritonPlaceholder 仅在被误用时报错）
    class _TritonPlaceholder:
        def __getattr__(self, name: str):
            raise RuntimeError("triton is not installed")

    triton = _TritonPlaceholder()  # type: ignore[assignment]
    tl = _TritonPlaceholder()  # type: ignore[assignment]
