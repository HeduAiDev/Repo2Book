# SOURCE: vllm/triton_utils/__init__.py
# HOST SEAM：Triton 可用性承载。本章消费面 = HAS_TRITON
# （topk_topp_sampler.py:L13/L15/L356/L360 的截断核分流判据）与
# `from vllm.triton_utils import tl, triton`（topk_topp_triton.py:L14 的
# kernel 语言面）。真实包（triton_utils/__init__.py + importing.py）按
# find_spec + 活跃驱动数裁决并给出 Placeholder 兜底；HOST SEAM 保留
# find_spec 探测 + 占位符结构，未装 triton 时 HAS_TRITON=False、
# topk_topp_sampler 跳过 triton import——与真实回退语义一致。
from __future__ import annotations

from importlib.util import find_spec

# SOURCE: vllm/triton_utils/importing.py:L13-L16 HAS_TRITON 探测 —— 逐字
#   （find_spec 双探测；真实另有活跃驱动数校验 L18-L60，host/CPU 无驱动
#   需求时探测结论一致）
HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)

if HAS_TRITON:
    # SOURCE: vllm/triton_utils/__init__.py:L8-L14 真实 triton 导入位 —— 逐字
    import triton
    import triton.language as tl
else:
    # SOURCE: vllm/triton_utils/__init__.py:L15-L23 Placeholder 兜底位
    #   —— HOST SEAM：最小占位（真实 TritonPlaceholder/TritonLanguagePlaceholder
    #   仅在被误用时报错；本章未装 triton 时根本不会 import 本包的 tl/triton）
    class _TritonPlaceholder:
        def __getattr__(self, name: str):
            raise RuntimeError("triton is not installed")

    triton = _TritonPlaceholder()  # type: ignore[assignment]
    tl = _TritonPlaceholder()  # type: ignore[assignment]
