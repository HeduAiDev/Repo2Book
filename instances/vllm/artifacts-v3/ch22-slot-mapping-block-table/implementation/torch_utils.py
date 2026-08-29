# SOURCE: vllm/utils/torch_utils.py
# ch22 消费面：PIN_MEMORY（HOST SEAM：CPU host 无 pinned memory，取 False——
# 只影响拷贝速度、不影响行为分支）与 _resolve_layer_name（attention 算子按
# layer_name 取上下文时的解包）。LayerName/opaque 注册面（L845-L875）归
# ch19 编译域，本章 str 直通。
from __future__ import annotations

# HOST SEAM：host 无 pinned memory（vllm/utils/torch_utils.py PIN_MEMORY 的
# CPU 设备取值）。
PIN_MEMORY = False


# SOURCE: vllm/utils/torch_utils.py:L882 _resolve_layer_name（LayerName 解包）
def _resolve_layer_name(layer_name) -> str:
    """Unwrap a LayerName to str, or return str unchanged."""
    # HOST SEAM：host 无 LayerName opaque 类型——getattr 兜底 value，否则原样。
    return getattr(layer_name, "value", layer_name)
