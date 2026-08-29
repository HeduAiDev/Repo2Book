# SOURCE: vllm/utils/torch_utils.py
# 本章消费面：_resolve_layer_name（逐层钩子装饰器把 LayerName 解回 str；
# HOST SEAM：LayerName 的 torch opaque 包装面删——str 直通，语义等价）。
# SUBTRACTED: LayerName opaque 类型/其余 torch 工具族（ch12/19 各章切面）。


# SOURCE: vllm/utils/torch_utils.py:L882 _resolve_layer_name
def _resolve_layer_name(layer_name: "str | str") -> str:
    """Unwrap a LayerName to str, or return str unchanged."""
    # SOURCE: vllm/utils/torch_utils.py:L884（opaque 包装面删——恒 str）
    return layer_name
