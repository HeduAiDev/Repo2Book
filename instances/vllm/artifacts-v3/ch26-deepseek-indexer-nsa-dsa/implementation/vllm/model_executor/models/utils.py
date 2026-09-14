# SOURCE: vllm/model_executor/models/utils.py
# ch26 消费面：extract_layer_index（L917-… 逐字）+ make_layers 的单进程
# 承载（PP 分段面 → ch17 域；HOST SEAM：prefix 编号 + 逐层构造）。
from __future__ import annotations


# SOURCE: vllm/model_executor/models/utils.py:L917 extract_layer_index —— 逐字
def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    Extract the layer index from the module name.
    Examples:
        model.layers.0.self_attn.q_a_proj -> 0
        model.layers.18.self_attn.q_a_proj -> 18
    """
    # SOURCE: vllm/model_executor/models/utils.py:L917（锚点双置）
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Cannot extract layer index from {layer_name}")


# SOURCE: vllm/model_executor/models/utils.py make_layers —— HOST SEAM
#   （单进程退化：start=0/end=num_hidden_layers + 逐层 prefix 构造；PP 分段
#   与 scratch 面归 ch17）
def make_layers(num_hidden_layers, layer_fn, prefix="", return_modules=True):
    # SOURCE: vllm/model_executor/models/utils.py make_layers —— HOST SEAM
    layers = [layer_fn(prefix=f"{prefix}.{idx}") for idx in range(num_hidden_layers)]
    return 0, num_hidden_layers, layers
