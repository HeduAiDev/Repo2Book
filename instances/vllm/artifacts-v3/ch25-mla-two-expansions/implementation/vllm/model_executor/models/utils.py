# SOURCE: vllm/model_executor/models/utils.py
# ch25 消费面：extract_layer_index（kv_cache_dtype_skip_layers 检查与 DSV4
# 逐层 compress_ratio 解析的层号提取）——逐字。
from __future__ import annotations


# SOURCE: vllm/model_executor/models/utils.py extract_layer_index —— 逐字
def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError if num_attn_module == 1
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )

        return int_vals[0]
    else:
        # SOURCE: num_attn_module > 1 的多头/尾段 —— 章界收窄（本章消费
        #   恒 num_attn_module=1）
        raise NotImplementedError("multi attn module names are out of ch25 scope")
