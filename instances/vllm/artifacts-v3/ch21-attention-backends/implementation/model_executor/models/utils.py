# Subtract-only companion for v3 ch21 — vllm/model_executor/models/utils.py
# (pin v0.27.1 / 6e448d0ea). 本章消费面：extract_layer_index——bind_kv_cache
# 按 layer_name 里的整数段排出 runner 侧 kv_caches 列表序；validate_kv_
# sharing_target 的目标层先后判别也吃它。模型拼装工具族其余 → ch23。
from __future__ import annotations


# SOURCE: vllm/model_executor/models/utils.py:L917-L946 extract_layer_index
#   ——（逐字）从模块名取层号
def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:  # SOURCE: vllm/model_executor/models/utils.py
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
        assert len(int_vals) <= 2, (
            f"layer name {layer_name} should contain most two integers"
        )
        layer_index = (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
        return layer_index


# SUBTRACTED: vllm/model_executor/models/utils.py 其余（stack_labels /
#   PPMissingLayer / 自动 TP 装配族等）——模型拼装域（ch23），本章零调用。
