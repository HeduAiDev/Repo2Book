# SOURCE: vllm/model_executor/layers/attention/attention.py
# 本章消费面：get_attention_context——逐层钩子装饰器（kv_transfer_utils）
# 从 forward context 抽出 (attn_metadata, 层实例, kv_cache, slot_mapping)
# 的访问器。契约最深的挂点长在模型层：装饰器经它拿层实例与张量。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   Attention/MLAAttention 层类族与 unified_kv_cache_update 等 kernel 面
#     （→ ch17/20）；DBO 双微批的 list 形态（spec decode → ch33——本章
#     单 dict 形态）。
from .forward_context import get_forward_context


# SOURCE: vllm/model_executor/layers/attention/attention.py:L732
#   get_attention_context
def get_attention_context(layer_name: str) -> tuple:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current forward pass
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L757-L763
    #   （dict 形态：层名→该层元数据；list 形态随 DBO 删）
    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    else:
        attn_metadata = attn_metadata_raw
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L765-L766
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L767-L772
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping
