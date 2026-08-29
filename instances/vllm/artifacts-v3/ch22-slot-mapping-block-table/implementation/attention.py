# SOURCE: vllm/model_executor/layers/attention/attention.py
# ch22 切面（站12 + WC4）：写腿通道三件——get_attention_context（按 layer_name
# 从 ForwardContext 取 attn_metadata/attn_layer/kv_cache/slot_mapping）、
# unified_kv_cache_update（KV 写独立成算子：返回空张量作 dummy 数据依赖保
# torch.compile 顺序）、unified_attention_with_output（读腿算子、吃同一 dummy）。
# Attention 层类本体/RoPE/量化投影面归 ch17/ch20/ch21。
from __future__ import annotations

import torch

from .forward_context import ForwardContext, get_forward_context
from .torch_utils import _resolve_layer_name


# SOURCE: vllm/model_executor/layers/attention/attention.py:L~730
#   get_attention_context —— 按 layer_name 取本层的执行环境（不透传、不污染
#   模型 forward 签名——ch19 已立的算子化纪律）
def get_attention_context(layer_name: str):
    """Get the attention context for a specific layer.

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
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L754-L764
    #   （attn_metadata 按 dict/list/裸值三态解包——DBO list 形态归 ch12/33）
    forward_context: ForwardContext = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    attn_metadata: object
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    elif isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding
        # where [0] is the base-model (non-speculative) metadata dict.
        attn_metadata = attn_metadata_raw[0][layer_name]
    else:
        attn_metadata = attn_metadata_raw
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L765-L771
    #   （no_compile_layers 按 layer_name 取层实例；slot_mapping 表同源）
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


# SOURCE: vllm/model_executor/layers/attention/attention.py:L775
#   unified_kv_cache_update —— KV 写独立算子（站12 写腿入口）
def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L784-L796（逐字）
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L798（空张量
    #   dummy——数据依赖的物理载体）
    return key.new_empty(0)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L801
#   unified_kv_cache_update_fake（torch.compile fake 模式实现）
def unified_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L806
    return torch.empty(0, device=key.device, dtype=key.dtype)

# SUBTRACTED: direct_register_custom_op 注册块（L809-L814）——torch.library
#   注册机制归 ch19（算子化全景）；本章直调 Python 函数体（同一控制流），
#   注册与否不改变逐 token 语义。


# SOURCE: vllm/model_executor/layers/attention/attention.py:L817
#   unified_attention_with_output —— attention 读腿算子
# SUBTRACTED: @eager_break_during_capture / @maybe_transfer_kv_layer 装饰器
#   （ch19 捕获纪律 / ch16 KV 层搬运）——装饰面归彼章，函数体逐字。
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L833-L846（逐字）
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


# SOURCE: vllm/model_executor/layers/attention/attention.py:L849
#   unified_attention_with_output_fake
# SOURCE: vllm/model_executor/layers/attention/attention.py:L849 unified_attention_with_output_fake
def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return
