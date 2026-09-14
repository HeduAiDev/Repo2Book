# SOURCE: vllm/forward_context.py
# ch26 消费面（站 6/8）：ForwardContext 容器 + get_forward_context +
# set_forward_context——模型前向一拍的上下文。sparse_attn_indexer op 从这里
# 按 k_cache.prefix 取 DeepseekV32IndexerMetadata（metadata 不走参数、隐式
# 契约——ch19/ch25 已立）；cudagraph_runtime_mode 为 compressor 的 c128
# 边界早退消费位（host 恒 NONE=非 FULL → 判定按真实非图形态走）。
# SUBTRACTED：DPMetadata/批尺寸观测计时面/ubatch（DBO）——ch12/ch19/ch34 域。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


# SOURCE: vllm/forward_context.py:L132 ForwardContext —— 前向上下文本体
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    # SOURCE: vllm/forward_context.py:L134 no_compile_layers —— static_
    #   forward_context 的每拍快照
    no_compile_layers: dict[str, Any]
    # SOURCE: vllm/forward_context.py:L135 attn_metadata —— dict[layer_name]
    #   或 spec decode 的 list[dict]
    attn_metadata: dict[str, Any] | list[dict[str, Any]]
    # SOURCE: vllm/forward_context.py:L136 slot_mapping —— dict[layer_name]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    # SOURCE: vllm/forward_context.py:L135-L141 docstring（v1/DBO 两形态说明）
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # SOURCE: vllm/forward_context.py:L147-L148 cudagraph_runtime_mode ——
    #   ch19 域字段；默认 NONE
    cudagraph_runtime_mode: Any = None
    # SUBTRACTED: dp_metadata/batch_descriptor/all_moe_layers/skip_compiled/
    #   additional_kwargs/is_padding/ubatch_slices（vllm/forward_context.py
    #   L145-L167）——DP/MoE/DBO 域


_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L199-L205 get_forward_context —— 逐字
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    # SOURCE: vllm/forward_context.py:L199-L205 get_forward_context（锚点双置）
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py set_forward_context —— 上下文管理器位
@contextmanager
def set_forward_context(context: ForwardContext):
    """Set the current forward context."""
    # SOURCE: vllm/forward_context.py set_forward_context —— 逐字主干
    global _forward_context
    saved = _forward_context
    _forward_context = context
    try:
        yield
    finally:
        _forward_context = saved
