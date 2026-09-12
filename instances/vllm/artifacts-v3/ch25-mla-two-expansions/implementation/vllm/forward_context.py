# SOURCE: vllm/forward_context.py
# ch25 消费面（站 10）：ForwardContext 容器 + get_forward_context +
# create_forward_context + set_forward_context——模型前向一拍的上下文。
# MLAAttention.forward 直调路径从这里取 attn_metadata（dict[layer_name]）/
# kv_cache（经 no_compile_layers）/slot_mapping（ch23 同款骨架）。
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
    #   forward_context 的每拍快照（MLAAttention 自注册的账本在此被读）
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
    # SOURCE: vllm/forward_context.py:L145 dp_metadata —— SUBTRACTED 字段位
    #   （DP 域 ch34；留 None 默认保形）
    dp_metadata: Any = None
    # SOURCE: vllm/forward_context.py:L147-L148 cudagraph_runtime_mode ——
    #   ch19 域；默认 NONE
    cudagraph_runtime_mode: Any = None
    # SOURCE: vllm/forward_context.py:L149 batch_descriptor —— ch19 域
    batch_descriptor: Any = None
    # SUBTRACTED: all_moe_layers/skip_compiled/additional_kwargs/is_padding/
    #   ubatch_slices（vllm/forward_context.py:L150-L167）——MoE/编译/DBO 域


_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L199-L205 get_forward_context —— 逐字
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    # SOURCE: vllm/forward_context.py:L199-L205 get_forward_context
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py:L207-L208 is_forward_context_available —— 逐字
def is_forward_context_available() -> bool:
    # SOURCE: vllm/forward_context.py:L207-L208 is_forward_context_available
    return _forward_context is not None


# SOURCE: vllm/forward_context.py:L212-L241 create_forward_context —— 减法子集
def create_forward_context(
    attn_metadata: Any,
    vllm_config: Any,
    dp_metadata: Any = None,
    cudagraph_runtime_mode: Any = None,
    batch_descriptor: Any = None,
    ubatch_slices: Any = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    # SUBTRACTED: fast_moe_cold_start 的 all_moe_layers 装配——MoE 域
    # SOURCE: vllm/forward_context.py:L212-L241 create_forward_context —— 减法子集
    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
    )


# SOURCE: vllm/forward_context.py:L244-L257 override_forward_context —— 逐字
@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    # SOURCE: vllm/forward_context.py:L244-L257 override_forward_context
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


# SOURCE: vllm/forward_context.py:L258-L344 set_forward_context —— 减法子集
@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: Any,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: Any = None,
    batch_descriptor: Any = None,
    ubatch_slices: Any = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    # SUBTRACTED: track_batchsize 计时/DPMetadata 装配/cudagraph 批描述符
    #   便利段/platform.set_additional_forward_context
    #   （vllm/forward_context.py:L271-L326）——观测/DP/ch19 域
    # SOURCE: vllm/forward_context.py:L258-L344 set_forward_context —— 减法子集
    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        slot_mapping=slot_mapping,
        skip_compiled=skip_compiled,
        is_padding=is_padding,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        # SUBTRACTED: 批尺寸日志尾段（vllm/forward_context.py:L337-L344）
        pass
