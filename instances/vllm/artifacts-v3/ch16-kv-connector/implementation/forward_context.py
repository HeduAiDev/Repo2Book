# SOURCE: vllm/forward_context.py
# ForwardContext 的最小镜像（worker 一拍生命周期的宿主之一）：no_compile_
# layers（层名→层实例——start_load_kv 逐层注入与 get_attention_context 都
# 从这取）、attn_metadata（逐层注意力元数据）、slot_mapping（逐层槽位表）；
# get/set_forward_context 的模块级全局（mixin 的 with 块进出的正是它）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   DPMetadata/cudagraph_runtime_mode/batch_descriptor/ubatch_slices/
#     is_padding/skip_compiled（DP/图捕获/ubatch 面——ch09/ch19）；
#   create_forward_context 的 compilation_config 装配链（set_forward_context
#     直构最小上下文——HOST SEAM）；
#   track_batchsize/perf 计时面。
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


# SOURCE: vllm/forward_context.py:L132 ForwardContext（最小面）
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    # SOURCE: vllm/forward_context.py:L134 no_compile_layers
    no_compile_layers: dict[str, Any]
    # SOURCE: vllm/forward_context.py:L135 attn_metadata
    attn_metadata: Any
    # set dynamically for each forward pass
    # SOURCE: vllm/forward_context.py:L136 slot_mapping
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # SUBTRACTED: dp_metadata/cudagraph_runtime_mode/batch_descriptor/
    #   ubatch_slices/is_padding/skip_compiled（L139-L160——DP/图捕获面）。


# SOURCE: vllm/forward_context.py:L~196 _forward_context 模块级全局
_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L199 get_forward_context
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    # SOURCE: vllm/forward_context.py:L201-L205
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py:L~207 is_forward_context_available
def is_forward_context_available() -> bool:
    # SOURCE: vllm/forward_context.py
    return _forward_context is not None


# SOURCE: vllm/forward_context.py:L~224 override_forward_context
@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    # SOURCE: vllm/forward_context.py（进出换全局、finally 还原）
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


# set_forward_context（HOST SEAM：真实版从 vllm_config.compilation_config.
#   static_forward_context 与 DP 协调装配；本章最小面直构——attn_metadata
#   =None 即 no_forward 的形态）
@contextmanager
# SOURCE: vllm/forward_context.py:L260
def set_forward_context(
    attn_metadata: Any,
    vllm_config: Any,
    num_tokens: int | None = None,
    slot_mapping: dict[str, torch.Tensor] | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    # SUBTRACTED: track_batchsize/DPMetadata/coordinate_batch_across_dp
    #   （L278-L310——观测与 DP 面）。
    with override_forward_context(
        ForwardContext(
            no_compile_layers=vllm_config.static_forward_context,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mapping if slot_mapping is not None else {},
        )
    ):
        yield
