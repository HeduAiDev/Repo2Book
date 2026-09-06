# Subtract-only companion for v3 ch21 — vllm/forward_context.py
# (pin v0.27.1 / 6e448d0ea). 本章切面（站 10-11）：ForwardContext 的
# attn_metadata（dict[layer_name]——_build_attention_metadata 铺设的产物）与
# slot_mapping（dict[layer_name]——写腿按层取槽位表）两通道 +
# set_forward_context 每拍推进入口（execute_model L4432-L4443 的调用点）。
# 模块级全局 _forward_context + override_forward_context 保存/恢复。
# torch.compile/DP/MoE 计数器面 → ch19/分布式/ch33（章界收窄）。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from ._host_seams import CUDAGraphMode


# SOURCE: vllm/forward_context.py:L29-L58 BatchDescriptor ——（逐字）cudagraph
#   派发的形状键（ch19 已立；本章 execute_model 由此取 padded 口径）
@dataclass(frozen=True)
class BatchDescriptor:  # SOURCE: vllm/forward_context.py:L29-L58
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly describe the padded
    batch for cudagraph.
    """

    num_tokens: int
    num_reqs: int | None = None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    uniform: bool = False
    """
    True if all the requests in the batch have the same number of tokens.
    """
    has_lora: bool = False
    """
    Whether this batch has active LoRA adapters.
    """
    num_active_loras: int = 0
    """
    Number of distinct active LoRA adapters in the batch.
    When cudagraph_specialize_lora_count is enabled, separate CUDA graphs are
    captured for each num_active_loras value. This allows kernels
    (like fused_moe_lora) whose grid size depends on num_active_loras
    to be properly captured.
    """


# SUBTRACTED: DPMetadata（L61-L~129——DP 域，ch12/分布式）。

# SUBTRACTED: UBatchSlices 类型面（DBO 域，ch12）——本章恒 None。


# SOURCE: vllm/forward_context.py:L131-L193 ForwardContext —— 前向上下文本体
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, Any] | list[dict[str, Any]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # set dynamically for each forward pass
    # SOURCE: vllm/forward_context.py:L145-L151（DP/cudagraph/ubatch 字段面）
    dp_metadata: Any | None = None
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    ubatch_slices: Any | None = None

    # Boolean mask over the token axis: True for padding rows that are not real
    # tokens. Consumers can use it to skip work for padded tokens. None when
    # the producer does not set it.
    is_padding: torch.Tensor | None = None

    # If True, bypass the compiled model call, e.g. by using .forward() directly
    skip_compiled: bool = False

    # SUBTRACTED: all_moe_layers/moe_layer_index（L161-L186）——MoE 冷启动
    #   计数器面 → ch33。

    # SOURCE: vllm/forward_context.py:L188
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    # SOURCE: vllm/forward_context.py:L190-L193 __post_init__ 的运行态校验
    def __post_init__(self):
        assert self.cudagraph_runtime_mode.is_valid_runtime_mode(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


# SOURCE: vllm/forward_context.py:L196 模块级全局（非 threading.local——通道
#   本身 ch19 已立）
_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L199-L205 get_forward_context ——（逐字）
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py:L208-L209 is_forward_context_available（逐字）
def is_forward_context_available() -> bool:
    return _forward_context is not None


# SOURCE: vllm/forward_context.py:L212-L241 create_forward_context ——（逐字
#   除 MoE 冷启动装配）
def create_forward_context(
    attn_metadata: Any,
    vllm_config,
    dp_metadata: Any | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: Any | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    # SUBTRACTED: fast_moe_cold_start 的 all_moe_layers 装配（L224-L227）
    #   ——ch33 MoE 域（恒 None）。

    # SOURCE: vllm/forward_context.py:L229-L241
    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
        additional_kwargs=additional_kwargs or {},
        is_padding=is_padding,
    )


# SOURCE: vllm/forward_context.py:L244-L256 override_forward_context ——（逐字）
@contextmanager
def override_forward_context(forward_context: ForwardContext | None):  # SOURCE: vllm/forward_context.py:L244-L256
    """A context manager to overrides the current forward context.
    This is used to override the forward context for a specific forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


# SOURCE: vllm/forward_context.py:L259-L344 set_forward_context —— 每拍把
#   attn_metadata/slot_mapping 推进模块级全局上下文的入口（站 10 调用点）
@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: Any | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    # SUBTRACTED: track_batchsize 计时观测（L276-L279）——观测域。

    # SUBTRACTED: DP metadata 装配块（L281-L310）——DP 域（单机不进；
    #   dp_metadata 恒 None）。
    dp_metadata: Any | None = None

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    # SOURCE: vllm/forward_context.py:L312-L316
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    # SUBTRACTED: current_platform.set_additional_forward_context（L318-L327）
    #   ——平台注入面（seam 平台无附加上下文，恒空 dict）。
    additional_kwargs: dict[str, Any] = {}

    # SOURCE: vllm/forward_context.py:L329-L344
    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
        is_padding=is_padding,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        # SUBTRACTED: batchsize 统计与周期日志（L346 起）——观测域。
        pass
