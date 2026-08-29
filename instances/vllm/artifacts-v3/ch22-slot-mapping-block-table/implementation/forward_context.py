# SOURCE: vllm/forward_context.py
# ch22 切面（站11 + WC4）：ForwardContext.slot_mapping（dict[layer_name]，L136）
# 与 set_forward_context(slot_mapping=...)——逐层铺设的通道；模型前向里每个
# Attention 层按 layer_name 从这里取表（不透传、不污染 forward 签名）。
# torch.compile/DP/MoE 计数器面 → ch19/分布式/ch33。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from ._host_seams import CUDAGraphMode, UBatchSlices


# SOURCE: vllm/forward_context.py:L29 BatchDescriptor —— cudagraph 派发的
#   形状键（ch19 已立；本章 execute_model 由此取 num_tokens/num_reqs padded）
@dataclass(frozen=True)
# SOURCE: vllm/forward_context.py:L30 BatchDescriptor
class BatchDescriptor:
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
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


# SOURCE: vllm/forward_context.py:L131 ForwardContext —— 前向上下文本体
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, Any] | list[dict[str, Any]]
    # SOURCE: vllm/forward_context.py:L136 slot_mapping —— 逐层 slot_mapping
    #   表（本章写腿的通道；set_forward_context 每拍推进）
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

    ubatch_slices: UBatchSlices | None = None

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


_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L199 get_forward_context
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py:L208 is_forward_context_available
def is_forward_context_available() -> bool:
    return _forward_context is not None


# SOURCE: vllm/forward_context.py:L212 create_forward_context
def create_forward_context(
    attn_metadata: Any,
    vllm_config,
    dp_metadata: Any | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
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


# SOURCE: vllm/forward_context.py:L244 override_forward_context
@contextmanager
# SOURCE: vllm/forward_context.py:L245 override_forward_context
def override_forward_context(forward_context: ForwardContext | None):
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


# SOURCE: vllm/forward_context.py:L259 set_forward_context —— 每拍把
#   attn_metadata/slot_mapping 推进 thread-local 上下文的入口（站11 调用点）
@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
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
