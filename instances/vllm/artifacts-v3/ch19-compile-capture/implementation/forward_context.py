# Subtract-only companion for v3 ch19 — vllm/forward_context.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`).
#
# Deletions here (dossier subtraction_plan.delete #8):
#   - set_forward_context 的 batchsize 统计（头部使用行 L276-L279 与 finally
#     尾段 L345-L376，连同 try/finally 壳——内层 with 去壳保留）与模块级
#     track_batchsize/last_logging_time/forward_start_time/batchsize_logging_
#     interval/batchsize_forward_time 全局量（L22-L26）；
#   - DPMetadata 的 SP 扩展（_compute_sp_num_tokens L61-L69、sp_local_sizes
#     L101-L113、cu_tokens_across_sp L119-L128——sequence-parallel 扩展态）。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from ._host_seams import (
    UBatchSlices,
    coordinate_batch_across_dp,
    current_platform,
    init_logger,
)
from .config import CUDAGraphMode, VllmConfig
from .v1.attention.backend import AttentionMetadata

logger = init_logger(__name__)

# SUBTRACTED: batchsize 统计的模块级全局量（L22-L26——观测域，delete[8]）。


# SOURCE: vllm/forward_context.py:L29-L58 BatchDescriptor —— frozen 五字段：
#   CUDA graph 查表的 key，『形状全等』的形式化
@dataclass(frozen=True)
class BatchDescriptor:  # SOURCE: vllm/forward_context.py:L29-L58
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
    Number of distinct active LoRA adapters in this batch.
    When cudagraph_specialize_lora_count is enabled, separate CUDA graphs
    are captured for each num_active_loras value. This allows kernels
    (like fused_moe_lora) whose grid size depends on num_active_loras
    to be properly captured.
    """


# SUBTRACTED: _compute_sp_num_tokens（L61-L69——SP 扩展态，delete[8]）。


# SOURCE: vllm/forward_context.py:L72-L99 DPMetadata —— DP>1 的跨 rank
#   token 数载体（make 的三断言原文保留；SP 两方法随 delete[8] 删）
@dataclass
class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor

    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None

    @staticmethod
    def make(  # SOURCE: vllm/forward_context.py:L79-L99
        parallel_config: Any,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert (
            parallel_config.data_parallel_size > 1
            or parallel_config.use_sequence_parallel_moe
        )
        assert parallel_config.is_moe_model is not False
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        return DPMetadata(num_tokens_across_dp_cpu)

    # SUBTRACTED: sp_local_sizes/get_chunk_sizes_across_dp_rank/cu_tokens_
    #   across_sp（L101-L128——SP 扩展态，delete[8]）。


# SOURCE: vllm/forward_context.py:L131-L193 ForwardContext —— 每拍执行环境
#   的 thread-local 载体（fast_moe_cold_start 的 all_moe_layers 计数器族
#   L161-L186 随 MoE 域注记删；__post_init__ 的运行期模式断言保留）
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None
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

    # SUBTRACTED: all_moe_layers/moe_layer_index（L161-L186——fast_moe_cold_
    #   start 旧方案的字符串弹出计数器，MoE 域；LayerName opaque 是其根治）。

    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):  # SOURCE: vllm/forward_context.py:L190-L193
        assert self.cudagraph_runtime_mode.is_valid_runtime_mode(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L199-L205 get_forward_context —— 未设即
#   assert 崩（算子实现入口的隐性契约）
def get_forward_context() -> ForwardContext:  # SOURCE: vllm/forward_context.py:L199-L205
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py:L208-L209 is_forward_context_available
def is_forward_context_available() -> bool:
    return _forward_context is not None


# SOURCE: vllm/forward_context.py:L212-L241 create_forward_context ——
#   no_compile_layers 即 static_forward_context 的拷贝（fast_moe 分支随
#   MoE 域删）
def create_forward_context(  # SOURCE: vllm/forward_context.py:L212-L241
    attn_metadata: Any,
    vllm_config: VllmConfig,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    # SUBTRACTED: fast_moe_cold_start 的 all_moe_layers 装配（L224-L227
    #   ——MoE 域）。
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


# SOURCE: vllm/forward_context.py:L244-L256 override_forward_context
@contextmanager
def override_forward_context(forward_context: ForwardContext | None):  # SOURCE: vllm/forward_context.py:L244-L256
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


# SOURCE: vllm/forward_context.py:L259-L344 set_forward_context —— runner
#   每拍注入：DP 元数据协调、cudagraph 用时顺手补建 batch_descriptor、
#   平台注入 additional_kwargs、override 全局 context 包住 yield
@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
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
    # SUBTRACTED: batchsize 统计头部（L276-L279——观测域，delete[8]）。

    dp_metadata: DPMetadata | None = None
    # SOURCE: vllm/forward_context.py:L282-L310
    if (
        (
            vllm_config.parallel_config.data_parallel_size > 1
            or vllm_config.parallel_config.use_sequence_parallel_moe
        )
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if (
            num_tokens_across_dp is None
            and vllm_config.parallel_config.data_parallel_size > 1
        ):
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
            )
            assert num_tokens_across_dp is not None
        elif num_tokens_across_dp is None:
            assert num_tokens is not None
            num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int32)
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp
        )

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    # SOURCE: vllm/forward_context.py:L312-L316
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    additional_kwargs = current_platform.set_additional_forward_context(
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        dp_metadata=dp_metadata,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )

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

    # SUBTRACTED: try/finally 统计尾段壳（L342-L376——观测域，delete[8]）；
    #   去壳后内层 with 原文保留。
    with override_forward_context(forward_context):
        yield
