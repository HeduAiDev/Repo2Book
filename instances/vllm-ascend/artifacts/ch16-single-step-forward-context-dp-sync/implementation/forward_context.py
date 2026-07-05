# vllm/forward_context.py —— subtract-only 精简版（ch15 配角：被昇腾包住的「基座」）
#
# 本文件只保留昇腾 set_ascend_forward_context 在 with set_forward_context(...) 这层
# 真正依赖的「基座语义」：建 ForwardContext（attn_metadata / dp_metadata /
# cudagraph_runtime_mode / batch_descriptor），dp_size>1 且 moe 时建 DPMetadata，
# 并经 current_platform.set_additional_forward_context 让平台注入 additional_kwargs
# ——这正是昇腾挂钩的入口。其余字段/计时/日志按 subtraction_plan.delete 折叠。
#
# 这些都是纯 Python 控制流，可在 host 跑（真实 all_reduce / DPMetadata 跨卡协调不真跑，
# 由测试以 num_tokens_across_dp 直接传入，对应「昇腾自己先算好再传进来」的真实路径）。
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

# SOURCE: vllm/forward_context.py:L1-L20（节选）
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.platforms import current_platform


@dataclass  # SOURCE: vllm/forward_context.py:L31
class BatchDescriptor:
    """Batch descriptor for cudagraph dispatching."""

    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0
    # SUBTRACTED: 字段上的大段 docstring（vllm/forward_context.py:L40-L60）—— 注释，删后语义不变。


@dataclass  # SOURCE: vllm/forward_context.py:L74
class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor
    local_sizes: list[int] | None = None

    @staticmethod  # SOURCE: vllm/forward_context.py:L80
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert parallel_config.data_parallel_size > 1
        assert parallel_config.is_moe_model is not False
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens
        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        return DPMetadata(num_tokens_across_dp_cpu)

    # SUBTRACTED: sp_local_sizes / get_chunk_sizes_across_dp_rank / cu_tokens_across_sp
    #   等 SP 切分辅助方法（vllm/forward_context.py:L98-L128）—— 与本章「基座注入 hook」正交。


@dataclass  # SOURCE: vllm/forward_context.py:L130
class ForwardContext:
    no_compile_layers: dict[str, Any]
    attn_metadata: Any
    slot_mapping: Any = None
    dp_metadata: DPMetadata | None = None
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None
    ubatch_slices: Any = None
    skip_compiled: bool = False
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    # SUBTRACTED: all_moe_layers / moe_layer_index 等 torch.compile 冷启动字符串规避字段
    #   （vllm/forward_context.py:L180-L181）—— 与本章 forward context 注入语义无关。


_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L192
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py:L205
def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: Any = None,
    slot_mapping: Any = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
):
    # SUBTRACTED: fast_moe_cold_start → static_all_moe_layers 分支（vllm/forward_context.py:L215-L218）。
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
    )


@contextmanager  # SOURCE: vllm/forward_context.py:L236
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context."""
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager  # SOURCE: vllm/forward_context.py:L251
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: Any = None,
    slot_mapping: Any = None,
    skip_compiled: bool = False,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    # SUBTRACTED: track_batchsize 计时 forward_start_time（vllm/forward_context.py:L266-L269）
    #   —— 性能日志统计，与上下文注入语义无关（subtraction_plan.delete）。

    dp_metadata: DPMetadata | None = None
    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if num_tokens_across_dp is None:
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
            )
            assert num_tokens_across_dp is not None
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp
        )

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given.
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    # 平台挂钩入口：昇腾就是经 current_platform.set_additional_forward_context 注入 additional_kwargs。
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
    )

    # SUBTRACTED: 收尾 with 内 kernel_config.ir_op_priority / vllm.ir.enable_torch_wrap 两层
    #   torch.compile 编译机制包裹（vllm/forward_context.py:L310-L330 区段，dossier embed elide 标注）
    #   —— 与本章「包基座再注入」的演示无关，只保留 override_forward_context 这层让上下文真正生效。
    with override_forward_context(forward_context):
        yield
    # SUBTRACTED: finally 中 batchsize 统计与周期性日志块（vllm/forward_context.py:L332-L362）
    #   —— 性能日志统计（subtraction_plan.delete）。


# SOURCE: vllm/forward_context.py（DP batch 协调入口；真实跨卡 all_reduce 不在 host 跑）
def coordinate_batch_across_dp(num_tokens_unpadded, parallel_config, allow_microbatching=False):
    # SUBTRACTED: 真实实现走 DPMetadata 的跨卡 all_reduce 协调。昇腾主路径**不经过这里**——
    #   它在 _sync_metadata_across_dp 里先把 num_tokens 算好、再以 num_tokens_across_dp 显式传入
    #   set_forward_context，故此分支在本章不被触达（保留签名以示基座原本的自动协调入口）。
    raise NotImplementedError(
        "Ascend pre-computes num_tokens_across_dp in _sync_metadata_across_dp; "
        "this auto-coordination path is not taken."
    )
