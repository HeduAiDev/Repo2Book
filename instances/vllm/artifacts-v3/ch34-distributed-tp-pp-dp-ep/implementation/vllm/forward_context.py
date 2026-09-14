# SOURCE: vllm/forward_context.py
# ch34 切面（m21/m23）：DPMetadata（MoE dispatch sizes 的载体）+ ForwardContext
# 载体 + holder 机制 + set_forward_context 的 DP-metadata 装配段。SUBTRACTED：
# batchsize 观测（L22-L26）、DBO/ubatch_slices 细节面、MoE 冷启动字符串表——各归
# 其域。全部保留行对 pin v0.27.1 现核。

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config import ParallelConfig


# SOURCE: vllm/forward_context.py:L30-L58 BatchDescriptor —— HOST SEAM 载体
#   （cudagraph 派发描述符；ch19 域，本章只要字段面）。
@dataclass(frozen=True)
class BatchDescriptor:  # HOST SEAM
    # SOURCE: vllm/forward_context.py:L30-L58（锚点双置）
    num_tokens: int


# SOURCE: vllm/forward_context.py:L61-L69 _compute_sp_num_tokens — 逐字（锚点双置）
def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()


# SOURCE: vllm/forward_context.py:L73-L128 DPMetadata — 逐字
@dataclass
class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor

    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None

    # SOURCE: vllm/forward_context.py:L80-L99 make —— 逐字（static 工厂：
    #   DP>1 或 SP-MoE 断言 + 本 rank 行对账）
    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        # SOURCE: vllm/forward_context.py:L80-L99（锚点双置）
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

    # SOURCE: vllm/forward_context.py:L102-L113 sp_local_sizes —— 逐字
    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int):
        """
        Context manager for setting self.local_sizes. Same as self.chunked_sizes
        but without any chunking.
        """
        # SOURCE: vllm/forward_context.py:L102-L113（锚点双置）
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    # SOURCE: vllm/forward_context.py:L115-L117 get_chunk_sizes_across_dp_rank
    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    # Get the cumulative tokens across sequence parallel ranks.
    # In this case the input to the MoEs will be distributed w.r.t both
    # DP and TP rank.
    # When sp_size==1, this is just the cumulative num tokens across DP.
    # SOURCE: vllm/forward_context.py:L123-L128 cu_tokens_across_sp
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)


# SOURCE: vllm/forward_context.py:L132-L193 ForwardContext —— 字段子集（attn/
#   cudagraph/编译冷启动轴按 SUBTRACTED 压缩为最小载体）
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    # SOURCE: vllm/forward_context.py:L132-L193（锚点双置）
    no_compile_layers: dict[str, Any]
    attn_metadata: Any
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: Any = None
    batch_descriptor: BatchDescriptor | None = None
    # SUBTRACTED: ubatch_slices/is_padding/skip_compiled/all_moe_layers/
    #   moe_layer_index/additional_kwargs 与 __post_init__ 校验（L146-L193）——
    #   DBO/编译冷启动/平台注入域。


# SOURCE: vllm/forward_context.py:L196 _forward_context 模块级 holder — 逐字
_forward_context: ForwardContext | None = None


# SOURCE: vllm/forward_context.py:L199-L205 get_forward_context — 逐字（锚点双置）
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


# SOURCE: vllm/forward_context.py:L208-L209 is_forward_context_available —— 逐字
def is_forward_context_available() -> bool:
    return _forward_context is not None


# SOURCE: vllm/forward_context.py:L245-L256 override_forward_context — 逐字
@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    # SOURCE: vllm/forward_context.py:L245-L256（锚点双置）
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


# SOURCE: vllm/forward_context.py:L260-L376 set_forward_context —— 签名与
#   DP-metadata 装配段逐字（L272-L313）；观测/平台注入/静态上下文查表按
#   SUBTRACTED 压缩（batchsize 跟踪 L273-L276、cudagraph descriptor 便利段
#   L315-L317、current_platform.set_additional_forward_context L319-L329、
#   create_forward_context 的编译冷启动查表 L224-L227）。
@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: Any = None,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: Any = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    # SOURCE: vllm/forward_context.py:L260-L376（锚点双置）
    dp_metadata: DPMetadata | None = None
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
            from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

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

    # SUBTRACTED: batchsize 观测 / cudagraph descriptor 便利段 / 平台注入 /
    #   MoE 冷启动字符串表（L273-L276、L315-L329、create_forward_context L212-
    #   L241 的编译查表）——观测与编译域。
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        # SUBTRACTED: batchsize 收尾观测（L339-L357）。
        pass
