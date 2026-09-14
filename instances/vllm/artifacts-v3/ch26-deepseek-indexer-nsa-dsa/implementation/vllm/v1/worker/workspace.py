# SOURCE: vllm/v1/worker/workspace.py
# HOST SEAM：WorkspaceManager 的 host 承载——get_simultaneous 按 (shape,
# dtype) 返回工作区张量（真实：单块 workspace 的视图复用；host：每次新分配，
# 数值语义等价——本章消费面为 indexer 的 K-gather workspace 与 topk 核
# RADIX_TOPK_WORKSPACE_SIZE）。真实锚 vllm/v1/worker/workspace.py:L31-…
# /L92-L…/L203-L213。
from __future__ import annotations

import torch

# HOST SEAM：真实实现对齐账 round_up(actual, 256)（v1/worker/workspace.py:
# L103-L104）；host 独立分配面数值等价，不引入该依赖。

_manager: "WorkspaceManager | None" = None


# SOURCE: vllm/v1/worker/workspace.py:L31 WorkspaceManager —— HOST SEAM
class WorkspaceManager:
    """Manager for workspace allocation.

    Manages one workspace buffer per active ubatch slot.
    Can be locked to prevent further growth during execution.
    """

    def __init__(self, device: torch.device, num_ubatches: int | None = None):
        # SOURCE: vllm/v1/worker/workspace.py:L38-… __init__ —— HOST SEAM
        self._device = device

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        """Get multiple workspace tensors simultaneously from a single allocation.

        Args:
            *shapes_and_dtypes: One or more (shape, dtype) tuples.

        Returns:
            List of tensor views into the workspace buffer, one per shape/dtype pair.
        """
        # SOURCE: vllm/v1/worker/workspace.py:L92-… get_simultaneous —— HOST
        #   SEAM（对齐账 round_up(., 256) 同式；分配面为独立张量）
        return [
            torch.empty(shape, dtype=dtype, device=self._device)
            for shape, dtype in shapes_and_dtypes
        ]


# SOURCE: vllm/v1/worker/workspace.py:L203-L213 current_workspace_manager
#   —— HOST SEAM：惰性单例（真实要求先 init_workspace_manager；host 自动
#   以 CPU 初始化——等价真实 runner 启动后的形态）
# SOURCE: vllm/v1/worker/workspace.py:L203-L213（锚点双置）
def current_workspace_manager() -> "WorkspaceManager":
    """Get the current workspace manager instance."""
    global _manager
    if _manager is None:
        _manager = WorkspaceManager(torch.device("cpu"))
    return _manager


# SOURCE: vllm/v1/worker/workspace.py init_workspace_manager —— HOST SEAM
def init_workspace_manager(device: torch.device) -> None:
    global _manager
    _manager = WorkspaceManager(device)
