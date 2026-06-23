# 只做减法的忠实精简版 —— 镜像 vllm/v1/sample/ops/logprobs.py（pin f3fef123）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: SPDX 版权头；`from vllm.platforms import current_platform` 及
# `@torch.compile(backend=current_platform.simple_compile_backend)` 装饰器 —— 仅做编译
# 后端选择，去掉后函数语义不变（torch.compile 只是生成优化内核，不改数值结果）。
import torch


def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/logprobs.py:L10-29
    """
    Counts elements in each row of x that are greater than the corresponding
    value in values.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.
    """
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)
