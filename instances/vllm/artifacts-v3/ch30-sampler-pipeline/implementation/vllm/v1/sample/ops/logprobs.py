# SOURCE: vllm/v1/sample/ops/logprobs.py
# ch29 step8 增量：rank=(x>=v).sum(-1) 不排序计数 + torch.compile 编译 +
# mark_unbacked 防批 1→2 重编译（m16；支路全景归 ch8）。
"""Some utilities for logprobs, including logits."""

import torch

from vllm.platforms import current_platform


@torch.compile(backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/logprobs.py:L10-L27 batched_count_greater_than —— 逐字（装饰器经 HOST SEAM platforms 的 simple_compile_backend， 真实值 "inductor"（vllm/platforms/interface.py:L165）、host 取 "eager" 同数学执行——ch8 已立同款 seam）
    """
    Counts elements in each row of x that are greater than the corresponding
    value in values.  Use torch.compile to generate an optimized kernel for
    this function. otherwise, it will create additional copies of the input
    tensors and cause memory issues.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.
    """
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)
