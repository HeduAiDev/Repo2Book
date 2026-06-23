# 精简版（subtract-only companion）—— vllm/distributed/communication_op.py
#
# 模型 forward 实际调用的公共 API：薄到只是 get_tp_group().<op>。
# 把『TP 维度的集合通信』钉死在一组自由函数上，模型代码无需知道
# GroupCoordinator 的存在——是连接模型层与本章分布式层的接缝。
from __future__ import annotations

import torch

from parallel_state import get_tp_group


# SOURCE: vllm/distributed/communication_op.py:L14
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


# SOURCE: vllm/distributed/communication_op.py:L19
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


# SOURCE: vllm/distributed/communication_op.py:L26
def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


# SUBTRACTED: tensor_model_parallel_gather（vllm/distributed/communication_op.py:
# L33-L44）与 all_reduce 等三原语同构、转发 get_tp_group().gather，本章用三大原语
# 讲清接缝即可。
