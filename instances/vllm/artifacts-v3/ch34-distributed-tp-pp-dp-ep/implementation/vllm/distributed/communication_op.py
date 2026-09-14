# SOURCE: vllm/distributed/communication_op.py
# ch34 切面（m7）：整文件逐字——模型 forward 实际调用的公共 API（薄封装把
# 『TP 维度的集合通信』钉死在一组自由函数上）。

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


# SOURCE: vllm/distributed/communication_op.py:L12-L14 tensor_model_parallel_all_reduce
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


# SOURCE: vllm/distributed/communication_op.py:L17-L21 tensor_model_parallel_all_gather
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


# SOURCE: vllm/distributed/communication_op.py:L24-L28 tensor_model_parallel_reduce_scatter
def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


# SOURCE: vllm/distributed/communication_op.py:L31-L35 tensor_model_parallel_gather
def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


# SOURCE: vllm/distributed/communication_op.py:L38-L43 broadcast_tensor_dict
def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
