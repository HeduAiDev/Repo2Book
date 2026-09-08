# SOURCE: vllm/distributed/__init__.py（真实为 parallel_state/utils/communication_op
# 的 re-export 门面）
# HOST SEAM：分布式面的单进程退化承载（ch34 领地——本章只消费层接口的切分
# 语义与 weight_loader 分片算术，单卡退化为恒等）。TP/PP rank 态集中在
# _TP_STATE（测试可拨动以构造 tp=2 等多重视角；真实源在进程组初始化里）。
from __future__ import annotations

from typing import Sequence

import torch

# SOURCE: vllm/distributed/parallel_state.py 进程组初始化态 —— HOST SEAM 载体
#   （rank/world 单进程默认 0/1；pp 同）
_TP_STATE = {"rank": 0, "size": 1, "pp_rank": 0, "pp_size": 1}


# SOURCE: vllm/distributed/utils.py:L60-L64 divide（逐字——层构造的整除账）
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    # SOURCE: vllm/distributed/utils.py:L53-L57 ensure_divisibility —— 逐字内联
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


# SOURCE: vllm/distributed/utils.py:L99-L137 split_tensor_along_last_dim
#   （RowParallelLinear input_is_parallel=False 支的split；逐字主体）
def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along the last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor into.
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    tensor_split = torch.split(tensor, last_dim_size, dim=last_dim)
    if not contiguous_split_chunks:
        return tensor_split

    # SOURCE: vllm/distributed/utils.py:L120-L130 contiguous 分支（逐字）
    chunk_sizes = [last_dim_size for _ in range(num_partitions)]
    chunk_sizes[-1] = tensor.size()[last_dim] - last_dim_size * (num_partitions - 1)
    return [
        chunk.contiguous() for chunk in torch.split(tensor, chunk_sizes, dim=last_dim)
    ]


# SOURCE: vllm/distributed/utils.py:L127-L175 get_pp_indices
#   均分 + 余层从尾往前摊给中段；VLLM_PP_LAYER_PARTITION 手工覆盖位保留
def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """Try to evenly distribute layers across partitions.

    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all but the last
    partition. The last partition is excluded because it often contains an
    additional norm layer and we are attempting to balance compute.

    If `pp_size > 2` and the number of remaining layers is
    `0 < x <= pp_size - 2` then the remaining layers are evenly distributed
    across the middle partitions. The first and last partitions are excluded
    because they contain the input and output embeddings respectively and
    we are attempting to reduce maximum memory consumption across partitions.
    """
    # SOURCE: vllm/distributed/utils.py:L127-L175 get_pp_indices
    from vllm import envs
    from vllm.logger import init_logger

    _logger = init_logger(__name__)
    partition_list_str = envs.VLLM_PP_LAYER_PARTITION
    if partition_list_str is not None:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError(
                "Invalid partition string: {}".format(partition_list_str)
            ) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
    else:
        layers_per_partition = num_hidden_layers // pp_size
        partitions = [layers_per_partition for _ in range(pp_size)]

        if remaining_layers := num_hidden_layers % pp_size:
            for i in range(2, remaining_layers + 2):
                partitions[-i] += 1
            _logger.info(
                "Hidden layers were unevenly partitioned: [%s]. "
                "This can be manually overridden using the "
                "VLLM_PP_LAYER_PARTITION environment variable",
                ",".join(str(p) for p in partitions),
            )

    start_layer = sum(partitions[:pp_rank])
    end_layer = start_layer + partitions[pp_rank]

    return (start_layer, end_layer)


# SOURCE: vllm/distributed/parallel_state.py:L2087-L2089 get_tensor_model_parallel_world_size
def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    return _TP_STATE["size"]


# SOURCE: vllm/distributed/parallel_state.py:L2092-L2094 get_tensor_model_parallel_rank
def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    return _TP_STATE["rank"]


# SOURCE: vllm/distributed/parallel_state.py:L409-L470 GroupCoordinator 的
#   rank 面 —— HOST SEAM：单组退化载体（rank_in_group/world_size 两个消费位）
class _GroupCoordinatorSeam:
    # SOURCE: vllm/distributed/parallel_state.py:L409 GroupCoordinator.__init__
    #   的 rank 面 —— HOST SEAM
    def __init__(self, rank: int, world_size: int):
        # SOURCE: vllm/distributed/parallel_state.py:L409 GroupCoordinator.__init__
        self.rank_in_group = rank
        self.world_size = world_size

    # SOURCE: vllm/distributed/parallel_state.py:L584-L586 first_rank —— HOST SEAM
    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L584-L586 first_rank —— HOST SEAM
        return 0

    # SOURCE: vllm/distributed/parallel_state.py:L588-L590 last_rank —— HOST SEAM
    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L588-L590 last_rank —— HOST SEAM
        return self.world_size - 1

    # SOURCE: vllm/distributed/parallel_state.py:L592-L595 is_first_rank
    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L592-L595 is_first_rank
        return self.rank_in_group == self.first_rank

    # SOURCE: vllm/distributed/parallel_state.py:L597-L600 is_last_rank
    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L597-L600 is_last_rank
        return self.rank_in_group == self.last_rank


# SOURCE: vllm/distributed/parallel_state.py:L1409-L1411 get_pp_group —— HOST SEAM
#   （真实断言 _PP 已初始化；host 退化返回单组协调器，rank 态从 _TP_STATE 读）
def get_pp_group() -> _GroupCoordinatorSeam:
    # SOURCE: vllm/distributed/parallel_state.py:L1409-L1411 get_pp_group —— HOST SEAM
    return _GroupCoordinatorSeam(_TP_STATE["pp_rank"], _TP_STATE["pp_size"])


# SOURCE: vllm/distributed/communication_op.py:L12-L14 tensor_model_parallel_all_reduce
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    # SOURCE: vllm/distributed/parallel_state.py GroupCoordinator 集合通信的
    #   world_size==1 旁路（同 gather L758-L761 的 bypass 形）——HOST SEAM 恒等
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    raise AssertionError("HOST SEAM: 多进程集合通信须进容器（ch34 领地）")


# SOURCE: vllm/distributed/communication_op.py:L17-L21 tensor_model_parallel_all_gather
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    # SOURCE: vllm/distributed/parallel_state.py all_gather 的 world_size==1
    #   旁路 —— HOST SEAM 恒等
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    raise AssertionError("HOST SEAM: 多进程集合通信须进容器（ch34 领地）")


# SOURCE: vllm/distributed/communication_op.py:L31-L35 tensor_model_parallel_gather
def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    # SOURCE: vllm/distributed/parallel_state.py:L758-L761 GroupCoordinator.gather
    #   ——「Bypass the function if we are using only 1 GPU」逐字语义
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    raise AssertionError("HOST SEAM: 多进程集合通信须进容器（ch34 领地）")
