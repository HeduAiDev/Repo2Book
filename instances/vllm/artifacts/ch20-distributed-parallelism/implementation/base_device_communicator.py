# 精简版（subtract-only companion）—— vllm/distributed/device_communicators/base_device_communicator.py
#
# device 端集合通信的抽象基类：默认实现全部基于 torch.distributed，作用在
# device_group 上。CudaCommunicator 会覆写它们以走 pynccl/CustomAllreduce，
# 但语义与这份默认实现一致——本章用这份默认实现讲清『device 后端』的语义。
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


# SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L118
class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L126
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name

        # SUBTRACTED: stateless process group 分支（弹性 EP 用，
        # base_device_communicator.py:L140-L154）——本章按非弹性主线，群组都由
        # torch.distributed.new_group 创建，直接走 dist.* 查询 rank/world_size。
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.rank_in_group = dist.get_group_rank(self.cpu_group, self.global_rank)

        # SUBTRACTED: use_ep / all2all_manager 的初始化（MoE 专家并行的 all2all
        # 后端，base_device_communicator.py:L163-L178）属 MoE 章节，删后集合原语
        # 与 P2P 不依赖它们。
        self.is_ep_communicator = unique_name.split(":")[0] == "ep"

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L180
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L184
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    # SUBTRACTED: all_gatherv（变长，base_device_communicator.py:L213-L226）——
    # 本章用等长 all_gather 讲清 concat-style 重组的语义。

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L228
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()
        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        # Perform reduce-scatter operation
        torch.distributed.reduce_scatter_tensor(
            output_tensor, input_tensor, group=self.device_group
        )

        # Reshape before returning
        return output_tensor.movedim(0, dim).contiguous()

    # SUBTRACTED: reduce_scatterv（变长，base_device_communicator.py:L250-L253，
    # 默认 NotImplementedError），理由同 all_gatherv。

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L255
    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L286
    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L293
    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L305
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor
        torch.distributed.broadcast(tensor, self.ranks[src], self.device_group)
        return tensor

    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L312
    def destroy(self):
        pass

    # SUBTRACTED: prepare_communication_buffer_for_model / all2all dispatch / combine
    # （MoE 专用，base_device_communicator.py:L315-L373）——本章不涉及 MoE 专家分发。
