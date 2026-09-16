# SOURCE: vllm/distributed/device_communicators/base_device_communicator.py
# ch34 切面：整文件近逐字（All2AllManagerBase 基类契约 + DeviceCommunicatorBase
# 的集合/P2P 用户面 + dispatch/combine no-op 兜底）。删除仅两处：ray_communicator
# 相关的 CACHE 键清理与 checkpoint 面按域裁除——见行内标记。

from __future__ import annotations

import threading
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.utils import is_moe_layer

logger = init_logger(__name__)


# SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L16-L30 Cache
class Cache:
    def __init__(self):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L17-L19（锚点双置）
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def get_or_create(self, kwargs, func):
        # Create a hashable key from the kwargs
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L21-L30（锚点双置）
        key = tuple(sorted((k, v) for k, v in kwargs.items()))

        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                instance = func(**kwargs)
                self._cache[key] = instance
            return instance


# SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L33-L159
#   All2AllManagerBase —— 逐字
class All2AllManagerBase:
    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L33-L159（锚点双置）
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L37-L68（锚点双置）
        self.cpu_group = cpu_group
        self.tcp_store_group = tcp_store_group

        # compute some common properties
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_tp_group,
            in_the_same_node_as,
        )

        # all2all lives in ep group, which is merged from dp and tp group
        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()

        # no self.ep_group since self.ep_group is still in construction
        # when we create this object
        self.dp_rank = self.dp_group.rank_in_group
        self.dp_world_size = self.dp_group.world_size
        self.rank = cpu_group.rank()
        self.world_size = cpu_group.size()

        # all2all communication often has separate implementations for
        # intra-node and inter-node communication
        if tcp_store_group is None:
            self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
        else:
            self.internode = not all(
                in_the_same_node_as(tcp_store_group, source_rank=0)
            )

        self.support_fault_tolerance = False

    def get_handle(self, kwargs):
        # get a handle for the all2all communication,
        # based on the kwargs.
        # different layers can have different configs,
        # e.g. one layer has hidden size 1024, another has 2048.
        # usually the underlying implementation caches the handle
        # and reuse it for the same config.
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L70-L77（锚点双置）
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        # Subclasses should either:
        # - implement handling for extra_tensors, or
        # - raise a clear error if extra_tensors is not supported.
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L79-L92（锚点双置）
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        # Subclasses should either:
        # - implement handling for extra_tensors, or
        # - raise a clear error if extra_tensors is not supported.
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L94-L108（锚点双置）
        raise NotImplementedError

    def query_active_mask(self) -> torch.Tensor:
        """Return the all2all liveness mask for the EP ranks.

        Returns:
            An int32 device tensor where 0 marks a live rank and 1 marks a
            masked (dead/unreachable) rank.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L110-L117（锚点双置）
        raise NotImplementedError

    def query_fault(self) -> torch.Tensor:
        """Return a scalar bool tensor, True if a new fault appeared.

        Compares the current mask against the baseline recorded at the last
        recovery point.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L119-L125（锚点双置）
        raise NotImplementedError

    def clean_buffers(self) -> None:
        """Reset this rank's RDMA buffers and all2all mask state (rank-local).

        Post-fault cleanup: a dispatch/combine that hit a dead peer or timed
        out can leave partially-written or stale tokens in the RDMA receive
        buffer, so it is zeroed to stop the next forward from reading that
        contaminated data.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L127-L135（锚点双置）
        raise NotImplementedError

    def set_num_sms(self, num_sms: int):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L137-L138（锚点双置）
        pass

    def max_sms_used(self) -> int | None:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L140-L141（锚点双置）
        return None  # None means it could use the whole GPU

    def checkpoint_prepare(self) -> None:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L143-L147（锚点双置）
        logger.warning_once(
            "%s.checkpoint_prepare is not implemented; skipping.",
            type(self).__name__,
        )

    def checkpoint_restore(self) -> None:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L149-L153（锚点双置）
        logger.warning_once(
            "%s.checkpoint_restore is not implemented; skipping.",
            type(self).__name__,
        )

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L155-L156（锚点双置）
        raise NotImplementedError

    def destroy(self):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L158-L159（锚点双置）
        pass


# SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L162-L419
#   DeviceCommunicatorBase —— 逐字
class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    It can use the `cpu_group` to initialize the communicator.
    If the device has PyTorch integration (PyTorch can recognize its
    communication backend), the `device_group` will also be given.
    """
    # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L162-L419（锚点双置）

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        use_all2all: bool = False,
    ):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L170-L218（锚点双置）
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name

        # Check if this is a stateless process group
        from torch.distributed.distributed_c10d import _world

        is_stateless = _world.pg_map.get(cpu_group, None) is None

        if is_stateless:
            # For stateless groups, we can't use torch.distributed methods
            self.rank = cpu_group.rank()
            self.world_size = cpu_group.size()
            assert global_ranks is not None
            assert global_world_size is not None
            self.ranks = global_ranks
            self.global_rank = self.ranks[self.rank]
            self.global_world_size = global_world_size
            self.rank_in_group = self.rank
        else:
            self.rank = dist.get_rank(cpu_group)
            self.world_size = dist.get_world_size(cpu_group)
            self.ranks = dist.get_process_group_ranks(cpu_group)
            self.global_rank = dist.get_rank()
            self.global_world_size = dist.get_world_size()
            self.rank_in_group = dist.get_group_rank(self.cpu_group, self.global_rank)

        all2all_backend = None
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None:
            all2all_backend = config.parallel_config.all2all_backend

        self.is_ep_communicator = unique_name.split(":")[0] == "ep"
        self.use_all2all = self.is_ep_communicator and use_all2all
        self.all2all_backend = all2all_backend
        self.all2all_manager: All2AllManagerBase | None = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L220-L222（锚点双置）
        dist.all_reduce(input_, group=self.device_group)
        return input_

    # DeviceCommunicatorBase 版 checkpoint 面（All2AllManagerBase 版在 L143-L153）：
    def checkpoint_prepare(self) -> None:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L224-L225（锚点双置）
        """Prepare reclaimable communicator state for checkpoint (default: no-op)."""

    def checkpoint_restore(self) -> None:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L227-L228（锚点双置）
        """Restore communicator state after checkpoint (default: no-op)."""

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L230-L253（锚点双置）
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

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L255-L261（锚点双置）
        raise NotImplementedError

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L263-L294（锚点双置）
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if dim < 0:
            # Convert negative dim to positive.
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

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L296-L299（锚点双置）
        raise NotImplementedError

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L301-L330（锚点双置）
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

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L332-L337（锚点双置）
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L339-L349（锚点双置）
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L351-L356（锚点双置）
        if self.world_size == 1:
            return tensor
        torch.distributed.broadcast(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L358-L359（锚点双置）
        pass

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module) -> None:
        """
        Prepare the communication buffer for the model.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L361-L370（锚点双置）
        if not self.is_ep_communicator:
            return

        moe_modules = [module for module in model.modules() if is_moe_layer(module)]
        for module in moe_modules:
            module.maybe_init_modular_kernel()

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L372-L388（锚点双置）
        if extra_tensors is not None:
            return hidden_states, router_logits, extra_tensors
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L390-L407（锚点双置）
        if extra_tensors is not None:
            return hidden_states, topk_weights, topk_ids, extra_tensors
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L409-L416（锚点双置）
        return hidden_states

    def batch_isend_irecv(self, p2p_ops: list):
        # SOURCE: vllm/distributed/device_communicators/base_device_communicator.py:L418-L419（锚点双置）
        raise NotImplementedError
