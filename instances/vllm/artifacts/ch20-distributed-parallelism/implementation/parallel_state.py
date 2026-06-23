# 精简版（subtract-only companion）—— vllm/distributed/parallel_state.py
#
# 忠实子集：与真实 vLLM 同名、同结构、同控制流，只删不增。
# 删除项见各处 `# SUBTRACTED:`；每个 def/class 标 `# SOURCE:`。
# 可在 host(CPU/gloo) 上运行：device_group/cpu_group 都用 gloo，
# self.device 走 CPU 分支，从而无需 CUDA 即可跑通三大集合原语与 P2P。
#
# 唯一的环境桥接：真实代码从 vllm.utils 引入 direct_register_custom_op、
# 从 vllm.platforms 引入 current_platform、从 device_communicators 引入具体
# DeviceCommunicatorBase 子类。host 上没有这些 vLLM 内部依赖，因此这里把它们
# 替换为等价的最小桥（语义一致，不杜撰任何 vLLM 没有的行为）。
from __future__ import annotations

import weakref
from collections import namedtuple
from typing import Any, Callable

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

from base_device_communicator import DeviceCommunicatorBase

# --- 环境桥接（替换 vllm.utils / vllm.platforms，语义等价，非 vLLM 新抽象）---
from _env_bridge import (
    current_platform,
    direct_register_custom_op,
    resolve_obj_by_qualname,
    suppress_stdout,
)


# SOURCE: vllm/distributed/parallel_state.py:L70
TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


# SOURCE: vllm/distributed/parallel_state.py:L81
def _split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any],
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: dict[str, int] = {}


# SOURCE: vllm/distributed/parallel_state.py:L110
def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: dict[str, Callable[[], "GroupCoordinator | None"]] = {}


# SOURCE: vllm/distributed/parallel_state.py:L126
def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


# SOURCE: vllm/distributed/parallel_state.py:L130
def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)


# SOURCE: vllm/distributed/parallel_state.py:L138
def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(tensor)


# SOURCE: vllm/distributed/parallel_state.py:L142
def reduce_scatter(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._reduce_scatter_out_place(tensor, dim)


# SOURCE: vllm/distributed/parallel_state.py:L152
def reduce_scatter_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] // world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


# SOURCE: vllm/distributed/parallel_state.py:L160
def all_gather(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_gather_out_place(tensor, dim)


# SOURCE: vllm/distributed/parallel_state.py:L170
def all_gather_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] * world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


# SUBTRACTED: patched_fused_scaled_matmul_reduce_scatter(_fake) 及其第 4 个
# direct_register_custom_op 注册（vllm/distributed/parallel_state.py:L178-L259,
# L280-L287）。这是 fp8 行内缩放 matmul+reduce_scatter 的融合算子，绕过 pytorch
# 2.9 的一个特定 bug，属边缘特性；删后三大集合原语的注册与行为完整。

# SOURCE: vllm/distributed/parallel_state.py:L262
direct_register_custom_op(
    op_name="all_reduce",
    op_func=all_reduce,
    fake_impl=all_reduce_fake,
)

# SOURCE: vllm/distributed/parallel_state.py:L268
direct_register_custom_op(
    op_name="reduce_scatter",
    op_func=reduce_scatter,
    fake_impl=reduce_scatter_fake,
)

# SOURCE: vllm/distributed/parallel_state.py:L274
direct_register_custom_op(
    op_name="all_gather",
    op_func=all_gather,
    fake_impl=all_gather_fake,
)


# SOURCE: vllm/distributed/parallel_state.py:L290
class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.

    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It manages both CPU and device
        communication.
    """

    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    device_communicator: DeviceCommunicatorBase | None
    mq_broadcaster: Any | None  # shared memory broadcaster

    # SOURCE: vllm/distributed/parallel_state.py:L309
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            # SUBTRACTED: xpu / out_of_tree 的 self.device 分支
            # (vllm/distributed/parallel_state.py:L363-L368)；本章按 CUDA 主线，
            # 保留 cuda 与 else=cpu 两支即可表达双群组语义，CPU 分支让精简版可在
            # host 上跑通。
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls()
            )
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        # SUBTRACTED: mq_broadcaster 的真正构造依赖 MessageQueue（共享内存广播，
        # vllm/distributed/parallel_state.py:L385-L389）；host 上无该依赖，这里只
        # 保留字段为 None（broadcast_object 因此走 cpu_group 主路径，语义不变）。
        self.mq_broadcaster = None

        self.use_custom_op_call = (
            current_platform.is_tpu() or current_platform.use_custom_op_collectives()
        )

        # SUBTRACTED: use_cpu_custom_send_recv（CPU 后端的同步 send/recv 旁路，
        # vllm/distributed/parallel_state.py:L397-L401）——本章按 GPU/通用主线讲。

    # SUBTRACTED: graph_capture / GraphCaptureContext（CUDA graph 捕获上下文，
    # vllm/distributed/parallel_state.py:L464-L500）属其它章节主题，删后集合原语
    # 调用链不变。

    # SOURCE: vllm/distributed/parallel_state.py:L502
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we always make the all-reduce operation
        out-of-place.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if self.use_custom_op_call:
            return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
        else:
            return self._all_reduce_out_place(input_)

    # SOURCE: vllm/distributed/parallel_state.py:L526
    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_reduce(input_)

    # SOURCE: vllm/distributed/parallel_state.py:L531
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if self.use_custom_op_call:
            return torch.ops.vllm.all_gather(
                input_, dim, world_size, group_name=self.unique_name
            )
        else:
            return self._all_gather_out_place(input_, dim)

    # SUBTRACTED: all_gatherv（带 per-rank 不等长 sizes 的变体，
    # vllm/distributed/parallel_state.py:L552-L560）直接转发 device_communicator，
    # 本章用等长 all_gather 讲清主线。

    # SOURCE: vllm/distributed/parallel_state.py:L562
    def _all_gather_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gather(input_, dim)

    # SOURCE: vllm/distributed/parallel_state.py:L567
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if self.use_custom_op_call:
            return torch.ops.vllm.reduce_scatter(
                input_, dim, world_size, group_name=self.unique_name
            )
        else:
            return self._reduce_scatter_out_place(input_, dim)

    # SUBTRACTED: reduce_scatterv（变长变体，
    # vllm/distributed/parallel_state.py:L578-L583），理由同 all_gatherv。

    # SOURCE: vllm/distributed/parallel_state.py:L585
    def _reduce_scatter_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.reduce_scatter(input_, dim)

    # SOURCE: vllm/distributed/parallel_state.py:L590
    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.gather(input_, dst, dim)

    # SOURCE: vllm/distributed/parallel_state.py:L606
    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(
            input_, src=self.ranks[src], group=self.device_group
        )
        return input_

    # SOURCE: vllm/distributed/parallel_state.py:L621
    def broadcast_object(self, obj: Any | None = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.mq_broadcaster is not None:
            assert src == 0, "Message queue broadcaster only supports src=0"
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list(
                [obj], src=self.ranks[src], group=self.cpu_group
            )
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=self.ranks[src], group=self.cpu_group
            )
            return recv[0]

    # SOURCE: vllm/distributed/parallel_state.py:L725
    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any] | None = None,
        src: int = 0,
        group: ProcessGroup | None = None,
        metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: list[tuple[Any, Any]] = []
            assert isinstance(tensor_dict, dict)
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor,
                        src=self.ranks[src],
                        group=metadata_group,
                        async_op=True,
                    )
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()
        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    if tensor.numel() == 0:
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True,
                        )
                    else:
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    # SUBTRACTED: send_tensor_dict / recv_tensor_dict 及其 isend/irecv 异步版本与
    # all_gather 优化分支（vllm/distributed/parallel_state.py:L807-L1038）——PP 传
    # 整个 IntermediateTensors 字典的批量封装；本章用单张量 send/recv 讲清 P2P 主线。

    # SOURCE: vllm/distributed/parallel_state.py:L1040
    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    # SOURCE: vllm/distributed/parallel_state.py:L1049
    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        self.device_communicator.send(tensor, dst)

    # SOURCE: vllm/distributed/parallel_state.py:L1056
    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.recv(size, dtype, src)

    # SUBTRACTED: dispatch / combine / dispatch_router_logits（MoE all2all 专家分发，
    # vllm/distributed/parallel_state.py:L1081-L1129）属 MoE 章节主题，删后双群组与
    # 集合原语主线不依赖它们。

    # SOURCE: vllm/distributed/parallel_state.py:L1064
    def destroy(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None


# === 全局单例 + 访问器 ============================================
# SUBTRACTED: _DCP/_PCP/_EPLB/_WORLD/_INNER_DP_WORLD 的同构单例与访问器
# (vllm/distributed/parallel_state.py:L1137-L1290)；本章按 TP/PP/DP/EP 四主维度讲，
# 其余维度同构、一句带过。下面保留 _WORLD（init 入口需要）。

_WORLD: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1132
def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


# SOURCE: vllm/distributed/parallel_state.py:L1145
def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,
        group_name="world",
    )


# SOURCE: vllm/distributed/parallel_state.py:L1159
def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
    use_device_communicator: bool = True,
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
    )


_TP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1229
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


_PP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1255
def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


_DP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1266
def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


_EP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1277
def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, (
        "expert parallel group is not initialized. "
        "EP group is only created for MoE models with num_experts > 0. "
        "This function should only be called for MoE models."
    )
    return _EP


# SOURCE: vllm/distributed/parallel_state.py:L1358
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: "Any | None" = None,
):
    # SUBTRACTED: DP 偏移 rank/world_size 的调整、enable_elastic_ep 分支、
    # nnodes_within_dp 的 _INNER_DP_WORLD（vllm/distributed/parallel_state.py:
    # L1374-L1408, L1439-L1491）——本章按非弹性、单 DP、单节点主线讲，保留
    # 『init_process_group(WORLD) → init_world_group(_WORLD)』核心骨架。
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        if not torch.distributed.is_backend_available(backend):
            assert torch.distributed.is_gloo_available(), (
                "Fallback Gloo backend is not available."
            )
            backend = "gloo"
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        local_rank = rank

    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size"
        )


# SOURCE: vllm/distributed/parallel_state.py:L1494
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """
    Initialize model parallel groups.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    # SUBTRACTED: 从 vllm.config 取 data_parallel_size / enable_elastic_ep 的逻辑
    # (vllm/distributed/parallel_state.py:L1527-L1552)——本章无 vllm.config，按
    # 非弹性、data_parallel_size=1 主线，直接用 torch world_size/rank。
    world_size = torch.distributed.get_world_size()
    data_parallel_size = 1
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group
    )

    # the layout order is: ExternalDP x DP x PP x PCP x TP
    # to get group_ranks for each dimension, transpose that dimension to the
    # last dimension, then reshape to 2D, then unbind the last dimension
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )

    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # SUBTRACTED: DCP/PCP 两维度的切分（vllm/distributed/parallel_state.py:
    # L1594-L1633），与 TP/PP/DP 同构；本章用 TP/PP/DP/EP 四主维度讲清套路。

    # Build the pipeline model-parallel groups.
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )

    global _DP
    assert _DP is None, "data parallel group is already initialized"
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    # SUBTRACTED: enable_elastic_ep 下用 StatelessGroupCoordinator/_init_stateless_group
    # 的分支（vllm/distributed/parallel_state.py:L1657-L1664）——本章按非弹性主线讲。
    _DP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="dp"
    )

    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    # Don't create EP group for dense models.
    # SUBTRACTED: config.model_config.is_moe 判定（本章无 vllm.config）。EP =
    # DP×PCP×TP 合并而成（transpose(1,2) 后 reshape）；保留切分套路、按需创建语义。
    # EPLB 群组（vllm/distributed/parallel_state.py:L1698-L1719）同样略去。
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            data_parallel_size
            * prefill_context_model_parallel_size
            * tensor_model_parallel_size,
        )
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    _EP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="ep"
    )


# SOURCE: vllm/distributed/parallel_state.py（测试便利：清空进程级单例）
def _reset_state_for_tests() -> None:
    # SUBTRACTED: 这是精简版为单进程多次 init 提供的测试钩子，真实 vLLM 用
    # destroy_model_parallel/destroy_distributed_environment 完成等价清理。
    global _WORLD, _TP, _PP, _DP, _EP
    for g in (_TP, _PP, _DP, _EP, _WORLD):
        if g is not None:
            try:
                g.destroy()
            except Exception:
                pass
    _WORLD = _TP = _PP = _DP = _EP = None
    _groups.clear()
    _group_name_counter.clear()
