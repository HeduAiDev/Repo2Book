# SOURCE: vllm/distributed/parallel_state.py
# ch34 切面（本章核心文件）：5 维 rank 张量四刀（initialize_model_parallel）+
# GroupCoordinator 双群组 + 集合原语 custom-op 注册与用户面 + P2P tensor_dict。
# 删除（dossier 删除项 1，全部逐处标 SUBTRACTED）：patched_fused_scaled_matmul_
# reduce_scatter(+fake+注册)、VLLM_DISTRIBUTED_USE_SPLIT_GROUP 全家、
# make_sibling_device_group、create_single_reader_mq_broadcasters、
# use_cpu_custom_send_recv 旁路、DCP 维度、EPLB 组、elastic EP 全家。
# 另按域界裁除（行内注明）：graph_capture 上下文（ch19）、checkpoint 面、
# cleanup_dist_env_and_memory（ch09 关停域）。

import contextlib
import pickle
import weakref
from collections import namedtuple
from collections.abc import Callable
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Any, Protocol
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup, Store

import vllm.envs as envs
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.network_utils import get_distributed_init_method
from vllm.utils.system_utils import suppress_stdout
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)

# SUBTRACTED: import gc / torch.distributed._functional_collectives /
#   torch.distributed._symmetric_memory / StatelessProcessGroup+get_cached_tcp_
#   store_client / get_worker_rank_suffix 相关（L27-L52）——split_group 与弹性 EP
#   的依赖面（删除项 1）。
# SUBTRACTED: TYPE_CHECKING 的 StatelessGroupCoordinator（L61-L62）——弹性 EP。


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

# SUBTRACTED: GraphCaptureContext dataclass（L66-L69）——CUDA graph 捕获的 stream
#   上下文载体；其唯一消费者 graph_capture 上下文管理器（L618-L660）已按 ch19 域
#   裁除，载体随之裁除。


# SOURCE: vllm/distributed/parallel_state.py:L73-L78 Handle Protocol — 逐字
class Handle(Protocol):
    """Minimal async work handle used by P2P send/recv methods."""

    # SOURCE: vllm/distributed/parallel_state.py:L76 is_completed（锚点双置）
    def is_completed(self) -> bool: ...

    # SOURCE: vllm/distributed/parallel_state.py:L78 wait（锚点双置）
    def wait(self) -> None: ...


# SOURCE: vllm/distributed/parallel_state.py:L81-L104 _split_tensor_dict — 逐字
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


# SOURCE: vllm/distributed/parallel_state.py:L110-L120 _get_unique_name — 逐字
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


# SOURCE: vllm/distributed/parallel_state.py:L126-L127 _register_group — 逐字
def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


# SUBTRACTED: _apply_to_device_comms（L130-L149）——checkpoint 面的遍历器（随
#   checkpoint_prepare/restore_distributed_state 裁除）。


# SOURCE: vllm/distributed/parallel_state.py:L152-L157 all_reduce — 逐字
def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)


# SOURCE: vllm/distributed/parallel_state.py:L160-L161 all_reduce_fake — 逐字
def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(tensor)


# SOURCE: vllm/distributed/parallel_state.py:L164-L171 reduce_scatter — 逐字
def reduce_scatter(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._reduce_scatter_out_place(tensor, dim)


# SOURCE: vllm/distributed/parallel_state.py:L174-L179 reduce_scatter_fake — 逐字
def reduce_scatter_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] // world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


# SOURCE: vllm/distributed/parallel_state.py:L182-L189 all_gather — 逐字
def all_gather(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_gather_out_place(tensor, dim)


# SOURCE: vllm/distributed/parallel_state.py:L192-L197 all_gather_fake — 逐字
def all_gather_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] * world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


# SUBTRACTED: patched_fused_scaled_matmul_reduce_scatter_fake（L200-L249）与
#   patched_fused_scaled_matmul_reduce_scatter（L320-L349）——fp8 融合算子绕
#   pytorch 2.9 bug（TODO 注释自认待删；删除项 1）。
# SUBTRACTED: _platform_device_type / _device_backend_str /
#   _create_subgroups_split_group（L252-L317）——VLLM_DISTRIBUTED_USE_SPLIT_GROUP
#   实验门全家（删除项 1，默认关）。


# SOURCE: vllm/distributed/parallel_state.py:L352-L368 三个集合算子注册 — 逐字
direct_register_custom_op(
    op_name="all_reduce",
    op_func=all_reduce,
    fake_impl=all_reduce_fake,
)

direct_register_custom_op(
    op_name="reduce_scatter",
    op_func=reduce_scatter,
    fake_impl=reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="all_gather",
    op_func=all_gather,
    fake_impl=all_gather_fake,
)

# SUBTRACTED: 第 4 个注册 patched_fused_scaled_matmul_reduce_scatter（L370-L377）
#   ——随其算子族裁除（删除项 1）。


# SOURCE: vllm/distributed/parallel_state.py:L380-L407 GroupCoordinator 类头与
#   4-rank/2-node 坐标表 —— 逐字
class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It manages both CPU and device
        communication.
    """

    # available attributes:
    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    # device communicator (if use_device_communicator=True)
    device_communicator: DeviceCommunicatorBase | None
    mq_broadcaster: Any | None  # shared memory broadcaster

    # SOURCE: vllm/distributed/parallel_state.py:L409-L527 __init__ —— 逐字 minus
    #   删除项（split_group 分支 L434-L445、xpu/out_of_tree 设备分支 L492-L497、
    #   use_cpu_custom_send_recv L529-L533）
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
        use_all2all: bool = False,
    ):
        # SOURCE: vllm/distributed/parallel_state.py:L409-L533（锚点双置）
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_index: int
        assert local_rank >= 0, (
            "local_rank must be provided when creating the world group"
        )
        self.device_index = local_rank

        self_device_group = None
        self_cpu_group = None

        # SUBTRACTED: VLLM_DISTRIBUTED_USE_SPLIT_GROUP 实验门分支（L434-L445，
        #   _create_subgroups_split_group 路径）——删除项 1；默认走 legacy new_group。
        from vllm.distributed.utils import (
            get_cpu_distributed_timeout_or_none,
            get_distributed_timeout_or_none,
        )

        timeout = get_cpu_distributed_timeout_or_none()
        device_timeout = get_distributed_timeout_or_none()

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks,
                backend=torch_distributed_backend,
                timeout=device_timeout,
            )
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(
                    ranks, backend="gloo", timeout=timeout
                )
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.group_ranks = group_ranks
        self.torch_distributed_backend = torch_distributed_backend

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(
                    self.device_index
                )
            )
            self.device = torch.device(f"cuda:{visible_device_index}")
        # SUBTRACTED: xpu / out_of_tree 设备分支（L492-L497）——非本章设备轴
        #   （embed elide 注：只保留 cuda 与 cpu 两支语义）。
        else:
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
                use_all2all=use_all2all,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self.mq_broadcaster: MessageQueue | None = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

        # TODO(#35915): Remove is_tpu() check once tpu_inference
        # overrides use_custom_op_collectives() to return True.
        self.use_custom_op_call = (
            current_platform.is_tpu() or current_platform.use_custom_op_collectives()
        )

        # SUBTRACTED: use_cpu_custom_send_recv（L529-L533）——CPU 后端旁路
        #   （删除项 1：isend/irecv 内 L1033-L1040/L1131-L1138 同步旁路一并裁）。

    # SUBTRACTED: make_sibling_device_group（L535-L555）与 create_single_reader_
    #   mq_broadcasters（L571-L582）——RL/弹性场景专用（删除项 1）。

    # SOURCE: vllm/distributed/parallel_state.py:L557-L569 create_mq_broadcaster
    def create_mq_broadcaster(
        self, writer_rank=0, external_writer_handle=None, blocking=True
    ):
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        return MessageQueue.create_from_process_group(
            self.cpu_group,
            1 << 22,
            6,
            writer_rank=writer_rank,
            external_writer_handle=external_writer_handle,
            blocking=blocking,
        )

    # SOURCE: vllm/distributed/parallel_state.py:L584-L616 rank 属性族 — 逐字
    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L585-L587（锚点双置）
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L590-L592（锚点双置）
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L595-L597（锚点双置）
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        # SOURCE: vllm/distributed/parallel_state.py:L600-L602（锚点双置）
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        # SOURCE: vllm/distributed/parallel_state.py:L605-L609（锚点双置）
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        # SOURCE: vllm/distributed/parallel_state.py:L612-L616（锚点双置）
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    # SUBTRACTED: graph_capture 上下文管理器（L618-L660）——CUDA graph 捕获的
    #   stream/ca_comm 上下文（ch19 域）。

    # SOURCE: vllm/distributed/parallel_state.py:L662-L684 all_reduce 用户面 — 逐字
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

    # SOURCE: vllm/distributed/parallel_state.py:L686-L689 _all_reduce_out_place
    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_reduce(input_)

    # SOURCE: vllm/distributed/parallel_state.py:L691-L705 all_gather 用户面 — 逐字
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

    # SOURCE: vllm/distributed/parallel_state.py:L707-L710 _all_gather_out_place
    def _all_gather_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gather(input_, dim)

    # SOURCE: vllm/distributed/parallel_state.py:L712-L720 all_gatherv — 逐字
    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gatherv(input_, dim, sizes)

    # SOURCE: vllm/distributed/parallel_state.py:L722-L736 reduce_scatter 用户面
    #   —— 逐字
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # SOURCE: vllm/distributed/parallel_state.py:L722-L736（锚点双置）
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

    # SOURCE: vllm/distributed/parallel_state.py:L738-L743 reduce_scatterv — 逐字
    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.reduce_scatterv(input_, dim, sizes)

    # SOURCE: vllm/distributed/parallel_state.py:L745-L748 _reduce_scatter_out_place
    def _reduce_scatter_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.reduce_scatter(input_, dim)

    # SOURCE: vllm/distributed/parallel_state.py:L750-L764 gather — 逐字
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

    # SOURCE: vllm/distributed/parallel_state.py:L766-L779 broadcast — 逐字
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

    # SOURCE: vllm/distributed/parallel_state.py:L781-L803 broadcast_object — 逐字
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

    # SOURCE: vllm/distributed/parallel_state.py:L805-L821 broadcast_object_list
    #   —— 逐字（删除项 1：『保留一句』的对照组承载）
    def broadcast_object_list(
        self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None
    ):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        # SOURCE: vllm/distributed/parallel_state.py:L805-L820（锚点双置）
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(
            obj_list, src=self.ranks[src], group=self.device_group
        )
        return obj_list

    # SOURCE: vllm/distributed/parallel_state.py:L822-L847 send_object — 逐字
    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # Send object size

        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)

        return None

    # SOURCE: vllm/distributed/parallel_state.py:L849-L883 recv_object — 逐字
    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.rank_in_group, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(
            size_tensor, src=self.ranks[src], group=self.cpu_group
        )

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )

        rank_object = torch.distributed.recv(
            object_tensor, src=self.ranks[src], group=self.cpu_group
        )

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank."
        )

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    # SOURCE: vllm/distributed/parallel_state.py:L885-L965 broadcast_tensor_dict
    #   —— 逐字（按张量 is_cpu 分流走 metadata_group/group——WC3 的双群组分工）
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
        # SOURCE: vllm/distributed/parallel_state.py:L885-L965（锚点双置）
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: list[tuple[Any, Any]] = []
            assert isinstance(tensor_dict, dict), (
                f"Expecting a dictionary, got {type(tensor_dict)}"
            )
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=metadata_group, async_op=True
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
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
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
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    # SOURCE: vllm/distributed/parallel_state.py:L967-L979 _should_use_all_gather
    #   —— 逐字（m11 的判定本体）
    def _should_use_all_gather(
        self,
        key: str,
        numel: int,
        all_gather_group: "GroupCoordinator | None",
        all_gather_tensors: dict[str, bool] | None,
    ) -> bool:
        # SOURCE: vllm/distributed/parallel_state.py:L967-L979（锚点双置）
        if all_gather_group is None:
            return False
        use_all_gather = numel % all_gather_group.world_size == 0
        if all_gather_tensors is not None:
            use_all_gather = all_gather_tensors.get(key, use_all_gather)
        return use_all_gather

    # SOURCE: vllm/distributed/parallel_state.py:L981-L1017 send_tensor_dict — 逐字
    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        handles = self.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )
        for handle in handles:
            handle.wait()
        return None

    # SOURCE: vllm/distributed/parallel_state.py:L1019-L1074 isend_tensor_dict
    #   —— 逐字 minus use_cpu_custom_send_recv 旁路（L1033-L1040，删除项 1）
    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Handle]:
        # SOURCE: vllm/distributed/parallel_state.py:L1019-L1074（锚点双置）
        if self.world_size <= 1:
            return []

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        # SUBTRACTED: use_cpu_custom_send_recv 的同步旁路（L1033-L1040）——删除项 1。

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)

        handles: list[Handle] = []
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                continue

            if self._should_use_all_gather(
                key, tensor.numel(), all_gather_group, all_gather_tensors
            ):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            comm_group = metadata_group if tensor.is_cpu else group
            handle = torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=comm_group
            )
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            handles.append(handle)

        return handles

    # SOURCE: vllm/distributed/parallel_state.py:L1076-L1112 recv_tensor_dict — 逐字
    def recv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None
        tensor_dict, handles, postprocess = self.irecv_tensor_dict(
            src=src,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )
        for handle in handles:
            handle.wait()
        for fn in postprocess:
            fn()
        return tensor_dict

    # SOURCE: vllm/distributed/parallel_state.py:L1114-L1198 irecv_tensor_dict
    #   —— 逐字 minus use_cpu_custom_send_recv 旁路（L1131-L1138，删除项 1）
    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor | Any] | None,
        list[Handle],
        list[Callable[[], None]],
    ]:
        # SOURCE: vllm/distributed/parallel_state.py:L1114-L1198（锚点双置）
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        # SUBTRACTED: use_cpu_custom_send_recv 的同步旁路（L1131-L1138）——删除项 1。

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        handles: list[Handle] = []
        postprocess: list[Callable[[], None]] = []

        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                full_tensor = torch.empty(
                    value.size, dtype=value.dtype, device=value.device
                )
                if full_tensor.numel() == 0:
                    tensor_dict[key] = full_tensor
                    continue

                if self._should_use_all_gather(
                    key, full_tensor.numel(), all_gather_group, all_gather_tensors
                ):
                    orig_shape = full_tensor.shape
                    slice_tensor = full_tensor.reshape(all_gather_size, -1)[
                        all_gather_rank
                    ]
                    comm_group = metadata_group if slice_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        slice_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)

                    def _postprocess(
                        key: str = key,
                        slice_tensor: torch.Tensor = slice_tensor,
                        orig_shape: tuple[int, ...] = tuple(orig_shape),
                        all_gather_group=all_gather_group,
                    ) -> None:
                        # SOURCE: vllm/distributed/parallel_state.py:L1175-L1184（锚点双置）
                        assert all_gather_group is not None
                        tensor_dict[key] = all_gather_group.all_gather(
                            slice_tensor, dim=0
                        ).reshape(orig_shape)

                    postprocess.append(_postprocess)
                    tensor_dict[key] = slice_tensor
                else:
                    comm_group = metadata_group if full_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        full_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)
                    tensor_dict[key] = full_tensor
            else:
                tensor_dict[key] = value

        return tensor_dict, handles, postprocess

    # SOURCE: vllm/distributed/parallel_state.py:L1200-L1207 barrier — 逐字
    #   （注释原话：NCCL barrier 的陷阱与刻意走 cpu_group）
    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        # SOURCE: vllm/distributed/parallel_state.py:L1200-L1207（锚点双置）
        torch.distributed.barrier(group=self.cpu_group)

    # SOURCE: vllm/distributed/parallel_state.py:L1209-L1214 send — 逐字
    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        self.device_communicator.send(tensor, dst)

    # SOURCE: vllm/distributed/parallel_state.py:L1216-L1223 recv — 逐字
    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.recv(size, dtype, src)

    # SOURCE: vllm/distributed/parallel_state.py:L1225-L1235 destroy — 逐字
    def destroy(self):
        if hasattr(self, "device_group"):
            torch.distributed.destroy_process_group(self.device_group)
            del self.device_group
        if hasattr(self, "cpu_group"):
            torch.distributed.destroy_process_group(self.cpu_group)
            del self.cpu_group
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None

    # SUBTRACTED: prepare_communication_buffer_for_model（L1237-L1239）——DeepEP
    #   等按模型形状预分配通信缓冲的面（真 A2A 后端域，删除项 3 同族）。

    # SOURCE: vllm/distributed/parallel_state.py:L1241-L1259 dispatch_router_logits
    #   —— 逐字
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
        # SOURCE: vllm/distributed/parallel_state.py:L1241-L1259（锚点双置）
        if self.device_communicator is not None:
            return self.device_communicator.dispatch_router_logits(
                hidden_states,
                router_logits,
                is_sequence_parallel,
                extra_tensors,
            )
        else:
            return hidden_states, router_logits

    # SOURCE: vllm/distributed/parallel_state.py:L1261-L1281 dispatch — 逐字
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.device_communicator is not None:
            return self.device_communicator.dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )
        else:
            return hidden_states, topk_weights, topk_ids

    # SOURCE: vllm/distributed/parallel_state.py:L1283-L1289 combine — 逐字
    def combine(
        self, hidden_states, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        if self.device_communicator is not None:
            return self.device_communicator.combine(hidden_states, is_sequence_parallel)
        else:
            return hidden_states


_WORLD: GroupCoordinator | None = None
_INNER_DP_WORLD: GroupCoordinator | None = None
_NODE_COUNT: int | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1297-L1299 get_world_group — 逐字
def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


# SOURCE: vllm/distributed/parallel_state.py:L1302-L1304 get_inner_dp_world_group
def get_inner_dp_world_group() -> GroupCoordinator:
    assert _INNER_DP_WORLD is not None, "inner dp world group is not initialized"
    return _INNER_DP_WORLD


# SOURCE: vllm/distributed/parallel_state.py:L1307-L1316 init_world_group — 逐字
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


# SOURCE: vllm/distributed/parallel_state.py:L1319-L1336 init_model_parallel_group
#   —— 逐字（构造单维度 GroupCoordinator 的工厂）
def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
    use_device_communicator: bool = True,
    use_all2all: bool = False,
) -> GroupCoordinator:
    # SOURCE: vllm/distributed/parallel_state.py:L1319-L1336（锚点双置）
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        use_all2all=use_all2all,
    )


# SUBTRACTED: _init_stateless_group（L1339-L1363）与 _replace_active_groups
#   （L1366-L1387）——elastic EP 全家（删除项 1）。


_TP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1393-L1395 get_tp_group — 逐字
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


# SUBTRACTED: _DCP 单例与 get_dcp_group（L1398-L1403）——DCP 维度（删除项 1）。


_PP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1409-L1411 get_pp_group — 逐字
def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


_DP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1417-L1419 get_dp_group — 逐字
def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


_EP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1425-L1431 get_ep_group — 逐字
#   （断言消息即『dense 不建 EP 组』的可观察证据）
def get_ep_group() -> GroupCoordinator:
    # SOURCE: vllm/distributed/parallel_state.py:L1425-L1431（锚点双置）
    assert _EP is not None, (
        "expert parallel group is not initialized. "
        "EP group is only created for MoE models with num_experts > 0. "
        "This function should only be called for MoE models."
    )
    return _EP


# SUBTRACTED: _EPLB 单例与 get_eplb_group（L1434-L1443）——EPLB 组（删除项 1）。


_PCP: GroupCoordinator | None = None


# SOURCE: vllm/distributed/parallel_state.py:L1449-L1451 get_pcp_group — 逐字
def get_pcp_group() -> GroupCoordinator:
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP


# SUBTRACTED: graph_capture 模块级上下文（L1454-L1479）——ch19 域。

logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True


# SOURCE: vllm/distributed/parallel_state.py:L1487-L1489 set_custom_all_reduce
#   —— 逐字
def set_custom_all_reduce(enable: bool):
    # SOURCE: vllm/distributed/parallel_state.py:L1487-L1489（锚点双置）
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


# SUBTRACTED: _init_process_group_for_split_group（L1492-L1524）与
#   _validate_default_pg_for_split_group（L1527-L1550）——split_group 全家（删除项 1）。
# SUBTRACTED: _init_elastic_ep_world（L1553-L1585）——elastic EP（删除项 1）。


# SOURCE: vllm/distributed/parallel_state.py:L1588-L1743 init_distributed_environment
#   —— 逐字 minus elastic EP / split_group 分支（删除项 1）；DP rank 偏移段
#   （L1608-L1638）逐字保留——站 5 的『worker 入全局 world』公式
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: timedelta | None = None,
):
    # SOURCE: vllm/distributed/parallel_state.py:L1588-L1743（锚点双置）
    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    # SUBTRACTED: enable_elastic_ep 的计算（L1607，`config is not None and
    #   config.parallel_config.enable_elastic_ep`）——elastic EP（删除项 1）。
    if (
        config is not None
        and config.parallel_config.distributed_executor_backend != "external_launcher"
        and (
            config.parallel_config.nnodes > 1
            or config.parallel_config.data_parallel_size > 1
        )
    ):
        parallel_config = config.parallel_config
        # adjust to take into account data parallelism
        # offset the rank by the data parallel rank
        rank = parallel_config.data_parallel_rank * world_size + rank
        # adjust the world size to take into account data parallelism
        world_size = parallel_config.world_size_across_dp

        # Use appropriate IP and port based on configuration
        if parallel_config.nnodes > 1:
            ip = parallel_config.master_addr
            port = parallel_config.master_port
            distributed_init_method = get_distributed_init_method(ip, port)
        else:
            ip = parallel_config.data_parallel_master_ip
            port = parallel_config.get_next_dp_init_port()
            distributed_init_method = get_distributed_init_method(ip, port)
            logger.debug(
                "Adjusting world_size=%d rank=%d distributed_init_method=%s for DP",
                world_size,
                rank,
                distributed_init_method,
            )
    if not torch.distributed.is_initialized():
        logger.info(
            "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
            world_size,
            rank,
            local_rank,
            distributed_init_method,
            backend,
        )
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        if not torch.distributed.is_backend_available(backend):
            logger.warning(
                "Distributed backend %s is not available; falling back to gloo.",
                backend,
            )
            assert torch.distributed.is_gloo_available(), (
                "Fallback Gloo backend is not available."
            )
            backend = "gloo"
        # SUBTRACTED: VLLM_DISTRIBUTED_USE_SPLIT_GROUP 的 eager-init 分支
        #   （L1661-L1678）——删除项 1；默认走 init_process_group。
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        # SUBTRACTED: elastic EP 的 tp_pp_cpu_group 多节点检查（L1688-L1698）。

    # SUBTRACTED: split_group 的 default PG 校验（L1700-L1701）——删除项 1。

    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        local_rank = envs.LOCAL_RANK if distributed_init_method == "env://" else rank

    global _WORLD, _NODE_COUNT, _INNER_DP_WORLD
    # SUBTRACTED: enable_elastic_ep 的 _init_elastic_ep_world 分支（L1712-L1714）。
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
        if config is not None and config.parallel_config.nnodes > 1:
            _NODE_COUNT = config.parallel_config.nnodes
        else:
            _NODE_COUNT = _node_count(_WORLD.cpu_group)
        logger.debug("Detected %d nodes in the distributed environment", _NODE_COUNT)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size"
        )
    if config is not None and config.parallel_config.nnodes_within_dp > 1:
        if parallel_config.data_parallel_size > 1:
            world_size_inner_dp = parallel_config.world_size
            group_ranks = [
                [dp_rank * world_size_inner_dp + i for i in range(world_size_inner_dp)]
                for dp_rank in range(parallel_config.data_parallel_size)
            ]
            _INNER_DP_WORLD = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="inner_dp_world",
                use_device_communicator=False,
            )
        else:
            _INNER_DP_WORLD = _WORLD


# SOURCE: vllm/distributed/parallel_state.py:L1746-L1775 initialize_model_parallel
#   的 docstring —— 逐字（8 GPU/TP2/PP4 的官方算例）
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used to parallelize
            the model pipeline.
        backend: name of torch distributed communication backend.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()

    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config()
    data_parallel_size = config.parallel_config.data_parallel_size
    # SUBTRACTED: enable_elastic_ep 的 coord_store/local_all_ranks 预备段
    #   （L1783-L1804）——elastic EP（删除项 1）；走非弹性 else 分支。
    parallel_config = config.parallel_config
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group
    )

    # the layout order is: ExternalDP x DP x PP x PCP x TP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    # DP is the data parallel group that is part of the model,
    # all the ranks in the same DP group should generate simultaneously,
    # i.e. the `generate` call in the same DP group should be called together,
    # otherwise it will cause deadlock.
    # to get group_ranks for each dimension, transpose that dimension to the
    # last dimension, then reshape to 2D, then unbind the last dimension
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )  # noqa

    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    # SUBTRACTED: enable_elastic_ep 的 local_all_ranks 覆写（L1834-L1836）。
    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # SUBTRACTED: DCP 维度建组段（L1846-L1862）——删除项 1。

    # SOURCE: vllm/distributed/parallel_state.py:L1864-L1881 PCP 刀 — 逐字
    #   minus elastic 覆写（L1872-L1878）
    global _PCP
    assert _PCP is None, "prefill context parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(3, 4)
        .reshape(-1, prefill_context_model_parallel_size)
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    _PCP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pcp"
    )

    # Build the pipeline model-parallel groups.
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    # SUBTRACTED: enable_elastic_ep 的 local_all_ranks 覆写（L1890-L1896）。
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )

    # SOURCE: vllm/distributed/parallel_state.py:L1901-L1916 DP 刀 — 逐字 minus
    #   elastic 分支（L1905-L1912）
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    # SUBTRACTED: enable_elastic_ep 的 _init_stateless_group 分支（L1905-L1912）。
    _DP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="dp"
    )

    # SOURCE: vllm/distributed/parallel_state.py:L1918-L1950 EP 刀 — 逐字 minus
    #   elastic 分支（L1934-L1942）；'Don't create EP group for dense models' 原话
    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    # Don't create EP group for dense models.
    if config.model_config is None or config.model_config.is_moe:
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
        use_all2all = parallel_config.use_all2all
        # SUBTRACTED: enable_elastic_ep 的 _init_stateless_group 分支
        #   （L1934-L1942）。
        _EP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="ep",
            use_all2all=use_all2all,
        )

        # SUBTRACTED: EPLB 组建组段（L1952-L1973）——删除项 1（隔离 EPLB 通信与
        #   MoE 前向集合通信、防 torch.distributed 死锁的注释随段裁除）。
    # If no EP group needed, _EP remains None
    # If no EPLB group needed, _EPLB remains None

    # SOURCE: vllm/distributed/parallel_state.py:L1977-L1989 建组完成日志 — 逐字
    #   minus DCP/EPLB 两项
    logger.info_once(
        "rank %s in world size %s is assigned as "
        "DP rank %s, PP rank %s, PCP rank %s, "
        "TP rank %s, EP rank %s",
        rank,
        world_size,
        _DP.rank_in_group,
        _PP.rank_in_group,
        _PCP.rank_in_group,
        _TP.rank_in_group,
        _EP.rank_in_group if _EP is not None else "N/A",
    )


# SOURCE: vllm/distributed/parallel_state.py:L1992-L2034 ensure_model_parallel_
#   initialized —— 逐字 minus DCP 断言段（L2035-L2041，_DCP 已按删除项 1 裁除）
def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    # SOURCE: vllm/distributed/parallel_state.py:L1992-L2041（锚点双置）
    world_group = get_world_group()
    if hasattr(world_group, "backend"):
        backend = backend or world_group.backend
    else:
        backend = backend or torch.distributed.get_backend(world_group.device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )
        return

    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size. "
        f"got: {get_tensor_model_parallel_world_size()=} vs. "
        f"wanted: {tensor_model_parallel_size=}"
    )
    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        "pipeline parallel group already initialized, but of unexpected size. "
        f"got: {pp_world_size=} vs. "
        f"wanted: {pipeline_model_parallel_size=}"
    )
    pcp_world_size = get_pcp_group().world_size
    assert pcp_world_size == prefill_context_model_parallel_size, (
        "prefill context parallel group already initialized, but of unexpected size: "
        f"{pcp_world_size=} vs. "
        f"{prefill_context_model_parallel_size=}"
    )


# SUBTRACTED: prepare_communication_buffer_for_model 模块级版本（L2044-L2062）。
# SUBTRACTED: checkpoint_prepare_distributed_state / checkpoint_restore_
#   distributed_state（L2065-L2076）——checkpoint 面（随 cuda_communicator 的
#   checkpoint 裁除联动）。


# SOURCE: vllm/distributed/parallel_state.py:L2079-L2081 model_parallel_is_
#   initialized —— 逐字
def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    # SOURCE: vllm/distributed/parallel_state.py:L2079-L2081（锚点双置）
    return _TP is not None and _PP is not None


# SOURCE: vllm/distributed/parallel_state.py:L2087-L2089
#   get_tensor_model_parallel_world_size —— 逐字
def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    # SOURCE: vllm/distributed/parallel_state.py:L2087-L2089（锚点双置）
    return get_tp_group().world_size


# SOURCE: vllm/distributed/parallel_state.py:L2092-L2094
#   get_tensor_model_parallel_rank —— 逐字
def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    # SOURCE: vllm/distributed/parallel_state.py:L2092-L2094（锚点双置）
    return get_tp_group().rank_in_group


# SOURCE: vllm/distributed/parallel_state.py:L2097-L2100 get_node_count — 逐字
def get_node_count() -> int:
    """Return the total number of nodes in the distributed environment."""
    assert _NODE_COUNT is not None, "distributed environment is not initialized"
    return _NODE_COUNT


# SOURCE: vllm/distributed/parallel_state.py:L2103-L2139 destroy_model_parallel
#   —— 逐字 minus DCP/EPLB 段（删除项 1）
def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    # SOURCE: vllm/distributed/parallel_state.py:L2103-L2139（锚点双置）
    global _TP

    if _TP:
        _TP.destroy()
    _TP = None

    global _PCP
    if _PCP:
        _PCP.destroy()
    _PCP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _DP
    if _DP:
        _DP.destroy()
    _DP = None

    global _EP
    if _EP:
        _EP.destroy()
    _EP = None


# SOURCE: vllm/distributed/parallel_state.py:L2142-L2149 destroy_distributed_
#   environment —— 逐字
def destroy_distributed_environment():
    # SOURCE: vllm/distributed/parallel_state.py:L2142-L2149（锚点双置）
    global _WORLD, _NODE_COUNT
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    _NODE_COUNT = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# SUBTRACTED: cleanup_dist_env_and_memory（L2152-L2191）——生产关停/缓存清理面
#   （ch09 域）。


# SOURCE: vllm/distributed/parallel_state.py:L2194-L2285 in_the_same_node_as
#   —— 逐字（_node_count 与 All2AllManagerBase 的节点判定基础）
def in_the_same_node_as(
    pg: ProcessGroup, source_rank: int = 0
) -> list[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    # SOURCE: vllm/distributed/parallel_state.py:L2194-L2285（锚点双置）
    # SUBTRACTED: StatelessProcessGroup 分支（L2212-L2215/L2236-L2237/L2241-L2248/
    #   L2268-L2269/L2276-L2283）——弹性 EP 域（删除项 1）；只走 ProcessGroup 面。
    assert torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL, (
        "in_the_same_node_as should be tested with a non-NCCL group."
    )
    # local rank inside the group
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # global ranks of the processes in the group
    ranks = torch.distributed.get_process_group_ranks(pg)

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor(
        [0] * world_size, dtype=torch.int32, device="cpu"
    )

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                assert shm.buf is not None, "Buffer was not created"
                shm.buf[: len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list(
                    [shm.name], src=ranks[source_rank], group=pg
                )
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(
                    recv, src=ranks[source_rank], group=pg
                )
                name = recv[0]
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                assert shm.buf is not None, "Buffer was not opened"
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()

    torch.distributed.all_reduce(is_in_the_same_node, group=pg)
    return [x == 1 for x in is_in_the_same_node.tolist()]


# SOURCE: vllm/distributed/parallel_state.py:L2288-L2316 is_global_first_rank
#   —— 逐字
def is_global_first_rank() -> bool:
    """
    Check if the current process is the first rank globally across all
    parallelism strategies (PP, TP, DP, EP, etc.).

    Unlike group-specific checks like `get_tensor_model_parallel_rank() == 0`
    or `get_pp_group().is_first_rank`, this function checks the global rank
    across all parallel parallelism dimensions.

    Returns:
        bool: True if this is the global first rank (rank 0), False otherwise.
              Returns True if distributed is not initialized (single process).
    """
    # SOURCE: vllm/distributed/parallel_state.py:L2288-L2316（锚点双置）
    try:
        # If world group is available, use it for the most accurate check
        global _WORLD
        if _WORLD is not None:
            return _WORLD.is_first_rank

        # If torch distributed is not initialized, assume single process
        if not torch.distributed.is_initialized():
            return True

        # Fallback to torch's global rank
        return torch.distributed.get_rank() == 0

    except Exception:
        # If anything goes wrong, assume this is the first rank
        return True


# SOURCE: vllm/distributed/parallel_state.py:L2319-L2339 is_local_first_rank
#   —— 逐字
def is_local_first_rank() -> bool:
    """
    Check if the current process is the first local rank (rank 0 on its node).
    """
    # SOURCE: vllm/distributed/parallel_state.py:L2319-L2339（锚点双置）
    try:
        # prefer the initialized world group if available
        global _WORLD
        if _WORLD is not None:
            return _WORLD.local_rank == 0

        if not torch.distributed.is_initialized():
            return True

        # fallback to environment-provided local rank if available
        # note: envs.LOCAL_RANK is set when using env:// launchers (e.g., torchrun)
        try:
            return int(envs.LOCAL_RANK) == 0  # type: ignore[arg-type]
        except Exception:
            return torch.distributed.get_rank() == 0
    except Exception:
        return True


# SOURCE: vllm/distributed/parallel_state.py:L2342-L2378 _node_count —— 逐字
#   minus StatelessProcessGroup 分支
def _node_count(pg: ProcessGroup) -> int:
    """
    Returns the total number of nodes in the process group.

    Args:
        pg: The process group to analyze

    Returns:
        int: The total number of nodes
    """
    # SOURCE: vllm/distributed/parallel_state.py:L2342-L2378（锚点双置）
    world_size = torch.distributed.get_world_size(group=pg)

    if world_size == 1:
        return 1

    # Build node assignment map
    node_assignment = [0] * world_size  # rank -> node_id
    next_node_id = 0

    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0:
            continue  # Already assigned to a node

        # Assign current rank to a new node
        next_node_id += 1
        node_assignment[current_rank] = next_node_id

        # Find all ranks on the same node as current_rank
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    return next_node_id
