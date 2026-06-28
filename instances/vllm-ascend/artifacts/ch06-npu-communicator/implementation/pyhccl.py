"""换底座线③：手写 pyhccl —— pynccl 的逐符号移植。

只做减法的忠实精简版。控制流与 vllm/distributed/device_communicators/pynccl.py
完全一致（unique_id 建组 / CommInitRank / warmup all_reduce / disabled 降级），
只换符号：NCCL→HCCL、cuda→npu、stream.cuda_stream→stream.npu_stream、枚举值按
HCCL 头文件。注意：本类当前【未接入】NPUCommunicator（npu_communicator.py 的
TODO(hz)），是为未来 custom-allreduce 预留的范式样本；NPU 实际集合通信走基类的
dist.*(group=device_group) + 进程组 HCCL backend，不经过 pyhccl。

host 可读：disabled 降级路径（world_size==1 / 缺库）是纯 Python，可单测；真正调
libhccl.so 的 all_reduce/broadcast 需 NPU，不在 host 跑。
"""
# SUBTRACTED: 文件头 Apache-2.0 许可证注释块（原 pyhccl.py:L1-L16）
import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from pyhccl_wrapper import (
    HCCLLibrary,
    aclrtStream_t,
    buffer_type,
    hcclComm_t,
    hcclDataTypeEnum,
    hcclRedOpTypeEnum,
    hcclUniqueId,
)

# SUBTRACTED: from vllm.logger import logger —— 用 stdlib logging 顶替（host 无 vllm）。
logger = logging.getLogger(__name__)


# SUBTRACTED: from vllm.distributed.utils import StatelessProcessGroup —— 基座类型；
#   host 无 vllm，给出忠实占位仅用于 isinstance 路由（精简版只跑「非 stateless」分支）。
#   原 import: vllm_ascend/distributed/device_communicators/pyhccl.py:L22
class StatelessProcessGroup:
    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl.py:L22 (import, 占位)
    pass


# SUBTRACTED: from vllm_ascend.utils import current_stream —— 取当前 NPU stream 的 helper；
#   host 无 NPU，占位（仅在真正发起 HCCL 调用的 NPU 路径被引用，host 不触达）。
def current_stream():
    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl.py:L34 (import, 占位)
    raise RuntimeError("current_stream: host 无 NPU（精简版占位，原为 vllm_ascend.utils）")


# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl.py:L37-L171
class PyHcclCommunicator:
    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl.py:L38-L128
    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        # SUBTRACTED: 原 12 行 docstring（L44-L54，解释 group/device/library_path 语义）——
        #   文档串不参与运行（plan 批准）。
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.HCCL, (
                "PyHcclCommunicator should be attached to a non-HCCL group."
            )
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return

        try:
            self.hccl = HCCLLibrary(library_path)
        except Exception:
            # disable because of missing HCCL library
            # e.g. in a non-NPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        logger.info("vLLM is using pyhccl")

        if isinstance(device, int):
            device = torch.device(f"npu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        if self.rank == 0:
            # get the unique id from HCCL
            with torch.npu.device(device):
                self.unique_id = self.hccl.hcclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = hcclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        # hccl communicator and stream will use this device
        # `torch.npu.device` is a context manager that changes the
        # current npu device to the specified one
        with torch.npu.device(device):
            self.comm: hcclComm_t = self.hccl.hcclCommInitRank(self.world_size, self.unique_id, self.rank)

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl.py:L130-L153
    def all_reduce(self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None) -> torch.Tensor:
        if self.disabled:
            return None
        # hccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, but the input tensor is on {in_tensor.device}"
        )

        out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.hccl.hcclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            hcclDataTypeEnum.from_torch(in_tensor.dtype),
            hcclRedOpTypeEnum.from_torch(op),
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )
        return out_tensor

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl.py:L155-L171
    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        buffer = buffer_type(tensor.data_ptr())
        self.hccl.hcclBroadcast(
            buffer,
            tensor.numel(),
            hcclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )
