"""换底座注入点②：子类化 DeviceCommunicatorBase，只改差异点。

只做减法的忠实精简版 —— 原文件本就只有 68 行，几乎无可删：整章「子类化=只改差异点」
的证据全在这。NPUCommunicator 继承基类后，所有集合通信（all_reduce/all_gather/
reduce_scatter/gather/send/recv/broadcast）零修改复用基类的 dist.xxx(group=device_group)
（底层走进程组的 HCCL backend）；只动两处：__init__ 把 self.device 设成 NPU 当前设备、
self.ca_comm=None 占位，以及【新增】MoE 用的 all_to_all（基类没有此方法）。

host 无 NPU/CANN：本文件依赖 torch.npu 与 vllm 基座，实际集合通信不在 host 跑；
精简版用于阅读 all_to_all 的形状代数控制流（见 tests 的结构性校验）。
"""
# SUBTRACTED: 文件头 Apache-2.0 许可证注释块（原 vllm_ascend/.../npu_communicator.py:L1-L16）
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase


# SOURCE: vllm_ascend/distributed/device_communicators/npu_communicator.py:L23-L68
class NPUCommunicator(DeviceCommunicatorBase):
    # SOURCE: vllm_ascend/distributed/device_communicators/npu_communicator.py:L24-L38
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        # TODO(hz): Refer to CudaCommunicator's implementation to integrate PyHcclCommunicator
        # init device according to rank
        self.device = torch.npu.current_device()

        # For compatibility (mainly for reusing graph capturing code in vllm),
        # init custom all-reduce implementation interface as in CUDACommunicator.
        self.ca_comm = None

    # SOURCE: vllm_ascend/distributed/device_communicators/npu_communicator.py:L40-L68
    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
            output_list = []
            tensor_shape_base = input_list[self.rank].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))

        else:
            input_list = [t.contiguous() for t in torch.tensor_split(input_, self.world_size, scatter_dim)]
            output_list = [torch.empty_like(input_list[i]) for i in range(self.world_size)]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor
