# SOURCE: vllm/platforms/__init__.py current_platform —— HOST SEAM：精简版触碰的
# 平台谓词子集。真实 current_platform 按宿主硬件解析（CUDA→nccl/CudaCommunicator）；
# 伴读本机无 CUDA 控制面，谓词如实报告（cpu 面），但 get_device_communicator_cls
# 仍解析到本章携带的 CudaCommunicator 切面（叙事对象；宿主上 pynccl seam 恒走
# 禁用路径，集合通信落 torch.distributed 兜底——真实源码自述的 testing 路径）。

from __future__ import annotations


# SOURCE: vllm/platforms/interface.py Platform —— HOST SEAM interface subset
class _PlatformSeam:
    # SOURCE: vllm/platforms/interface.py Platform.dispatch_key
    dispatch_key = "CPU"

    # SOURCE: vllm/platforms/interface.py Platform.is_cuda_alike
    def is_cuda_alike(self) -> bool:
        # HOST SEAM：宿主无 CUDA 控制面（GroupCoordinator 设备分支走 cpu 端）
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cpu
    def is_cpu(self) -> bool:
        return True

    # SOURCE: vllm/platforms/interface.py Platform.is_tpu
    def is_tpu(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_rocm
    def is_rocm(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py Platform.use_custom_op_collectives
    def use_custom_op_collectives(self) -> bool:
        # HOST SEAM：CPU 平台不启用 custom-op 集合调用（真实语义按平台而定）
        return False

    # SOURCE: vllm/platforms/cuda.py CudaPlatform.logical_device_id_to_visible_device_id
    def logical_device_id_to_visible_device_id(self, local_rank: int) -> int:
        return local_rank


# SOURCE: vllm/platforms/cuda.py CudaPlatform.get_device_communicator_cls —— 解析到
# 本章携带的 CudaCommunicator 切面（规范 qualname；真 vllm 在场时由真平台接管）。
def get_device_communicator_cls() -> str:  # HOST SEAM
    return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"


_PlatformSeam.get_device_communicator_cls = staticmethod(get_device_communicator_cls)

current_platform = _PlatformSeam()
