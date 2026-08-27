# SOURCE: vllm/v1/utils.py
# 本章消费面：CpuGpuBuffer——worker 侧 block_table 的 CPU/GPU 双镜像容器
# （.np 写 / commit 拷 .gpu[:n] 活跃行）。
# HOST SEAM：CPU host 上 .gpu 与 .cpu 同为 CPU 张量（同构造、真拷贝）——
#   双镜像契约逐字成立，容器内为真 GPU。
# SUBTRACTED: get_engine_client_zmq_addr / APIServerProcessManager 等引擎
#   装配族（ch05）与本链路无关；PIN_MEMORY 默认值折入 host False。
import numpy as np
import torch

# SOURCE: vllm/utils/torch_utils.py:L72 PIN_MEMORY（HOST SEAM：CPU host 无
#   pinned memory；容器内真 GPU 环境为 True）
PIN_MEMORY = False


# SOURCE: vllm/v1/utils.py:L110 CpuGpuBuffer
class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    # SOURCE: vllm/v1/utils.py:L113-L137 __init__
    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device | None = None,
        pin_memory: bool = PIN_MEMORY,
        with_numpy: bool = True,
    ) -> None:
        # SUBTRACTED: device 形参校验（真实断言 device 已初始化——切面
        #   device 直供，host 上同 CPU）。
        # these buffers are mutable runtime state, so allocate them as normal
        # SOURCE: vllm/v1/utils.py:L122-L126
        with torch.inference_mode(False):
            self.cpu = torch.zeros(
                *size, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self.gpu = torch.zeros_like(self.cpu, device=self.cpu.device)
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        # SOURCE: vllm/v1/utils.py:L131-L137
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    # SOURCE: vllm/v1/utils.py:L139 copy_to_gpu
    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        # SOURCE: vllm/v1/utils.py:L140-L142
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
