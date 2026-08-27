# SOURCE: vllm/v1/utils.py
# 本章消费面：CpuGpuBuffer——worker 侧 block_table/slot_mapping 的 CPU/GPU
# 双镜像容器（m15「commit 先行拷贝」的载体；容器内景深讲归 ch18）。
# HOST SEAM：CPU host 上 .gpu 与 .cpu 同为 CPU 张量（同构造、真拷贝）——
# 双镜像契约（CPU 写 .np / commit 拷 .gpu[:n]）逐字成立，容器内为真 GPU。
# SUBTRACTED: get_engine_client_zmq_addr / APIServerProcessManager 等引擎
#   装配族（ch05）与本链路无关。
import numpy as np
import torch

from .torch_utils import PIN_MEMORY


# SOURCE: vllm/v1/utils.py:L110 CpuGpuBuffer
class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    # SOURCE: vllm/v1/utils.py:L113-L137 __init__
    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = PIN_MEMORY,
        with_numpy: bool = True,
    ) -> None:
        # these buffers are mutable runtime state, so allocate them as normal
        # SOURCE: vllm/v1/utils.py:L122-L126
        with torch.inference_mode(False):
            self.cpu = torch.zeros(
                *size, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self.gpu = torch.zeros_like(self.cpu, device=device)
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

    # SOURCE: vllm/v1/utils.py:L144 copy_to_cpu
    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        # SOURCE: vllm/v1/utils.py:L147-L149
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)
