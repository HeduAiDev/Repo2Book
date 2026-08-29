# SOURCE: vllm/v1/utils.py
# ch22 切面（m4）：CpuGpuBuffer——块表双镜像与 slot_mapping 持久缓冲的共同
# 基座（cpu pinned + gpu + np 三视图一体；copy_to_gpu(n) 只传活跃前缀、
# non_blocking）。ch18 已全文立过；本章块表这一份照用，全文逐字。
from __future__ import annotations

import numpy as np
import torch

from .torch_utils import PIN_MEMORY


# SOURCE: vllm/v1/utils.py:L110 CpuGpuBuffer
class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    # SOURCE: vllm/v1/utils.py:L113-L120 __init__ 签名
    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = PIN_MEMORY,
        with_numpy: bool = True,
    ) -> None:
        # SOURCE: vllm/v1/utils.py:L121-L126（cpu pinned / gpu 同形零张量）
        # these buffers are mutable runtime state, so allocate them as normal
        with torch.inference_mode(False):
            self.cpu = torch.zeros(
                *size, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self.gpu = torch.zeros_like(self.cpu, device=device)
        # SOURCE: vllm/v1/utils.py:L127-L137（np 视图按需建；bfloat16 拒 numpy）
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    # SOURCE: vllm/v1/utils.py:L139 copy_to_gpu —— 只传活跃前缀 non_blocking
    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        # SOURCE: vllm/v1/utils.py:L141-L142
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
