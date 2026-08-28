# SOURCE: vllm/v1/utils.py
# 本章主角文件之一：CpuGpuBuffer（m05）——固定形状 cpu(pinned)+gpu+np 三视图
# 双端缓冲，copy_to_gpu(n) 只传活跃前缀。全 runner per-step 张量的地址稳定器。
# 另收 copy_slice（_make_sampling_metadata 的前缀拷贝）与
# record_function_or_nullcontext（execute_model/sample_tokens 的剖面上下文）。
# SUBTRACTED: ConstantList / get_engine_client_zmq_addr /
#   APIServerProcessManager 等引擎装配族（ch05/ch09 域，与本章链路无关）。
from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager

import numpy as np
import torch

from .torch_utils import PIN_MEMORY
from ._host_seams import envs


# SOURCE: vllm/v1/utils.py:L110 CpuGpuBuffer —— 全文逐字（m05 本章标题之一）
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

    # SOURCE: vllm/v1/utils.py:L139 copy_to_gpu(n) —— 活跃前缀语义（m05 核心）
    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        # SOURCE: vllm/v1/utils.py:L140-L142
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    # SOURCE: vllm/v1/utils.py:L144 copy_to_cpu(n)
    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        # SOURCE: vllm/v1/utils.py:L147-L149
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


# SOURCE: vllm/v1/utils.py:L647 copy_slice
def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    # SOURCE: vllm/v1/utils.py:L656-L657
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


# SOURCE: vllm/v1/utils.py:L758 record_function_or_nullcontext
# （envs 三开关由 HOST SEAM 提供：全 False → nullcontext，与真实默认环境一致）
def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    # SOURCE: vllm/v1/utils.py:L760-L762 —— fast path assume it is set
    global _PROFILER_FUNC
    if _PROFILER_FUNC is not None:
        return _PROFILER_FUNC(name)

    # SOURCE: vllm/v1/utils.py:L764-L771（真实在文件头 L27 顶层 import
    #   from torch.autograd.profiler import record_function——HOST 移植改惰性
    #   import，默认全 False 分支不受影响）
    func = contextlib.nullcontext
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        from torch.autograd.profiler import record_function

        func = record_function
    elif envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        import nvtx

        func = nvtx.annotate

    _PROFILER_FUNC = func
    return func(name)


# SOURCE: vllm/v1/utils.py:L755 _PROFILER_FUNC 全局（剖面钩子，默认 None）
_PROFILER_FUNC = None
