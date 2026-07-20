# SOURCE: vllm/v1/worker/gpu/buffer_utils.py
# 只做减法的忠实精简版。本章只需要 async_copy_to_gpu——StructuredOutputsWorker
# 用它把掩码 np.ndarray 异步搬到预分配的 GPU 缓冲上。
#
# SUBTRACTED: SPDX 版权头、UvaBuffer 类（统一虚拟地址缓冲，与掩码搬运无关）。
import numpy as np
import torch


def async_copy_to_gpu(
    x: "torch.Tensor | np.ndarray",
    out: "torch.Tensor | None" = None,
    device: "torch.device | None" = None,
) -> "torch.Tensor":
    # SOURCE: vllm/v1/worker/gpu/buffer_utils.py:L17-33
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # Copy directly to GPU — explicit pin_memory() causes sporadic stalls
    # under high concurrency due to CUDA driver contention. The driver
    # handles the transfer efficiently without manual pinning.
    return out.copy_(x, non_blocking=True)
