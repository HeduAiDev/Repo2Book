# SOURCE: vllm/v1/worker/gpu/buffer_utils.py
# HOST SEAM：本章消费面一件——async_copy_to_gpu（V2 StructuredOutputsWorker
# 的掩码异步 H2D，双 H2D 链之一）。逐字（该文件其余的 UvaBuffer/池化/流
# 管理归 ch12 的传输域）。
from __future__ import annotations

import numpy as np
import torch


# SOURCE: vllm/v1/worker/gpu/buffer_utils.py:L26-L44 async_copy_to_gpu —— 逐字
def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # pin_memory() is no-op if the memory is already pinned.
    pinned = x.pin_memory()
    return out.copy_(pinned, non_blocking=True)
