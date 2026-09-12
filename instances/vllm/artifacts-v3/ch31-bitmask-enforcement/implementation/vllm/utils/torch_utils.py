# SOURCE: vllm/utils/torch_utils.py
# HOST SEAM：本章消费面两件——PIN_MEMORY（utils.py pinned sorted_bitmask 与
# V2 copy_stream 双 H2D 的物质前提，L72）与 async_tensor_h2d（out_indices 的
# 搬运工，L573-L586——自己搬 tensor 免 xgrammar 内 cpu sync）。逐字。
from __future__ import annotations

import numpy as np
import torch

from vllm.utils.platform_utils import is_pin_memory_available

# SOURCE: vllm/utils/torch_utils.py:L72 PIN_MEMORY = is_pin_memory_available()
#   —— 逐字（派生链经 HOST SEAM platforms）
PIN_MEMORY = is_pin_memory_available()


# SOURCE: vllm/utils/torch_utils.py:L573-L586 async_tensor_h2d —— 逐字
def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Copy list/numpy array/tensor async from host to device."""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        t = data.pin_memory() if PIN_MEMORY else data
    else:
        t = torch.tensor(data, dtype=dtype, pin_memory=PIN_MEMORY, device="cpu")
    assert t.is_cpu
    return t.to(device=device, dtype=dtype, non_blocking=True)
