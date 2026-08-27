# SOURCE: vllm/utils/torch_utils.py
# 本章消费面：get_dtype_size（real_page_size_bytes 公式的 dtype 字节项）、
# PIN_MEMORY（CpuGpuBuffer 默认）、async_tensor_h2d（KVBlockZeroer 的块 id
# 异步上载——CUDA 分支用）。
import numpy as np
import torch

# SOURCE: vllm/utils/torch_utils.py:L72 PIN_MEMORY
# HOST SEAM：CPU host 无 pinned memory（真实 = is_pin_memory_available()；
# 容器内真 GPU 环境为 True）。
PIN_MEMORY = False


# SOURCE: vllm/utils/torch_utils.py:L212 get_dtype_size
def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    # SOURCE: vllm/utils/torch_utils.py:L214
    return torch.tensor([], dtype=dtype).element_size()


# SOURCE: vllm/utils/torch_utils.py:L573 async_tensor_h2d
def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Copy list/numpy array/tensor async from host to device."""
    # SOURCE: vllm/utils/torch_utils.py:L579-L586
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        t = data.pin_memory() if PIN_MEMORY else data
    else:
        t = torch.tensor(data, dtype=dtype, pin_memory=PIN_MEMORY, device="cpu")
    assert t.is_cpu
    return t.to(device=device, dtype=dtype, non_blocking=True)
