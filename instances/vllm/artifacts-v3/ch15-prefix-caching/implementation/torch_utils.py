# SOURCE: vllm/utils/torch_utils.py
# 本章只消费 async_tensor_h2d（worker 侧 CoW 拷贝把块号对搬上设备）。
# SUBTRACTED: 该模块其余函数（is_pin_memory_available/copy 等——ch13 已建
#   全量切面）；PIN_MEMORY 以 host 无 CUDA 的 False 定值（HOST SEAM——
#   GPU 面（pin memory + non_blocking）在容器验）。
import numpy as np
import torch

# SOURCE: vllm/utils/torch_utils.py:L72 PIN_MEMORY（HOST SEAM：host 无 CUDA
#   恒 False；容器内为 is_pin_memory_available()）
PIN_MEMORY = False


# SOURCE: vllm/utils/torch_utils.py:L573 async_tensor_h2d
def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Copy list/numpy array/tensor async from host to device."""
    # SOURCE: vllm/utils/torch_utils.py:L578-L585
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        t = data.pin_memory() if PIN_MEMORY else data
    else:
        t = torch.tensor(data, dtype=dtype, pin_memory=PIN_MEMORY, device="cpu")
    assert t.is_cpu
    return t.to(device=device, dtype=dtype, non_blocking=True)
