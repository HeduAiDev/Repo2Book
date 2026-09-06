# SOURCE: vllm/utils/torch_utils.py
# HOST SEAM：本章消费面五件——PIN_MEMORY（sampler.py:L9 / penalties 的
# make_tensor_with_pad）、guard_cuda_initialization（插件加载器，
# logits_processor/__init__.py:L77/L133）、async_tensor_h2d（两个内置
# 处理器的 _device_tensor）、make_tensor_with_pad + make_ndarray_with_pad +
# TORCH_DTYPE_TO_NUMPY_DTYPE（penalties 的历史张量化）。均逐字。
from __future__ import annotations

import contextlib
import os
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

from vllm.utils.platform_utils import is_pin_memory_available

T = TypeVar("T")

# SOURCE: vllm/utils/torch_utils.py:L54-L61 TORCH_DTYPE_TO_NUMPY_DTYPE —— 逐字
#   （make_tensor_with_pad 的 torch→numpy dtype 换算表）
TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

# SOURCE: vllm/utils/torch_utils.py:L72 PIN_MEMORY = is_pin_memory_available()
#   —— 逐字（派生链经 HOST SEAM platforms）
PIN_MEMORY = is_pin_memory_available()


@contextlib.contextmanager
def guard_cuda_initialization():
    # SOURCE: vllm/utils/torch_utils.py:L185-L209 guard_cuda_initialization —— 逐字
    """Avoid unexpected CUDA initialization."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        yield
        return

    old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    except Exception as e:
        if "No CUDA GPUs are available" in str(e):
            err_msg = "CUDA initialization is blocked."
        else:
            err_msg = str(e)
        raise RuntimeError(err_msg) from e
    finally:
        if old_value is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value


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


# SOURCE: vllm/utils/torch_utils.py:L594-L616 make_ndarray_with_pad —— 逐字
def make_ndarray_with_pad(
    x: list[list[T]],
    pad: T,
    dtype: npt.DTypeLike,
    *,
    max_len: int | None = None,
) -> npt.NDArray:
    """
    Make a padded array from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    if max_len is None:
        # Unlike for most functions, map is faster than a genexpr over `len`
        max_len = max(map(len, x), default=0)

    padded_x = np.full((len(x), max_len), pad, dtype=dtype)
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, : len(blocktb)] = blocktb

    return padded_x


# SOURCE: vllm/utils/torch_utils.py:L619-L640 make_tensor_with_pad —— 逐字
def make_tensor_with_pad(
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: int | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor
