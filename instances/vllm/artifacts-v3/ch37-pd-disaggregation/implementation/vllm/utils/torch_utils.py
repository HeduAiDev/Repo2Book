# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章用到的 torch 小工具。

# SOURCE: vllm/utils/torch_utils.py:L1-L120
# SUBTRACTED: 自定义 op 注册/编译期工具/显存工具——本章的 NIXL 数据面不触编译栈。
"""

import numpy as np
import torch


# SOURCE: vllm/utils/torch_utils.py:L212-L230
# SUBTRACTED: 预建张量的查表实现——改用 torch 自带的 element_size()，语义相同
# （AttentionSpec.real_page_size_bytes 用它把 head_dim 折算成字节数）。
# SOURCE: vllm/utils/torch_utils.py:L212-L230
def get_dtype_size(dtype: torch.dtype) -> int:
    # SOURCE: vllm/utils/torch_utils.py:L212-L230
    """Get the size of a torch dtype in bytes."""
    return torch.empty((), dtype=dtype).element_size()


# SOURCE: vllm/utils/torch_utils.py:L573-L586
# SUBTRACTED: PIN_MEMORY 分支（非阻塞 pinned 拷贝）——本章 host 无 CUDA，
#   落到普通 CPU 张量；调用点拿到的东西不变（device 上的 index 张量）。
# SOURCE: vllm/utils/torch_utils.py:L573-L586
def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # SOURCE: vllm/utils/torch_utils.py:L573-L586
    """Copy list/numpy array/tensor async from host to device."""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        t = data
    else:
        t = torch.tensor(data, dtype=dtype, device="cpu")
    return t.to(device=device, dtype=dtype)


__all__ = ["get_dtype_size", "async_tensor_h2d"]
