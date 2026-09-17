# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章用到的 torch 小工具。

# SOURCE: vllm/utils/torch_utils.py:L1-L120
# SUBTRACTED: 自定义 op 注册/编译期工具/显存工具——本章的数据面不触编译栈。
"""

import torch


# SOURCE: vllm/utils/torch_utils.py:L212-L230
# SUBTRACTED: 预建张量的查表实现——改用 torch 自带 element_size()，语义相同。
# SOURCE: vllm/utils/torch_utils.py:L212-L230
def get_dtype_size(dtype: torch.dtype) -> int:
    # SOURCE: vllm/utils/torch_utils.py:L212-L230
    """Get the size of a torch dtype in bytes."""
    return torch.empty((), dtype=dtype).element_size()


# SOURCE: vllm/utils/torch_utils.py:PIN_MEMORY 条目
# SUBTRACTED: torch.cuda.is_available() 探测——host 无 CUDA 时恒 False，描述符
#   缓冲落普通页锁内存语义之外的路径（真实部署在 CUDA host 上为 True）。
PIN_MEMORY: bool = False


__all__ = ["get_dtype_size", "PIN_MEMORY"]
