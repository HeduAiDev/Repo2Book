# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA 自定义算子的 host 替身（本章唯一的 kernel 面 SEAM）。

# SOURCE: vllm/_custom_ops.py:swap_blocks_batch（真实实现在 C++ 扩展
#   csrc/cache_kernels.cu 的 batch 版，经 torch.library 注册进 `vllm._custom_ops`）
# SEAM: 真源是 CUDA C++ 批量拷贝 kernel（吃描述符三缓冲：src 指针/dst 指针/size，
#   每描述符一次 memcpy）；host 无 CUDA，本替身用 ctypes.memmove 执行**同一套
#   描述符语义**——指针从哪来、搬多少字节、搬到哪，逐描述符等价。
#   is_src_access_order_any 只是 CUDA 驱动的源读流水线提示，host 路径无驱动
#   可提示，参数保留但不改变行为（真源在 GPU 上也不改变拷贝内容）。
"""

import ctypes

import torch


# SOURCE: vllm/_custom_ops.py:swap_blocks_batch（C++ kernel 的 Python 绑定签名）
def swap_blocks_batch(
    src: torch.Tensor,
    dst: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
) -> None:
    """Batch copy: for each i, copy sizes[i] bytes from src[i] to dst[i]."""
    src_ptrs = src.detach().cpu().numpy()
    dst_ptrs = dst.detach().cpu().numpy()
    size_arr = sizes.detach().cpu().numpy()
    for s, d, n in zip(src_ptrs, dst_ptrs, size_arr):
        if n > 0:
            ctypes.memmove(int(d), int(s), int(n))


__all__ = ["swap_blocks_batch"]
