# SOURCE: vllm/v1/worker/utils.py
# **worker 侧 CoW 执行**（m14 管线终点）：copy_kv_cache_blocks_inplace——
# 调度器只给 (src,dst) 块号对，真拷贝在 worker：块主序底层存储上
# blocks[dst] = blocks[src]（块 i 拥有连续字节区间 [i*page,(i+1)*page)）。
# HOST SEAM：torch CPU 张量等价复现（GPU 带宽语义不变；容器内验真 GPU）。
# SUBTRACTED: 该模块其余函数（bind_kv_caches/is_residual_scattered 等——
#   ch13/ch17 各章切面）。
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from .kv_cache_utils import KVCacheBlockCopy
from .torch_utils import async_tensor_h2d


# SOURCE: vllm/v1/worker/utils.py:L528 copy_kv_cache_blocks_inplace
def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    # SOURCE: vllm/v1/worker/utils.py:L533-L534
    if not kv_cache_block_copies:
        return

    # SOURCE: vllm/v1/worker/utils.py:L536-L547（去重共享底层存储——
    #   Mamba 层持状态张量列表、注意力层单张量，都别名同一块主序后备）
    storage_tensors: list[torch.Tensor] = []
    seen_storage: set[int] = set()
    for entry in kv_caches:
        # Mamba layers hold a list of state tensors; attention layers a single
        # tensor. Both alias the shared block-major backing storage.
        tensors = entry if isinstance(entry, (list, tuple)) else (entry,)
        for tensor in tensors:
            ptr = tensor.untyped_storage().data_ptr()
            if ptr in seen_storage:
                continue
            seen_storage.add(ptr)
            storage_tensors.append(tensor)

    # SOURCE: vllm/v1/worker/utils.py:L549-L554
    if not storage_tensors:
        return
    device = storage_tensors[0].device
    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)
    indices = async_tensor_h2d(indices_np, device=device)
    src_indices, dst_indices = indices.unbind(dim=1)

    # SOURCE: vllm/v1/worker/utils.py:L556-L564（块主序存储：块 i 拥有连续
    #   字节区间 [i*page, (i+1)*page)——整块搬运）
    for tensor in storage_tensors:
        assert tensor.device == device
        blocks = torch.empty(0, dtype=torch.uint8, device=device)
        blocks.set_(tensor.untyped_storage())
        # Block-major backing storage: block i owns the contiguous byte range
        # [i * page_size, (i + 1) * page_size).
        assert blocks.numel() % num_blocks == 0
        blocks = blocks.view(num_blocks, -1)
        blocks[dst_indices] = blocks[src_indices]
