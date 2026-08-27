# SOURCE: vllm/v1/worker/utils.py
# KVBlockZeroer——新块清零器（m8）：构造期预计算各段绝对地址表，每步
# zero_block_ids 用 Triton kernel 把新分块的显存清零——块是从自由队列回收
# 的，上一任主人留下的字节还躺在显存里（gpu_model_runner.py:L1219-L1221
# 注释原话 "to prevent stale NaN/data from corrupting attention or SSM
# computation"）。
# HOST SEAM：CPU host 无 CUDA launch——zero_block_ids 在 CPU 设备经**同一张
#   绝对地址表**用 ctypes.memset 置零（kernel 的 ptr 写零语义逐行对应）；
#   CUDA 分支逐字保留，容器内真跑。
# SUBTRACTED: 本文件其余工具族（copy_kv_cache_blocks/CoW 拷贝 → ch15、
#   is_pin_memory_available 等）；AttentionGroup 的类型面（backend 的
#   AttentionBackend/AttentionMetadataBuilder 类型导入 → ch21，换 Any 账位）。
import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product as iprod
from typing import Any

import torch
import triton
import triton.language as tl

from .kv_cache_interface import FullAttentionSpec, KVCacheSpec
from .math_utils import largest_power_of_2_divisor
from .torch_utils import async_tensor_h2d


# SOURCE: vllm/v1/worker/utils.py:L44 _zero_kv_blocks_kernel
@triton.jit(do_not_specialize=["n_blocks"])
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    Segments may have different page sizes (e.g. models with multiple KV
    cache groups like MLA + DSA indexer).  Each segment's page size is
    read from seg_page_sizes_ptr; programs whose chunk_index falls beyond
    their segment's page size early-exit.

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped as (block_index, seg_index, chunk_index).
    """
    # SOURCE: vllm/v1/worker/utils.py:L71-L90
    pid = tl.program_id(0)
    work_per_block = N_SEGS * MAX_CHUNKS
    block_index = pid // work_per_block
    if block_index >= n_blocks:
        return
    remainder = pid % work_per_block
    seg_index = remainder // MAX_CHUNKS
    chunk_index = remainder % MAX_CHUNKS
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    if chunk_index >= page_size_el // BLOCK_SIZE:
        return
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    offset = (
        block_id.to(tl.int64) * page_size_el.to(tl.int64)
        + chunk_index.to(tl.int64) * BLOCK_SIZE
    )
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


# SOURCE: vllm/v1/worker/utils.py:L93 KVBlockZeroer
class KVBlockZeroer:
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Construct once after KV caches are allocated to precompute segment
    addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    # SOURCE: vllm/v1/worker/utils.py:L101 __init__
    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable[Any],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        static_forward_context: dict[str, Any],
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        """Precompute the absolute-address table for the Triton zeroing kernel.

        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        Each segment's page_size_el accounts for this ratio so that
        ``block_id * page_size_el`` lands at the correct offset.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        # SUBTRACTED: AttentionGroup 类型标注换 Any（注意力后端族导入 → ch21；
        #   测试侧以最小 duck type 提供 group.kv_cache_spec/backend/layer_
        #   names/kv_cache_group_id）。
        # SOURCE: vllm/v1/worker/utils.py:L122-L129
        self.device = device
        self._meta: tuple[torch.Tensor, torch.Tensor, int, int, int] | None = None

        # SOURCE: vllm/v1/worker/utils.py:L125-L129
        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        seg_page_sizes: list[int] = []

        # SOURCE: vllm/v1/worker/utils.py:L131-L173 段表预计算（逐层 kv_cache
        #   张量按 block_dim 切段：块外层 → 1 段/缓冲，K/V 外层 → 2 段/缓冲）
        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            ratio = spec.block_size // kernel_bs
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_bs,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv = static_forward_context[layer_name].kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                dp = kv.data_ptr()
                if dp in seen_ptrs:
                    continue
                seen_ptrs.add(dp)

                el = kv.element_size()
                cur_bytes = kv.stride(block_dim) * el
                assert cur_bytes % 4 == 0
                kernel_block_el = cur_bytes // 4
                cur_page_el = kernel_block_el * ratio

                block_stride_bytes = cur_bytes
                outer_dims = [
                    d
                    for d in range(block_dim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    seg_addrs.append(dp + off_bytes)
                    seg_page_sizes.append(cur_page_el)

        # SOURCE: vllm/v1/worker/utils.py:L175-L190
        if not seg_addrs:
            self._meta = None
            return

        max_page_size_el = max(seg_page_sizes)
        blk_size = min(
            min(largest_power_of_2_divisor(ps) for ps in seg_page_sizes),
            1024,
        )
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            max_page_size_el // blk_size,
            blk_size,
            len(seg_addrs),
        )

    # SOURCE: vllm/v1/worker/utils.py:L192 zero_block_ids —— 清零入口
    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        # SOURCE: vllm/v1/worker/utils.py:L194-L195
        if not block_ids or self._meta is None:
            return
        seg_addrs, seg_page_sizes, max_chunks, blk_size, n_segs = self._meta

        # HOST SEAM：CPU host 无 CUDA——经同一张绝对地址表按 (block, seg) 置
        #   零（kernel L85-L88 的 offset = block_id*page_size_el + chunk*
        #   BLOCK_SIZE 逐行对应；int32 段 = page_size_el*4 字节）。
        if self.device.type == "cpu":
            for block_id in block_ids:
                for seg_index in range(n_segs):
                    page_el = int(seg_page_sizes[seg_index].item())
                    addr = int(seg_addrs[seg_index].item()) + block_id * page_el * 4
                    ctypes.memset(addr, 0, page_el * 4)
            return

        # SOURCE: vllm/v1/worker/utils.py:L196-L208（CUDA 分支逐字）
        n_blocks = len(block_ids)
        idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
        grid = (n_blocks * n_segs * max_chunks,)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            seg_page_sizes,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            MAX_CHUNKS=max_chunks,
            BLOCK_SIZE=blk_size,
        )

    # SOURCE: vllm/v1/worker/utils.py:L210 warmup
    def warmup(self, num_kv_blocks: int) -> None:
        """JIT-compile the zeroing kernel before the first real request."""
        # SOURCE: vllm/v1/worker/utils.py:L212-L213
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


# SOURCE: vllm/v1/worker/utils.py:L216 AttentionGroup
@dataclass
class AttentionGroup:
    """One attention group: backend + layers + spec + group id.

    真实的 metadata_builders（ubatch 的 per-builder 持久缓冲，L222-L227）
    归 ch21/22；backend 类型面（AttentionBackend）→ ch21。
    """

    # SOURCE: vllm/v1/worker/utils.py:L218-L221
    backend: Any
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # SUBTRACTED: metadata_builders（L222-L227——ubatch 元数据构建器，ch21/22）。
