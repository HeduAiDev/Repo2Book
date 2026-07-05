# vllm_ascend/simple_kv_offload/npu_mem_ops.py —— subtract-only companion（极简路径底层原语）
#
# 两条路径最终都收口到同一个昇腾批量搬运算子 torch.ops._C_ascend.swap_blocks_batch（底层
# aclrtMemcpyBatchAsync）。本文件是极简路径的指针拼装：
#   · build_params —— 预算每子张量 base_ptr 与 bytes-per-block（bpb = stride(0)*element_size），
#     方向码 DIRECTION_H2D=0(load) / DIRECTION_D2H=1(store) 与 C++ 绑定约定一致。
#   · copy_blocks —— numpy 广播把 (num_sub_tensors, n) 的 src/dst 指针与尺寸拼平、发一次算子。
# 注意：极简路径 CPU/NPU block 1:1 同粒度，无 block_size_factor 展开（与标准路径根本区别）。
#
# host 无 NPU：swap_blocks_batch 由 runtime_stub 补丁记录调用（不真搬字节）；指针算术纯 Python，可跑可验。
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

import runtime_stub  # noqa: F401 —— 接缝：装上 torch.ops._C_ascend.swap_blocks_batch host 补丁

# Direction codes shared with csrc/torch_binding.cpp::swap_blocks_batch.
DIRECTION_H2D = 0  # SOURCE: vllm_ascend/simple_kv_offload/npu_mem_ops.py:L17
DIRECTION_D2H = 1  # SOURCE: vllm_ascend/simple_kv_offload/npu_mem_ops.py:L18


class BatchMemcpyParams(NamedTuple):  # SOURCE: vllm_ascend/simple_kv_offload/npu_mem_ops.py:L21
    """Pre-computed per-tensor descriptors for batched block copy."""

    src_bases: np.ndarray  # [num_sub_tensors] int64 — data_ptr per tensor
    dst_bases: np.ndarray  # [num_sub_tensors] int64
    bpb: np.ndarray  # [num_sub_tensors] int64 — bytes per block
    num_sub_tensors: int
    direction: int  # DIRECTION_H2D or DIRECTION_D2H


def _ordered_tensors(caches: dict[str, torch.Tensor]) -> list[torch.Tensor]:  # SOURCE: vllm_ascend/simple_kv_offload/npu_mem_ops.py:L31
    """Return values in insertion order (kept as a function for clarity)."""
    return list(caches.values())


def build_params(  # SOURCE: vllm_ascend/simple_kv_offload/npu_mem_ops.py:L36
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    direction: int,
) -> BatchMemcpyParams:
    """Build cached pointer/stride descriptors for all sub-tensors.

    Both ``src_caches`` and ``dst_caches`` must have identical keys and a
    matching ``[num_blocks, block_bytes]`` layout (already prepared by
    :class:`SimpleCPUOffloadNPUWorker.register_kv_caches`).
    """
    assert list(src_caches.keys()) == list(dst_caches.keys()), "src/dst cache key order must match"
    src_tensors = _ordered_tensors(src_caches)
    dst_tensors = _ordered_tensors(dst_caches)

    src_bases: list[int] = []
    dst_bases: list[int] = []
    bpb: list[int] = []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        d_bpb = d.stride(0) * d.element_size()
        assert s_bpb == d_bpb, f"per-block bytes mismatch src={s_bpb} dst={d_bpb}"
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.int64),
        dst_bases=np.array(dst_bases, dtype=np.int64),
        bpb=np.array(bpb, dtype=np.int64),
        num_sub_tensors=len(src_tensors),
        direction=direction,
    )


def copy_blocks(  # SOURCE: vllm_ascend/simple_kv_offload/npu_mem_ops.py:L71
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Issue a batched async DMA on the *current* NPU stream.

    The caller is expected to be inside a ``torch.npu.stream(...)``
    context so the issued copies bind to the dedicated transfer stream.
    """
    n = len(src_block_ids)
    if n == 0:
        return
    assert n == len(dst_block_ids), "src/dst block counts must match"

    src_ids = np.asarray(src_block_ids, dtype=np.int64)
    dst_ids = np.asarray(dst_block_ids, dtype=np.int64)

    # Layout: (num_sub_tensors, n) flattened — contract of swap_blocks_batch.
    bpb_col = params.bpb[:, None]
    src_all = (params.src_bases[:, None] + src_ids[None, :] * bpb_col).ravel()
    dst_all = (params.dst_bases[:, None] + dst_ids[None, :] * bpb_col).ravel()
    sz_all = np.broadcast_to(bpb_col, (params.num_sub_tensors, n)).ravel().copy()

    batch_src = torch.from_numpy(src_all)
    batch_dst = torch.from_numpy(dst_all)
    batch_sizes = torch.from_numpy(sz_all)

    torch.ops._C_ascend.swap_blocks_batch(batch_src, batch_dst, batch_sizes, params.direction)
