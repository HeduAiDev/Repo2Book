"""Reference model -- coalesced vs. scattered global-memory access.

PAPER: NVIDIA CUDA C++ Best Practices Guide, "Coalesced Access to Global
Memory" (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
a warp's simultaneous memory accesses are coalesced by hardware into the
minimum number of transactions when they fall into the same aligned segment;
scattered/strided addresses split into one transaction per distinct segment,
dividing effective bandwidth by (roughly) the number of transactions.
"""
from __future__ import annotations

from typing import Sequence

WARP_SIZE = 32
TRANSACTION_BYTES = 128  # PAPER: CUDA C++ Best Practices Guide -- Coalesced Access to Global Memory


def count_transactions(byte_addresses: Sequence[int], transaction_bytes: int = TRANSACTION_BYTES) -> int:
    # PAPER: CUDA C++ Best Practices Guide -- "Coalesced Access to Global Memory"
    """Count how many ``transaction_bytes``-aligned memory transactions a
    single warp's simultaneous byte addresses require: one per distinct
    aligned segment the addresses fall into (the hardware coalescing unit)."""
    segments = {addr // transaction_bytes for addr in byte_addresses}
    return len(segments)


def warp_offsets_bytes(block_start_offset: int, warp_lane_ids: Sequence[int], element_bytes: int) -> list:
    # PAPER: CUDA C++ Best Practices Guide -- "Coalesced Access to Global Memory"
    """Byte addresses a warp touches when accessing ``element_bytes``-wide
    elements at contiguous logical offsets ``block_start_offset + lane_id`` --
    the pattern ``tl.load(x_ptr + offsets)`` in vector-add produces, since
    ``offsets = block_start + tl.arange(0, BLOCK_SIZE)`` is contiguous."""
    return [(block_start_offset + lane_id) * element_bytes for lane_id in warp_lane_ids]


def strided_offsets_bytes(base_offset: int, warp_lane_ids: Sequence[int], stride_elements: int, element_bytes: int) -> list:
    # PAPER: CUDA C++ Best Practices Guide -- "Coalesced Access to Global Memory"
    """Byte addresses a warp touches under a *strided/gather* access pattern
    -- the counter-example that shows what coalescing buys you when it is
    absent (vector-add's own ``offsets`` never takes this shape; this
    function exists purely to make the contrast measurable)."""
    return [(base_offset + lane_id * stride_elements) * element_bytes for lane_id in warp_lane_ids]
