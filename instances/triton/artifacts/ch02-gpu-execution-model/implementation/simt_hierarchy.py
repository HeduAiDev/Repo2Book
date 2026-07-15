"""Reference model -- SIMT execution hierarchy: grid -> block(CTA) -> warp ->
lane.

PAPER: NVIDIA CUDA C++ Programming Guide, "Thread Hierarchy" and "Hardware
Implementation: SIMT Architecture"
(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html): a kernel
launch's threads are grouped into blocks (CTAs); the hardware further
partitions each block into warps of 32 threads, the unit that is actually
scheduled and executed in lockstep (SIMT).

Triton's programmer-facing model stops one level higher than raw CUDA: a
kernel author writes to the *program* (== block/CTA/tile) level and never
names a warp or a lane directly -- ``tl.program_id`` is the only place a
program instance's identity appears
(``python/triton/language/core.py:L1148-L1163``, lowering to
``create_get_program_id`` -> PTX ``%ctaid``,
``python/triton/language/semantic.py:L28-L31``). Warp/lane partitioning below
the program level is entirely the compiler's and hardware's job; this module
exists only to make that invisible layer visible for the reader.
"""
from __future__ import annotations

WARP_SIZE = 32  # PAPER: CUDA C++ Programming Guide -- Hardware Implementation: SIMT Architecture


def num_warps_per_block(threads_per_block: int, warp_size: int = WARP_SIZE) -> int:
    # PAPER: CUDA C++ Programming Guide -- "Hardware Implementation: SIMT Architecture"
    """How many warps a block of ``threads_per_block`` logical threads is cut
    into. Real hardware always rounds a block's thread count up to a whole
    number of warps."""
    return -(-threads_per_block // warp_size)  # ceil division


def partition_into_warps(threads_per_block: int, warp_size: int = WARP_SIZE) -> list:
    # PAPER: CUDA C++ Programming Guide -- "Thread Hierarchy" / "Hardware Implementation: SIMT Architecture"
    """Split a block's lane ids ``[0, threads_per_block)`` into consecutive
    groups of ``warp_size`` -- the same grouping the hardware scheduler
    applies. The last warp may be a *partial* warp (padded with inactive
    lanes on real hardware) when ``threads_per_block`` is not a multiple of
    ``warp_size``."""
    return [
        list(range(start, min(start + warp_size, threads_per_block)))
        for start in range(0, threads_per_block, warp_size)
    ]
