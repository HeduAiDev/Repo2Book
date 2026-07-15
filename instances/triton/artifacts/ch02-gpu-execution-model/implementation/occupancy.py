"""Reference model -- occupancy: active warps / max warps per SM, gated by
per-thread register usage and per-block shared-memory usage.

PAPER: NVIDIA CUDA C++ Programming Guide, "Hardware Implementation: Hardware
Multithreading" (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html):
the number of blocks and warps resident on each multiprocessor for a given
kernel launch is capped by the multiprocessor's register file and shared
memory, among other resources. The two resource gates (registers, shared
memory) are combined by taking the smaller of the two resulting occupancies
-- an SM cannot exceed either budget simultaneously.

Constants below are Ampere-class order-of-magnitude hardware limits (the
same numbers used in the chapter's worked numeric example); real limits are
per-architecture and should be read from ``cudaDeviceProp`` / the CUDA
Occupancy Calculator in production code, not hardcoded like this.
"""
from __future__ import annotations

WARP_SIZE = 32
# PAPER: CUDA C++ Programming Guide -- Hardware Implementation: Hardware Multithreading
MAX_REGISTERS_PER_SM = 65536              # 64K 32-bit registers/SM (Ampere-class order of magnitude)
MAX_THREADS_PER_SM = 2048                 # => 64 warps/SM max
MAX_SHARED_MEM_PER_SM_BYTES = 164 * 1024  # 164 KiB/SM (Ampere-class order of magnitude)
MAX_RESIDENT_BLOCKS_PER_SM = 32


def max_warps_per_sm(max_threads_per_sm: int = MAX_THREADS_PER_SM, warp_size: int = WARP_SIZE) -> int:
    # PAPER: CUDA C++ Programming Guide -- Hardware Implementation: Hardware Multithreading
    """The SM's hard ceiling on resident warps."""
    return max_threads_per_sm // warp_size


# PAPER: CUDA C++ Programming Guide -- Hardware Implementation: Hardware Multithreading
def occupancy_by_registers(
    registers_per_thread: int,
    *,
    max_registers_per_sm: int = MAX_REGISTERS_PER_SM,
    max_threads_per_sm: int = MAX_THREADS_PER_SM,
    warp_size: int = WARP_SIZE,
) -> float:
    """Occupancy ceiling imposed by register pressure alone: how many threads
    the SM's fixed register file can keep resident at ``registers_per_thread``
    registers each, expressed as a fraction of the SM's max resident threads.

    This is the simplified, block-granularity-free form used in the book's
    worked numeric example (registers/thread -> resident threads -> warps ->
    occupancy); it treats threads as the atom rather than quantizing to whole
    blocks, matching how the Programming Guide first introduces the
    register/occupancy relationship before layering in block-size rounding.
    ``occupancy_by_shared_memory`` below does quantize by whole blocks, since
    shared memory is allocated per block, not per thread.
    """
    if registers_per_thread <= 0:
        raise ValueError("registers_per_thread must be positive")
    resident_threads = min(max_threads_per_sm, max_registers_per_sm // registers_per_thread)
    return resident_threads / max_threads_per_sm


# PAPER: CUDA C++ Programming Guide -- Hardware Implementation: Hardware Multithreading
def occupancy_by_shared_memory(
    shared_mem_per_block_bytes: int,
    threads_per_block: int,
    *,
    max_shared_mem_per_sm_bytes: int = MAX_SHARED_MEM_PER_SM_BYTES,
    max_threads_per_sm: int = MAX_THREADS_PER_SM,
    max_resident_blocks_per_sm: int = MAX_RESIDENT_BLOCKS_PER_SM,
) -> float:
    """Occupancy ceiling imposed by shared-memory usage alone: how many whole
    blocks of ``shared_mem_per_block_bytes`` fit in the SM's shared-memory
    budget (capped by the thread-count budget too), expressed as a fraction
    of the SM's max resident threads."""
    if shared_mem_per_block_bytes <= 0:
        blocks_by_smem = max_resident_blocks_per_sm
    else:
        blocks_by_smem = min(max_resident_blocks_per_sm, max_shared_mem_per_sm_bytes // shared_mem_per_block_bytes)
    blocks_by_threads = max_threads_per_sm // threads_per_block
    resident_blocks = min(blocks_by_smem, blocks_by_threads)
    resident_threads = resident_blocks * threads_per_block
    return resident_threads / max_threads_per_sm


# PAPER: CUDA C++ Programming Guide -- Hardware Implementation: Hardware Multithreading
def occupancy(
    registers_per_thread: int,
    threads_per_block: int,
    shared_mem_per_block_bytes: int = 0,
    **hw_limits,
) -> dict:
    """Combine the two resource gates by taking their minimum (an SM must
    respect both budgets simultaneously) and report which gate binds.

    Returns a dict with ``occupancy`` (0..1), ``limiting_factor``
    (``"registers"`` | ``"shared_memory"`` | ``"tie"``), and the two
    component occupancies for diagnostics.
    """
    reg_kwargs = {k: v for k, v in hw_limits.items() if k in ("max_registers_per_sm", "max_threads_per_sm", "warp_size")}
    smem_kwargs = {k: v for k, v in hw_limits.items() if k in ("max_shared_mem_per_sm_bytes", "max_threads_per_sm", "max_resident_blocks_per_sm")}
    occ_reg = occupancy_by_registers(registers_per_thread, **reg_kwargs)
    occ_smem = occupancy_by_shared_memory(shared_mem_per_block_bytes, threads_per_block, **smem_kwargs)
    if occ_reg < occ_smem:
        limiting = "registers"
    elif occ_smem < occ_reg:
        limiting = "shared_memory"
    else:
        limiting = "tie"
    return {
        "occupancy": min(occ_reg, occ_smem),
        "limiting_factor": limiting,
        "occupancy_by_registers": occ_reg,
        "occupancy_by_shared_memory": occ_smem,
    }
