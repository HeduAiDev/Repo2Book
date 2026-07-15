"""Reference model -- register spill: registers beyond the compiler's/
hardware's per-thread budget move to *local memory*, which despite its name
is backed by the same DRAM as global memory (with an L1/L2 cache reprieve).

PAPER: NVIDIA CUDA C++ Programming Guide, "Device Memory Accesses" / "Local
Memory" (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html):
local memory resides in device (DRAM) memory, so it has the same high
latency and low bandwidth as global memory accesses, subject to the same
requirements for memory coalescing.
"""
from __future__ import annotations

from memory_hierarchy import LATENCY_CYCLES


def spilled_registers(registers_needed_per_thread: int, register_budget_per_thread: int) -> int:
    # PAPER: CUDA C++ Programming Guide -- "Device Memory Accesses: Local Memory"
    """How many of the per-thread registers a kernel *wants* don't fit in the
    budget the compiler/hardware actually grants it, and therefore spill."""
    if registers_needed_per_thread <= register_budget_per_thread:
        return 0
    return registers_needed_per_thread - register_budget_per_thread


def effective_access_cycles(registers_needed_per_thread: int, register_budget_per_thread: int) -> dict:
    # PAPER: CUDA C++ Programming Guide -- "Device Memory Accesses: Local Memory"
    """Split a kernel's per-thread register accesses into the ones that stay
    in real registers (fast) and the ones that spill to local memory (DRAM
    latency) -- the two-sided cost that trades off against occupancy:
    shedding registers to raise occupancy (see ``occupancy.py``) is only a
    win if it doesn't create spills whose per-access latency collapses back
    to the global-memory tier.
    """
    spilled = spilled_registers(registers_needed_per_thread, register_budget_per_thread)
    resident = registers_needed_per_thread - spilled
    reg_lo, reg_hi = LATENCY_CYCLES["register"]
    spill_lo, spill_hi = LATENCY_CYCLES["global"]
    return {
        "resident_registers": resident,
        "spilled_registers": spilled,
        "resident_access_cycles": (reg_lo, reg_hi),
        "spilled_access_cycles": (spill_lo, spill_hi) if spilled > 0 else (0, 0),
        "spill_latency_multiplier": (
            ((spill_lo + spill_hi) / 2) / ((reg_lo + reg_hi) / 2) if spilled > 0 else 1.0
        ),
    }
