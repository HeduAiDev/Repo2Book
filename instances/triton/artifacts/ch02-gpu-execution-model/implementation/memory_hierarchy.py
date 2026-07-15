"""Reference model -- GPU memory-hierarchy latency ladder.

No academic paper backs the GPU execution model itself; the facts here come
from vendor architecture documentation:
  - NVIDIA CUDA C++ Programming Guide, "Memory Hierarchy" and
    "Device Memory Accesses"
    (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

The absolute cycle counts vary by GPU generation; the numbers below are the
Ampere-class order of magnitude used throughout the book's occupancy/spill
worked examples (register ~1 cycle, shared memory ~20-30, L2 ~200, global
HBM/DRAM ~400-800). What matters for the reasoning in this chapter is the
*ladder* -- each level roughly an order of magnitude slower/bigger than the
one above it -- not the exact constant, which is architecture-dependent.
"""
from __future__ import annotations

from typing import Tuple

LEVEL_ORDER = ["register", "shared", "l2", "global"]

# PAPER: CUDA C++ Programming Guide -- "Memory Hierarchy" / "Device Memory Accesses"
# Ampere-class order-of-magnitude latencies in core clock cycles (low, high), architecture-dependent.
LATENCY_CYCLES = {
    "register": (1, 1),      # ~1 cycle
    "shared": (20, 30),      # ~20-30 cycles (SMEM)
    "l2": (200, 200),        # ~200 cycles
    "global": (400, 800),    # ~400-800 cycles (HBM/DRAM)
}


def latency_cycles(level: str) -> Tuple[int, int]:
    # PAPER: CUDA C++ Programming Guide -- "Memory Hierarchy"
    """Return the (low, high) order-of-magnitude cycle range for one level of
    the memory hierarchy. Raises on an unknown level name."""
    if level not in LATENCY_CYCLES:
        raise ValueError(f"unknown memory level: {level!r} (expected one of {LEVEL_ORDER})")
    return LATENCY_CYCLES[level]


def slower_than(level_a: str, level_b: str) -> bool:
    # PAPER: CUDA C++ Programming Guide -- "Memory Hierarchy" (ladder ordering)
    """True if ``level_a`` sits below (slower than) ``level_b`` in the
    hierarchy."""
    return LEVEL_ORDER.index(level_a) > LEVEL_ORDER.index(level_b)
