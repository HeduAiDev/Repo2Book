"""Reference model -- SPMD tile launch: grid sizing (cdiv), per-program tile
offsets, and the boundary mask.

Faithful to Triton's own vector-add tutorial and its underlying primitives
(the chapter's code spine): ``python/tutorials/01-vector-add.py:L27-L76``
(``add_kernel`` / ``add``), ``python/triton/__init__.py:L59-L60`` (``cdiv``).

PAPER: Triton -- "Triton: An Intermediate Language and Compiler for Tiled
Neural Network Computations", MAPL 2019 (DOI:10.1145/3315508.3329973) --
title/abstract-level thesis only: the *tile* (a statically-shaped
multi-dimensional sub-array) is the unit of both the program and its IR; a
Triton kernel author writes one program per tile (SPMD) and leaves the
mapping onto warps/lanes/shared memory to the compiler. No section numbers
from this paper are cited -- only the title/abstract thesis, per the
chapter's paper-grounding note (no invented section numbers).

Grid/thread-hierarchy vocabulary otherwise follows the NVIDIA CUDA C++
Programming Guide, "Thread Hierarchy"
(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
"""
from __future__ import annotations

from typing import Sequence


def cdiv(x: int, y: int) -> int:
    # PAPER: MAPL 2019 (DOI:10.1145/3315508.3329973) -- title/abstract: SPMD blocked-program launch
    """Ceiling division: how many ``y``-sized tiles are needed to cover ``x``
    elements. This is the SPMD launch grid's size -- how many program
    instances get dispatched (mirrors ``python/triton/__init__.py``'s
    ``cdiv``, which computes the same ``(x + y - 1) // y``)."""
    return (x + y - 1) // y


def tile_offsets(program_id: int, block_size: int) -> list:
    # PAPER: MAPL 2019 (DOI:10.1145/3315508.3329973) -- title/abstract: tile as the unit of the program
    """The contiguous logical offsets one SPMD program instance
    (``program_id``) is responsible for:
    ``[program_id*block_size, (program_id+1)*block_size)``. Consecutive
    lanes within the tile get consecutive offsets -- the source of coalesced
    memory access (see ``coalescing.py``). Mirrors
    ``block_start = pid * BLOCK_SIZE; offsets = block_start +
    tl.arange(0, BLOCK_SIZE)``."""
    block_start = program_id * block_size
    return [block_start + i for i in range(block_size)]


def boundary_mask(offsets: Sequence[int], n_elements: int) -> list:
    # PAPER: CUDA C++ Programming Guide -- "Hardware Implementation: SIMT Architecture" (predicated execution)
    """Which of a tile's offsets are actually in bounds -- Triton's
    ``mask=offsets < n_elements`` guard against the ragged last tile when
    ``n_elements`` is not a multiple of ``block_size``. Lanes whose mask is
    False perform no memory access (predicated execution, not a branch)."""
    return [offset < n_elements for offset in offsets]


def spmd_grid(n_elements: int, block_size: int) -> list:
    # PAPER: MAPL 2019 (DOI:10.1145/3315508.3329973) -- title/abstract: SPMD blocked-program launch
    """The full launch grid: ``cdiv(n_elements, block_size)`` program
    instances, each identified by a ``program_id`` in ``[0, grid_size)`` --
    the SPMD analogue of a CUDA launch grid's block count. Mirrors
    ``grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )``."""
    return list(range(cdiv(n_elements, block_size)))
