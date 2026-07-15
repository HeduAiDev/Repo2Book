# ch02 implementation notes — GPU execution model (primer, paper-grounded reference implementation)

This chapter's implementation is **not** a subtract-only slice of a code
repository — it's the primer-chapter exemption (see `.claude/agents/implementer.md`
"primer 原理章分支"). GPU execution model has no host academic paper; the
grounding is vendor architecture documentation (NVIDIA CUDA C++ Programming
Guide / Best Practices Guide) plus the title/abstract-level Triton MAPL 2019
thesis (`DOI:10.1145/3315508.3329973`), per `dossier.json.paper_grounding_note`.

Every module is a small, dependency-free, pure-Python reference model of one
mechanism from `dossier.json.mechanisms` — sized so `explainer` can run it
and produce a teachable trace, not a full simulator.

## Source map

| Reference implementation | Grounding (# PAPER) | Anchor / code spine cross-reference | Notes |
|---|---|---|---|
| `spmd_tile.cdiv` | MAPL 2019 (DOI:10.1145/3315508.3329973), title/abstract: SPMD blocked-program launch | `python/triton/__init__.py:L59-L60` | identical `(x+y-1)//y`; grid size |
| `spmd_tile.tile_offsets` | MAPL 2019, title/abstract: tile as unit of program | `python/tutorials/01-vector-add.py:L42-L43` (`block_start`, `offsets`) | contiguous offsets = coalescing's first cause |
| `spmd_tile.boundary_mask` | CUDA C++ Programming Guide — Hardware Implementation: SIMT Architecture (predicated execution) | `python/tutorials/01-vector-add.py:L45-L52` | mirrors `mask = offsets < n_elements` |
| `spmd_tile.spmd_grid` | MAPL 2019, title/abstract: SPMD blocked-program launch | `python/tutorials/01-vector-add.py:L68-L73` | grid = `cdiv(n_elements, BLOCK_SIZE)` program instances |
| `simt_hierarchy.num_warps_per_block` / `partition_into_warps` | CUDA C++ Programming Guide — Thread Hierarchy / Hardware Implementation: SIMT Architecture | `python/triton/language/core.py:L1148-L1163`, `semantic.py:L28-L31` (`program_id` → `%ctaid`) | makes the compiler/hardware-managed warp layer visible |
| `memory_hierarchy.LATENCY_CYCLES` / `latency_cycles` / `slower_than` | CUDA C++ Programming Guide — Memory Hierarchy / Device Memory Accesses | `python/tutorials/01-vector-add.py:L48-L52` (load/store touch this ladder) | Ampere-class order-of-magnitude cycle counts, architecture-dependent |
| `coalescing.count_transactions` / `warp_offsets_bytes` / `strided_offsets_bytes` | CUDA C++ Best Practices Guide — Coalesced Access to Global Memory | `python/tutorials/01-vector-add.py:L43-L52`, `core.py:L1184-L1200` (`arange`) | contiguous offsets → 1 transaction; strided → 32 |
| `occupancy.occupancy_by_registers` / `occupancy_by_shared_memory` / `occupancy` | CUDA C++ Programming Guide — Hardware Implementation: Hardware Multithreading | `python/tutorials/01-vector-add.py:L32-L33,L73` (`BLOCK_SIZE`) | reproduces the chapter's 32/64/128-regs-per-thread → 100%/50%/25% worked example |
| `register_spill.spilled_registers` / `effective_access_cycles` | CUDA C++ Programming Guide — Device Memory Accesses: Local Memory | `python/tutorials/01-vector-add.py:L48-L52` | local memory is physically DRAM; spill collapses latency to the global tier |

## What is deliberately not implemented

- `constexpr` (dossier `m09`) and `mask`'s PTX cache-operator plumbing
  (`load`/`store`'s `cache_modifier`) are **not** given their own reference
  module — the dossier's own note says these are supporting/carried-forward
  concepts ("承 ch01，本章不展开"), not new mechanisms this chapter needs to
  demonstrate numerically. They're covered in prose via the embedded source
  excerpts, not re-implemented.
- No CUDA/PTX compilation or execution — everything here is a pure-Python
  arithmetic/bookkeeping model of what the hardware and compiler do (grid
  sizing, warp partitioning, transaction counting, occupancy arithmetic,
  spill bookkeeping). It is explicitly *not* claiming to be a cycle-accurate
  simulator; the goal is a runnable, steppable model of the three judging
  tools this chapter hands the reader (occupancy / coalescing / spill), not
  a GPU simulator.
- Hardware constants (`MAX_REGISTERS_PER_SM=65536`, `MAX_THREADS_PER_SM=2048`,
  `MAX_SHARED_MEM_PER_SM_BYTES=164*1024`) are Ampere-class order-of-magnitude
  figures used consistently with the chapter's own worked numeric example
  (dossier `theory` field) — not a specific GPU's exact `cudaDeviceProp`
  values, and documented as such in each module's docstring.

## Test-to-mechanism map

| Test file | Mechanism(s) | Book worked example reproduced |
|---|---|---|
| `test_spmd_tile.py` | m02, m07, m08 | N=98432, BLOCK=1024 → grid=97, tail masks 896/1024 lanes |
| `test_simt_hierarchy.py` | m01 | BLOCK_SIZE=1024 → 32 warps/block |
| `test_memory_hierarchy.py` | m03 | ladder ordering + ≥100x register↔global gap |
| `test_coalescing.py` | m04 | contiguous warp = 1 transaction; strided = 32 |
| `test_occupancy.py` | m05 | 32/64/128 regs/thread → 100%/50%/25%; shared-memory-bound case |
| `test_register_spill.py` | m06 | spill count + latency collapse to global tier |

Run: `python3 -m pytest tests/` (no CUDA/Triton runtime needed — pure Python,
host-runnable).
