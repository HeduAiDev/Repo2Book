# SPDX-License-Identifier: Apache-2.0
"""
Offload Spec — the data contract layer.

This module mirrors `vllm/v1/kv_offload/base.py` (398 LOC at commit 98661fe).
It defines the abstract types that every concrete offload backend implements:

    OffloadKey                    ← packed (block_hash || group_idx) bytes
    LoadStoreSpec  (ABC)          ← worker-side locator
    BlockIDsLoadStoreSpec         ← block-id-based concrete locator
    GPULoadStoreSpec              ← GPU-side locator
    CPULoadStoreSpec              ← CPU-side locator
    PrepareStoreOutput            ← (keys_to_store, store_spec, evicted_keys)
    OffloadingEvent               ← scheduler event for prom-metrics
    ReqContext                    ← per-request context (kv_transfer_params)
    OffloadingSpec  (ABC)         ← factory contract for managers + handlers
    CPUOffloadingSpec             ← concrete CPU DRAM tier spec

Why these abstractions exist (HARD GATE design decision 1):
  vLLM ships ~18 connector backends (LMCache, Mooncake, Nixl, hf3fs, p2p, ...).
  Without a uniform spec/manager/handler split, every backend would duplicate
  ref-counting + eviction + event emission. The split lets a backend swap
  ONLY the transport (handler) and storage tier (spec) while reusing
  manager-side keyspace logic (ref_cnt, events, allocation).

Why OffloadKey is bytes (HARD GATE design decision 2):
  vLLM hashes prefixes 1000+ times per second. Each hash is a 32-byte digest.
  Storing keys as `tuple[bytes, int]` would cost a tuple-allocation per lookup
  (~120 ns) plus dict hashing of two objects (~400 ns total). Packing
  block_hash (28-32 B) + group_idx (4 B big-endian uint) into one bytes object
  lets the dict use the bytes hash directly (~80 ns) — 5x speedup.
  REFERENCE: vllm/v1/kv_offload/base.py:L24-L44 (NewType + make_offload_key)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NewType


# REFERENCE: vllm/v1/kv_offload/base.py:L27 — NewType("OffloadKey", bytes)
# An OffloadKey is the packed bytes of a block hash followed by a 4-byte
# big-endian group index. Use the helpers below to construct/decompose.
OffloadKey = NewType("OffloadKey", bytes)


def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    """Pack (block_hash, group_idx) into a single bytes object.

    REFERENCE: vllm/v1/kv_offload/base.py:L32-L34
    """
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


def get_offload_block_hash(key: OffloadKey) -> bytes:
    """Extract the block_hash prefix from an OffloadKey.

    REFERENCE: vllm/v1/kv_offload/base.py:L37-L39
    """
    return bytes(key[:-4])


def get_offload_group_idx(key: OffloadKey) -> int:
    """Extract the 4-byte big-endian group_idx suffix.

    REFERENCE: vllm/v1/kv_offload/base.py:L42-L44
    """
    return int.from_bytes(bytes(key[-4:]), "big", signed=False)


# REFERENCE: vllm/v1/kv_offload/base.py:L47-L49 — ReqContext dataclass
@dataclass
class ReqContext:
    """Per-request context. Today carries kv_transfer_params (used by
    LMCache/Mooncake to attach do_remote_decode flags etc.); kept as a
    dataclass for forward-compat (more fields will land here)."""

    kv_transfer_params: dict[str, Any] | None = None


# REFERENCE: vllm/v1/kv_offload/base.py:L52-L66 — LoadStoreSpec ABC
class LoadStoreSpec(ABC):
    """Abstract metadata describing where a block lives on a particular medium.

    Each concrete subclass exposes a `medium()` static method returning
    a unique string (e.g. "GPU", "CPU"). The OffloadingWorker dispatches
    transfers by (src_medium, dst_medium) tuple.
    """

    @staticmethod
    @abstractmethod
    def medium() -> str:
        """Return the medium identifier (used in transfer-type dispatch)."""
        raise NotImplementedError


# REFERENCE: vllm/v1/kv_offload/base.py:L68-L72 — PrepareStoreOutput
@dataclass
class PrepareStoreOutput:
    """Returned by OffloadingManager.prepare_store.

    Three fields, each load-bearing:
      * keys_to_store  — keys that were not already present (skip duplicates)
      * store_spec     — worker-side locator for the destination
      * evicted_keys   — proactive eviction (computed BEFORE the actual store
                         so the scheduler can release worker-side state)

    The evicted_keys field is the key invariant of "proactive eviction":
    if eviction were reactive (computed at allocate time), the scheduler
    would not know which downstream blocks were freed and would risk
    reading stale data. See impl-notes O15 + Trap C.
    """

    keys_to_store: list[OffloadKey]
    store_spec: LoadStoreSpec
    evicted_keys: list[OffloadKey]


# REFERENCE: vllm/v1/kv_offload/base.py:L75-L80 — OffloadingEvent
@dataclass
class OffloadingEvent:
    """Scheduler-side event for prometheus metrics + KV cache event stream.

    `removed=True`  : these keys were just evicted from the medium
    `removed=False` : these keys were just stored (became loadable)
    """

    keys: list[OffloadKey]
    medium: str
    removed: bool


# REFERENCE: vllm/v1/kv_offload/base.py:L219-L228 — BlockIDsLoadStoreSpec
class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """LoadStoreSpec that locates blocks by integer block IDs into a
    medium-specific physical buffer.

    Used by both GPU and CPU specs. The `block_ids` are stored as np.int64
    rather than a Python list in production because the worker passes them
    straight into the C-level swap_blocks_batch call.

    SIMPLIFIED: this educational impl uses a Python list to avoid the numpy
    dependency. The semantics (iterable of int IDs) are identical.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids: list[int] = list(block_ids)

    def __repr__(self) -> str:
        return repr(self.block_ids)


# REFERENCE: vllm/v1/kv_offload/base.py:L231-L266 — GPULoadStoreSpec
class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """GPU-side locator. Carries group_sizes + block_indices for HMA hybrids.

    SIMPLIFIED: we still preserve the group_sizes and block_indices invariants
    (sum(group_sizes) == len(block_ids); len(block_indices) == len(group_sizes))
    because Ch24 layerwise-connectors composes on top of these, so we must
    not strip them away. Real vLLM uses these to skip part of the first CPU
    block when (offloaded_block_size > gpu_block_size). REFERENCE: same range.
    """

    def __init__(
        self,
        block_ids: list[int],
        group_sizes: list[int] | None = None,
        block_indices: list[int] | None = None,
    ):
        super().__init__(block_ids)
        if group_sizes is None:
            group_sizes = [len(block_ids)]
        if block_indices is None:
            block_indices = [0]
        assert sum(group_sizes) == len(block_ids), (
            f"sum(group_sizes)={sum(group_sizes)} != len(block_ids)={len(block_ids)}"
        )
        assert len(block_indices) == len(group_sizes)
        self.group_sizes: list[int] = list(group_sizes)
        self.block_indices: list[int] = list(block_indices)

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """CPU-side locator. Symmetric to GPULoadStoreSpec, no group/sliding info.

    REFERENCE: vllm/v1/kv_offload/cpu/common.py — `CPULoadStoreSpec(block_ids)`.
    The original keeps it deliberately minimal because CPU blocks are
    always full-size (= offloaded_block_size); only the GPU side may
    need partial-block math.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"


# REFERENCE: vllm/v1/kv_offload/base.py:L269-L316 — Canonical KV cache types
@dataclass
class CanonicalKVCacheTensor:
    """A KV cache tensor canonicalized to (num_blocks, page_size_bytes)."""

    tensor: Any  # torch.Tensor in production; Any here so demo runs without torch
    page_size_bytes: int


@dataclass
class CanonicalKVCacheRef:
    """Reference into the list of CanonicalKVCacheTensor entries."""

    tensor_idx: int
    page_size_bytes: int


@dataclass
class CanonicalKVCaches:
    """Canonicalized KV caches: tensors + per-group references.

    The (num_blocks, page_size_bytes) shape lets the C-level swap_blocks_batch
    treat any backend layout uniformly. FlashAttention's (2, num_blocks, ...)
    layout is split into 2 entries (K and V). FlashInfer's (num_blocks, ...)
    becomes 1 entry. See vllm/v1/kv_offload/base.py:L300+ comment.
    """

    tensors: list[CanonicalKVCacheTensor]
    group_data_refs: list[list[CanonicalKVCacheRef]]


# REFERENCE: vllm/v1/kv_offload/base.py:L319-L398 — OffloadingSpec ABC
class OffloadingSpec(ABC):
    """Spec for an offloading connector.

    `get_manager()`  → returns an OffloadingManager (scheduler-side keyspace)
    `get_handlers()` → yields (src_type, dst_type, handler) tuples (worker-side)

    The split is deliberate: the manager runs in the scheduler process
    (single-thread, dict-based, no GPU); the handlers run in the worker
    process (per-rank, GPU-attached, CUDA streams). They communicate ONLY
    via LoadStoreSpec objects. This is what makes connector composition
    via MultiConnector possible — see Ch22-Ch25 for PD-disagg variations.
    """

    def __init__(
        self,
        hash_block_size: int,
        gpu_block_size: tuple[int, ...],
        block_size_factor: int = 1,
        cpu_bytes_to_use: int = 0,
    ):
        # SIMPLIFIED: we accept primitive args here instead of (vllm_config,
        # kv_cache_config). The original constructs gpu_block_size from
        # `kv_cache_config.kv_cache_groups[i].kv_cache_spec.block_size *
        #  context_parallel_factor`. Logic is identical.
        # REFERENCE: vllm/v1/kv_offload/base.py:L322-L374
        self.hash_block_size = hash_block_size
        self.gpu_block_size: tuple[int, ...] = tuple(gpu_block_size)
        self.block_size_factor: int = block_size_factor
        self.cpu_bytes_to_use: int = cpu_bytes_to_use

        for block_size in self.gpu_block_size:
            assert block_size % self.hash_block_size == 0, (
                f"gpu_block_size={block_size} not divisible by "
                f"hash_block_size={self.hash_block_size}. Hybrid models "
                f"(Mamba+Attention) need --enable-prefix-caching to align."
            )

    @abstractmethod
    def get_manager(self) -> Any:
        """Return the scheduler-side OffloadingManager."""
        raise NotImplementedError

    @abstractmethod
    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], Any]]:
        """Yield (src_type, dst_type, handler) tuples for the worker."""
        raise NotImplementedError


class CPUOffloadingSpec(OffloadingSpec):
    """Concrete spec for the CPU DRAM tier.

    REFERENCE: vllm/v1/kv_offload/cpu/spec.py:L22-L102

    Computes `num_blocks` from the requested `cpu_bytes_to_use` and the
    per-block byte size derived from the GPU KV cache config. Then exposes
    a CPUOffloadingManager backed by the configured eviction policy
    ("lru" or "arc").
    """

    def __init__(
        self,
        hash_block_size: int,
        gpu_block_size: tuple[int, ...],
        kv_bytes_per_block: int,
        cpu_bytes_to_use: int,
        block_size_factor: int = 1,
        eviction_policy: str = "lru",
        store_threshold: int = 0,
    ):
        super().__init__(
            hash_block_size=hash_block_size,
            gpu_block_size=gpu_block_size,
            block_size_factor=block_size_factor,
            cpu_bytes_to_use=cpu_bytes_to_use,
        )

        # REFERENCE: vllm/v1/kv_offload/cpu/spec.py:L33-L47 — num_blocks calc
        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        self.num_blocks: int = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        self.eviction_policy: str = eviction_policy
        self.store_threshold: int = store_threshold
        self._manager: Any | None = None
        self._handlers: Any | None = None

    def get_manager(self) -> Any:
        # Local import to avoid circulars at module-load time.
        from .offload_manager import CPUOffloadingManager
        from .reuse_manager import FilterReusedOffloadingManager

        if self._manager is None:
            mgr = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                enable_events=True,
            )
            # REFERENCE: vllm/v1/kv_offload/cpu/spec.py:L70-L82
            if self.store_threshold >= 2:
                mgr = FilterReusedOffloadingManager(
                    backing=mgr,
                    store_threshold=self.store_threshold,
                )
            self._manager = mgr
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], Any]]:
        from .cpu_gpu_worker import CpuGpuOffloadingHandlers

        if self._handlers is None:
            self._handlers = CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
            )
        # REFERENCE: vllm/v1/kv_offload/cpu/spec.py:L100-L102 — yield both directions
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


# Trap-aware constants used by the demos (see Trap A in impl-notes).
# These are the per-tier numbers from the brief; they are deterministic
# (formula-driven), NOT measured. See K17 OR-skip discipline.
HBM3_BANDWIDTH_GB_PER_S: float = 3000.0  # ~3 TB/s on H100
DDR5_BANDWIDTH_GB_PER_S: float = 96.0
PCIE_GEN5_BANDWIDTH_GB_PER_S: float = 64.0
NVME_GEN5_BANDWIDTH_GB_PER_S: float = 14.0  # sequential SSD bandwidth
HBM_CAPACITY_GB: float = 80.0  # H100 80 GB SKU
DDR5_CAPACITY_GB: float = 512.0  # typical 2-socket server
NVME_CAPACITY_GB: float = 4000.0  # 4 TB consumer SSD
KV_BLOCK_BYTES: int = 16 * 1024 * 1024  # 16 MB conservative per brief §1
DECODE_STEP_MS: float = 50.0  # 70B model on H100 ~ 50 ms / step
PREFILL_STEP_MS: float = 200.0
PCIE_OVERHEAD_ALPHA_US: float = 10.0  # transfer overhead intercept
PCIE_OVERHEAD_BETA_US_PER_BYTE: float = 1.5e-5
