# SPDX-License-Identifier: Apache-2.0
"""
SimpleCPUOffloadScheduler — educational variant.

REFERENCE: vllm/v1/simple_kv_offload/manager.py (742 LOC at 98661fe)

vLLM ships TWO CPU offload paths at this commit:
  v1/kv_offload/             — production v1 (pluggable policies, ARC, events,
                                multi-worker shared mmap region, 18-connector
                                composability via OffloadingSpec). Used by
                                OffloadingConnector.
  v1/simple_kv_offload/      — minimal reference impl (single-process LRU via
                                BlockPool, single mode select between lazy/eager
                                offload). Used by SimpleCPUOffloadConnector;
                                serves as pedagogical anchor + drop-in baseline.

Why TWO implementations exist (HARD GATE design decision 9):
  The production path is general-purpose (multi-rank, ARC, ghost lists,
  shared mmap, multi-tier composition). It is hard to read because
  every method handles every backend. The simple path is single-rank,
  LRU only, no ghost lists, no shared mmap — readable in one sitting.
  Production teaches us "what's possible"; simple teaches us "what's
  fundamental".
  REFERENCE: vllm/v1/simple_kv_offload/manager.py:L67 docstring.

This module implements a teaching-grade subset of SimpleCPUOffloadScheduler:
  * Lazy mode: track GPU free queue cursor; offload only when free count
    drops below `target_free`.
  * Eager mode: offload every block at scheduler step.
  * Two CUDA streams (load_stream + store_stream) — modeled here as two
    independent SingleDirectionOffloadingHandler instances.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .cpu_gpu_worker import (
    CpuGpuOffloadingHandlers,
    OffloadingWorker,
    SingleDirectionOffloadingHandler,
)
from .offload_spec import (
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    CPULoadStoreSpec,
    GPULoadStoreSpec,
)


class OffloadMode(Enum):
    """Mode used by SimpleCPUOffloadScheduler.

    LAZY  → only offload when GPU free count < target_free
    EAGER → offload every block as it becomes hashable
    """

    LAZY = "lazy"
    EAGER = "eager"


@dataclass
class TransferMeta:
    """Pair of GPU/CPU block-id arrays that the worker copies.

    REFERENCE: vllm/v1/simple_kv_offload/manager.py:L41-L44
    """

    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class LoadRequestState:
    """Per-request load tracking. Distinct from STORE state because
    a request can have at most ONE in-flight load (block_hashes are
    contiguous prefix), but multiple in-flight stores.

    REFERENCE: vllm/v1/simple_kv_offload/manager.py:L47-L52
    """

    request_id: str
    transfer_meta: TransferMeta
    load_event: Optional[int] = None
    finished: bool = False


@dataclass
class StoreRequestState:
    """Per-request store tracking (eager mode only).

    `num_stored_blocks` is a CURSOR, NOT a count: it tracks where in the
    request's block list we are. The cursor + scheduler_output let us
    compute "what is newly hashable this step".
    REFERENCE: vllm/v1/simple_kv_offload/manager.py:L55-L64
    """

    request_id: str
    block_ids: tuple[list[int], ...]
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False


class SimpleCPUOffloadScheduler:
    """Educational variant of OffloadingConnectorScheduler.

    Differences from the production v1 path:
      * Single CachePolicy (LRU via OrderedDict — no pluggable ARC).
      * Single rank (no shared mmap region; no DCP/PCP composition).
      * No `OffloadingManager.lookup` / `prepare_load` ABC indirection;
        we go straight from request → block_id pair → worker.
      * Two explicit modes (lazy/eager) controlled by a flag, vs the
        production code's protocol-based composition.

    REFERENCE: vllm/v1/simple_kv_offload/manager.py:L67-L742
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        gpu_kv_bytes_per_block: int,
        cpu_capacity_bytes: int,
        mode: OffloadMode = OffloadMode.EAGER,
        watermark_ratio: float = 1.0,
    ):
        # REFERENCE: vllm/v1/simple_kv_offload/manager.py:L70-L157
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.gpu_kv_bytes_per_block = gpu_kv_bytes_per_block
        # Derive the CPU block count from the byte ratio.
        # REFERENCE: vllm/v1/simple_kv_offload/manager.py:L173 — same scaling.
        gpu_total_bytes = num_gpu_blocks * gpu_kv_bytes_per_block
        self.num_cpu_blocks = max(
            1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes
        )
        self.mode = mode
        # SIMPLIFIED: single OrderedDict acts as the "BlockPool LRU" used
        # by the production simple variant. Production uses the v1
        # `BlockPool` data structure (Ch02) which has additional features.
        self.cpu_block_pool: OrderedDict[bytes, int] = OrderedDict()
        self.cpu_free_blocks: list[int] = list(range(self.num_cpu_blocks))

        # In LAZY mode, this is the watermark on GPU free blocks below
        # which we start offloading proactively.
        # REFERENCE: vllm/v1/simple_kv_offload/manager.py:L189-L210
        self.target_free: int = (
            int(num_gpu_blocks * watermark_ratio)
            if mode == OffloadMode.LAZY
            else 0
        )

        # Per-req state (only used by eager mode in production; we keep
        # both maps live so demos can switch modes).
        self.reqs_to_load: dict[str, LoadRequestState] = {}
        self.reqs_to_store: dict[str, StoreRequestState] = {}

        # Event counters — used by the worker to ack completion.
        # REFERENCE: vllm/v1/simple_kv_offload/manager.py:L150-L153
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

    # --- block-pool primitives ---

    def _allocate_cpu_blocks(self, n: int) -> Optional[list[int]]:
        # If we don't have n free blocks, try to evict from the LRU end.
        if len(self.cpu_free_blocks) >= n:
            return [self.cpu_free_blocks.pop(0) for _ in range(n)]
        deficit = n - len(self.cpu_free_blocks)
        evicted = self._evict_lru(deficit)
        if evicted is None:
            return None
        # `evicted` returns block_ids; combine with whatever was already free.
        out = list(self.cpu_free_blocks)
        self.cpu_free_blocks.clear()
        out.extend(evicted)
        return out

    def _evict_lru(self, n: int) -> Optional[list[int]]:
        # REFERENCE: vllm/v1/simple_kv_offload/manager.py — internal LRU eviction.
        if len(self.cpu_block_pool) < n:
            return None
        evicted_ids: list[int] = []
        # OrderedDict iteration order = insertion order = LRU oldest first
        keys_to_evict = list(self.cpu_block_pool.keys())[:n]
        for k in keys_to_evict:
            evicted_ids.append(self.cpu_block_pool.pop(k))
        return evicted_ids

    # --- public API ---

    def lookup(self, block_hash: bytes) -> Optional[int]:
        """Return CPU block id for hash (and refresh LRU position), or None."""
        # REFERENCE: vllm/v1/simple_kv_offload/manager.py — block-hash → cpu_block.
        if block_hash in self.cpu_block_pool:
            self.cpu_block_pool.move_to_end(block_hash)
            return self.cpu_block_pool[block_hash]
        return None

    def queue_store(
        self,
        request_id: str,
        block_hashes: list[bytes],
        gpu_block_ids: list[int],
    ) -> Optional[TransferMeta]:
        """Allocate CPU slots + bind hash→cpu_block; build transfer pair.

        Mirrors the eager path in `SimpleCPUOffloadScheduler` where every
        new hashable block becomes a transfer. Returns None if we cannot
        allocate enough CPU blocks (caller must skip / retry).
        """
        assert len(block_hashes) == len(gpu_block_ids)
        # Skip already-stored hashes (idempotent).
        new_hashes: list[bytes] = []
        new_gpu_ids: list[int] = []
        for h, gid in zip(block_hashes, gpu_block_ids):
            if h not in self.cpu_block_pool:
                new_hashes.append(h)
                new_gpu_ids.append(gid)
        if not new_hashes:
            return TransferMeta(gpu_block_ids=[], cpu_block_ids=[])

        cpu_ids = self._allocate_cpu_blocks(len(new_hashes))
        if cpu_ids is None:
            return None

        for h, cid in zip(new_hashes, cpu_ids):
            self.cpu_block_pool[h] = cid

        meta = TransferMeta(gpu_block_ids=new_gpu_ids, cpu_block_ids=cpu_ids)
        self.reqs_to_store[request_id] = StoreRequestState(
            request_id=request_id,
            block_ids=(new_gpu_ids,),
            num_stored_blocks=[len(new_gpu_ids)],
        )
        return meta

    def queue_load(
        self,
        request_id: str,
        block_hashes: list[bytes],
        gpu_block_ids: list[int],
    ) -> Optional[TransferMeta]:
        """Build a load transfer for a request given block hashes and the
        already-allocated GPU dst block ids.

        REFERENCE: vllm/v1/simple_kv_offload/manager.py — load path.
        """
        assert len(block_hashes) == len(gpu_block_ids)
        cpu_ids: list[int] = []
        gpu_ids_used: list[int] = []
        for h, gid in zip(block_hashes, gpu_block_ids):
            cid = self.lookup(h)
            if cid is None:
                # missed — stop at first miss (prefix lookup semantics)
                break
            cpu_ids.append(cid)
            gpu_ids_used.append(gid)

        if not cpu_ids:
            return None

        meta = TransferMeta(gpu_block_ids=gpu_ids_used, cpu_block_ids=cpu_ids)
        self.reqs_to_load[request_id] = LoadRequestState(
            request_id=request_id,
            transfer_meta=meta,
        )
        return meta

    # --- inspectors used by tests / demos ---

    def num_cpu_used(self) -> int:
        return len(self.cpu_block_pool)

    def num_cpu_free(self) -> int:
        return self.num_cpu_blocks - len(self.cpu_block_pool)

    def evict_oldest(self, n: int) -> int:
        """Test helper — pop n oldest entries; return how many were evicted."""
        ids = self._evict_lru(n) or []
        self.cpu_free_blocks.extend(ids)
        return len(ids)


def estimate_lazy_target_blocks(
    num_attention_groups: int,
    num_mamba_groups: int,
    num_sliding_window_groups: int,
    sliding_window_blocks: int,
    max_num_batched_tokens: int,
    block_size: int,
) -> int:
    """Estimate the watermark number of free GPU blocks for lazy mode.

    REFERENCE: vllm/v1/simple_kv_offload/manager.py:L189-L210
    Formula approx mirrors production heuristic:
      target = num_attention * cdiv(max_batched, block_size)
             + num_mamba * 2
             + num_sliding * sliding_window_blocks
    """
    from math import ceil

    return (
        num_attention_groups * ceil(max_num_batched_tokens / block_size)
        + num_mamba_groups * 2
        + num_sliding_window_groups * sliding_window_blocks
    )
