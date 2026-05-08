# SPDX-License-Identifier: Apache-2.0
"""
OffloadingManager — scheduler-side keyspace owner.

This module mirrors `vllm/v1/kv_offload/base.py:L110-L218` (the ABC) and
`vllm/v1/kv_offload/cpu/manager.py:L25-L200` (the concrete impl).

Why the manager runs in the scheduler (HARD GATE design decision 5):
  Eviction decisions need GLOBAL knowledge of every offloaded block —
  which req holds it, when it was last touched, whether it's loadable.
  The scheduler is the only single-threaded process with this view.
  Putting the manager in a worker would force cross-rank synchronization
  on every prepare_store call (= sched-step latency disaster).

The manager exposes 8 verbs:
  lookup        — is this key offloaded and ready?
  prepare_load  — protect blocks from eviction; return load locator
  touch         — mark recently used (LRU/ARC organize)
  complete_load — drop ref count after worker finishes
  prepare_store — allocate slots + decide evictions BEFORE store
  complete_store— mark as loadable after worker confirms upload
  take_events   — flush prom-metrics event queue
  shutdown      — release resources

The `prepare_store` returns evicted_keys EAGERLY (before the actual store)
so the scheduler can release worker-side state in the same step. Reactive
eviction would force a second round-trip. See impl-notes O15.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from typing import Optional

from .offload_spec import (
    CPULoadStoreSpec,
    LoadStoreSpec,
    OffloadKey,
    OffloadingEvent,
    PrepareStoreOutput,
    ReqContext,
)
from .policies import (
    CACHE_POLICIES,
    BlockStatus,
    CachePolicy,
)


# REFERENCE: vllm/v1/kv_offload/base.py:L110-L218 — OffloadingManager ABC
class OffloadingManager(ABC):
    """Scheduler-side keyspace owner. Tracks which blocks are offloaded
    and orchestrates evictions / ref counting / event emission.

    Subclasses are responsible for choosing how blocks are organized
    (e.g. LRU, ARC, sieve) and where they live (CPU DRAM, GPU mem, NVMe).
    The contract here is uniform across all backends.
    """

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> Optional[bool]:
        """Return True/False/None.
        True  → block is offloaded AND loadable (ref_cnt >= 0).
        False → block is not offloaded, or not yet loadable.
        None  → backend deferred lookup; scheduler should retry next step.
                (LMCache uses this for async cache-line warming.)
        """

    @abstractmethod
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        """Bump ref_cnt on each key (protects from eviction). Return a
        LoadStoreSpec that the worker uses to locate the data."""

    def touch(self, keys: Collection[OffloadKey]) -> None:
        """Mark keys as recently used. Default no-op (subclasses override)."""

    def complete_load(self, keys: Collection[OffloadKey]) -> None:
        """Drop ref_cnt on each key. Default no-op."""

    @abstractmethod
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> Optional[PrepareStoreOutput]:
        """Allocate slots + decide evictions. None means cannot satisfy
        the store request (e.g. no idle blocks to evict)."""

    def complete_store(
        self, keys: Collection[OffloadKey], success: bool = True
    ) -> None:
        """Flip blocks from not-ready (ref_cnt=-1) to loadable (ref_cnt=0).
        On failure, remove + free the block."""

    def take_events(self) -> Iterable[OffloadingEvent]:
        """Drain queued events. Default empty."""
        return ()

    def shutdown(self) -> None:
        """Release manager resources."""
        return


# REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L25-L200 — CPUOffloadingManager
class CPUOffloadingManager(OffloadingManager):
    """OffloadingManager backed by a pluggable CachePolicy (LRU or ARC).

    The manager owns SHARED logic:
      * physical block-id pool (free list + alloc cursor)
      * ref counting on ABC's PrepareStoreOutput
      * event queue for prom-metrics
      * the prepare_store / complete_store atomic skeleton

    Policy-specific organization (T1/T2 lists, ghost lists) lives in
    CachePolicy. This split lets us swap LRU↔ARC without touching the
    manager — see impl-notes O01 + O08.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        enable_events: bool = False,
    ):
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L36-L52
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: Optional[list[OffloadingEvent]] = (
            [] if enable_events else None
        )

        policy_cls = CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy {cache_policy!r}. "
                f"Supported: {list(CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)

    # --- block pool primitives ---

    def _get_num_free_blocks(self) -> int:
        # free_list size + un-allocated capacity
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L56-L57
        return (
            len(self._free_list)
            + self._num_blocks
            - self._num_allocated_blocks
        )

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        # Allocate up to (num_blocks - num_allocated_blocks) fresh ids first,
        # then pop from free_list for the remainder.
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L59-L73
        num_fresh = min(
            len(keys), self._num_blocks - self._num_allocated_blocks
        )
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused, (
            f"need {num_reused} reused blocks but free_list has "
            f"{len(self._free_list)}"
        )

        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(block_id=self._num_allocated_blocks))
            self._num_allocated_blocks += 1
        for _ in range(num_reused):
            blocks.append(BlockStatus(block_id=self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L75-L76
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L78-L83
        return CPULoadStoreSpec([blk.block_id for blk in blocks])

    # --- OffloadingManager interface ---

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> Optional[bool]:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L87-L89
        block = self._policy.get(key)
        return block is not None and block.is_ready

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L91-L103
        blocks: list[BlockStatus] = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.is_ready, f"Block {key!r} not loadable"
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    def touch(self, keys: Collection[OffloadKey]) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L105-L106
        self._policy.touch(keys)

    def complete_load(self, keys: Collection[OffloadKey]) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L108-L113
        for key in keys:
            block = self._policy.get(key)
            assert block is not None
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> Optional[PrepareStoreOutput]:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L115-L168

        # 1) drop already-stored keys (idempotent re-stores cost nothing).
        keys_to_store = [k for k in keys if self._policy.get(k) is None]
        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        # 2) compute eviction count.
        # CRITICAL: we evict BEFORE we allocate. If we cannot evict
        # enough idle blocks, we abort atomically (no state mutation).
        num_blocks_to_evict = (
            len(keys_to_store) - self._get_num_free_blocks()
        )
        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            # Don't evict any of the input keys (they are the ones we
            # are about to store; if they were already in the cache we
            # already filtered them out above; the ones we are storing
            # NOW must remain after this call).
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None  # cannot satisfy
            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        # 3) emit eviction event for prom-metrics.
        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        # 4) allocate slots + insert into policy. Each block's ref_cnt
        # remains at -1 (not yet ready) until complete_store flips it.
        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store)
        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)

        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def complete_store(
        self, keys: Collection[OffloadKey], success: bool = True
    ) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L170-L195
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    # Flip from -1 (not ready) → 0 (loadable).
                    block.ref_cnt = 0
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    # Failed upload — remove + free block id.
                    self._policy.remove(key)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        # REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L197-L200
        if self.events is not None:
            yield from self.events
            self.events.clear()

    # --- inspectors used by tests / demos ---

    def num_blocks(self) -> int:
        return self._num_blocks

    def num_offloaded(self) -> int:
        return len(self._policy)

    def policy_name(self) -> str:
        return type(self._policy).__name__
