# SPDX-License-Identifier: Apache-2.0
"""
Eviction policies — LRU + ARC (Megiddo-Modha 2003).

This module mirrors `vllm/v1/kv_offload/cpu/policies/` (3 files at 98661fe):
    base.py  (76 LOC)  — BlockStatus + CachePolicy ABC
    lru.py   (46 LOC)  — single-OrderedDict LRU
    arc.py   (156 LOC) — T1 + T2 + B1 + B2 ghost lists with adaptive split

Outline-vs-source reframe (Trap B):
  The outline says §12.2 should walk LRU / LFU / attention-score-based.
  vLLM at 98661fe ships LRU + ARC. There is NO `lfu.py`, NO attention-score
  policy file. ARC IS the production sophisticated alternative; the chapter
  walks LRU + ARC honestly. See impl-notes O02.

Why ARC instead of LFU (HARD GATE design decision 3):
  Pure LFU stores per-key counters that grow unbounded — under steady state
  a frequently-accessed block accumulates a counter so high it cannot be
  evicted even after access stops. ARC sidesteps this by using *position*
  in T2 (frequent list) instead of a counter; eviction is from the LRU end
  of T2, not the lowest-counter block. ARC is also self-tuning: it learns
  the right LRU/LFU split from miss-stream signal (B1 vs B2 ghost hits).
  REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L17-L46 (docstring).

Why ref_cnt = -1 means "not ready" (HARD GATE design decision 4):
  When prepare_store inserts a block, the data has NOT been copied yet —
  the worker is still uploading. Setting ref_cnt = 0 would mark the block
  as evictable by lookup() (which checks `block.ref_cnt >= 0`). Using -1
  as a sentinel lets prepare_store insert + complete_store flip to 0 on
  success, atomically transitioning the block from "reserved" to "loadable".
  REFERENCE: vllm/v1/kv_offload/cpu/policies/base.py:L21-L33 (W10 wisdom).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from .offload_spec import OffloadKey


# REFERENCE: vllm/v1/kv_offload/cpu/policies/base.py:L10-L33 — BlockStatus
# Original uses ctypes.Structure for compactness (16 bytes per block in C
# layout vs ~56 bytes for a Python class). At 1 M blocks that is 40 MB
# per scheduler — not negligible. We use a dataclass for educational
# clarity since 1 M blocks is not the demo's regime.
@dataclass
class BlockStatus:
    """Per-block status. Holds (ref_cnt, block_id).

    ref_cnt = -1  → block is reserved but not yet loadable (data still copying)
    ref_cnt = 0   → block is idle (eligible for eviction)
    ref_cnt > 0   → block is being read (protected from eviction)

    The is_ready property returns (ref_cnt >= 0) — a convenience used by
    `OffloadingManager.lookup`. See impl-notes O04 + W10.
    """

    block_id: int
    ref_cnt: int = -1  # SENTINEL: -1 means "not ready"

    @property
    def is_ready(self) -> bool:
        return self.ref_cnt >= 0


class CachePolicy(ABC):
    """Encapsulates BOTH organization and eviction.

    The split is unusual — most cache libraries separate "what's in the cache"
    from "which to evict". vLLM intentionally fuses them because ARC's
    ghost lists (B1, B2) and the adaptive target_t1_size live at the
    intersection: a B1 hit during touch() bumps target_t1_size, which biases
    later evict() decisions. You CANNOT separate organization from eviction
    cleanly in ARC.
    REFERENCE: vllm/v1/kv_offload/cpu/policies/base.py:L36-L77 (docstring).
    """

    @abstractmethod
    def __init__(self, cache_capacity: int) -> None:
        ...

    @abstractmethod
    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        """Find block; None if absent."""

    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        """Add a fresh block. ARC also strips key from B1/B2 ghost lists."""

    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        """Remove a block (used on failed store)."""

    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        """Mark blocks as recently used. ARC also processes ghost-list hits."""

    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> Optional[list[tuple[OffloadKey, BlockStatus]]]:
        """Atomically evict exactly n blocks. None if n cannot be satisfied
        (e.g. fewer than n idle non-protected blocks)."""

    def __len__(self) -> int:
        # Convenience for tests; default counts via .get() probe is wrong
        # (subclasses override).
        raise NotImplementedError


class LRUCachePolicy(CachePolicy):
    """LRU policy backed by a single OrderedDict.

    Invariant: insertion / move-to-end / popitem(last=False) are all O(1).
    REFERENCE: vllm/v1/kv_offload/cpu/policies/lru.py:L10-L46
    """

    def __init__(self, cache_capacity: int):
        # cache_capacity is unused by LRU; accepted for uniform constructor
        self.cache_capacity = cache_capacity
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/lru.py:L17-L18
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/lru.py:L20-L21
        self.blocks[key] = block

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        # IMPORTANT: iterate in REVERSE so that the LAST key in `keys` ends up
        # at the MRU position (end of OrderedDict). The scheduler passes keys
        # in chronological order; the last block touched is the most recent.
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/lru.py:L26-L29
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> Optional[list[tuple[OffloadKey, BlockStatus]]]:
        # Scan from LRU end; pick first n idle (ref_cnt == 0) non-protected.
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/lru.py:L31-L46
        if n == 0:
            return []
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((key, block))
                if len(candidates) == n:
                    break
        if len(candidates) < n:
            return None  # cannot satisfy — atomic abort, no state changes
        for key, _ in candidates:
            del self.blocks[key]
        return candidates

    def __len__(self) -> int:
        return len(self.blocks)


class ARCCachePolicy(CachePolicy):
    """Adaptive Replacement Cache (Megiddo & Modha, 2003).

    Four lists:
        T1  — recent: blocks accessed once, ordered LRU→MRU
        T2  — frequent: blocks accessed more than once
        B1  — GHOST list of recently-evicted T1 keys (no data, just key)
        B2  — GHOST list of recently-evicted T2 keys

    Adaptive parameter: target_t1_size, in [0, cache_capacity].

    Promotion rules (in touch()):
      key in T1   → move to T2  (one→multiple promotion)
      key in T2   → move to T2 MRU end
      key in B1   → target_t1_size += max(1, |B2|/|B1|)   (recency wins)
      key in B2   → target_t1_size -= max(1, |B1|/|B2|)   (frequency wins)

    Eviction rules (in evict()):
      |T1| >= target_t1_size → evict from T1 LRU end → push to B1
      else                   → evict from T2 LRU end → push to B2
      Then trim ghost lists to cache_capacity.

    Why this beats LRU on real LLM workloads: prefix-cache hits are
    skewed (HOT prefixes hit 100x; COLD ones never). Pure LRU sees the
    HOT block once, decides "recent", and evicts it. ARC's T2 explicitly
    represents "I have seen this twice → keep" so HOT blocks survive
    longer at the cost of one extra access.
    REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L10-L156
    """

    def __init__(self, cache_capacity: int):
        self.cache_capacity: int = cache_capacity
        self.target_t1_size: float = 0.0
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L48-L55
        self.t1: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.t2: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.b1: OrderedDict[OffloadKey, None] = OrderedDict()  # ghost
        self.b2: OrderedDict[OffloadKey, None] = OrderedDict()  # ghost

    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L57-L58
        return self.t1.get(key) or self.t2.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        # New blocks always go to T1; B1/B2 are stripped (we just promoted
        # the key from "evicted" back to "live").
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L60-L63
        self.t1[key] = block
        self.b1.pop(key, None)
        self.b2.pop(key, None)

    def remove(self, key: OffloadKey) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L65-L67
        if self.t1.pop(key, None) is None:
            self.t2.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L69-L95
        for key in reversed(list(keys)):
            if key in self.t1:
                block = self.t1.pop(key)
                if not block.is_ready:
                    # block was just inserted by prepare_store and not yet
                    # loaded — it is being "touched" because the scheduler
                    # is reading the same prefix again. Keep in T1 (haven't
                    # truly been used twice yet).
                    self.t1[key] = block
                else:
                    # Promotion: T1 → T2 (block now seen at least twice).
                    self.t2[key] = block
            elif key in self.t2:
                self.t2.move_to_end(key)
            elif key in self.b1:
                # Ghost hit on B1 → recency wins → grow target_t1_size.
                # The delta = max(1, |B2|/|B1|) makes the adaptation
                # symmetric: when B1 is small relative to B2, a B1 hit
                # is "rare" and so worth more.
                delta = max(1.0, len(self.b2) / len(self.b1))
                self.target_t1_size = min(
                    self.target_t1_size + delta, float(self.cache_capacity)
                )
                self.b1.move_to_end(key)
            elif key in self.b2:
                # Ghost hit on B2 → frequency wins → shrink target_t1_size.
                delta = max(1.0, len(self.b1) / len(self.b2))
                self.target_t1_size = max(self.target_t1_size - delta, 0.0)
                self.b2.move_to_end(key)
            # otherwise: key absent from all 4 lists, ignore.

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> Optional[list[tuple[OffloadKey, BlockStatus]]]:
        # REFERENCE: vllm/v1/kv_offload/cpu/policies/arc.py:L97-L156
        if n == 0:
            return []

        # PHASE 1: dry-run select n candidates.
        # The dry-run is needed because picking from T1 reduces |T1|, which
        # can flip the |T1| >= target_t1_size predicate for the next pick.
        # We must simulate this WITHOUT mutating the actual data structures
        # (so a None return = no state change = atomic).
        candidates: list[tuple[OffloadKey, BlockStatus, bool]] = []  # (k, blk, from_t1)
        already_selected: set[OffloadKey] = set()
        virtual_t1_size = len(self.t1)

        for _ in range(n):
            chosen: Optional[tuple[OffloadKey, BlockStatus, bool]] = None

            if virtual_t1_size >= int(self.target_t1_size):
                # Try T1 first (recency partition is overweight).
                for key, block in self.t1.items():
                    if (
                        block.ref_cnt == 0
                        and key not in protected
                        and key not in already_selected
                    ):
                        chosen = (key, block, True)
                        virtual_t1_size -= 1
                        break

            if chosen is None:
                # Fall back to T2 (or T1 was already at target).
                for key, block in self.t2.items():
                    if (
                        block.ref_cnt == 0
                        and key not in protected
                        and key not in already_selected
                    ):
                        chosen = (key, block, False)
                        break
                if chosen is None:
                    return None  # cannot satisfy n evictions

            candidates.append(chosen)
            already_selected.add(chosen[0])

        # PHASE 2: apply evictions; push evicted keys to ghost lists.
        result: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block, from_t1 in candidates:
            if from_t1:
                del self.t1[key]
                self.b1[key] = None
            else:
                del self.t2[key]
                self.b2[key] = None
            result.append((key, block))

        # PHASE 3: trim ghost lists to cache_capacity (bounded memory).
        for ghost in (self.b1, self.b2):
            while len(ghost) > self.cache_capacity:
                ghost.popitem(last=False)

        return result

    def __len__(self) -> int:
        return len(self.t1) + len(self.t2)


# Lookup table — used by CPUOffloadingManager to instantiate by string.
# REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L19-L22
CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}
