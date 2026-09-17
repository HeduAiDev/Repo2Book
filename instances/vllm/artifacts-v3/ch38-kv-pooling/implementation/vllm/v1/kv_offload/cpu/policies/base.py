# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CachePolicy 基类与 BlockStatus（可插拔驱逐策略的契约面）。"""

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext


# SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L10-L33
class BlockStatus(ctypes.Structure):
    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L10-L20
    """
    Offloading status for a single block of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    block_id - index of the physical CPU buffer slot.
    """

    _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L22-L26
    def __init__(self, block_id: int):
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L22-L26
        super().__init__()
        # initialize block as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1
        self.block_id = block_id

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L28-L33
    @property
    def is_ready(self) -> bool:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L28-L33
        """
        Returns whether the block is ready to be read.
        """
        return self.ref_cnt >= 0


# SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L36-L98
class CachePolicy(ABC):
    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L37-L44
    """
    Encapsulates both block organization (data structures) and replacement
    decisions (which block to evict). LRU and ARC differ in both dimensions —
    ARC's ghost lists and target_t1_size live at the intersection of storage
    and eviction, so they cannot be separated cleanly.
    """

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L44-L45
    def __init__(self, cache_capacity: int) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L44-L45
        self.cache_capacity = cache_capacity

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L47-L49
    @abstractmethod
    def get(self, key: OffloadKey) -> BlockStatus | None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L48-L49
        """Find block in data structures. Returns None if not present."""

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L51-L53
    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L52-L53
        """Add a newly allocated block. For ARC: also removes from ghost lists."""

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L55-L57
    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L56-L57
        """Remove a block (used to clean up after a failed store)."""

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L59-L67
    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L59-L67
        """
        Mark blocks as recently used.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context for the request touching these blocks.
        """

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L69-L82
    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L70-L82
        """
        Evict exactly n blocks, skipping any in protected.

        Returns a list of (key, block) for the evicted blocks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L84-L90
    @abstractmethod
    def clear(self) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L84-L90
        """
        Remove ALL blocks regardless of ref_cnt.

        Ghost lists and adaptive state are also reset.
        """

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L92-L93
    def mark_evictable(self, key: OffloadKey) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L92-L93
        """Called when a block's ref_cnt transitions to 0."""
        return

    # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L95-L98
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/base.py:L96-L98
        """Called when a block's ref_cnt transitions from 0."""
        return
