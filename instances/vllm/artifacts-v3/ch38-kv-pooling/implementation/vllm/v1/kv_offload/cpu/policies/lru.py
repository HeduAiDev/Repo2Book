# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU 驱逐策略（专设 evictable 链表换 O(1) 驱逐）。"""

from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


# SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L12-L90
class LRUCachePolicy(CachePolicy):
    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L13-L19
    """
    LRU Caching policy that keeps a dedicated evictable list for fast eviction.
    A use is indicated by,
     - First time the key is added (store).
     - Load job completion
     - touch
    """

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L21-L25
    def __init__(self, cache_capacity: int):
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L21-L25
        super().__init__(cache_capacity)
        # Blocks with ref_cnt 0 (not participating in any loads/stores) ordered in LRU
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L27-L29
    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L28-L29
        return self.blocks.get(key)

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L31-L35
    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L32-L35
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L37-L40
    @override
    def remove(self, key: OffloadKey) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L38-L40
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L42-L48
    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L43-L48
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)
            # active blocks are untouched as they are non-evictable now. They
            # will eventually reach the end of evictable_blocks when they finish.

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L50-L53
    @override
    def clear(self) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L51-L53
        self.evictable_blocks.clear()
        self.blocks.clear()

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L55-L78
    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L56-L78
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, _ in self.evictable_blocks.items():
            if key in protected:
                continue

            block = self.blocks[key]
            assert block.ref_cnt == 0
            candidates.append((key, block))
            if len(candidates) == n:
                break

        if len(candidates) < n:
            return None
        for key, _ in candidates:
            del self.evictable_blocks[key]
            del self.blocks[key]
        return candidates

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L80-L85
    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L81-L85
        # blocks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self.evictable_blocks[key] = None

    # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L87-L90
    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/policies/lru.py:L88-L90
        # key must have been in the evictable list.
        del self.evictable_blocks[key]
