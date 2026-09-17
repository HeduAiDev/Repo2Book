# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 块视图：connector 从这里拿请求的块表（update_state_after_alloc 的入参）。

# SOURCE: vllm/v1/core/kv_cache_manager.py:L33-L127（KVCacheBlocks）
# SUBTRACTED: KVCacheManager 主体（分配/回收/前缀缓存哈希/驱逐/CoW）——
#   ch13/ch15/ch16 的面；本章只需要 `KVCacheBlocks` 这层「请求的块表」。
"""

from collections.abc import Sequence
from typing import Any


# SOURCE: vllm/v1/core/kv_cache_manager.py:L33-L127
class KVCacheBlocks:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L33-L60
    """The allocation result of KVCacheManager（组 × 块 的二维视图）。"""

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L47-L60
    def __init__(self, blocks: tuple[Sequence[Any], ...]):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L47-L60
        self.blocks = blocks

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L62-L69
    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L62-L69
        """Adds two KVCacheBlocks instances."""
        from itertools import chain

        return KVCacheBlocks(
            tuple(
                list(chain(blk1, blk2)) for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L71-L98
    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L82-L98
        """Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L100-L115
    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L106-L115
        """Get block_ids of unhashed blocks（skip padding blocks）."""
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L123
    def new_empty(self) -> "KVCacheBlocks":
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L123
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


__all__ = ["KVCacheBlocks"]
