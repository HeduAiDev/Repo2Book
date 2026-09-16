# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 块池的「块视图」：connector 从这里拿**未入哈希表的块**作落地缓冲。

# SOURCE: vllm/v1/core/kv_cache_manager.py:L60-L115（KVCacheBlocks）
# SUBTRACTED: KVCacheManager 主体（分配/回收/前缀缓存哈希/驱逐/CoW，L117-L900）——
#   那是 ch13/ch15/ch16 的面；本章只需要 `KVCacheBlocks` 这一层「请求的块表」。
"""

from typing import Any


# SOURCE: vllm/v1/core/kv_cache_manager.py:L33-L115
class KVCacheBlocks:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L34-L35
    def __init__(self, blocks: tuple[list[Any], ...]):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L34-L35
        self.blocks = blocks

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L98-L108
    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L110-L114
    def new_empty(self) -> "KVCacheBlocks":
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


__all__ = ["KVCacheBlocks"]
