# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 对应 vllm/v1/core/block_pool.py：BlockHashToBlockMap 去重映射 + BlockPool 全局块管家
# （分配/touch/free/惰性驱逐/缓存满块）。
from collections.abc import Iterable, Sequence
from typing import Any

from .kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from .request import Request


# SOURCE: vllm/v1/core/block_pool.py:L34 (class BlockHashToBlockMap)
class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        # SOURCE: vllm/v1/core/block_pool.py:L57 (BlockHashToBlockMap.__init__)
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        # SOURCE: vllm/v1/core/block_pool.py:L62 (get_one_block)
        """Gets any block with the given block hash key."""
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        # SOURCE: vllm/v1/core/block_pool.py:L75 (insert)
        """Inserts the KVCacheBlock to the cache."""
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        # SOURCE: vllm/v1/core/block_pool.py:L93 (pop)
        """Checks if block_hash exists and pop block_id from the cache."""
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contains blocks, put it back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        # SOURCE: vllm/v1/core/block_pool.py:L123 (__len__)
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        # SOURCE: vllm/v1/core/block_pool.py:L126 (_unexpected_blocks_type)
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


# SOURCE: vllm/v1/core/block_pool.py:L130 (class BlockPool)
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.
    """

    # SOURCE: vllm/v1/core/block_pool.py:L149 (BlockPool.__init__)
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
    ):
        # SUBTRACTED: enable_kv_cache_events / metrics_collector 参数与
        # kv_event_queue / metrics_collector 字段（L154-L155, L179-L182）—— KV 事件
        # 订阅与统计旁路，默认关闭，不影响分配/命中/驱逐控制流。
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # SOURCE: vllm/v1/core/block_pool.py:L184 (get_cached_block)
    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group."""
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    # SOURCE: vllm/v1/core/block_pool.py:L211 (cache_full_blocks)
    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This updates each block's hash metadata and registers it in the
        `cached_block_hash_to_block`. The block hash values are computed by the
        Request object immediately when it is created and when new tokens are
        appended."""
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        # Common case: block_size == hash_block_size.
        # SUBTRACTED: block_size != hash_block_size 的多组重算分支
        # （L244-L252，BlockHashListWithBlockSize）—— 仅当不同 KV cache group 块大小
        # 不同才触发；单组下 block_size == hash_block_size，走 common case。
        block_hashes = request.block_hashes

        new_block_hashes = block_hashes[num_cached_blocks:]
        for i, blk in enumerate(new_full_blocks):
            # SUBTRACTED: is_null 跳过（L262-L263）—— 仅滑窗/Mamba 对齐模式产生 null
            # 满块；全注意力主路径不出现。原 vllm/v1/core/block_pool.py:L259-L263。
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and add the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
        # SUBTRACTED: enable_kv_cache_events 发布 BlockStored 事件（L255-L257,
        # L273-L320）—— 外部 KV 事件订阅旁路，默认关闭，不改变缓存登记行为。

    # SOURCE: vllm/v1/core/block_pool.py:L322 (get_new_blocks)
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.
        Note that we do not check block cache in this function."""
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                # SUBTRACTED: metrics_collector.on_block_allocated（L344-L345）。
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                # SUBTRACTED: metrics_collector.on_block_allocated（L350-L351）。
        return ret

    # SOURCE: vllm/v1/core/block_pool.py:L354 (_maybe_evict_cached_block)
    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """If a block is cached in `cached_block_hash_to_block`, we reset its
        hash metadata and evict it from the cache."""
        # SUBTRACTED: metrics_collector.on_block_evicted（L365-L367）。
        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block, eviction not needed
            return False

        block.reset_hash()
        # SUBTRACTED: enable_kv_cache_events 发布 BlockRemoved（L381-L388）。
        return True

    # SOURCE: vllm/v1/core/block_pool.py:L391 (touch)
    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix."""
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            # SUBTRACTED: metrics_collector.on_block_accessed（L405-L406）。

    # SOURCE: vllm/v1/core/block_pool.py:L408 (free_blocks)
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first."""
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    # SUBTRACTED: evict_blocks（L424-L...）—— KV connector / offloading 主动驱逐接口，
    # 与本地前缀缓存主流程正交。原 vllm/v1/core/block_pool.py:L424。

    # SOURCE: vllm/v1/core/block_pool.py:L478 (get_num_free_blocks)
    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool."""
        return self.free_block_queue.num_free_blocks

    # SUBTRACTED: get_usage / reset_prefix_cache / take_events / 其余运维接口
    # —— 统计与运维旁路，非分页/命中核心控制流。原 vllm/v1/core/block_pool.py:L486+。
