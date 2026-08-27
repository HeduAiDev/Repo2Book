# SOURCE: vllm/v1/core/block_pool.py
# BlockPool——一个池的本体（ch13 全文已建；本章组化后**全组共用这一个池**）：
# 块数组一次预构、自由队列（null_block popleft 占 0 号、ref_cnt 不维护
# 处处特判）、get_new_blocks/touch/free_blocks 引用计数生命周期、
# get_usage（分母扣 null_block——账本的读数出口）。
# SUBTRACTED（dossier.delete 批准项的落点 + ch15 边界）：哈希表全套
#   （cached_block_hash_to_block/get_cached_block/cache_full_blocks/
#   _maybe_evict_cached_block/evict_blocks/reset_prefix_cache——链式哈希/
#   前缀命中/LRU 驱逐 → ch15；free_blocks 的哈希劈分**原样保留**：本章
#   False 支全走 append_n）；events/metrics 贯穿调用（第 9 条）；
#   KVCacheBlockCopy/migrate 相关（CoW 管线 → ch15）。
from collections.abc import Iterable

from .kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


# SOURCE: vllm/v1/core/block_pool.py:L143 BlockPool
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
    """

    # SOURCE: vllm/v1/core/block_pool.py:L162 __init__
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
    ):
        # SUBTRACTED: enable_kv_cache_events/metrics_collector 参数与
        #   kv_event_queue 字段（L167-L168、L193-L196——dossier.delete 第 9 条）。
        # SOURCE: vllm/v1/core/block_pool.py:L170-L173
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        # SOURCE: vllm/v1/core/block_pool.py:L174-L181（块数组一次预构 + 自由
        #   队列）
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # SUBTRACTED: 哈希表两件（L183-L185——ch15；enable_caching=False 支
        #   恒空表）。

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        # SOURCE: vllm/v1/core/block_pool.py:L187-L191（null_block 占 0 号）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # SUBTRACTED: get_cached_block / cache_full_blocks（L198-L290——ch15
    #   哈希命中与登记）。

    # SOURCE: vllm/v1/core/block_pool.py:L647 get_new_blocks
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L658-L659
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # SOURCE: vllm/v1/core/block_pool.py:L661-L677（caching 支的驱逐调用
        #   随哈希表删；metrics 调用随第 9 条删——ref_cnt += 1 主干逐字）
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return ret

    # SOURCE: vllm/v1/core/block_pool.py:L702 touch
    def touch(self, blocks: Iterable[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L710-L715（ref_cnt=0 且非 null
        #   → 从自由队列摘出；metrics 调用随第 9 条删）
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    # SOURCE: vllm/v1/core/block_pool.py:L719 free_blocks
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L727-L742（哈希劈分两行**原样
        #   保留**——本章 False 支全走 append_n；free 掉的满块不清哈希的
        #   语义 → ch15）
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_with_hash = []
        blocks_without_hash = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                # When caching is disabled we always append for better
                # GPU cache locality from reusing recently used blocks
                if block.block_hash is None and self.enable_caching:
                    blocks_without_hash.append(block)
                else:
                    blocks_with_hash.append(block)

        # Blocks without hash get evicted first - prepend them last to the tail
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    # SOURCE: vllm/v1/core/block_pool.py:L799 get_num_free_blocks
    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L805
        return self.free_block_queue.num_free_blocks

    # SOURCE: vllm/v1/core/block_pool.py:L807 get_usage
    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        # SOURCE: vllm/v1/core/block_pool.py:L814-L818（分母扣 null——null
        #   占位不接客）
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)
