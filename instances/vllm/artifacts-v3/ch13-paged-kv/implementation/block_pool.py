# SOURCE: vllm/v1/core/block_pool.py
# BlockPool——全局块管家（m2/m4 主角）：blocks 数组 + 侵入式自由队列 +
# null_block；原语 get_new_blocks（取块+ref_cnt+1）/ touch（+1 出队）/
# free_blocks（-1 归零入队）；get_num_free_blocks/get_usage 供容量检查与观测。
# 本章精简版跑 enable_caching=False（正交开关；哈希链语义 → ch15 精简版）：
# get_new_blocks 走免摘哈希支、free_blocks 劈分跳过、get_new_blocks 超额抛错。
# SUBTRACTED: 哈希侧全链（dossier.delete 第 3 条）：BlockHashToBlockMap、
#   cached_block_hash_to_block/cached_block_hashes_by_block、get_cached_block、
#   cache_full_blocks、cache_partial_block、_get_partial_block_hash(_and_start)、
#   _remove_cached_block_hashes、_insert_block_hash、move_block_hashes、
#   _maybe_evict_cached_block（→ ch15）；KV cache events 全套（第 1 条：
#   enable_kv_cache_events/kv_event_queue/_build_block_stored_event/
#   emit_cached_block_events/_emit_block_removed_events/take_events）；
#   metrics_collector 全部调用（第 2 条）；evict_blocks（第 7 条）、
#   reset_prefix_cache（第 11 条）。
from collections.abc import Iterable, Sequence

from .kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


# SOURCE: vllm/v1/core/block_pool.py:L143 BlockPool
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their hash.

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
        # SUBTRACTED: enable_kv_cache_events / metrics_collector 参数与字段
        #   （L167-L168、L193-L196——观测旁路，dossier.delete 第 1/2 条）。
        # SOURCE: vllm/v1/core/block_pool.py:L170-L173
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        # SOURCE: vllm/v1/core/block_pool.py:L174-L177（对象数组一次预构）
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        # SOURCE: vllm/v1/core/block_pool.py:L178-L181
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # SUBTRACTED: 前缀缓存查找表 cached_block_hash_to_block /
        #   cached_block_hashes_by_block（L183-L185——第 3 条，→ ch15；本章当
        #   它空着不动）。

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        # SOURCE: vllm/v1/core/block_pool.py:L187-L191（null_block 占 block_id=0）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # SOURCE: vllm/v1/core/block_pool.py:L647 get_new_blocks
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L658-L659（容量护栏）
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # SOURCE: vllm/v1/core/block_pool.py:L661
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        # SOURCE: vllm/v1/core/block_pool.py:L664-L677（caching 开时先摘旧哈希
        # ——本章 False 支跳过；metrics 调用删除）
        if self.enable_caching:
            for block in ret:
                # SUBTRACTED: self._maybe_evict_cached_block(block)（L666——
                #   取走被缓存块时摘哈希，→ ch15；本章哈希表恒空、无哈希可摘）
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    # SUBTRACTED: _maybe_evict_cached_block（L679-L700）——哈希表摘条目 +
    #   reset_hash（→ ch15，dossier.delete 第 3 条）。

    # SOURCE: vllm/v1/core/block_pool.py:L702 touch
    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L710-L715（+1 且救回驱逐候选；
        #   场景 = ch15 前缀命中，本章讲语义）
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
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        # SOURCE: vllm/v1/core/block_pool.py:L727-L742
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
        # SOURCE: vllm/v1/core/block_pool.py:L740-L742（无哈希块 prepend_n 到
        #   队头先驱逐、有哈希块 append_n 进 LRU 尾 = ch15 的 LRU 双不变量；
        #   本章 False 支全走 append_n）
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
        # SOURCE: vllm/v1/core/block_pool.py:L814-L818（减 1 记 null 块——
        # vLLM 运行日志 "GPU KV cache usage" 的出处）
        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)
