# SOURCE: vllm/v1/core/block_pool.py
# 池（ch13/15 已立、ch16 消费的本地缓存底座）+ 本章三处开口：
# touch（partial-tail 交接的钉住原语 L869）、free_blocks（延迟归还的
# 逆序释放原语）、evict_blocks（失败块的缓存逐出 L744-L761——sync 加载
# 命中过哈希表的坏块只有逐出才能被安全复用）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 3 条观测旁路：kv_event_queue/_build_block_stored_event/
#     emit_cached_block_events/take_events、maybe_convert_block_hash、
#     metrics_collector 全部调用行；
#   reset_prefix_cache（L763-L797——ch15 m20）；
#   move_block_hashes（L629-L645——CoW 拷贝重指，第 10 条归 ch15）；
#   get_usage（调度器水位面消费——本章不触，账位减法）。
import logging
from collections.abc import Iterable, Sequence

from .kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    FreeKVCacheBlockQueue,
    make_block_hash_with_group_id,
    resolve_block_hashes,
)
from .request import Request

logger = logging.getLogger(__name__)  # LOGGER SEAM：vllm.logger.init_logger → stdlib


# SOURCE: vllm/v1/core/block_pool.py:L33 BlockHashToBlockMap
class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    # SOURCE: vllm/v1/core/block_pool.py:L56 __init__
    def __init__(self):
        # SOURCE: vllm/v1/core/block_pool.py:L57-L59
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    # SOURCE: vllm/v1/core/block_pool.py:L61 get_one_block
    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given hash hash key.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L65-L72
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    # SOURCE: vllm/v1/core/block_pool.py:L74 contain
    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        """
        Checks whether the key maps to the given block ID.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L78-L86
        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        if isinstance(blocks, dict):
            return block_id in blocks
        self._unexpected_blocks_type(blocks)
        return False

    # SOURCE: vllm/v1/core/block_pool.py:L88 insert
    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        # SOURCE: vllm/v1/core/block_pool.py:L92-L104
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

    # SOURCE: vllm/v1/core/block_pool.py:L106 pop
    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        # SOURCE: vllm/v1/core/block_pool.py:L110-L133
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
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    # SOURCE: vllm/v1/core/block_pool.py:L136 __len__
    def __len__(self) -> int:
        # SOURCE: vllm/v1/core/block_pool.py:L137
        return len(self._cache)

    # SOURCE: vllm/v1/core/block_pool.py:L139 _unexpected_blocks_type
    def _unexpected_blocks_type(self, blocks) -> None:
        # SOURCE: vllm/v1/core/block_pool.py:L140
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


# SOURCE: vllm/v1/core/block_pool.py:L143 BlockPool
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.
    """

    # SOURCE: vllm/v1/core/block_pool.py:L162 __init__
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        # SUBTRACTED: enable_kv_cache_events/metrics_collector 参数与账位
        #   （第 3 条观测旁路）。
    ):
        # SOURCE: vllm/v1/core/block_pool.py:L170-L171
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        # SOURCE: vllm/v1/core/block_pool.py:L173-L175
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        # SOURCE: vllm/v1/core/block_pool.py:L176-L178
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        # SOURCE: vllm/v1/core/block_pool.py:L180-L182
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        # SOURCE: vllm/v1/core/block_pool.py:L184-L187
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # SOURCE: vllm/v1/core/block_pool.py:L198 get_cached_block
    def get_cached_block(
        self, block_hash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L210-L223
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

    # SOURCE: vllm/v1/core/block_pool.py:L225 cache_full_blocks
    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks).
        """
        # SOURCE: vllm/v1/core/block_pool.py:L259-L267
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        new_block_hashes = block_hashes[num_cached_blocks:]
        # SUBTRACTED: new_hashes 事件账与 L298-L299 事件行、L301-L342 事件
        #   发布段（第 3 条观测旁路）。
        # SOURCE: vllm/v1/core/block_pool.py:L271-L297 核心环（null/mask 跳过、
        #   partial→full 晋升、insert）
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            block_hash = new_block_hashes[i]
            num_hash_tokens = (num_cached_blocks + i + 1) * block_size

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            if blk.block_hash is not None:
                # The only valid case where a "new full block" already has a
                # hash is partial->full promotion of the same cache block.
                assert (
                    blk.block_hash_num_tokens is not None
                    and blk.block_hash_num_tokens < num_hash_tokens
                )
                # SUBTRACTED: removed_hashes 事件行（L291-L292——第 3 条）。
                self._remove_cached_block_hashes(blk)
            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )

    # SOURCE: vllm/v1/core/block_pool.py:L445 cache_partial_block
    def cache_partial_block(
        self,
        request: Request,
        block: KVCacheBlock,
        num_tokens: int,
        kv_cache_group_id: int,
        block_size: int,
    ) -> BlockHashWithGroupId | None:
        """Register a partial prefix-cache entry for an existing block.

        Prefix-cache keys normally identify full cache blocks. A partial entry
        makes an existing cache block reachable from a fine-grained prefix
        boundary inside that block without allocating or copying a new
        ``KVCacheBlock``.

        The partial entry is lookup metadata owned by ``block``. If ``block``
        has no primary hash, the key becomes its primary hash. If the block
        already has a primary hash, the partial entry is tracked in
        ``cached_block_hashes_by_block`` so eviction, reset, and promotion can
        remove every hash key that points to the block.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L484-L512 核心段（null 早退、
        #   粒度断言、already_cached 判定、旧更短条目摘除、insert）
        if block.is_null:
            return None

        assert block_size > self.hash_block_size
        assert block_size % self.hash_block_size == 0
        assert num_tokens % block_size != 0
        block_hash = self._get_partial_block_hash(request, num_tokens)
        num_hash_blocks = num_tokens // self.hash_block_size
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        already_cached = block.block_hash == block_hash_with_group_id or (
            self.cached_block_hash_to_block.contain(
                block_hash_with_group_id, block.block_id
            )
        )
        if (
            not already_cached
            and block.block_hash is not None
            and block.block_hash_num_tokens is not None
            and block.block_hash_num_tokens < num_hash_blocks * self.hash_block_size
        ):
            # SUBTRACTED: 事件行（L506-L507——第 3 条）。
            self._remove_cached_block_hashes(block)
        self._insert_block_hash(
            block_hash_with_group_id,
            block,
            num_tokens=num_hash_blocks * self.hash_block_size,
        )
        # SUBTRACTED: L513-L543 事件发布段（第 3 条观测旁路）。
        # SOURCE: vllm/v1/core/block_pool.py:L544
        return block_hash_with_group_id

    # SOURCE: vllm/v1/core/block_pool.py:L546 _get_partial_block_hash
    def _get_partial_block_hash(
        self,
        request: Request,
        num_tokens: int,
    ):
        # SOURCE: vllm/v1/core/block_pool.py:L555-L562（链尾即前缀指纹：
        #   部分条目的哈希 = num_tokens 边界上的链式哈希）
        num_hash_blocks = num_tokens // self.hash_block_size
        assert num_hash_blocks >= 1
        assert len(request.block_hashes) >= num_hash_blocks
        return request.block_hashes[num_hash_blocks - 1]

    # SOURCE: vllm/v1/core/block_pool.py:L571 _remove_cached_block_hashes
    def _remove_cached_block_hashes(
        self,
        block: KVCacheBlock,
    ):
        # SOURCE: vllm/v1/core/block_pool.py:L575-L590（主哈希+反向索引
        #   双向清账、reset_hash）
        block_hashes: list[BlockHashWithGroupId] = []
        if block.block_hash is not None:
            block_hashes.append(block.block_hash)
        block_hashes.extend(self.cached_block_hashes_by_block.pop(block.block_id, ()))
        if not block_hashes:
            return []

        removed_hashes: list[BlockHashWithGroupId] = []
        for block_hash in block_hashes:
            if (
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ):
                removed_hashes.append(block_hash)
        block.reset_hash()
        return removed_hashes

    # SUBTRACTED: _emit_block_removed_events（L592-L605——第 3 条观测旁路）。

    # SOURCE: vllm/v1/core/block_pool.py:L607 _insert_block_hash
    def _insert_block_hash(
        self,
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int | None,
    ) -> None:
        # SOURCE: vllm/v1/core/block_pool.py:L613-L627（单条目 union 省 GC；
        #   已有主哈希 → 反向索引追加）
        if block.block_hash == block_hash_with_group_id:
            return

        if self.cached_block_hash_to_block.contain(
            block_hash_with_group_id, block.block_id
        ):
            return

        if block.block_hash is None:
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
                block_hash_with_group_id
            )
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)

    # SUBTRACTED: move_block_hashes（L629-L645——CoW 拷贝重指，第 10 条归 ch15）。

    # SOURCE: vllm/v1/core/block_pool.py:L647 get_new_blocks
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L657-L659
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # SOURCE: vllm/v1/core/block_pool.py:L661
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        # SOURCE: vllm/v1/core/block_pool.py:L663-L676（caching 开：惰性驱逐
        #   +引用计数；metrics 行删）
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    # SOURCE: vllm/v1/core/block_pool.py:L679 _maybe_evict_cached_block
    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L691-L701（metrics 行删；free
        #   不清哈希、复用才摘——惰性驱逐）
        evicted_hashes = self._remove_cached_block_hashes(block)
        if not evicted_hashes:
            # The block doesn't have hash, eviction is not needed
            return False
        return True

    # SOURCE: vllm/v1/core/block_pool.py:L702 touch
    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L710-L717（ref_cnt=0 → 出 free
        #   queue；+1——partial-tail 交接的钉住走同一原语）
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
        # SOURCE: vllm/v1/core/block_pool.py:L727-L742（劈分：无哈希块 prepend
        #   队头先驱逐、有哈希块 append LRU 尾；缓存关恒 prepend——本章
        #   NoPrefixCache 支的归还序）
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

    # SOURCE: vllm/v1/core/block_pool.py:L744 evict_blocks
    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L753-L761（失败块逐出——sync
        #   加载命中过哈希表的坏块安全复用前提）
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    # SUBTRACTED: reset_prefix_cache（L763-L797——ch15 m20 切面）。

    # SOURCE: vllm/v1/core/block_pool.py:L799 get_num_free_blocks
    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L805
        return self.free_block_queue.num_free_blocks

    # SUBTRACTED: get_usage / take_events（L807-L830——水位面/观测面）。
