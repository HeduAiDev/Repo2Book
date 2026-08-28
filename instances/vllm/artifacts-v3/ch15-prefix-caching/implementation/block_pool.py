# SOURCE: vllm/v1/core/block_pool.py
# **前缀缓存登记与驱逐的管家**（m2/m6/m8/m9/m10/m20）：
# BlockHashToBlockMap——平面哈希表（{hash+group_id(bytes) → block | {block_id:
# block}}；故意不去重 NOTE #1 保块表 append-only、union 省 GC NOTE #2）；
# BlockPool 前缀面——cache_full_blocks（满块+block_mask 入表、partial→full
# 晋升摘旧插新）、cache_partial_block（块内细粒度条目、不分配新块）、
# _insert_block_hash/_remove_cached_block_hashes（主哈希+反向索引双向维护）、
# move_block_hashes（CoW 后重指）、get_new_blocks+_maybe_evict_cached_block
# （惰性驱逐：复用才摘）、touch（命中救回）、free_blocks（劈分挂回）、
# reset_prefix_cache（RLHF 失效面）。
# 池构造/get_new_blocks 的免缓存支/free_blocks 的 False 支 ch13 已建——本章
# 打开哈希面（caching=True 主路径）。
# SUBTRACTED（dossier.delete 批准项的落点）：
#   第 1 条 kv events 全套：enable_kv_cache_events/kv_event_queue/
#     _build_block_stored_event/emit_cached_block_events/_emit_block_removed_
#     events/take_events 及 cache_full_blocks/cache_partial_block/reset 的
#     事件段（观测旁路，默认 False）；
#   第 2 条 metrics_collector（构造参数与 on_block_* 调用——纯统计旁路）；
#   第 5 条 connector 面：evict_blocks（KV connector 的按号驱逐）、
#     get_partial_block_parent_hash_and_start 的事件段（L559-L569 随事件删）。
from collections.abc import Iterable, Sequence
import logging

from .kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
    resolve_block_hashes,
)
from .request import Request

logger = logging.getLogger(__name__)  # LOGGER SEAM：init_logger 同构账位


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
        # SOURCE: vllm/v1/core/block_pool.py:L57-L59（平面 dict 本体——
        #   没有 radix 树、没有节点对象）
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    # SOURCE: vllm/v1/core/block_pool.py:L61 get_one_block
    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L65-L72（重复块时任取一块）
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
        # SOURCE: vllm/v1/core/block_pool.py:L92-L104（单块→dict 的退化路径：
        #   同键第二块起合并成 dict——不去重的实现面）
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
        # SOURCE: vllm/v1/core/block_pool.py:L110-L134
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
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
    def _unexpected_blocks_type(self, blocks: object) -> None:
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

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    # SOURCE: vllm/v1/core/block_pool.py:L162 __init__
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector=None,
    ):
        # SUBTRACTED: enable_kv_cache_events/metrics_collector 两观测参数的
        #   账面使用（第 1/2 条；签名保留默认值——装配面不破）
        # SOURCE: vllm/v1/core/block_pool.py:L170-L191
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
        # SOURCE: vllm/v1/core/block_pool.py:L184-L185（平面表 + 反向索引：
        #   部分条目时代的别名账本）
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        # SOURCE: vllm/v1/core/block_pool.py:L187-L191（null_block 占 0 号）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # SOURCE: vllm/v1/core/block_pool.py:L198 get_cached_block
    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
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
        # SOURCE: vllm/v1/core/block_pool.py:L212-L223（逐 group 打包键查
        #   map——任一 group miss 整体 miss：多组一致才命中）
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
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L259-L267（幂等闸 + 哈希视图）
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        new_block_hashes = block_hashes[num_cached_blocks:]
        # SUBTRACTED: new_hashes 事件收集变量（L268-L270——第 1 条观测旁路）
        # SOURCE: vllm/v1/core/block_pool.py:L271-L299 核心环（事件行 L292/
        #   L298-L299 随第 1 条删）
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
                removed_hashes = self._remove_cached_block_hashes(blk)
                # SUBTRACTED: _emit_block_removed_events(removed_hashes)
                #   （L292——第 1 条）
            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )

        # SUBTRACTED: kv_event_queue 发布段（L301-L342——第 1 条观测旁路：
        #   parent_block_hash/extra_keys_list 组装与 BlockStored 事件）。

    # SUBTRACTED: _build_block_stored_event（L344-L371）/emit_cached_block_
    #   events（L373-L443）——第 1 条 kv events 全套（供网关/外部订阅的
    #   观测旁路，默认 False）。

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

        Args:
            request: Request whose token IDs and block hashes define the
                partial entry.
            block: Existing cache block to make reachable from the partial
                prefix boundary.
            num_tokens: Prefix length represented by the partial entry. It
                must be a positive multiple of ``self.hash_block_size`` and
                cannot exceed the request's computed block hashes.
            kv_cache_group_id: KV cache group that owns the partial entry.
            block_size: Cache block size for the owning group. The partial
                entry hash itself is always the prefix-chain hash at
                ``num_tokens``; ``block_size`` is used to assert that the
                entry is partial within the owning cache block.

        Returns:
            The hash key with group ID if a partial entry can be registered;
            otherwise ``None`` for null blocks.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L484-L512 核心（断言组 L487-L489
        #   说明前提：block_size>hash_block_size 且边界不落整块；事件行
        #   L507 随第 1 条删）
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
            removed_hashes = self._remove_cached_block_hashes(block)
            # SUBTRACTED: _emit_block_removed_events(removed_hashes)
            #   （L507——第 1 条）
        self._insert_block_hash(
            block_hash_with_group_id,
            block,
            num_tokens=num_hash_blocks * self.hash_block_size,
        )
        # SUBTRACTED: kv_event_queue 发布段（L513-L543——第 1 条观测旁路；
        #   _get_partial_block_parent_hash_and_start（L559-L569）只服务它，
        #   一并删）
        return block_hash_with_group_id

    # SOURCE: vllm/v1/core/block_pool.py:L546 _get_partial_block_hash
    def _get_partial_block_hash(
        self,
        request: Request,
        num_tokens: int,
    ) -> BlockHash:
        # SOURCE: vllm/v1/core/block_pool.py:L551-L557（哈希=该边界的前缀链
        #   哈希：block_hashes[num_tokens/hash_bs − 1]——每个 hash_block_size
        #   哈希已链住整条前缀）
        assert num_tokens % self.hash_block_size == 0
        num_hash_blocks = num_tokens // self.hash_block_size
        assert 0 < num_hash_blocks <= len(request.block_hashes)

        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        return request.block_hashes[num_hash_blocks - 1]

    # SOURCE: vllm/v1/core/block_pool.py:L571 _remove_cached_block_hashes
    def _remove_cached_block_hashes(
        self,
        block: KVCacheBlock,
    ) -> list[BlockHashWithGroupId]:
        # SOURCE: vllm/v1/core/block_pool.py:L575-L590（摘全部别名：主哈希 +
        #   反向索引——摘干净才不留悬空键）
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

    # SUBTRACTED: _emit_block_removed_events（L592-L605——第 1 条 kv events）。

    # SOURCE: vllm/v1/core/block_pool.py:L607 _insert_block_hash
    def _insert_block_hash(
        self,
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int | None,
    ) -> None:
        # SOURCE: vllm/v1/core/block_pool.py:L613-L627（一票主、多票别名：
        #   块已有主哈希时新条目进反向索引）
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

    # SOURCE: vllm/v1/core/block_pool.py:L629 move_block_hashes
    def move_block_hashes(
        self,
        src_block: KVCacheBlock,
        dst_block: KVCacheBlock,
    ) -> None:
        """Re-point ``src_block``'s prefix-cache entries to ``dst_block``.

        Used when the request owning ``src_block`` keeps writing into it
        : the prefix cache holds a private copy (``dst_block``)
        under the same hashes instead. Entries stay live; no events emitted.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L640-L645（CoW 后哈希条目重指：
        #   条目活着不摘，请求侧块表 append-only 由它兜住）
        assert dst_block.block_hash is None
        assert dst_block.block_id not in self.cached_block_hashes_by_block
        num_tokens = src_block.block_hash_num_tokens
        for block_hash in self._remove_cached_block_hashes(src_block):
            # `num_tokens` only applies to the first (primary) insertion.
            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)

    # SOURCE: vllm/v1/core/block_pool.py:L647 get_new_blocks
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L658-L677（metrics 调用行
        #   L669-L670/L675-L676 随第 2 条删）
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
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
        # SOURCE: vllm/v1/core/block_pool.py:L690-L700（驱逐是惰性隐式的：
        #   块被复用才摘哈希，不发生在 free 时；metrics 行 L691-L692、
        #   事件行 L699 随第 1/2 条删）
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
        # SOURCE: vllm/v1/core/block_pool.py:L710-L717（救回本体：ref_cnt==0
        #   在 free list 当驱逐候选 → O(1) remove 出队；metrics 行删）
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
        # SOURCE: vllm/v1/core/block_pool.py:L727-L738 劈分（#42656）
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
        # SOURCE: vllm/v1/core/block_pool.py:L740-L742（无哈希块 prepend 队头
        #   先驱逐、有哈希块 append LRU 尾——LRU 不变量二）
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    # SUBTRACTED: evict_blocks（L744-L761——KV connector 的按号驱逐面，
    #   第 5 条 connector → ch16）。

    # SOURCE: vllm/v1/core/block_pool.py:L763 reset_prefix_cache
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        # SOURCE: vllm/v1/core/block_pool.py:L772-L797（要求全部块空闲才清；
        #   metrics/events 段随第 1/2 条删）
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block.clear()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        return True

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
        # SOURCE: vllm/v1/core/block_pool.py:L814-L818
        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    # SUBTRACTED: take_events（L820-L830——第 1 条 kv events 观测旁路）。
