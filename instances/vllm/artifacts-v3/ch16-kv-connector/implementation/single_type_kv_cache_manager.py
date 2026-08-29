# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py
# 每注意力类型一份块账本（ch13/14 已立）。本章切面 = **ext_comp 段**：
# allocate_external_computed_blocks（外部已算 token 的新块分配——『已分配
# 未缓存』窗口的分配半边，L291-L328）+ 零清账（records_new_block_ids/
# take_new_block_ids——站 6 跳过清零与站 10 补登记的账本）+ partial-tail
# offload 手递手队列（take_pending_partial_tail_offloads——m15）。
# FullAttentionManager.find_longest_cache_hit 的 phase 1/phase 2（本地命中
# ——get_computed_blocks_for_connector 的非混合路径直接消费）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 10 条 CoW 管线：_partial_hit_reqs/_pending_cow_copies/take_pending_
#     cow_copies/_apply_cow、get_num_blocks_to_allocate 的 partial-hit +1
#     预留与 allocate_new_blocks 的换尾重定向（L347-L357）——本章 m3 的
#     仲裁正是『免 CoW』；块内 CoW 三件套 → ch15 m13；
#   SWA/Chunked/RSWA/SinkFull/Cross/Mamba 管理器（混合命中调和 → ch15；
#     本章单 full 组 + NoPrefixCache 多组两形态）；mamba align 分配内部；
#   eagle/dcp/pcp 乘子与 use_eagle（L42、L51-L52、L76-L79、L701-L704——
#     ch33/上下文并行）；metrics 贯穿；
#   KVCacheSpecRegistry 注册表（ch14 全量切面——直配映射）。
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

import itertools

from .block_pool import BlockPool
from .kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec
from .kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    resolve_block_hashes,
)
from .math_utils import cdiv
from .request import Request


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L36 SingleTypeKVCacheManager
class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L42
    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L44 __init__
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            scheduler_block_size: The scheduling granularity (LCM of all group
                block sizes); a multiple of this manager's ``block_size``.
            needs_kv_cache_zeroing: Whether worker-side KV cache zeroing needs
                newly allocated block IDs from this manager.
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every
                block until the request finishes.
        """
        # SUBTRACTED: dcp_world_size/pcp_world_size 参数与乘子放大
        #   （L51-L52、L76-L79——单卡恒 1 烘干）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L73-L75
        self.scheduler_block_size = scheduler_block_size
        # The block size for this manager; used for actual block allocation.
        self.block_size = kv_cache_spec.block_size
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L80-L83
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        # SOURCE: vllm/v1/core/kv_transfer 面的零清类型表（L86-L92 缩到
        #   FullAttentionSpec——TQ/MLA/HiddenState 族归邻章）
        self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (
            FullAttentionSpec,
        )
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L94-L97（逻辑
        #   块表本体——_update_requests_with_invalid_blocks/交接块表都读它）
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track
        # the data for the preempted ones.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L99-L103
        self.num_cached_block: dict[str, int] = {}

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L105-L106
        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

        # SUBTRACTED: use_eagle（L108-L112——ch33）、_partial_hit_reqs/
        #   _pending_cow_copies（L114-L117——CoW 管线第 10 条归 ch15）。
        # Partial-tail offload hand-off for external KV connectors: when a
        # producer registers its last-prompt-boundary partial tail and the
        # durable boundary block is not on the append-only request block table
        # (mamba "align" CoW target), record the request, group, block, and
        # exact token boundary so a connector can offload it under the right
        # hash. Populated only by mamba "align".
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L118-L126
        #   _pending_partial_tail_offloads（m15 的队列本体）
        self._pending_partial_tail_offloads: list[
            tuple[str, int, KVCacheBlock, int]
        ] = []

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L128 _get_num_evictable_blocks
    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L130
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    # SUBTRACTED: _has_partial_local_hit（L132-L142——CoW 管线随第 10 条删：
    #   本章 m3 仲裁后要么砍掉子块尾（免 CoW）、要么保尾不走分配重定向）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L144 get_num_blocks_to_allocate
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, clamp by `num_required_blocks` by
                `_max_admission_blocks_per_request`for recycling-aware specs
                (SWA, chunked-local).

        Returns:
            The number of blocks to allocate.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L178-L191（SWA/
        #   chunked 回收型上限分支随管理器族删；cap 参数面保留）
        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L194-L200
        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L202-L213（SWA
        #   窗外跳过算术随族删后 num_skipped 恒 0——full 不跳）
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        # SUBTRACTED: partial-hit CoW 的 +1 预留（L226-L229——第 10 条）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L230
        return num_new_blocks + num_evictable_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L232 add_local_computed_blocks
    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the locally cached (prefix-hit) blocks to the request:
        1. Touch the computed blocks (paired with adding them to `req_blocks`)
           so their ref_cnt exactly tracks the referencing requests.
        1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
        2. Add the remaining computed blocks.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L253-L265
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

        # Touch the computed blocks to make sure they won't be evicted.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L267-L273
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L275-L282
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)
        # SUBTRACTED: partial-hit CoW 登记（L283-L289——第 10 条归 ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L291
    #   allocate_external_computed_blocks——ext_comp 段的分配半边（本章核心）
    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Allocate new blocks for external (KV-connector) computed tokens.

        Must run only after every group's local blocks have been touched via
        `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
        evict another group's cache-hit blocks (issue #33775).

        Args:
            request_id: The request ID.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L309-L320（SWA
        #   窗外扣减随族删后直通；<=0 早退——『已分配未缓存』窗口开在这）
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L322-L328
        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L330 allocate_new_blocks
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        # SUBTRACTED: partial-hit CoW 换尾重定向（L347-L357——第 10 条归
        #   ch15；重定向与 get_num_blocks_to_allocate 的 +1 预留成对删除，
        #   分配算术保持一致）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L359-L369
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks)
        return new_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L371 records_new_block_ids
    @property
    def records_new_block_ids(self) -> bool:
        """Whether this manager's new blocks are zeroed by the worker."""
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L373-L374
        return self._record_new_block_ids

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L376 take_new_block_ids
    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L377-L380
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    # SUBTRACTED: take_pending_cow_copies（L382-L388——第 10 条归 ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L390
    #   take_pending_partial_tail_offloads——m15 手递手 drain
    def take_pending_partial_tail_offloads(
        self,
    ) -> list[tuple[str, int, KVCacheBlock, int]]:
        """Drain producer partial-tail hand-offs.

        Entries are ``(req_id, group_id, block, boundary_tokens)``.

        Only mamba "align" populates this. The block lives off the request
        block table, so the caller must pin it until the connector has read
        it — nothing else keeps it alive once the CoW retention is released.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L401-L403
        pending = self._pending_partial_tail_offloads
        self._pending_partial_tail_offloads = []
        return pending

    # SUBTRACTED: _apply_cow（L405-L425——第 10 条归 ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L427 cache_blocks
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including the tokens that are already cached).
            retention_interval: Sparse local-checkpoint granularity. ``None``
                keeps dense checkpointing; ``0`` keeps only the latest replay
                boundary; a positive multiple of ``scheduler_block_size`` keeps
                a tail once per that-sized segment. Only SWA acts on it.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L445-L449
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L451-L466
        # Token boundaries whose reachable tail must be retained under sparse
        # retention: the replay boundary (``num_prompt - 1``, capped by
        # ``get_computed_blocks``) and any detected shared-prefix junction.
        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)

        block_mask = self.reachable_block_mask(
            start_block=num_cached_blocks,
            end_block=num_full_blocks,
            alignment_tokens=self.scheduler_block_size,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=False,  # SUBTRACTED: eagle 位（第 6 条——恒 False）
            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
        )
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
            block_mask=block_mask,
        )

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L477
        self.num_cached_block[request.request_id] = num_full_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L479 reachable_block_mask
    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
    ) -> list[bool] | None:
        """Per-block mask for ``cache_full_blocks``. ``None`` means cache
        every (non-null) block — the default for full attention.

        Subclasses with sparse hit semantics (SWA / Mamba) override this to skip
        blocks that can never serve a hit at any alignment-aligned prefix length.
        ``reachable_boundaries`` are token positions whose reachable tail must be
        retained; the base (dense) policy ignores them.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L498
        return None

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L500 pop_blocks_for_free
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        Pop the request's bookkeeping and return its blocks without yet
        returning them to the block pool. The caller is responsible for
        eventually passing the returned blocks to `block_pool.free_blocks`,
        freeing them in reverse order (so that tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L513-L517
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        return req_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L519 free
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Free blocks in reverse order so that the tail blocks are freed first.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L526-L527
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L529
    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache.

        Args:
            running_request_id: The request ID.

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L545（抽象面）

        raise NotImplementedError

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L545 find_longest_cache_hit
    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.

        Returns:
            A tuple containing cached blocks and the exact cache-hit length in
            tokens.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L545-L593（抽象
        #   面——详版 docstring 的 SWA/mamba 例子随覆写族删除收编）
        raise NotImplementedError

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L595 _remove_blocks_in_range
    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        """Free blocks in ``[first_block, last_block)`` and replace with null_block.

        Iterates backward so newly-evictable tail blocks are reached even after
        earlier blocks in the range were nulled in a prior call.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L606-L620
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L622 remove_skipped_blocks
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L643-L659（full
        #   注意力 get_num_skipped_tokens 恒 0 → 早退）
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L661 get_num_skipped_tokens
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L671-L672
        return 0

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L674 new_step_starts
    def new_step_starts(self) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L675
        return None


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L678 FullAttentionManager
class FullAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L679
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L681 find_longest_cache_hit
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L694-L699
        assert isinstance(kv_cache_spec, FullAttentionSpec), (
            "FullAttentionManager can only be used for full attention "
            "groups"
        )
        # SUBTRACTED: dcp 分片视图（L700-L704——恒 1）。
        block_size = kv_cache_spec.block_size
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L705-L711
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        # Fine-grained mode (alignment_tokens == hash_block_size <
        # block_size): resolve_block_hashes kept the raw hash-granularity
        # list so interior boundaries can be probed.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L713-L726
        fine_grained = (
            alignment_tokens < block_size and block_size % alignment_tokens == 0
        )
        if fine_grained:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            full_block_hashes: BlockHashList = BlockHashListWithBlockSize(
                block_hashes, alignment_tokens, block_size
            )
        else:
            full_block_hashes = block_hashes

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L728-L739
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        # Phase 1: longest run of cached full blocks from the start. A missing
        # block implies every later block misses too (chained hashes).
        for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
            cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached_block:
                break
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        hit_length = len(computed_blocks[0]) * block_size

        # Phase 2 (fine-grained only): extend into the first non-full block by
        # probing its interior hash boundaries high-to-low (longest hit first).
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L741-L762
        if fine_grained:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            scale_factor = block_size // alignment_tokens
            first_partial_idx = len(computed_blocks[0]) * scale_factor
            max_partial_idx = min(
                first_partial_idx + scale_factor - 1,
                max_length // alignment_tokens,
                len(block_hashes),
            )
            for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
                cached_tail = block_pool.get_cached_block(
                    block_hashes[fine_idx], kv_cache_group_ids
                )
                if not cached_tail:
                    continue
                for computed, cached in zip(computed_blocks, cached_tail):
                    computed.append(cached)
                hit_length = (fine_idx + 1) * alignment_tokens
                break

        # SUBTRACTED: eagle 丢尾（L764-L769——drop_eagle_block 恒 False 支
        #   删；参数面保留）。
        # Round down to the alignment; a no-op when fine-grained (hits land on
        # hash boundaries by construction) and when alignment_tokens ==
        # block_size. Then trim blocks past the new tail.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L770-L777
        hit_length -= hit_length % alignment_tokens
        num_blocks = cdiv(hit_length, block_size)
        for computed in computed_blocks:
            del computed[num_blocks:]
        return computed_blocks, hit_length

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L779 cache_blocks
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L785-L789
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
        hash_block_size = self.block_pool.hash_block_size
        if self.block_size == hash_block_size:
            return
        self._cache_partial_tail_block(request, num_tokens)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L791 _cache_partial_tail_block
    def _cache_partial_tail_block(
        self,
        request: Request,
        num_tokens: int,
    ) -> None:
        """Cache the prompt tail when it ends inside a cache block.

        Only the final prompt hash boundary is registered as a partial
        prefix-cache entry; intermediate hash boundaries inside the same cache
        block are intentionally skipped.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L802-L819
        hash_block_size = self.block_pool.hash_block_size
        boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size
        if boundary_tokens == 0 or boundary_tokens > num_tokens:
            return
        if boundary_tokens % self.block_size == 0:
            return

        blocks = self.req_to_blocks[request.request_id]
        block_idx = boundary_tokens // self.block_size
        if block_idx >= len(blocks):
            return
        self.block_pool.cache_partial_block(
            request=request,
            block=blocks[block_idx],
            num_tokens=boundary_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L821 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L822-L829
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


# SUBTRACTED: RSWA/SWA/ChunkedLocal/SinkFull/Cross 管理器族
#   （L832-L1252、L1747-L1833——SWA 窗外回收 → ch14/15、cross → ch13）。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1253 MambaManager
#   （最小面：find 的 fine 分支——enable_partial_hash_hits 的装配前提；
#   align 分配内部/状态块滚动/producer partial-tail 登记 → 邻章/ch15）
class MambaManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1254
    #   （细粒度哈希查找支持——fine 分支的装配位）
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1256 __init__
    def __init__(self, kv_cache_spec, block_pool: BlockPool, **kwargs) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1259-L1264
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.block_size = kv_cache_spec.block_size
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        # SUBTRACTED: num_speculative_blocks/cached_blocks_this_step 与
        #   align 内部四账（L1265-L1277——last_state_block_idx 状态块滚动/
        #   _producer_partial_tail_reqs 登记，ch15/邻章；_pending_partial_
        #   tail_offloads 的队列与 drain 在基类，本章消费端保留）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1279 find_longest_
    #   cache_hit（fine 分支逐字：自高向低探测 hash 粒度边界、命中即回填
    #   null 占位 + 状态块；粗分支右到左早停）
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1292-L1296
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1297-L1308
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        hit_length = 0

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1309-L1330
        #   fine 分支（alignment < block 且整除——enable_partial_hash_hits 开
        #   时的形态）
        block_size = kv_cache_spec.block_size
        if alignment_tokens < block_size and block_size % alignment_tokens == 0:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            hash_block_size = alignment_tokens
            scale_factor = block_size // hash_block_size
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
                num_tokens = (fine_idx + 1) * hash_block_size
                block_hash = block_hashes[fine_idx]
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    block_idx = fine_idx // scale_factor
                    for computed, cached in zip(computed_blocks, cached_block):
                        computed.extend([block_pool.null_block] * block_idx)
                        computed.append(cached)
                    hit_length = num_tokens
                    break
            return computed_blocks, hit_length

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1332-L1356
        #   粗分支（右到左早停——只要最后一个状态块）
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure
                # the prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                hit_length = (i + 1) * block_size
                break  # we just need the last match - early stopping
        return computed_blocks, hit_length

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1446
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1447-L1450
        # SUBTRACTED: mamba 专属实现——基类不可达的空账
        return 0

    # SUBTRACTED: align 分配内部四重写/_cache_partial_tail_block mamba 版/
    #   reachable_block_mask/cached_blocks_this_step 族/get_num_skipped
    #   （L1359-L1444、L1452-L1744——状态块滚动与稀疏驻留 → ch15；本章
    #   基类回退：cache_blocks 走 dense、分配走基类按块分配——impl≠pin
    #   边界见 impl-notes）。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1836 get_manager_for_
#   kv_cache_spec（注册表删——直配映射；SWA/Chunked 回收型上限注入随族删）
_MANAGER_CLASSES: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MambaSpec: MambaManager,
}


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1836 get_manager_for_kv_cache_spec
def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_in_flight_tokens: int,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """
    Get the appropriate manager for a given KVCacheSpec.

    Args:
        kv_cache_spec: The KVCacheSpec instance
        max_in_flight_tokens: The max tokens scheduled but not yet settled
        max_model_len: The maximum context length the model could serve
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass
    """
    # SUBTRACTED: KVCacheSpecRegistry 查建（L1857-L1860——ch14 全量切面）
    #   与 SWA/Chunked 上限注入（L1861-L1876）。
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1877-L1878
    manager_class = _MANAGER_CLASSES.get(type(kv_cache_spec))
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"
    )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
