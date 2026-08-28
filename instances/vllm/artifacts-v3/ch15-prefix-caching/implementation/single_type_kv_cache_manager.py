# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py
# **各注意力类型块管理器**（m4/m5/m6/m13/m16 的宿主）：
# SingleTypeKVCacheManager ABC——基类账本（req_to_blocks/num_cached_block/
# get_num_blocks_to_allocate/allocate_new_blocks 的 CoW 换尾/_apply_cow/
# cache_blocks 的 reachable_boundaries 组装/free 逆序/pop_blocks_for_free
# 分离）+ FullAttentionManager——命中查找主算法（phase 1 沿链 miss 即停 +
# phase 2 块内细粒度自高向低探测）+ prompt 尾部分条目 + SlidingWindowManager
# ——右到左窗口连续段 finder + reachable_block_mask 稀疏驻留两段 + MambaManager
# ——边界状态 finder + Marconi 特赦 mask。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 3 条 eagle/投机解码：find 的丢尾（L764-L769）、SWA mask 的 shift=1、
#     _contiguous_blocks_for_hit 的 +1、manager 侧 use_eagle 传播（签名参数
#     与 use_eagle=False 账位保留——非投机路径恒 False/0）；
#   第 4 条 DCP/PCP：dcp_world_size/pcp_world_size 参数与 ×dcp 分支（单卡恒 1）；
#   第 5 条 connector：allocate_external_computed_blocks（外部块第二相）、
#     _pending_partial_tail_offloads/take_pending_partial_tail_offloads；
#   第 6 条 Mamba align 分配内部：_allocated_block_reqs/cached_blocks_this_step/
#     num_speculative_blocks、get_num_blocks_to_allocate/allocate_new_blocks/
#     pop_blocks_for_free/cache_blocks 四重写、_cache_partial_tail_block 的
#     mamba 版与 producer 记账（find 的 fine 分支与 reachable_block_mask
#     全保留——Marconi 面）；
#   第 7 条 RSWA/ChunkedLocal/Cross/SinkFull 内部（保留类壳与注册表条目）；
#   第 1/2 条 events/metrics（本文件无涉或已随 block_pool 删）。
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

from .block_pool import BlockPool
from .kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from .kv_cache_spec_registry import KVCacheSpecRegistry
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
    logic of one specific attention layer.
    """

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
        # SUBTRACTED: dcp_world_size/pcp_world_size 两参数与 ×dcp 缩放
        #   （L51-L52、L76-L79——第 4 条：单卡恒 1，分支不触发）
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L73-L83
        self.scheduler_block_size = scheduler_block_size
        # The block size for this manager; used for actual block allocation.
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L86-L92（TQ/MLA/
        #   HiddenState 三型随第 7 条删——剩 FullAttentionSpec 一型）
        self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (
            FullAttentionSpec,
        )
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L94-L97 req_to_blocks
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L99-L103 num_cached_block
        self.num_cached_block: dict[str, int] = {}

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L105-L106
        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

        # Whether this group's prefix-cache hits drop the EAGLE/MTP lookahead
        # block. Only consulted by managers whose hit logic is sparse within an
        # aligned segment (SWA). Initialized lazily by the coordinator after
        # determining the attention groups.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L108-L112
        #   use_eagle（eagle 传播链删——第 3 条；恒 False 账位保留，
        #   cache_blocks 的掩码实参原样引用它）
        self.use_eagle = False

        # Partial-hit copy-on-write bookkeeping. Populated only by fine-grained
        # managers (full attention, mamba "align"); harmlessly empty elsewhere.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L114-L117 部分命中
        #   两账本（_partial_hit_reqs 触发 CoW；_pending_cow_copies 拷贝对源）
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        # SUBTRACTED: _pending_partial_tail_offloads（L118-L126——第 5 条
        #   connector 的 partial-tail offload 记账，只有 mamba align 生产者
        #   会填 → ch16）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L128 _get_num_evictable_blocks
    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L130
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L132 _has_partial_local_hit
    def _has_partial_local_hit(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> bool:
        # The local prefix-cache hit ends inside one of this manager's
        # blocks: the shared tail block needs CoW.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L137-L142
        return (
            len(new_computed_blocks) > 0
            and num_local_computed_tokens % self.block_size != 0
        )

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

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L178-L191
        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L194-L200 fast-path
        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L202-L218（skipped
        #   推导——SWA 窗外段被跳过，命中块里前 num_skipped_blocks 块不算新容量）
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to
        # the required capacity; otherwise, skipped blocks dominate.
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L220-L225
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L226-L230（部分
        #   命中 +1：给 CoW 私有块预留的那块）
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Reserve the extra block that allocate_new_blocks pulls for the
            # partial-hit CoW redirect.
            num_new_blocks += 1
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
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L253-L265
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L267-L273（touch
        #   救回——ref_cnt+1 出队）
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L275-L289
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Record the partial tail for the CoW redirect in
            # allocate_new_blocks; cap the cached count at the full blocks so
            # cache_blocks() re-caches the private copy once full.
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
            self.num_cached_block[request_id] = block_idx

    # SUBTRACTED: allocate_external_computed_blocks（L291-L328——第 5 条
    #   connector 的外部块相（issue #33775 的第二半）；本地前缀缓存主流程
    #   num_external_computed_tokens 恒 0 不经过）。

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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L347-L357（CoW
        #   换尾：_partial_hit_reqs 有账 → 取 1 块私有 cow 块、_apply_cow
        #   原地替换共享尾块——预算里多要的那 1 块在这用掉）
        cow_blocks: list[KVCacheBlock] = []
        if request_id in self._partial_hit_reqs:
            # Partial hit: redirect the shared tail to a private CoW block.
            # Replacing in place keeps the length-based allocation below
            # correct; the extra block was reserved by
            # get_num_blocks_to_allocate.
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L359-L369（正常
        #   补块：按长度差值要新块）
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return cow_blocks
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return cow_blocks + new_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L371 records_new_block_ids
    @property
    def records_new_block_ids(self) -> bool:
        """Whether this manager's new blocks are zeroed by the worker."""
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L374
        return self._record_new_block_ids

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L376 take_new_block_ids
    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L378-L380
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L382 take_pending_cow_copies
    def take_pending_cow_copies(
        self,
    ) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Drain pending CoW source and destination block pairs."""
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L386-L388
        pending_copies = self._pending_cow_copies
        self._pending_cow_copies = []
        return pending_copies

    # SUBTRACTED: take_pending_partial_tail_offloads（L390-L403——第 5 条
    #   connector 的 producer partial-tail 手递手 → ch16）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L405 _apply_cow
    def _apply_cow(
        self,
        request_id: str,
        block_idx: int,
        source_block: KVCacheBlock,
        cow_block: KVCacheBlock,
    ) -> None:
        """Redirect a partial prefix-cache hit to a private CoW block.

        Both copy endpoints stay retained until the copy has run on the worker,
        so a same-step free cannot recycle them: ``source_block`` keeps its
        hit-ref, ``cow_block`` takes an extra ref beyond the one handed to the
        request.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L419-L425（原地
        #   换尾 + 登记拷贝对 + cow 额外 +1 引用）
        req_blocks = self.req_to_blocks[request_id]
        assert block_idx < len(req_blocks)
        assert req_blocks[block_idx] is source_block
        assert not source_block.is_null and source_block.ref_cnt > 0
        req_blocks[block_idx] = cow_block
        self._pending_cow_copies.append((source_block, cow_block))
        cow_block.ref_cnt += 1

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
                (including tokens that are already cached).
            retention_interval: Sparse local-checkpoint granularity. ``None``
                keeps dense checkpointing; ``0`` keeps only the latest replay
                boundary; a positive multiple of ``scheduler_block_size`` keeps
                a tail once per that-sized segment. Only SWA acts on it.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L445-L449 幂等闸
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        # Token boundaries whose reachable tail must be retained under sparse
        # retention: the replay boundary (``num_prompt - 1``, capped by
        # ``get_computed_blocks``) and any detected shared-prefix junction.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L451-L456
        #   reachable_boundaries（Marconi 钉住的载体）
        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L458-L475（算
        #   入表掩码后委托 cache_full_blocks）
        block_mask = self.reachable_block_mask(
            start_block=num_cached_blocks,
            end_block=num_full_blocks,
            alignment_tokens=self.scheduler_block_size,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
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

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L477（防重复
        #   登记的进度账）
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L498（基类稠密）
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L513-L517（先摘
        #   账后还块的分离——延迟释放面用）
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        self._partial_hit_reqs.pop(request_id, None)
        return req_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L519 free
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Free blocks in reverse order so that the tail blocks are freed first.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L526-L527（LRU
        #   不变量一：逆序传入）
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L529 get_num_common_prefix_blocks
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L541-L543

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
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.
        If eagle is enabled, drop the last matched block to force recompute the
        last block to get the required hidden states for eagle drafting head.
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
                Always False for non-EAGLE/MTP groups, but can be False for EAGLE/MTP
                groups too if the last block is already dropped (e.g., in a
                convergence loop in `find_longest_cache_hit`).
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens). By default, it should
                be set to the block_size.

        Returns:
            A tuple containing cached blocks and the exact cache-hit length in
            tokens. The cached block tuple has skipped blocks replaced by null
            blocks for each kv cache group in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L591-L593

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

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for attention types (e.g.
                R-SWA) that evict a middle gap rather than a head prefix. Ignored
                by the default implementation.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L643-L659（SWA
        #   窗外回收——ch14 已讲机制，本章消费 [NULL,...] 形态）
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus, we do not need to free any token outside attention window.
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L675（基类 no-op；
        #   mamba 版的 cached_blocks_this_step 清账随第 6 条删）
        return None


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L678 FullAttentionManager
class FullAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L679
    #   supports_fine_grained_hash_lookup（细粒度命中的能力位）
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
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # SUBTRACTED: dcp_world_size/pcp_world_size 参数与 ×dcp 分支
        #   （L691-L692、L701-L704——第 4 条：单卡恒 1）；assert 里的
        #   ChunkedLocalAttentionSpec 分型也随第 7 条删（Full 专属）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L694-L699
        assert isinstance(
            kv_cache_spec, FullAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        block_size = kv_cache_spec.block_size
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L705-L711（哈希
        #   视图按块尺寸解析——细粒度查找时保留原始哈希粒度列表）
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L713-L726 fine_grained
        #   判定旗标（alignment < block_size 且整除）
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

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L728-L739 phase 1
        #   满块链：沿链逐块查、miss 即断（链式保证 miss 后必 miss、无需回溯）
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L741-L762 phase 2
        #   块内细粒度探测（自高向低=先试最长）
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

        # SUBTRACTED: eagle 丢尾段（L764-L769——第 3 条：drop_eagle_block 恒
        #   False，签名参数保留默认值）
        # Round down to the alignment; a no-op when fine-grained (hits land on
        # hash boundaries by construction) and when alignment_tokens ==
        # block_size. Then trim blocks past the new tail.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L770-L777（对齐
        #   取整裁块——命中长度的出口整形）
        hit_length -= hit_length % alignment_tokens
        num_blocks = cdiv(hit_length, block_size)
        for computed in computed_blocks:
            del computed[num_blocks:]
        return computed_blocks, hit_length

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L779 cache_blocks（重载）
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L785-L789（先走
        #   基类满块，再补 prompt 尾部分条目）
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L802-L819（只登
        #   最后一个哈希边界——『下一个请求最可能对齐』的 prompt 边界；
        #   中间边界故意不登）
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L822-L829（ref_cnt
        #   == 全体请求数的连续前段）
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L832 RSWAManager
# SUBTRACTED: 类内部（__init__ 的 rswa_window 存储 + remove_skipped_blocks 的
#   gap 回收重写，L841-L875——dossier.delete 第 7 条：R-SWA 特化；最小壳
#   继承 FullAttentionManager 保留注册表条目）。
class RSWAManager(FullAttentionManager):
    pass  # SOURCE: ...:L832（最小壳）


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L878 SlidingWindowManager
class SlidingWindowManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L879 __init__
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L880-L881
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L883 _contiguous_blocks_for_hit
    @classmethod
    def _contiguous_blocks_for_hit(
        cls, window_size: int, block_size: int, use_eagle: bool
    ) -> int:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L887-L894
        #   （eagle 的 +1 随第 3 条删——非投机路径 use_eagle 恒 False）
        blocks = cdiv(window_size - 1, block_size)
        return blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L896 find_longest_cache_hit
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
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # SUBTRACTED: dcp/pcp 两 assert（L912-L913——第 4 条）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L909-L911（assert
        #   分型）+ L914-L917（SWA 不支持细粒度部分命中——明言）
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        # Fine-grained partial hits are not supported for sliding window now
        assert alignment_tokens % kv_cache_spec.block_size == 0, (
            "SlidingWindowManager does not support fine-grained (partial) cache hits"
        )
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L918-L924（哈希
        #   视图 + 窗口连续块数）
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        # The number of contiguous blocks needed for a prefix cache hit.
        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block
        )

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L936-L968（右到左
        #   找窗口连续段、match 即停；eagle 对齐微调段 L951-L954/L980-L991
        #   的 drop_eagle 支随第 3 条删）
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            # SUBTRACTED: 对齐回退 while 环（L974-L979——hybrid 粗块对齐
            #   （block_size != alignment_tokens）时的 pop 微调；本章混合
            #   用例块尺寸相等为 no-op，eagle 支 L980-L991 一并删）
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L992-L993
        hit_length = len(computed_blocks[0]) * block_size
        return computed_blocks, hit_length

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L995 reachable_block_mask
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1006-L1010
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        if alignment_tokens is None:
            # Fast path: when the coordinator imposes no alignment constraint.
            return None
        assert alignment_tokens % kv_cache_spec.block_size == 0

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1012-L1018 need
        #   = 窗口连续块数（eagle 的 +1 随第 3 条删）
        block_size = kv_cache_spec.block_size
        # Contiguous blocks a hit needs at a boundary (incl. the EAGLE peek).
        need = cls._contiguous_blocks_for_hit(
            window_size=kv_cache_spec.sliding_window,
            block_size=block_size,
            use_eagle=use_eagle,
        )
        # The matched run's right edge sits on the aligned boundary block when
        # EAGLE peeks one block past it (shift=1), otherwise on the last block
        # before the boundary (shift=0).
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1019-L1022
        #   （eagle 的 shift=1 随第 3 条删——恒 shift=0）
        shift = 0

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1024 mask 初始化
        mask = [False] * (end_block - start_block)

        # (1) Segment-boundary tails. ``retention_interval``:
        #   None -> dense (a tail at every ``alignment_tokens`` boundary);
        #   0    -> no dense tails (only the replay boundary below);
        #   >0   -> a tail once per ``retention_interval``-sized segment.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1026-L1042 分段尾
        segment_tokens = (
            alignment_tokens
            if retention_interval is None
            else (None if retention_interval == 0 else retention_interval)
        )
        if segment_tokens is not None:
            per_segment = segment_tokens // block_size
            if need >= per_segment:
                # Every block is reachable; cache them all.
                return None
            for i in range(start_block, end_block):
                if i >= shift and (i - shift) % per_segment >= per_segment - need:
                    mask[i - start_block] = True

        # (2) Reachable-boundary tails: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction. Both
        # land before segments would cover them under sparse retention, so keep
        # the ``need``-block tail ending on each boundary explicitly.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1044-L1053 特赦段
        #   （Marconi junction 落地——稀疏不掉复用点）
        if retention_interval is not None:
            for boundary_tokens in reachable_boundaries:
                aligned = boundary_tokens // alignment_tokens * alignment_tokens
                end = aligned // block_size + shift
                for j in range(max(start_block, end - need), min(end_block, end)):
                    mask[j - start_block] = True

        return mask

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1057 get_num_skipped_tokens
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1083
        return max(0, num_computed_tokens - self.sliding_window + 1)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1085 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1092
        return 0


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1095 ChunkedLocalAttentionManager
# SUBTRACTED: 类内部（find_longest_cache_hit 的窗外 null 标记扫描 +
#   get_num_skipped_tokens 的 chunk 左界，L1100-L1250——dossier.delete 第 7 条：
#   特化注意力类型，主路径不触发；最小壳保留注册表条目）。
class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    pass  # SOURCE: ...:L1095（最小壳）


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1253 MambaManager
class MambaManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1254
    #   supports_fine_grained_hash_lookup（mamba align 也支持细粒度）
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1256 __init__
    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        # SUBTRACTED: DCP 缩放回退行（L1260-L1263——第 4 条：mamba 不乘 dcp
        #   的说明，dcp 删后为 no-op）；num_speculative_blocks/cached_blocks_
        #   this_step/align 三账本（L1265-L1277——第 6 条：align 分配内部与
        #   partial-tail offload 记账）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1264
        #   mamba_cache_mode（"align" 是 enable_partial_hash_hits 的前提）
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1279 find_longest_cache_hit
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
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # SUBTRACTED: dcp/pcp 两 assert（L1295-L1296——第 4 条）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1292-L1303
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1304-L1307
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        hit_length = 0

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1309-L1330 fine
        #   分支（概念壳保留：块内自高向低探边界——mamba 只要那一个状态块）
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

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1332-L1356（整块
        #   版：右到左找边界状态块、early stop——mamba 命中只要边界上那一个
        #   状态块，前段 null 占位）
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
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

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1358 reachable_block_mask
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
        """Sparse Mamba state-snapshot retention.

        ``retention_interval``:

          ``None`` -> dense (cache every block; default, unchanged behavior)
          ``0``    -> keep only the ``reachable_boundaries`` states
          ``> 0``  -> keep one state per ``retention_interval``-sized segment

        ``reachable_boundaries`` are proven reuse points (the replay boundary and
        any cross-request shared-prefix junction, Marconi-style APC); their
        boundary state is always kept so sparse retention does not defeat reuse.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1381-L1386
        if retention_interval is None or alignment_tokens is None:
            # Dense caching (default) or no alignment constraint imposed.
            return None
        assert isinstance(kv_cache_spec, MambaSpec)
        block_size = kv_cache_spec.block_size
        mask = [False] * (end_block - start_block)

        # (1) Segment-boundary states. A Mamba hit needs exactly the single
        # state block ending on the boundary (no window, and draft models have
        # no mamba layers, so no eagle shift). Block ``i`` ends at token
        # ``(i + 1) * block_size``.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1388-L1402 分段状态
        segment_tokens = None if retention_interval == 0 else retention_interval
        if segment_tokens is not None:
            per_segment = segment_tokens // block_size
            if per_segment <= 1:
                # Interval at/below the block size: every block is a boundary.
                return None
            first_boundary = (
                start_block + per_segment
            ) // per_segment * per_segment - 1
            for i in range(first_boundary - start_block, len(mask), per_segment):
                mask[i] = True

        # (2) Reachable-boundary states: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction, both
        # of which segments would otherwise skip under sparse retention. A Mamba
        # hit needs exactly the single state block ending on the boundary.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1404-L1412 特赦段
        #   （Marconi 落地：junction 的边界状态永远保留）
        for boundary_tokens in reachable_boundaries:
            aligned = boundary_tokens // alignment_tokens * alignment_tokens
            boundary_block = aligned // block_size - 1
            if start_block <= boundary_block < end_block:
                mask[boundary_block - start_block] = True

        return mask

    # SUBTRACTED: remove_skipped_blocks 重写（L1416-L1444——align 的
    #   last_state_block_idx 状态块回收，第 6 条）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1446 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1450
        return 0

    # SUBTRACTED: get_num_blocks_to_allocate/allocate_new_blocks/pop_blocks_for_
    #   free/cache_blocks 四重写 + _cache_partial_tail_block 的 mamba 版 +
    #   new_step_starts（L1452-L1744——dossier.delete 第 6 条：Mamba align
    #   分配内部（last_state_block_idx 状态块滚动/_allocated_block_reqs 复用/
    #   cached_blocks_this_step 防同步命中/producer partial-tail 记账）；
    #   走基类版本即得全量重算语义）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1667 get_num_skipped_tokens
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens whose mamba state are not needed anymore. Mamba only
        need to keep the state of the last computed token, so we return
        num_computed_tokens - 1.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1673
        return num_computed_tokens - 1


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1747 CrossAttentionManager
# SUBTRACTED: 类内部（L1750-L1807——第 7 条：encoder-decoder 的 cross-attention
#   不参与跨请求共享，全部方法为拒绝/空实现；最小壳保留注册表条目）。
class CrossAttentionManager(SingleTypeKVCacheManager):
    pass  # SOURCE: ...:L1747（最小壳）


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1810 SinkFullAttentionManager
# SUBTRACTED: 类内部（L1811-L1833——第 7 条：sink 块预占的特化构造；最小壳
#   继承 FullAttentionManager 保留注册表条目）。
class SinkFullAttentionManager(FullAttentionManager):
    pass  # SOURCE: ...:L1810（最小壳）


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1836 get_manager_for_kv_cache_spec
def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_in_flight_tokens: int,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """
    Get the appropriate manager for a given KVCacheSpec.

    Uses the KVCacheSpecRegistry to look up the manager class, supporting
    both built-in and custom specs registered via @register_kv_cache_spec
    and KVCacheSpecRegistry.register.

    Args:
        kv_cache_spec: The KVCacheSpec instance
        max_in_flight_tokens: The max tokens scheduled but not yet settled
            (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`
        max_model_len: The maximum context length the model could serve
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass
    """
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1857-L1878
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"
    )
    # SlidingWindow / ChunkedLocalAttention managers recycle blocks;
    # the runtime admission cap must match the recycling-aware bound the
    # startup pool sizer uses (single source of truth: the spec method).
    # R-SWA also recycles gap blocks but peak physical KV still fits the
    # full-attention bound (prefix + window <= max_model_len), so it inherits
    # FullAttentionSpec sizing without a separate admission cap.
    if isinstance(
        kv_cache_spec,
        (SlidingWindowSpec, ChunkedLocalAttentionSpec),
    ):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1881 register_all_kvcache_specs
def register_all_kvcache_specs(vllm_config=None):
    """Built-in spec registration"""
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1883-L1938（TQ/
    #   MLA/SlidingWindowMLA/HiddenState 四条随第 7 条 spec 族删；其余条目
    #   壳全保留——注册表完整性）
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    KVCacheSpecRegistry.register(
        SlidingWindowSpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )

    KVCacheSpecRegistry.register(
        MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec
    )
    KVCacheSpecRegistry.register(
        ChunkedLocalAttentionSpec,
        ChunkedLocalAttentionManager,
        uniform_type_base_spec=ChunkedLocalAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        CrossAttentionSpec,
        CrossAttentionManager,
        uniform_type_base_spec=CrossAttentionSpec,
    )

    KVCacheSpecRegistry.register(
        RSWASpec, RSWAManager, uniform_type_base_spec=FullAttentionSpec
    )
    KVCacheSpecRegistry.register(
        SinkFullAttentionSpec,
        SinkFullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    # SUBTRACTED: current_platform.register_custom_kv_cache_specs（L1940-L1942
    #   ——平台自定义 spec 外挂面）。
