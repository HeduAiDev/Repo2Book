# SOURCE: vllm/v1/core/kv_cache_coordinator.py
# 命中查找协调层（ch15 已立、ch16 消费）+ 本章的两处开口：
# allocate_new_computed_blocks 的**外部块第二相**（L230-L236——ext_comp
# 段的『先全组 touch、再全组分配』两阶段，防 A 组 get_new_blocks 驱逐
# B 组未 touch 的命中块，issue #33775）与 find_longest_cache_hit_per_group
# （L819-L848——get_computed_blocks_for_connector 混合感知命中的逐组
# 独立查找半边）。三态分派（NoPrefixCache/Unitary/Hybrid）逐字。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 3 条观测旁路：enable_kv_cache_events/metrics_collector 参数与透传；
#   eagle：eagle_group_ids/兜底全标/use_eagle 传播/SpecGroup.use_eagle 字段
#     （find 签名的 drop_eagle_block 实参 False——恒假）；
#   第 5 条 cross-layer/uniform 布局；dcp/pcp 乘子（单卡恒 1 烘干）；
#   cross-attention 的 encoder 静态分配支（L168-L179——本章 encoder-decoder
#     在调度器构造期被拒，恒不可达；num_encoder_tokens 参数面保留）。
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NamedTuple

from .block_pool import BlockPool
from .envs import VLLM_PREFIX_CACHE_RETENTION_INTERVAL
from .kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
)
from .kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)
from .math_utils import cdiv
from .request import Request
from .single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L30 _validate_prefix_cache_retention_interval
def _validate_prefix_cache_retention_interval(
    retention_interval: int | None,
    scheduler_block_size: int,
    kv_cache_config: KVCacheConfig,
) -> None:
    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L35-L36
    if retention_interval is None:
        return

    # Retention sparsifies sliding-window and Mamba (linear-attention)
    # checkpoints; full-attention and chunked-local groups cache densely and
    # ignore it (their hit granularity must stay fine).
    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L38-L50
    # SUBTRACTED: SWA 组谓词（族删——只剩 Mamba 位）
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_config.kv_cache_groups
    ):
        raise ValueError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL is set but this model has "
            "no sliding-window or Mamba KV cache group, so retention has no "
            "effect. Unset it (it only applies to sliding-window and Mamba "
            "attention)."
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L52-L57
    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:
        raise ValueError(
            f"VLLM_PREFIX_CACHE_RETENTION_INTERVAL ({retention_interval}) "
            "must be non-negative and a multiple of scheduler_block_size "
            f"({scheduler_block_size})."
        )


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L60 KVCacheCoordinator
class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L65 __init__
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        enable_caching: bool,
        scheduler_block_size: int,
        hash_block_size: int,
    ):
        # SUBTRACTED: use_eagle/enable_kv_cache_events/dcp_world_size/
        #   pcp_world_size/metrics_collector 五参数（观测旁路/eagle/上下文并行）。
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L79-L88
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # The scheduling granularity (LCM of all group block sizes), must be a multiple
        # of the hash_block_size and the block size of each group.
        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L90-L96 建 BlockPool
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
        )

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L106-L120 逐组建
        #   single_type_manager（needs_kv_cache_zeroing 实参——站 6/10 的
        #   零清账从这里通电到每个 manager）
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        # A positive retention interval must be a multiple of the base hit granularity
        # (``scheduler_block_size``) to land on real cache-hit boundaries.
        # 0 = keep only the latest replay boundary; None = dense;
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L122-L128
        self.retention_interval = VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L130 get_num_blocks_to_allocate
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
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
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, apply the recycling-aware
                per-request admission cap (SWA / chunked-local).

        Returns:
            The number of blocks to allocate.
        """
        # SUBTRACTED: cross-attention 静态分配支（L168-L179——encoder-decoder
        #   构造期被拒）。
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L180-L190 逐 manager
        #   扇出求和
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[i],
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        return num_blocks_to_allocate

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L192 allocate_new_computed_blocks
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # A running request is already tracked in num_cached_block and won't
        # have new prefix-cache hits, so this is a no-op for it.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L210-L217（running 短路）
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # Two-phase allocation (issue #33775): first touch every group's local
        # cache-hit blocks, then allocate external blocks for every group. This
        # ensures an earlier group's external `get_new_blocks` cannot evict a
        # later group's not-yet-touched cache-hit blocks.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L219-L229 第一相：先
        #   全组 touch 本地命中块
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L230-L236 第二相：
        #   外部块分配（ext_comp 段——『已分配未缓存』窗口的分配半边）
        if num_external_computed_tokens > 0:
            for manager in self.single_type_managers:
                manager.allocate_external_computed_blocks(
                    request_id,
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L238 allocate_new_blocks
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.
        """
        # SUBTRACTED: cross-attention 的 num_encoder_tokens 分派（L262-L271
        #   的三目——encoder-decoder 恒拒）。
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L262-L271 逐 manager 扇出
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L273 cache_blocks
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L283-L288
        for manager in self.single_type_managers:
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.retention_interval,
            )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L290 free
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L297-L298
        for manager in self.single_type_managers:
            manager.free(request_id)

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L300 pop_blocks_for_free
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        Pop the request's bookkeeping from all single-type managers and
        return its blocks without returning them to the block pool. The
        caller must eventually pass the returned blocks to
        `block_pool.free_blocks`, freeing them in reverse order (so that
        tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L314-L317
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L319 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache for each kv cache group.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L331-L334
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L336 remove_skipped_blocks
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L354-L357
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L359 get_blocks
    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L363-L366
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L368 find_longest_cache_hit
    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """Returns the per-group hit blocks, the hit length, and the number of
        ``num_uncached_common_prefix_tokens`` (a shared prefix that a
        sparse-retention group has not cached yet; 0 unless hybrid)."""
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L377
        pass

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L379 new_step_starts
    def new_step_starts(self) -> None:
        """Notify each manager that a new step is starting."""
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L381-L382
        for manager in self.single_type_managers:
            manager.new_step_starts()


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L385 KVCacheCoordinatorNoPrefixCache
class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L393 __init__
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        scheduler_block_size: int,
        hash_block_size: int,
    ):
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L406-L419（enable_caching
        #   固定 False——本章『外部缓存是唯一缓存』配置的原生路径）
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            False,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L419
        self.num_single_type_manager = len(self.single_type_managers)

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L421 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L422
        return [0] * self.num_single_type_manager

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L424 find_longest_cache_hit
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L429-L432（关缓存：
        #   本地命中恒空——双查退化为纯外部查询）
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0, 0


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L435 UnitaryKVCacheCoordinator
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L442 __init__
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        enable_caching: bool,
        scheduler_block_size: int,
        hash_block_size: int,
    ):
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L456-L479
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            enable_caching,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L469-L470
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        # For models using only Mamba, block_size is set to max_model_len when
        # prefix caching is disabled, and hash_block_size validation is skipped.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L475-L482
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )
        # SUBTRACTED: dcp 缩放与 use_eagle 传播（L471-L474、L483-L484）。

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L486 find_longest_cache_hit
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L491-L501（直接委托
        #   唯一 manager；单组无块内边界——partial_tail 恒 0 的根源）
        hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=False,
            alignment_tokens=self.block_size,
        )
        # Single group: nothing "uncached common" -- no other group to lag it.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L502-L503
        return hit_blocks, hit_length, 0


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L506 SpecGroup
class SpecGroup(NamedTuple):
    """KV cache groups that share one spec, batched together for a single
    cache-hit lookup.
    """

    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    # SUBTRACTED: use_eagle 字段（L518——eagle）。
    # SUBTRACTED: is_eagle_group 字段（L519——eagle）。


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L521 HybridKVCacheCoordinator
class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L527 __init__
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        enable_caching: bool,
        scheduler_block_size: int,
        hash_block_size: int,
    ):
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L541-L589
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            enable_caching,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L554-L569
        self.hash_block_size = hash_block_size
        group_block_sizes = [
            manager.block_size for manager in self.single_type_managers
        ]
        assert all(
            block_size % hash_block_size == 0 for block_size in group_block_sizes
        ), (
            "Each KV cache group's real block_size must be divisible by "
            f"hash_block_size. block_sizes={group_block_sizes}, "
            f"hash_block_size={hash_block_size}"
        )
        # Partial hash hits are limited to full-attention + mamba ("align")
        # without context parallelism.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L581-L588
        #   enable_partial_hash_hits（块内边界命中的装配前提——m3 子块尾的
        #   出生地：块大小 > 哈希粒度的组才有『子块尾』可言）
        self.enable_partial_hash_hits = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            and g.kv_cache_spec.mamba_cache_mode == "align"
            and g.kv_cache_spec.block_size > hash_block_size
            for g in kv_cache_config.kv_cache_groups
        )
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L589
        self.verify_and_split_kv_cache_groups()

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L591 _cache_hit_alignment_tokens
    @property
    def _cache_hit_alignment_tokens(self) -> int:
        # Fine-grained partial hits may return hash-block-aligned lengths;
        # otherwise it must stay scheduler-block-aligned.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L595-L599
        return (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.scheduler_block_size
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L601 verify_and_split_kv_cache_groups
    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L606-L625（同 spec 并桶）
        self.attention_groups: list[SpecGroup] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for group in self.attention_groups:
                if group.spec == spec:
                    assert manager_cls is group.manager_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group.group_ids.append(i)
                    break
            else:
                self.attention_groups.append(SpecGroup(spec, [i], manager_cls))

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L627-L629
        assert len(self.attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L631-L635
        self.attention_groups.sort(
            key=lambda g: not isinstance(g.spec, FullAttentionSpec)
        )

        # Dense reference group for per-group lookups (None when the model
        # has no full-attention layers): full attention is downward-closed,
        # so any group reporting a longer per-group hit implies the union of
        # per-group hits is not consistent at a single boundary (#46453).
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L637-L644
        first = self.attention_groups[0]
        self.full_attention_group_id: int | None = (
            first.group_ids[0] if isinstance(first.spec, FullAttentionSpec) else None
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L652 cache_blocks（重载）
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L653-L665（对齐区取整）
        if self.enable_partial_hash_hits:
            aligned_num_computed_tokens = num_computed_tokens
        else:
            # Cache hits in this coordinator are always a multiple of
            # ``scheduler_block_size`` tokens (see ``find_longest_cache_hit``).
            # Within an aligned region, SWA groups may only consult a subset of
            # blocks per ``scheduler_block_size``-segment so the unused blocks
            # also stay out of the prefix-cache hash map.
            aligned_num_computed_tokens = (
                num_computed_tokens
                // self.scheduler_block_size
                * self.scheduler_block_size
            )
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L666-L683
        for manager in self.single_type_managers:
            num_tokens_to_cache = aligned_num_computed_tokens
            # The manager already knows the fine hit granularity
            # (``scheduler_block_size``); retention is passed separately so it
            # can keep both the coarse segment tails and the fine replay
            # boundary (which needs the fine value).
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L685 find_longest_cache_hit
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the reconciled (combined) cache hit.
                - ``num_uncached_common_prefix_tokens``: a shared prefix that a
                  sparse-retention group has not cached yet (0 unless hybrid).
        """

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L710-L714 状态初始化
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L716-L720
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L727-L796 主循环
        #   （不动点：每类型接受或缩短）
        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls) in enumerate(
                self.attention_groups
            ):
                first_group_id = group_ids[0]
                group_block_size = self.single_type_managers[first_group_id].block_size
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L766-L779
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=False,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                )
                # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L785-L790
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L792-L796（无缩短
            #   → 收敛退出；simple hybrid 一轮早停）
            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L798-L808
        first_group = self.attention_groups[0]
        if isinstance(first_group.spec, FullAttentionSpec):
            group_block_size = self.single_type_managers[
                first_group.group_ids[0]
            ].block_size
            num_blocks = cdiv(hit_length, group_block_size)
            for group_id in first_group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        # Uncached shared prefix detection: if any attn. group cached a longer
        # prefix than the reconciled hit, it is an uncached common prefix across
        # requests that a sparse-retention group hasn't cached yet.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L810-L817
        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L819
    #   find_longest_cache_hit_per_group——逐组独立查找（本章 m3 的发散
    #   判定半边：get_computed_blocks_for_connector 消费）
    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        """Like find_longest_cache_hit but evaluates each group independently.

        Returns:
            (blocks_per_group, hit_lengths_per_group)
        """

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L830-L846（逐组各查
        #   各的、不做不动点调和——发散与否交还调用方）
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups

        for spec, group_ids, manager_cls in self.attention_groups:
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=False,
                alignment_tokens=self._cache_hit_alignment_tokens,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit

        return tuple(hit_blocks), tuple(hit_lengths)


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L851 get_kv_cache_coordinator
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    enable_caching: bool,
    scheduler_block_size: int,
    hash_block_size: int,
) -> KVCacheCoordinator:
    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L864-L876（关缓存 →
    #   NoPrefixCache——本章『外部缓存是唯一缓存』的配置支）
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L877-L890（单组 → Unitary）
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            enable_caching,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L891-L903（多组 → Hybrid）
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        enable_caching,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )
