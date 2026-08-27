# SOURCE: vllm/v1/core/kv_cache_coordinator.py
# KVCacheCoordinator——把不同 KV 组的块账本扇出给各 SingleTypeKVCacheManager
# 的协调层（KVCacheManager 对它的调用 = allocate_slots 三段式的内部段）。
# 本章精简版跑 enable_prefix_caching=False → get_kv_cache_coordinator 返回
# KVCacheCoordinatorNoPrefixCache（源码原生路径，L864-L876——支持任意组数，
# 含 0 组；不实现任何前缀缓存特性）。
# SUBTRACTED（dossier.delete 批准项的落点）：
#   第 3 条 哈希侧：find_longest_cache_hit 抽象（L368-L377）；
#   第 4 条 混合/多组：HybridKVCacheCoordinator/SpecGroup/verify_and_split_
#     kv_cache_groups/_cache_hit_alignment_tokens/find_longest_cache_hit_
#     per_group（L506-L848）；UnitaryKVCacheCoordinator（L435-L503——唯一
#     实体方法 find_longest_cache_hit 是哈希侧、eagle 装配是第 5 条，删后
#     仅剩基类行为；单组开缓存的命中查找 → ch15）；SWA/Mamba 准入上限与
#     retention_interval/_validate_prefix_cache_retention_interval
#     （L30-L57、L122-L128）；
#   第 1/2 条 观测：enable_kv_cache_events/metrics_collector 贯穿参数；
#   第 5 条 eagle：eagle_group_ids 装配；
#   第 6 条 DCP/PCP：dcp_world_size/pcp_world_size 乘子（单卡恒 1 烘干）；
#   第 7 条 connector：allocate_external_computed_blocks 扇出（L230-L236）；
#   第 11 条 get_num_common_prefix_blocks（L319-L334）。
from abc import ABC
from collections.abc import Sequence

from .block_pool import BlockPool
from .kv_cache_interface import KVCacheConfig
from .kv_cache_utils import KVCacheBlock
from .request import Request
from .single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
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
        enable_caching: bool,
        scheduler_block_size: int,
        hash_block_size: int,
    ):
        # SUBTRACTED: use_eagle/enable_kv_cache_events/dcp_world_size/
        #   pcp_world_size/metrics_collector 参数（L70-L77——第 1/2/5/6 条）。
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

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L90-L96 块池构造
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
        )

        # SUBTRACTED: eagle_group_ids 装配（L98-L104——第 5 条）。

        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L106-L120 每组一份
        #   manager（扇出账本）
        self.single_type_managers: tuple[SingleTypeKVCacheManager, ...] = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_model_len=max_model_len,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        # SUBTRACTED: retention_interval 装配与校验（L122-L128——第 4 条）。

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

        Returns:
            The number of blocks to allocate.
        """
        # SUBTRACTED: apply_admission_cap 参数与 CrossAttentionManager 静态
        #   分配分支（L139、L168-L179——第 4 条；编码器 token 归 ch14 的跨注意
        #   力族，参数保留作调用面账位，decoder-only 恒 0）。
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L166-L190 扇出求和
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[i],
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
            )
        return num_blocks_to_allocate

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L192 allocate_new_computed_blocks
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of the local computed tokens.
        """
        # SUBTRACTED: num_external_computed_tokens 参数与外部块补充分支
        #   （L196、L231-L236——第 7 条 → ch16；两阶段分配的注释脉络保留）。
        # A running request is already tracked in num_cached_block and won't
        # have new prefix-cache hits, so this is a no-op for it.
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L210-L217
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # Touch every group's cache-hit blocks before any external allocation
        # could evict them (issue #33775 的两阶段次序：先全组 touch).
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L223-L229
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
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

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        # SUBTRACTED: CrossAttentionManager 的 encoder token 分支（L264-L267
        #   ——第 4 条；参数保留作调用面账位）。
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L262-L271 扇出分配
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

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        # SUBTRACTED: retention_interval 透传（L287——第 4 条）。
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L283-L288 扇出写回
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L290 free
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
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

    # SUBTRACTED: get_num_common_prefix_blocks（L319-L334——第 11 条）。

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

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length. R-SWA managers use this to
                free gap blocks between the prefill tail and decode window; other
                manager types ignore it.
        """
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L354-L357（单组全注意力
        #   下各 manager no-op——滑窗回收 → ch14）
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

    # SUBTRACTED: find_longest_cache_hit 抽象（L368-L377——第 3 条 → ch15）。

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
        enable_kv_cache_events: bool,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector=None,
    ):
        # SUBTRACTED: use_eagle/dcp_world_size/pcp_world_size 透传实参（L398-
        #   L401、L413-L414——第 5/6 条）；本类签名保留源参数面
        #   （enable_kv_cache_events/metrics_collector 已无人供给，账位保留）。
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L406-L418
        super().__init__(
            kv_cache_config,
            max_model_len,
            False,  # enable_caching：本协调器按定义恒 False
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L419
        self.num_single_type_manager = len(self.single_type_managers)

    # SUBTRACTED: get_num_common_prefix_blocks（L421-L422——第 11 条）。

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L424 find_longest_cache_hit
    def find_longest_cache_hit(
        self,
        block_hashes: list,
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L429-L432（无缓存 →
        #   恒空命中——NoPrefixCache 的原生契约）
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0, 0


# SUBTRACTED: UnitaryKVCacheCoordinator（L435-L503——唯一实体方法
#   find_longest_cache_hit 属哈希侧（第 3 条 → ch15）、eagle 装配属第 5 条、
#   DCP block_size 放大属第 6 条；删后仅剩基类行为。单组开缓存的命中查找
#   → ch15）、HybridKVCacheCoordinator/SpecGroup（L506-L848——第 4 条 → ch14）。


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L851 get_kv_cache_coordinator
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    enable_caching: bool,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector=None,
) -> KVCacheCoordinator:
    # SUBTRACTED: use_eagle/enable_kv_cache_events/dcp_world_size/
    #   pcp_world_size 参数（L855-L858——第 1/5/6 条）。
    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L864-L876（enable_caching=
    #   False → NoPrefixCache：本章精简版的源码原生路径）
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            enable_kv_cache_events=False,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    # SUBTRACTED: 单组（Unitary，L877-L890）与混合（Hybrid，L891-L903）分支
    #   ——开缓存的命中查找 → ch15；多组混合 → ch14。本章精简版不开缓存。
    raise AssertionError(
        "ch13 精简版跑 enable_prefix_caching=False（NoPrefixCache 原生路径）；"
        "开缓存的协调器 → ch15"
    )
