# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 对应 vllm/v1/core/kv_cache_coordinator.py：协调多个 KV cache group。本章保留完整的
# 三态拓扑：get_kv_cache_coordinator 工厂 + KVCacheCoordinator 基类（逐组转发）+
# 三个具体协调器（NoPrefixCache / Unitary / Hybrid）。Hybrid 的不动点迭代命中查找
# 是本章核心。
from collections.abc import Sequence
from math import lcm

from .block_pool import BlockPool
from .kv_cache_utils import BlockHash, KVCacheBlock
from .request import FullAttentionSpec, KVCacheConfig, KVCacheSpec, Request
from .single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L28 (class KVCacheCoordinator)
class KVCacheCoordinator:
    """Coordinate the KV cache of different KV cache groups."""
    # SUBTRACTED: ABC / @abstractmethod（L28, L262）—— 精简版保留全部三个具体子类，
    # find_longest_cache_hit 由各子类提供实体，无需抽象约束。

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L33 (__init__)
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        enable_caching: bool,
        hash_block_size: int,
    ):
        # SUBTRACTED: use_eagle / eagle_group_ids（L38, L58-L64）—— 投机解码草稿头丢尾块。
        # SUBTRACTED: enable_kv_cache_events / metrics_collector / dcp / pcp 透传
        # （L40-L44, L52-L56, L74-L75）—— 事件/统计/上下文并行旁路。
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
        )

        # SUBTRACTED: eagle_group_ids 集合（L58-L64）—— 投机解码组标记；非投机路径恒空。
        # 为保留 Unitary.find_longest_cache_hit 的 `0 in self.eagle_group_ids` 表达式可读，
        # 这里以空集合占位（恒不命中），等价 use_eagle=False。
        self.eagle_group_ids: set[int] = set()

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L80 (get_num_blocks_to_allocate)
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """Get the number of blocks needed to be allocated for the request."""
        # SUBTRACTED: CrossAttentionManager 分支（L115-L125）—— 编码器-解码器交叉注意力
        # 按 num_encoder_tokens 静态分配，decoder-only（num_encoder_tokens==0）不触发。
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[i],
                total_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        return num_blocks_to_allocate

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L137 (allocate_new_computed_blocks)
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Add the new computed blocks to the request for each group, optionally
        allocating new blocks for external computed tokens."""
        for i, manager in enumerate(self.single_type_managers):
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L163 (allocate_new_blocks)
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """Allocate new blocks for the request to give it at least
        `num_tokens` token slots."""
        # SUBTRACTED: CrossAttentionManager 用 num_encoder_tokens 替代 num_tokens
        # （L190-L192）—— decoder-only 不触发。
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L198 (cache_blocks)
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request."""
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L211 (free)
    def free(self, request_id: str) -> None:
        """Free the blocks for the request."""
        for manager in self.single_type_managers:
            manager.free(request_id)

    # SUBTRACTED: get_num_common_prefix_blocks（L221-L236）—— 级联前缀统计，非命中核心。

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L238 (remove_skipped_blocks)
    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, total_computed_tokens)

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L253 (get_blocks)
    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """Get the blocks for the request."""
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    # find_longest_cache_hit: 由各子类提供实体（原为抽象方法 @abstractmethod, L262-L268）。

    # SUBTRACTED: new_step_starts（L270-L273）—— 逐步引流接口，非分配/命中核心控制流。


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L276 (KVCacheCoordinatorNoPrefixCache)
class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L284 (__init__)
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        hash_block_size: int,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            False,
            hash_block_size=hash_block_size,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    # SUBTRACTED: get_num_common_prefix_blocks（L310-L311）—— 见基类同名说明。

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L313 (find_longest_cache_hit)
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L324 (UnitaryKVCacheCoordinator)
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L331 (__init__)
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        enable_caching: bool,
        hash_block_size: int,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            enable_caching,
            hash_block_size=hash_block_size,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        # SUBTRACTED: dcp/pcp block_size 乘子（L358-L363）—— 上下文并行，单卡为 1。
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L373 (find_longest_cache_hit)
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=0 in self.eagle_group_ids,
            alignment_tokens=self.block_size,
        )
        # SUBTRACTED: dcp_world_size / pcp_world_size 透传（L386-L387）—— 单卡为 1。
        return hit_blocks, len(hit_blocks[0]) * self.block_size


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L392 (HybridKVCacheCoordinator)
class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L398 (__init__)
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        enable_caching: bool,
        hash_block_size: int,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            enable_caching,
            hash_block_size=hash_block_size,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        # SUBTRACTED: dcp/pcp == 1 的 assert（L432-L433）—— Hybrid 不支持上下文并行。
        self.verify_and_split_kv_cache_groups()

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L436 (verify_and_split_kv_cache_groups)
    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        block_sizes = [spec.block_size for spec, _, _ in attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

        # SUBTRACTED: eagle_attn_group_indices（L479-L485）—— 投机解码草稿层标记，
        # 非投机路径不触发不动点中的 eagle 分支。

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L487 (find_longest_cache_hit)
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Returns:
            A tuple of (per-group cache hit blocks, longest cache hit length).
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> list[BlockHash]:
            # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L510 (_get_block_hashes)
            # SUBTRACTED: block_size != hash_block_size 的 BlockHashListWithBlockSize
            # 换算（L511-L515）—— 不同组不同块尺寸的罕见配置；常见情形等块尺寸直接复用。
            assert kv_cache_spec.block_size == self.hash_block_size
            return block_hashes

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        # SUBTRACTED: eagle_verified 集合与 eagle 的 +block_size 多匹配/丢尾分支
        # （L527-L530, L546-L570 散点）—— 投机解码草稿头专用；非投机路径每类型对候选
        # 长度只"接受或缩短"，恒走主分支。

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = (
                        curr_hit_length // spec.block_size * spec.block_size
                    )
                    continue

                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=False,
                    alignment_tokens=self.lcm_block_size,
                )
                _new_hit_length = len(hit_blocks[0]) * spec.block_size
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // spec.block_size
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L594 (get_kv_cache_coordinator)
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    enable_caching: bool,
    hash_block_size: int,
) -> KVCacheCoordinator:
    # SUBTRACTED: use_eagle / enable_kv_cache_events / dcp / pcp / metrics_collector
    # 透传参数（L598-L604）—— 投机解码、事件、上下文并行、统计旁路。
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            hash_block_size=hash_block_size,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            enable_caching,
            hash_block_size=hash_block_size,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_num_batched_tokens,
        enable_caching,
        hash_block_size=hash_block_size,
    )
