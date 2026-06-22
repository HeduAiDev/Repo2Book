# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 对应 vllm/v1/core/kv_cache_coordinator.py：协调各 KV cache group。本章主路径只保留
# KVCacheCoordinator 基类 + 单组 UnitaryKVCacheCoordinator；HybridKVCacheCoordinator
# （多组）与 KVCacheCoordinatorNoPrefixCache 属可省略分支。
from collections.abc import Sequence

from .block_pool import BlockPool
from .kv_cache_utils import BlockHash, KVCacheBlock
from .request import FullAttentionSpec, Request
from .single_type_kv_cache_manager import FullAttentionManager


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L28 (class KVCacheCoordinator)
class KVCacheCoordinator:
    """Coordinate the KV cache of different KV cache groups."""
    # SUBTRACTED: ABC / @abstractmethod —— 本章只保留 UnitaryKVCacheCoordinator 一个
    # 具体子类，无需抽象约束。原 L28, L262。

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L33 (KVCacheCoordinator.__init__)
    def __init__(
        self,
        kv_cache_spec: FullAttentionSpec,
        num_blocks: int,
        max_model_len: int,
        enable_caching: bool,
        hash_block_size: int,
    ):
        # SUBTRACTED: use_eagle / eagle_group_ids（L38, L58-L64）—— 投机解码草稿头丢尾块。
        # SUBTRACTED: enable_kv_cache_events / metrics_collector / dcp / pcp 透传
        # （L40-L44, L52-L56, L74-L75）—— 事件/统计/上下文并行旁路。
        # SUBTRACTED: 多组 single_type_managers 构造（L66-L78）—— 单组直接构造唯一的
        # FullAttentionManager；get_manager_for_kv_cache_spec 的类型分派属混合模型维度。
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = BlockPool(num_blocks, enable_caching, hash_block_size)

        self.single_type_managers = (
            FullAttentionManager(
                kv_cache_spec=kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=0,
            ),
        )

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L80 (get_num_blocks_to_allocate)
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """Get the number of blocks needed to be allocated for the request."""
        # SUBTRACTED: num_encoder_tokens / CrossAttentionManager 分支（L85, L115-L125）
        # —— 编码器-解码器交叉注意力的静态分配，decoder-only 主路径不触发。
        # SUBTRACTED: apply_admission_cap 透传 —— 见 single_type 侧删除说明。
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[i],
                total_computed_tokens,
                num_tokens_main_model,
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
        """Add the new computed blocks to the request for each group."""
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
    ) -> tuple[list[KVCacheBlock], ...]:
        """Allocate new blocks for the request to give it at least
        `num_tokens` token slots."""
        # SUBTRACTED: num_encoder_tokens / CrossAttentionManager 分支（L168, L190-L192）。
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

    # SUBTRACTED: get_num_common_prefix_blocks / remove_skipped_blocks / get_blocks /
    # new_step_starts（L221-L273）—— 级联前缀统计、滑窗跳块（全注意力 no-op）、
    # 逐步引流接口，非分页/命中核心控制流。


# SOURCE: vllm/v1/core/kv_cache_coordinator.py:L324 (UnitaryKVCacheCoordinator)
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models where all attention layers use full attention.
    """

    # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L331 (UnitaryKVCacheCoordinator.__init__)
    def __init__(
        self,
        kv_cache_spec: FullAttentionSpec,
        num_blocks: int,
        max_model_len: int,
        enable_caching: bool,
        hash_block_size: int,
    ):
        super().__init__(
            kv_cache_spec,
            num_blocks,
            max_model_len,
            enable_caching,
            hash_block_size,
        )
        # SUBTRACTED: dcp/pcp block_size 乘子（L358-L363）单卡为 1。
        self.kv_cache_spec = kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
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
            use_eagle=False,
            alignment_tokens=self.block_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size
