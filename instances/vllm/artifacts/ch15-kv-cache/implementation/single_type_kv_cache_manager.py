# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 对应 vllm/v1/core/single_type_kv_cache_manager.py：单一注意力类型的块账本。
# 本章锁定单组全注意力主路径：保留 SingleTypeKVCacheManager 基类 +
# FullAttentionManager。SlidingWindow/ChunkedLocal/Mamba/CrossAttention/Sink
# 等其他子类属混合模型维度，整体 SUBTRACTED。
import itertools
from collections import defaultdict
from collections.abc import Sequence

from .block_pool import BlockPool
from .kv_cache_utils import BlockHash, KVCacheBlock
from .request import FullAttentionSpec, Request


def cdiv(a: int, b: int) -> int:
    # SOURCE: vllm/utils/math_utils.py (cdiv — 向上取整除)
    return -(a // -b)


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L30 (SingleTypeKVCacheManager)
class SingleTypeKVCacheManager:
    """
    A manager that handles the kv cache management logic of one specific type
    of attention layer.
    """
    # SUBTRACTED: ABC / @abstractmethod 抽象基类语义 —— 单组全注意力只保留一个具体
    # 子类（FullAttentionManager），不再需要抽象约束。原 L30, L320, L336。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L36 (__init__)
    def __init__(
        self,
        kv_cache_spec: FullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
    ) -> None:
        # SUBTRACTED: dcp_world_size / pcp_world_size 乘子（L42-L43, L60-L63）—— 上下文
        # 并行，单卡为 1，block_size 乘子无效果。
        # SUBTRACTED: max_admission_blocks_per_request（L44, L67）—— SWA/chunked-local
        # 的回收感知准入上限，全注意力恒 None 无效果。
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}.
        # Only used to track the RUNNING requests.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L84
    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L85
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L88
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """Get the number of blocks needed to be allocated for the request."""
        # SUBTRACTED: apply_admission_cap / _max_admission_blocks_per_request
        # 子句（L95, L120-L132）—— SWA/chunked-local 回收感知准入闸，全注意力恒 no-op。
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped (full attention), this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L169
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Add the new computed blocks to the request: touch them so they
        won't be evicted, then add them to req_to_blocks."""
        # SUBTRACTED: num_external_computed_tokens 外部块分配（L181, L201-L214,
        # L233-L240）—— KV connector / P-D 路径，本地前缀缓存主流程恒为 0。
        # SUBTRACTED: num_skipped_blocks 与 null 填充（L204-L214, L224-L225）—— 滑窗跳块，
        # 全注意力恒 0。
        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Add the computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits are already cached; mark them so cache_blocks() will
        # not try to re-cache blocks that already have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L242
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """Allocate new blocks for the request to give it at least
        `num_tokens` token slots."""
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            self.new_block_ids.extend(b.block_id for b in new_blocks)
            return new_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L277
    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """Cache the full blocks for the request."""
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L303
    def free(self, request_id: str) -> None:
        """Free the blocks for the request."""
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L428 (get_num_skipped_tokens)
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Get the number of tokens that will be skipped for attention
        computation. The default behavior is to not skip any tokens."""
        # 全注意力从不跳过块，基类默认返回 0，相关逻辑空转。
        return 0

    # SUBTRACTED: remove_skipped_blocks（L385-L426）的滑窗回收实现 —— 全注意力
    # num_skipped_tokens==0 时该函数提前 return，等价 no-op；故精简版不需要它，
    # KVCacheManager 侧也已 SUBTRACTED 对应调用。原 vllm/v1/core/single_type_kv_cache_manager.py:L385。
    # SUBTRACTED: get_num_common_prefix_blocks / take_new_block_ids / new_step_starts
    # —— 级联前缀统计与逐步块 ID 引流接口，非分页/命中核心控制流。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L446 (FullAttentionManager)
class FullAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L447 (find_longest_cache_hit)
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: FullAttentionSpec,
        use_eagle: bool,
        alignment_tokens: int,
    ) -> tuple[list[KVCacheBlock], ...]:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L447 (find_longest_cache_hit)
        # SUBTRACTED: dcp_world_size / pcp_world_size 乘子（L457-L458, L470-L471）单组为 1。
        assert isinstance(kv_cache_spec, FullAttentionSpec), (
            "FullAttentionManager can only be used for full attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_block, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        # SUBTRACTED: use_eagle 丢尾块（L484-L487）—— 投机解码草稿头专用。
        # SUBTRACTED: alignment_tokens 对齐裁剪（L488-L493）—— 单组下 block_size ==
        # alignment_tokens，while 条件恒 False 不触发。
        return computed_blocks

    # SUBTRACTED: get_num_common_prefix_blocks（L496-...）—— 级联前缀统计，非命中核心。
