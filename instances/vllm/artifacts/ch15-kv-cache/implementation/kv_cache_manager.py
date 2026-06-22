# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 对应 vllm/v1/core/kv_cache_manager.py：对 Scheduler 的门面。KVCacheBlocks 隐藏内部
# 结构；get_computed_blocks 查前缀命中；allocate_slots 三段式分配。
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from .kv_cache_coordinator import UnitaryKVCacheCoordinator
from .kv_cache_utils import KVCacheBlock
from .request import FullAttentionSpec, Request


# SOURCE: vllm/v1/core/kv_cache_manager.py:L21 (class KVCacheBlocks)
@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group and the j-th block of
    tokens.
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L44 (__add__)
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L65 (get_block_ids)
    def get_block_ids(self) -> tuple[list[int], ...]:
        """Converts the KVCacheBlocks instance to block_ids, grouped by KV
        cache group."""
        # SUBTRACTED: allow_none 重载（L53-L63, L78-L79）—— 仅为返回 None 的便捷形态，
        # 不改变 block_id 提取逻辑。
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    # SUBTRACTED: get_unhashed_block_ids / get_unhashed_block_ids_all_groups /
    # new_empty（L82-L103）—— worker 侧未哈希块引流与空实例工厂，非分页/命中核心。


# SOURCE: vllm/v1/core/kv_cache_manager.py:L106 (class KVCacheManager)
class KVCacheManager:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L107 (KVCacheManager.__init__)
    def __init__(
        self,
        kv_cache_spec: FullAttentionSpec,
        num_blocks: int,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
    ) -> None:
        # SUBTRACTED: use_eagle / log_stats / metrics_collector / prefix_cache_stats /
        # enable_kv_cache_events / dcp / pcp（L114-L135, L143-L147）—— 投机解码、统计、
        # 事件、上下文并行旁路。
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        # SUBTRACTED: get_kv_cache_coordinator 工厂分派（L137-L148）—— 单组全注意力 +
        # 前缀缓存开启时它返回 UnitaryKVCacheCoordinator；精简版直接构造。
        self.coordinator = UnitaryKVCacheCoordinator(
            kv_cache_spec=kv_cache_spec,
            num_blocks=num_blocks,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
        )
        self.num_kv_cache_groups = 1
        self.block_pool = self.coordinator.block_pool

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L183 (get_computed_blocks)
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full."""
        # We skip finding the prefix cache hit when prefix caching is disabled
        # or the request is marked as skipping kv cache read.
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        # SUBTRACTED: log_stats 记录前缀命中统计（L215-L221）—— 统计旁路。

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L225 (allocate_slots)
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        The allocation has three stages:
        - Check if we have sufficient free blocks (return None if not).
        - Handle prefix tokens: touch the new computed (prefix-hit) blocks.
        - Allocate new blocks for tokens to be computed.
        """
        # SUBTRACTED: num_lookahead_tokens / num_external_computed_tokens /
        # delay_cache_blocks / num_encoder_tokens / full_sequence_must_fit 参数与分支
        # （L231-L235, L312-L318, L335-L349）—— 投机解码、KV connector/P-D、编码器输入、
        # SWA 准入闸；本地全注意力主路径均为 0 / no-op。
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits.
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(num_local_computed_tokens, self.max_model_len)

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(num_tokens_main_model, self.max_model_len)

        # SUBTRACTED: coordinator.remove_skipped_blocks（L362-L364）—— 滑窗跳块释放，
        # 单组全注意力 num_skipped_tokens==0 时该调用为 no-op。

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            total_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        if new_computed_block_list is not self.empty_kv_cache_blocks.blocks:
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=0,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
        )

        # SUBTRACTED: not enable_caching or delay_cache_blocks 早退（L402-L403）——
        # delay_cache_blocks 属 P/D 路径；enable_caching=False 时本方法仍可调，但本章
        # 主路径恒开启前缀缓存，故直接进入缓存满块。
        if not self.enable_caching:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We cap the number at `request.num_tokens` to ensure only
        # "finalized" tokens are cached (excluding e.g. unverified draft tokens).
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L418 (free)
    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled."""
        self.coordinator.free(request.request_id)

    # SUBTRACTED: remove_skipped_blocks / evict_blocks / reset_prefix_cache /
    # get_num_common_prefix_blocks / make_prefix_cache_stats / take_new_block_ids /
    # new_step_starts / usage（L428+）—— 滑窗、connector、运维统计、逐步引流旁路接口。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L526 (create_kv_cache_blocks)
    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
