# vllm_ascend/core/single_type_kv_cache_manager.py —— subtract-only 精简版
#
# 昇腾对 KV 管理的「唯一新增」：CompressAttentionManager（压缩 MLA 的 block 管理）+
# get_manager_for_kv_cache_spec（重映射工厂）。FullAttentionManager / BlockPool /
# SingleTypeKVCacheManager / spec_manager_map 全部原样复用 vLLM（只 import，不改）。
#
# 本文件几乎逐字保留——它正是「复用 vs 特化」边界的核心，没有可安全删除的复刻段。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


# SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L28
class CompressAttentionManager(FullAttentionManager):
    # SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L29
    def __init__(self, kv_cache_spec: MLAAttentionSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block

    # SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L34
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        # assert isinstance(self.kv_cache_spec, (CompressAttentionSpec, C4IndexerSpec))

        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio

        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )

    # SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L59
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.
        3. (Optional) For KV connectors, allocate new blocks for external computed
            tokens (if any).
        """

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_total_computed_tokens //= self.compress_ratio
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), "Computed blocks should be empty when prefix caching is disabled"

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    # SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L129
    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.
        """
        num_tokens //= self.compress_ratio
        ## TODO: check spec decode
        num_tokens_main_model //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    # SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L156
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        alignment_tokens: int | None = None,
    ) -> None:
        """
        Cache the blocks for the request.
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // (self.block_size * self.compress_ratio)

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size * self.compress_ratio,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        self.num_cached_block[request.request_id] = num_full_blocks

    @classmethod
    # SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L188
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        # SUBTRACTED: L201-L205 被注释掉的 isinstance dead assert（原文即注释，无运行作用）。
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        logical_block_size = block_size * kv_cache_spec.compress_ratio
        logical_block_hashes = BlockHashListWithBlockSize(block_hashes, block_size, logical_block_size)
        max_num_blocks = max_length // logical_block_size
        for block_hash in itertools.islice(logical_block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()

        while (
            logical_block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * logical_block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks


# SOURCE: vllm_ascend/core/single_type_kv_cache_manager.py:L236
def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """Build the per-spec KV cache manager.

    For DSv4 / DSA path (``MLAAttentionSpec`` with ``compress_ratio>1``), align
    the runtime admission gate with the startup pool-sizing bound the same way
    vLLM PR #40946 does for ``SlidingWindowSpec`` / ``ChunkedLocalAttentionSpec``.
    Without this cap, an admitted request can demand more blocks than the pool
    was sized to back, and ``allocate_slots`` silently returns ``None`` from
    the ``full_sequence_must_fit`` branch, leaving long-input requests stuck
    in the waiting queue (see vLLM issue #40863).
    """
    manager_class = spec_manager_map[type(kv_cache_spec)]
    if isinstance(kv_cache_spec, MLAAttentionSpec) and kv_cache_spec.compress_ratio > 1:
        manager_class = CompressAttentionManager
        if max_model_len is not None:
            # Compressed-MLA peak in blocks: ceil(max_model_len/compress/block).
            compress_ratio = kv_cache_spec.compress_ratio
            block_size = kv_cache_spec.block_size
            max_compressed_tokens = max_model_len // compress_ratio
            kwargs["max_admission_blocks_per_request"] = cdiv(max_compressed_tokens, block_size) + 1
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        # Replicate the upstream PR #40946 cap setting for recycling specs.
        # We override the vLLM factory above, so the upstream block that does
        # this lives in dead code (never reached); without re-applying it here
        # SlidingWindow / ChunkedLocalAttention groups have no cap and
        # ``full_sequence_must_fit`` admission reserves the full ``max_model_len``
        # worth of blocks per request, exhausting the pool at cc>=2 (vLLM #40863).
        if max_num_batched_tokens is not None and max_model_len is not None:
            kwargs["max_admission_blocks_per_request"] = kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
