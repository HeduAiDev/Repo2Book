# SOURCE: vllm/v1/core/kv_cache_manager.py
# KVCacheManager 是另一章（KV cache / 分页）的主角。本章只需要它对调度器暴露的
# **接口契约**（allocate_slots 成功返回块/显存不足返回 None、free、get_blocks、
# get_computed_blocks、new_step_starts），因此这里给出一个按‘块数’计数的最小实现，
# 保留与真实 vllm/v1/core/kv_cache_manager.py 完全一致的方法签名与‘满则返回 None’语义，
# 让 schedule() 的抢占分支（allocate_slots→None→_preempt_request）能被真实驱动。
#
# SUBTRACTED: 真实的前缀缓存哈希匹配、块池、引用计数、cascade/common-prefix、
#   KVConnector 外部命中等全部分页细节（原 kv_cache_manager.py 全文）——
#   属 KV cache 章，dossier 将 KVConnector/前缀缓存细节列为 delete 批准项。
from __future__ import annotations

import math


# SOURCE: vllm/v1/core/kv_cache_manager.py:KVCacheBlocks
class KVCacheBlocks:
    """轻量块句柄：持有该请求当前已分配的块 id 列表。"""

    def __init__(self, block_ids: list[int]) -> None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:KVCacheBlocks
        self._block_ids = block_ids

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L54 get_block_ids
    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...]:
        return (list(self._block_ids),)


class KVCacheManager:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:__init__
    def __init__(self, num_gpu_blocks: int = 1 << 30, block_size: int = 16) -> None:
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.num_free_blocks = num_gpu_blocks
        # req_id -> 已分配块数
        self._blocks: dict[str, list[int]] = {}
        self._next_block_id = 0
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L158 empty_kv_cache_blocks
        self.empty_kv_cache_blocks = KVCacheBlocks([])

    # SOURCE: vllm/v1/core/kv_cache_manager.py: new_step_starts
    def new_step_starts(self) -> None:
        # SUBTRACTED: 每拍重置内部缓存命中统计（可观测性，dossier.delete 批准）
        pass

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L183 get_computed_blocks
    def get_computed_blocks(self, request) -> tuple[KVCacheBlocks, int]:
        # SUBTRACTED: 真实前缀缓存哈希匹配（KV cache 章）。精简版无命中：
        #   返回空块 + 0 命中 token，等价于关闭前缀缓存的行为。
        return self.empty_kv_cache_blocks, 0

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L225 allocate_slots
    def allocate_slots(
        self,
        request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        **kwargs,
    ) -> KVCacheBlocks | None:
        # 计算这次新增 token 需要的‘新块’数：按 (已算+新增) 跨越的块边界算增量。
        # SUBTRACTED: 真实分页里的前缀块复用/lookahead 细节，这里用块数守恒近似，
        #   保留‘所需块 > 空闲块则返回 None（触发抢占）’这一与调度耦合的关键语义。
        cur = self._blocks.get(request.request_id, [])
        total_tokens = (
            request.num_computed_tokens + num_new_tokens + num_lookahead_tokens
        )
        need_blocks = math.ceil(max(total_tokens, 1) / self.block_size)
        delta = need_blocks - len(cur)
        if delta > self.num_free_blocks:
            # The request cannot be scheduled: out of KV blocks.
            return None
        if delta > 0:
            new_ids = list(range(self._next_block_id, self._next_block_id + delta))
            self._next_block_id += delta
            self.num_free_blocks -= delta
            cur = cur + new_ids
            self._blocks[request.request_id] = cur
            return KVCacheBlocks(new_ids)
        self._blocks.setdefault(request.request_id, cur)
        return KVCacheBlocks([])

    # SOURCE: vllm/v1/core/kv_cache_manager.py: get_blocks
    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return KVCacheBlocks(self._blocks.get(request_id, []))

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L418 free
    def free(self, request) -> None:
        ids = self._blocks.pop(request.request_id, [])
        self.num_free_blocks += len(ids)

    # SOURCE: vllm/v1/core/kv_cache_manager.py: cache_blocks
    def cache_blocks(self, request, num_tokens: int) -> None:
        # SUBTRACTED: 真实的块哈希登记入前缀缓存（KV cache 章）；async 路径在
        #   _update_request_with_output 里调用它兑现占位，这里保留为 no-op 桩。
        pass
