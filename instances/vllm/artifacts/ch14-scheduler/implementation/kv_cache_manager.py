# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版。本章只关心 allocate_slots 的*可观察契约*：
#   - 有空闲块 → 返回 new_blocks（非 None）
#   - 块池耗尽 → 返回 None（这是触发抢占循环的因果信号）
# 真实 KVCacheManager 含前缀缓存命中、块哈希、滑窗、混合 mamba 等大量机制，
# 本章不涉及，仅保留 allocate_slots/free 的块计数语义以驱动抢占/回流。
from request import Request


# SOURCE: vllm/v1/core/kv_cache_manager.py (class KVCacheManager) —— 精简到块计数语义
# SUBTRACTED: 前缀缓存（get_computed_blocks / cache_blocks 的命中逻辑）、块哈希、
# 滑动窗口、混合 mamba 块对齐、num_lookahead_tokens 的实际块预留等 —— 全属 KV 缓存
# 子系统（后续章节），与本章抢占·回流主线正交。这里只保留「分配失败返回 None /
# free 归还块」这一对本章因果可观察的契约。原 vllm/v1/core/kv_cache_manager.py。
class KVCacheManager:
    # SOURCE: vllm/v1/core/kv_cache_manager.py (KVCacheManager.__init__ — 精简到块计数)
    def __init__(self, block_capacity: int, block_size: int = 1):
        self.block_capacity = block_capacity
        self.block_size = block_size
        self.num_free_blocks = block_capacity
        # request_id -> 已分配块数
        self._allocated: dict[str, int] = {}

    # SOURCE: vllm/v1/core/kv_cache_manager.py (allocate_slots) —— must_keep
    def allocate_slots(self, request: Request, num_new_tokens: int,
                       num_lookahead_tokens: int = 0):
        # SUBTRACTED: num_lookahead_tokens 的 spec-decode 额外块预留计算 —— 精简版
        # 按 token 数粗算块需求。原 allocate_slots 内的 lookahead 块数推导。
        # 为本拍 num_new_tokens 个新 token 申请增量块（decode 逐步增长，prefill 一次性）。
        delta = max(1, (num_new_tokens + self.block_size - 1) // self.block_size)
        if delta > self.num_free_blocks:
            # The request cannot be scheduled. (块池耗尽 → 触发抢占)
            return None
        self.num_free_blocks -= delta
        self._allocated[request.request_id] = (
            self._allocated.get(request.request_id, 0) + delta
        )
        # 返回非 None 的「new_blocks」占位（真实为 KVCacheBlocks 对象）
        return self._allocated[request.request_id]

    # SOURCE: vllm/v1/core/kv_cache_manager.py (free)
    def free(self, request: Request) -> None:
        n = self._allocated.pop(request.request_id, 0)
        self.num_free_blocks += n

    # SOURCE: vllm/v1/core/kv_cache_manager.py (get_blocks)
    def get_blocks(self, request_id: str):
        return self._allocated.get(request_id, 0)

    # SOURCE: vllm/v1/core/kv_cache_manager.py (cache_blocks) —— AsyncScheduler 用
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        # SUBTRACTED: 实际把已算 token 对应块登记进前缀缓存 —— 属 KV 缓存子系统。
        # 精简版保留调用点（AsyncScheduler 只对 RUNNING 请求 cache_blocks）作 no-op。
        return None
