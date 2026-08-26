# SOURCE: vllm/v1/core/kv_cache_manager.py
# KV cache 管理器——本章消费面：allocate_slots（schedule 两阶段的块预约）、
# free（抢占/完成归还）、cache_blocks（AsyncScheduler._update_request_with_output
# 的『乐观块转正式』调用位——参数 computed−placeholders 是不变式的化身）、
# get_blocks/new_step_starts/get_computed_blocks。
# 块池内景（哈希/前缀缓存/ref_cnt/水位）归 ch13/ch15；前缀命中在精简版恒 0。
from __future__ import annotations

from .request import Request


# SOURCE: vllm/v1/core/kv_cache_manager.py:L33 KVCacheBlocks（消费面镜像：
# blocks 元组 + get_block_ids）
class KVCacheBlocks:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L34-L50 __init__（块元组承载）
    def __init__(self, blocks: tuple[int, ...] | list[int] = ()):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L34-L50
        self.blocks: tuple[int, ...] = tuple(blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py get_block_ids
    def get_block_ids(self, allow_none: bool = False) -> tuple[int, ...]:
        # SOURCE: vllm/v1/core/kv_cache_manager.py get_block_ids
        return self.blocks

    # SOURCE: vllm/v1/core/kv_cache_manager.py __len__
    def __len__(self) -> int:
        # SOURCE: vllm/v1/core/kv_cache_manager.py __len__
        return len(self.blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py __bool__
    def __bool__(self) -> bool:
        # SOURCE: vllm/v1/core/kv_cache_manager.py __bool__
        return bool(self.blocks)


# SOURCE: vllm/v1/core/kv_cache_manager.py:L117 KVCacheManager（接口子集镜像）
class KVCacheManager:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L118-L135 __init__
    # SUBTRACTED: 真实签名（kv_cache_config/hash_block_size/coordinator/
    #   watermark/max_in_flight_tokens 装配，L118-L185）——块池内景归 ch13/ch15；
    #   HOST SEAM：num_gpu_blocks/block_size 直供。
    def __init__(
        self,
        num_gpu_blocks: int = 1 << 30,
        block_size: int = 16,
        max_model_len: int = 4096,
        enable_caching: bool = True,
    ) -> None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L118-L185（约束字段镜像）
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L185-L188 empty_kv_cache_blocks
        self.empty_kv_cache_blocks = KVCacheBlocks(())
        # req_id -> 已持块
        self.req_to_blocks: dict[str, tuple[int, ...]] = {}
        self._next_block = 0
        self._free_blocks: list[int] = []
        # ENGINE SEAM observation：cache_blocks 调用账（m7 测试对账位）
        self.cache_blocks_calls: list[tuple[str, int]] = []

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L344 allocate_slots（消费面：
    # None=放不下 → 调度侧抢占/放弃）
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        full_sequence_must_fit: bool = True,
        has_scheduled_reqs: bool = False,
    ) -> KVCacheBlocks | None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L344-L560（预约算术镜像：
        # 目标块数 = ceil((computed + new) / block_size)）
        # SUBTRACTED: lookahead/encoder/reserved_blocks 参数面与水位 headroom
        #   （ch11 立过 watermark；本章不展开准入深水）。
        total_tokens = request.num_computed_tokens + num_new_tokens
        if full_sequence_must_fit and total_tokens > self.num_gpu_blocks * self.block_size:
            return None
        num_blocks = (total_tokens + self.block_size - 1) // self.block_size
        # SUBTRACTED: 前缀重命中块的复用（L369-L451——ch15）；本章恒新分配。
        while len(self.req_to_blocks.get(request.request_id, ())) < num_blocks:
            if self._free_blocks:
                block_id = self._free_blocks.pop()
            elif self._next_block < self.num_gpu_blocks:
                block_id = self._next_block
                self._next_block += 1
            else:
                # 放不下：真实会走 preemption（调度侧抢占环处理），这里回 None。
                return None
            self.req_to_blocks.setdefault(request.request_id, tuple())
            blocks = list(self.req_to_blocks[request.request_id])
            blocks.append(block_id)
            self.req_to_blocks[request.request_id] = tuple(blocks)
        return KVCacheBlocks(self.req_to_blocks[request.request_id])

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L567 free
    def free(self, request: Request) -> None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L567-L590（归还块池；
        #   哈希保留与否归 ch15——精简版直接归还）
        blocks = self.req_to_blocks.pop(request.request_id, ())
        self._free_blocks.extend(blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L703 get_blocks
    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L703-L704
        return KVCacheBlocks(self.req_to_blocks.get(request_id, ()))

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L760 cache_blocks ——『乐观块
    # 转正式』调用位（m7：num_computed_tokens 含占位，差值=真实已算）
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks for.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L768-L770
        if self.enable_caching:
            # SUBTRACTED: coordinator.cache_blocks（哈希登记/CoW——ch15）。
            #   HOST SEAM：记调用账（精简版块池无哈希面，转正在此记账）。
            self.cache_blocks_calls.append((request.request_id, num_computed_tokens))

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L229 get_computed_blocks（前缀
    # 命中——本章恒 0：命中深水归 ch15）
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L249-L251 无缓存管理器快返
        return self.empty_kv_cache_blocks, 0, 0

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L876 new_step_starts
    def new_step_starts(self) -> None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L876-L884（步起清理；
        #   精简版块池无步态）
        pass
