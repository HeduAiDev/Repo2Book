"""ch13 分页 KV —— 单元+契约测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测行为**
（锚点 = vllm/... 行号，基线 v0.27.1 现核，非 v2 资产的 v0.21.0 旧行号）。
本章精简版跑 enable_prefix_caching=False 支（cache_config 的正交开关，
NoPrefixCache 协调器是源码原生路径 kv_cache_coordinator.py:L864-L876）。

行为清单（按 dossier.mechanisms 对账）：
- m1/m2 池的构造：blocks 数组一次预构、null_block 从队头 popleft 占 block_id=0
  且 is_null=True、ref_cnt 不维护（block_pool.py:L175-L191）
- m3 侵入式自由队列：fake head/tail 哨兵、popleft/popleft_n 队头取、
  remove O(1) 中间摘、append_n/prepend_n 归还（kv_cache_utils.py:L184-L413）
- m4 引用计数生命周期：get_new_blocks +1 / free_blocks −1 归零入队 /
  touch +1 出队救回（block_pool.py:L647-L742）
- m5 需块预测：cdiv 主算术 + running fast-path + 可驱逐命中块计数
  （single_type_kv_cache_manager.py:L144-L230）
- m6 allocate_slots 三段式：容量检查（不够 None）→ 挂命中块 → 分新块 →
  （caching 关）早退；None = ch11 抢占唯一触发信号的内因
  （kv_cache_manager.py:L344-L565）
- m7 block_id 跨进程契约：新请求全量块表 NewRequestData.from_request /
  在跑请求增量 new_block_ids（get_block_ids(allow_none=True) 空则 None）/
  worker block_ids.extend + block_table.append_row
  （scheduler.py:L1144-L1149 / L1451-L1453 / gpu_model_runner.py:L1441-L1474）
- m8 新块清零：take_new_block_ids 每步排干 → _get_new_block_ids_to_zero →
  KVBlockZeroer 清零（防陈旧 NaN/data）
  （scheduler.py:L1260-L1272 / gpu_model_runner.py:L1219-L1222 / worker/utils.py）
- m9 槽位恒等式：slot = block_table[req][pos//block_size]*block_size +
  pos%block_size；尾部 PAD（block_table.py:L379-L442）
- m10 页物理形状：real_page_size_bytes = 2×block_size×kv_heads×head_dim×
  dtype 字节；num_blocks = numel // page_size_bytes
  （kv_cache_interface.py:L184-L226 / gpu_model_runner.py:L7400-L7413）
- m11 decode 稳态长块：每 block_size 个 token 多要一块（RUNNING fast-path）
- m12 终局逆序 free："tail blocks are freed first" → 逆序归还自由队列
  （single_type_kv_cache_manager.py:L519-L527 / block_pool.py:L719-L742）
- m13 DEFAULT_BLOCK_SIZE=16（vllm/config/cache.py:L43-L51）
- m15 commit_block_table 每拍只拷活跃行（block_table.py:L213-L214）
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.block_pool import BlockPool  # noqa: E402
from implementation.block_table import (  # noqa: E402
    PAD_SLOT_ID,
    BlockTable,
    SlotMappingMode,
)
from implementation.cache import CacheConfig  # noqa: E402
from implementation.gpu_input_batch import CachedRequestState, InputBatch  # noqa: E402
from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.kv_cache_utils import (  # noqa: E402
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)
from implementation.output import NewRequestData, SchedulerOutput  # noqa: E402
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.worker_utils import KVBlockZeroer  # noqa: E402


# --------------------------------------------------------------------------- #
# 构造辅助：真实装配的最小镜像（单组全注意力 + enable_prefix_caching=False）
# --------------------------------------------------------------------------- #

BLOCK_SIZE = 16
LAYER = "model.layers.0.self_attn.attn"


def make_spec(
    block_size: int = BLOCK_SIZE,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )


def make_config(
    num_blocks: int = 10,
    spec: FullAttentionSpec | None = None,
    num_groups: int = 1,
) -> KVCacheConfig:
    spec = spec or make_spec()
    if num_groups == 2:
        # 混合精度两组（fp16 + fp32）：needs_kv_cache_zeroing=True 的合法构造
        spec_b = make_spec(dtype=torch.float32)
        layer_b = "model.layers.1.self_attn.attn"
        groups = [
            KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec),
            KVCacheGroupSpec(layer_names=[layer_b], kv_cache_spec=spec_b),
        ]
        tensors = [
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[LAYER]),
            KVCacheTensor(size=num_blocks * spec_b.page_size_bytes, shared_by=[layer_b]),
        ]
    else:
        groups = [KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec)]
        tensors = [
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[LAYER])
        ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def make_manager(
    num_gpu_blocks: int = 10,
    max_model_len: int = 4096,
    config: KVCacheConfig | None = None,
) -> KVCacheManager:
    config = config or make_config(num_blocks=num_gpu_blocks)
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=max_model_len,
        scheduler_block_size=config.kv_cache_groups[0].kv_cache_spec.block_size,
        hash_block_size=config.kv_cache_groups[0].kv_cache_spec.block_size,
        enable_caching=False,  # 本章精简版跑正交开关的 False 支（→ ch15）
    )


def make_request(
    req_id: str = "req-0",
    prompt_tokens: int = 100,
    num_computed_tokens: int = 0,
) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(prompt_tokens)))
    req.status = RequestStatus.WAITING
    req.num_computed_tokens = num_computed_tokens
    return req


def make_runner(
    num_blocks: int = 10,
    max_num_reqs: int = 4,
    max_blocks_per_req: int = 8,
    max_num_batched_tokens: int = 128,
) -> GPUModelRunner:
    config = make_config(num_blocks=num_blocks)
    spec = config.kv_cache_groups[0].kv_cache_spec
    return GPUModelRunner(
        kv_cache_config=config,
        block_size=spec.block_size,
        max_num_reqs=max_num_reqs,
        max_blocks_per_req=max_blocks_per_req,
        max_num_batched_tokens=max_num_batched_tokens,
        device=torch.device("cpu"),  # HOST SEAM：CPU host 无 CUDA（容器内为真 GPU）
    )


# --------------------------------------------------------------------------- #
# A. 池的构造与 null_block（m1/m2，block_pool.py:L162-L191）
# --------------------------------------------------------------------------- #


class TestPoolConstruction:
    def test_blocks_preallocated_and_null_block_takes_id_zero(self):
        # SOURCE 行为：blocks 数组一次预构（L175-L177）；null_block 从队头
        # popleft 占 block_id=0 并置 is_null（L190-L191）。
        pool = BlockPool(
            num_gpu_blocks=10, enable_caching=False, hash_block_size=BLOCK_SIZE
        )
        assert len(pool.blocks) == 10
        assert [b.block_id for b in pool.blocks] == list(range(10))
        assert pool.null_block is pool.blocks[0]
        assert pool.null_block.is_null is True
        # null_block 的 ref_cnt 不维护（L188-L189 注释原话 "not maintained"）
        assert pool.null_block.ref_cnt == 0

    def test_free_blocks_count_excludes_null(self):
        pool = BlockPool(
            num_gpu_blocks=10, enable_caching=False, hash_block_size=BLOCK_SIZE
        )
        # null_block 占掉一块：空闲 = 10 - 1 = 9
        assert pool.get_num_free_blocks() == 9

    def test_usage_subtracts_null_block(self):
        # SOURCE 行为：get_usage 减 1 记 null 块（L814-L818）
        pool = BlockPool(
            num_gpu_blocks=11, enable_caching=False, hash_block_size=BLOCK_SIZE
        )
        assert pool.get_usage() == 0.0
        pool.get_new_blocks(5)
        # 10 可用中被占 5 → 0.5
        assert pool.get_usage() == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# B. 侵入式自由队列（m3，kv_cache_utils.py:L184-L413）
# --------------------------------------------------------------------------- #


class TestFreeQueue:
    def make_queue(self, n: int = 5) -> tuple[FreeKVCacheBlockQueue, list]:
        blocks = [KVCacheBlock(idx) for idx in range(n)]
        return FreeKVCacheBlockQueue(blocks), blocks

    def test_init_links_consecutive_and_sentinels(self):
        # SOURCE 行为：__init__ 串起相邻块（L210-L214）、fake head/tail 哨兵
        # 挂两端（L222-L230）——哨兵 block_id=-1、真实块 prev/next 齐备。
        q, blocks = self.make_queue(3)
        assert q.fake_free_list_head.block_id == -1
        assert q.fake_free_list_tail.block_id == -1
        assert q.fake_free_list_head.next_free_block is blocks[0]
        assert blocks[0].prev_free_block is q.fake_free_list_head
        assert blocks[-1].next_free_block is q.fake_free_list_tail
        assert q.num_free_blocks == 3

    def test_popleft_takes_head_and_nulls_pointers(self):
        q, blocks = self.make_queue(3)
        first = q.popleft()
        assert first is blocks[0]
        assert q.num_free_blocks == 2
        # 弹出块的侵入式指针被清（L268）
        assert first.prev_free_block is None and first.next_free_block is None
        # fake head 直连第二块（L264-L265）
        assert q.fake_free_list_head.next_free_block is blocks[1]
        assert blocks[1].prev_free_block is q.fake_free_list_head

    def test_popleft_empty_raises(self):
        # SOURCE 行为：队空 popleft 抛 ValueError（L246-L250）
        q, _ = self.make_queue(1)
        q.popleft()
        with pytest.raises(ValueError):
            q.popleft()

    def test_popleft_n_batch_and_zero(self):
        # SOURCE 行为：popleft_n 批量取队头（L273-L304）；n==0 早退（L282-L283）
        q, blocks = self.make_queue(5)
        assert q.popleft_n(0) == []
        ret = q.popleft_n(2)
        assert [b.block_id for b in ret] == [0, 1]
        assert q.num_free_blocks == 3
        # 摘除后 fake head 连到 blocks[2]
        assert q.fake_free_list_head.next_free_block is blocks[2]

    def test_remove_o1_from_middle_keeps_chain(self):
        # SOURCE 行为：remove 从中间摘块、前后互连（L306-L324）——touch 救回
        # 命中块的关键原语（ch15 前置）
        q, blocks = self.make_queue(5)
        q.remove(blocks[2])
        assert q.num_free_blocks == 4
        assert blocks[1].next_free_block is blocks[3]
        assert blocks[3].prev_free_block is blocks[1]
        # 被摘块指针清零
        assert blocks[2].prev_free_block is None and blocks[2].next_free_block is None
        # 队序保持 [0,1,3,4]
        assert [b.block_id for b in q.get_all_free_blocks()] == [0, 1, 3, 4]

    def test_append_n_returns_to_tail_in_order(self):
        # SOURCE 行为：append_n 挂队尾、保持传入序（L370-L393）——free_blocks
        # 归还路径（m12 逆序语义的载体）；队头仍是最旧未用块
        q, blocks = self.make_queue(5)
        taken = q.popleft_n(2)
        q.append_n(taken)
        assert q.num_free_blocks == 5
        # 归还块挂到队尾（LRU 序：最近归还最后被再分配）
        assert [b.block_id for b in q.get_all_free_blocks()] == [2, 3, 4, 0, 1]

    def test_prepend_n_puts_at_front(self):
        # SOURCE 行为：prepend_n 挂队头（L349-L368）——free_blocks 劈分时
        # 无哈希块先驱逐的挂点（ch15 LRU 双不变量；caching 关时不触发）
        q, blocks = self.make_queue(5)
        taken = q.popleft_n(2)
        q.prepend_n(taken)
        assert [b.block_id for b in q.get_all_free_blocks()] == [0, 1, 2, 3, 4]
        # 再取队头应拿到刚 prepend 的 [0,1]
        assert [b.block_id for b in q.popleft_n(2)] == [0, 1]


# --------------------------------------------------------------------------- #
# C. 引用计数生命周期（m4，block_pool.py:L647-L742）
# --------------------------------------------------------------------------- #


class TestRefcountLifecycle:
    def make_pool(self, n: int = 10) -> BlockPool:
        return BlockPool(
            num_gpu_blocks=n, enable_caching=False, hash_block_size=BLOCK_SIZE
        )

    def test_get_new_blocks_plus_one_and_fifo_ids(self):
        # SOURCE 行为：popleft_n 取最旧空闲块、每块 ref_cnt+1（L661-L676）；
        # 块 id 从 1 起（0 被 null_block 占）
        pool = self.make_pool()
        blocks = pool.get_new_blocks(3)
        assert [b.block_id for b in blocks] == [1, 2, 3]
        assert all(b.ref_cnt == 1 for b in blocks)
        assert pool.get_num_free_blocks() == 6

    def test_get_new_blocks_over_capacity_raises(self):
        # SOURCE 行为：超额取块抛 ValueError（L658-L659）
        pool = self.make_pool(4)  # 3 usable
        with pytest.raises(ValueError):
            pool.get_new_blocks(4)

    def test_free_blocks_decref_zero_returns_to_tail(self):
        # SOURCE 行为：free_blocks 逐块 −1，归零且非 null 回自由队列
        # （L730-L742）；caching 关时全部 append_n 到队尾（L733-L734 注释）
        pool = self.make_pool()
        blocks = pool.get_new_blocks(3)
        pool.free_blocks(blocks)
        assert all(b.ref_cnt == 0 for b in blocks)
        assert pool.get_num_free_blocks() == 9
        # 归还序保持 free 传入序（append_n 按序挂尾）
        assert [b.block_id for b in pool.free_block_queue.get_all_free_blocks()[-3:]] == [1, 2, 3]

    def test_free_blocks_null_block_never_returns(self):
        # SOURCE 行为：is_null 的块即使 ref_cnt 归零也不回自由队列（L732）
        pool = self.make_pool()
        pool.free_blocks([pool.null_block])
        assert pool.get_num_free_blocks() == 9
        assert pool.null_block.next_free_block is None

    def test_touch_rescues_eviction_candidate(self):
        # SOURCE 行为：ref_cnt==0 且非 null 的块被 touch 时先从自由队列
        # remove 再 +1（L710-L715）——前缀命中救回驱逐候选（ch15 场景）
        pool = self.make_pool()
        blocks = pool.get_new_blocks(2)
        pool.free_blocks(blocks)  # 回自由队列，成为驱逐候选
        assert pool.get_num_free_blocks() == 9
        pool.touch(blocks)
        assert all(b.ref_cnt == 1 for b in blocks)
        # 出队：空闲减 2
        assert pool.get_num_free_blocks() == 7

    def test_shared_block_two_owners_freed_once_stays(self):
        # 引用计数共享语义：两个主人各 touch/持有一块，free 一次后仍被占用
        pool = self.make_pool()
        (block,) = pool.get_new_blocks(1)
        pool.touch([block])  # 第二个主人
        assert block.ref_cnt == 2
        pool.free_blocks([block])
        assert block.ref_cnt == 1
        assert pool.get_num_free_blocks() == 8  # 未归零未回池


# --------------------------------------------------------------------------- #
# D. 需块预测 get_num_blocks_to_allocate（m5，single_type:L144-L230）
# --------------------------------------------------------------------------- #


class TestNumBlocksToAllocate:
    def make_mgr(self, n: int = 10):
        mgr = KVCacheManager(
            kv_cache_config=make_config(num_blocks=n),
            max_model_len=4096,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            enable_caching=False,
        )
        single = mgr.coordinator.single_type_managers[0]
        return mgr, single

    def test_cdiv_main_arithmetic(self):
        # SOURCE 行为：num_required_blocks = cdiv(num_tokens, block_size)
        # （L178）——100 token / 16 = 7 块（worked example 的算术底座）
        _, single = self.make_mgr()
        n = single.get_num_blocks_to_allocate(
            request_id="req-0",
            num_tokens=100,
            new_computed_blocks=[],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=100,
        )
        assert n == 7

    def test_running_fastpath_difference(self):
        # SOURCE 行为：running 请求走 fast-path 差值 max(need − held, 0)
        # （L194-L200）；spec decode 拒绝草稿时 need 可小于 held → 0
        mgr, single = self.make_mgr()
        req = make_request("req-0", 100)
        mgr.allocate_slots(req, 100)
        req.status = RequestStatus.RUNNING
        single.num_cached_block["req-0"] = 7  # running 的入口账位
        # 已持 7 块再要 100 token 覆盖 → 0 新块
        assert (
            single.get_num_blocks_to_allocate(
                request_id="req-0",
                num_tokens=100,
                new_computed_blocks=[],
                total_computed_tokens=100,
                num_local_computed_tokens=100,
                num_tokens_main_model=100,
            )
            == 0
        )
        # 长到 113 token → cdiv=8，差 1 块（m11：每 block_size 个 token 多要一块）
        assert (
            single.get_num_blocks_to_allocate(
                request_id="req-0",
                num_tokens=113,
                new_computed_blocks=[],
                total_computed_tokens=113,
                num_local_computed_tokens=113,
                num_tokens_main_model=113,
            )
            == 1
        )
        # spec 拒绝草稿回退：目标反而更小 → 钳 0（L197-L199 NOTE）
        assert (
            single.get_num_blocks_to_allocate(
                request_id="req-0",
                num_tokens=64,
                new_computed_blocks=[],
                total_computed_tokens=100,
                num_local_computed_tokens=100,
                num_tokens_main_model=64,
            )
            == 0
        )

    def test_evictable_hit_blocks_counted(self):
        # SOURCE 行为：命中块里还躺在自由队列（ref_cnt==0、非 null）的
        # 可驱逐块也要数进容量检查（L220-L225）
        mgr, single = self.make_mgr()
        # 造一块 ref_cnt==0 的"可驱逐命中块"
        (free_block,) = mgr.block_pool.get_new_blocks(1)
        mgr.block_pool.free_blocks([free_block])
        n = single.get_num_blocks_to_allocate(
            request_id="req-new",
            num_tokens=32,
            new_computed_blocks=[free_block],
            total_computed_tokens=16,
            num_local_computed_tokens=16,
            num_tokens_main_model=32,
        )
        # num_new = max(cdiv(32,16)=2 − max(0, len(hit)=1), 0) = 1；evictable = 1
        assert n == 2


# --------------------------------------------------------------------------- #
# E. allocate_slots 三段式（m6，kv_cache_manager.py:L344-L565）
# --------------------------------------------------------------------------- #


class TestAllocateSlots:
    def test_basic_allocation_seven_blocks(self):
        # worked example：prompt 100 token / block_size 16 → 7 块（112 槽）
        mgr = make_manager(num_gpu_blocks=10)
        req = make_request("req-0", 100)
        result = mgr.allocate_slots(req, 100)
        assert result is not None
        block_ids = result.get_block_ids()
        assert block_ids == ([1, 2, 3, 4, 5, 6, 7],)  # 0 号是 null_block
        assert mgr.block_pool.get_num_free_blocks() == 2
        # 挂账：逻辑块表加长一段
        single = mgr.coordinator.single_type_managers[0]
        assert [b.block_id for b in single.req_to_blocks["req-0"]] == [1, 2, 3, 4, 5, 6, 7]

    def test_insufficient_free_returns_none_no_partial_state(self):
        # SOURCE 行为：需块 > 空闲 → return None（L510-L527）；且不留半截账
        mgr = make_manager(num_gpu_blocks=10)  # 9 usable
        req0 = make_request("req-0", 128)  # 8 块
        assert mgr.allocate_slots(req0, 128) is not None
        # 只剩 1 块：第二条 128-token 请求要 8 块 → None
        req1 = make_request("req-1", 128)
        assert mgr.allocate_slots(req1, 128) is None
        single = mgr.coordinator.single_type_managers[0]
        assert "req-1" not in single.req_to_blocks
        assert mgr.block_pool.get_num_free_blocks() == 1

    def test_decode_growth_one_block_per_block_size(self):
        # m11：decode 每多 block_size 个 token 多要一块（fast-path 差值）
        mgr = make_manager(num_gpu_blocks=10)
        req = make_request("req-0", 32)  # 2 块
        first = mgr.allocate_slots(req, 32)
        assert first.get_block_ids() == ([1, 2],)
        req.num_computed_tokens = 32
        req.status = RequestStatus.RUNNING
        # 生成 16 个新 token → 需 cdiv(48,16)=3 块，已持 2 → 增量 1 块
        second = mgr.allocate_slots(req, 16)
        assert second.get_block_ids() == ([3],)
        single = mgr.coordinator.single_type_managers[0]
        assert [b.block_id for b in single.req_to_blocks["req-0"]] == [1, 2, 3]

    def test_zero_new_tokens_raises(self):
        # SOURCE 行为：无外部 token 时 num_new_tokens==0 抛 ValueError
        # （L442-L446）
        mgr = make_manager()
        req = make_request()
        with pytest.raises(ValueError):
            mgr.allocate_slots(req, 0)

    def test_running_request_none_is_preemption_signal(self):
        # ch11 抢占唯一触发信号的内因：RUNNING 长大时池已干 → None
        mgr = make_manager(num_gpu_blocks=3)  # 2 usable
        req = make_request("req-0", 32)
        assert mgr.allocate_slots(req, 32) is not None  # 占满 2 块
        req.num_computed_tokens = 32
        req.status = RequestStatus.RUNNING
        assert mgr.allocate_slots(req, 16) is None  # 再要 1 块没有 → None

    def test_caching_disabled_early_return_before_cache_blocks(self):
        # SOURCE 行为：not enable_caching → 提前返回，不进 cache_blocks
        # （L549-L552）——本章 False 支的控制流闭合点
        mgr = make_manager()
        req = make_request("req-0", 16)
        mgr.allocate_slots(req, 16)
        single = mgr.coordinator.single_type_managers[0]
        # num_cached_block 未被 allocate_slots 路径推进（cache_blocks 没跑）
        assert "req-0" not in single.num_cached_block
        # 门面 cache_blocks 调用点在，但 enable_caching=False 门守住不进
        mgr.cache_blocks(req, 16)
        assert "req-0" not in single.num_cached_block
        # manager 侧写回账位（幂等闸 + num_cached_block 推进）：ch15 的入口
        mgr.coordinator.cache_blocks(req, 16)
        assert single.num_cached_block.get("req-0") == 1
        # 幂等：num_cached_blocks >= num_full_blocks 时早退（L448-L449）
        mgr.coordinator.cache_blocks(req, 16)
        assert single.num_cached_block.get("req-0") == 1


# --------------------------------------------------------------------------- #
# F. 新块 id 记录与排干（m8 上半，single_type:L367-L368 / manager:L796-L801）
# --------------------------------------------------------------------------- #


class TestNewBlockIdsChannel:
    def test_record_and_drain(self):
        # needs_kv_cache_zeroing=True 时新块 id 记进 manager.new_block_ids
        # （single_type L367-L368），take_new_block_ids 排干（L376-L380）
        config = make_config()  # 单组 uniform → zeroing False
        assert config.needs_kv_cache_zeroing is False
        # 混合精度两组 → zeroing True（kv_cache_interface.py:L1013-L1022）
        config2 = make_config(num_groups=2)
        assert config2.needs_kv_cache_zeroing is True
        mgr = KVCacheManager(
            kv_cache_config=config2,
            max_model_len=4096,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            enable_caching=False,
        )
        req = make_request("req-0", 32)
        mgr.allocate_slots(req, 32)
        ids = mgr.take_new_block_ids()
        # 两组各分 2 块：块 id 1,2（组0）+ 3,4（组1）——popleft 顺序分配
        assert sorted(ids) == [1, 2, 3, 4]
        # 排干语义：第二次取为空（"does not grow unbounded"）
        assert mgr.take_new_block_ids() == []


# --------------------------------------------------------------------------- #
# G. 调度器侧站点（scheduler.py:L973-L985 / L1144-L1149 / L1260-L1272 / L2329-L2354）
# --------------------------------------------------------------------------- #


class TestSchedulerStations:
    def make_scheduler(self, num_blocks: int = 10, num_groups: int = 1) -> Scheduler:
        config = make_config(num_blocks=num_blocks, num_groups=num_groups)
        return Scheduler(
            kv_cache_config=config,
            max_model_len=4096,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            enable_caching=False,
        )

    def test_waiting_entry_allocate_or_none(self):
        # 第 2 站：WAITING 侧 allocate_slots 入口（L973-L985）——
        # 前缀命中作 new_computed_blocks 参数传入（本章恒空 → ch15）
        sched = self.make_scheduler()
        req = make_request("req-0", 100)
        blocks = sched.allocate_slots_for_waiting(
            req,
            100,
            num_new_computed_tokens=0,
            new_computed_blocks=sched.kv_cache_manager.empty_kv_cache_blocks,
        )
        assert blocks is not None
        assert blocks.get_block_ids() == ([1, 2, 3, 4, 5, 6, 7],)

    def test_waiting_breaks_when_none(self):
        sched = self.make_scheduler(num_blocks=5)  # 4 usable
        req0 = make_request("req-0", 64)
        assert sched.allocate_slots_for_waiting(req0, 64, 0, None) is not None
        req1 = make_request("req-1", 64)
        # 拿不到块 → None（ch10 只见 break，这里是 None 的出生地）
        assert sched.allocate_slots_for_waiting(req1, 64, 0, None) is None

    def test_running_loop_none_triggers_preempt_oldest(self):
        # 第 11 站：RUNNING 循环 while True（L576-L629）——None → 抢占最低
        # 优先级（FCFS = 队尾最新者被 pop），被抢者块全还、computed 归零
        sched = self.make_scheduler(num_blocks=5)  # 4 usable
        r_old = make_request("req-old", 32)
        r_new = make_request("req-new", 32)
        sched.running.extend([r_old, r_new])
        for r in sched.running:
            r.status = RequestStatus.RUNNING
            assert sched.allocate_slots_for_running(r, 32) is not None
        # 池干：req-old 再长 1 块 → 抢占环 pop 队尾 req-new 给它腾块
        r_old.num_computed_tokens = 32
        result = sched.allocate_slots_for_running(r_old, 16)
        assert result is not None
        # req-new 被抢：PREEMPTED、块全还、num_computed_tokens=0（ch11 外部行为
        # 的块侧内景）
        assert r_new.status == RequestStatus.PREEMPTED
        assert r_new.num_computed_tokens == 0
        single = sched.kv_cache_manager.coordinator.single_type_managers[0]
        assert "req-new" not in single.req_to_blocks

    def test_new_request_full_block_table_crossing(self):
        # 第 5 站：新请求全量块表随首帧过线（L1144-L1149）
        sched = self.make_scheduler()
        req = make_request("req-0", 100)
        blocks = sched.allocate_slots_for_waiting(req, 100, 0, None)
        data = sched.make_new_reqs_data([req], {req.request_id: blocks})
        assert len(data) == 1
        assert data[0].req_id == "req-0"
        assert data[0].block_ids == ([1, 2, 3, 4, 5, 6, 7],)
        assert data[0].num_computed_tokens == 0

    def test_running_request_incremental_only(self):
        # 第 5 站：在跑请求只带增量 new_block_ids（L1451-L1453
        # get_block_ids(allow_none=True)，空则 None 不占带宽）
        sched = self.make_scheduler()
        req = make_request("req-0", 100)
        blocks = sched.allocate_slots_for_waiting(req, 100, 0, None)
        assert blocks.get_block_ids() == ([1, 2, 3, 4, 5, 6, 7],)
        req.status = RequestStatus.RUNNING
        req.num_computed_tokens = 100
        # 本步无新块（allocate_slots 返回预构空对象）→ None
        empty = sched.kv_cache_manager.empty_kv_cache_blocks
        cached = sched._make_cached_request_data(
            [req], [], {req.request_id: 100}, {req.request_id: empty}
        )
        assert cached.new_block_ids == [None]
        # 本步长 1 块 → 只带增量
        grown = sched.allocate_slots_for_running(req, 16)
        assert grown.get_block_ids() == ([8],)
        cached2 = sched._make_cached_request_data(
            [req], [], {req.request_id: 16}, {req.request_id: grown}
        )
        assert cached2.new_block_ids == [([8],)]

    def test_get_new_block_ids_to_zero_drains_each_step(self):
        # 第 6 站：_get_new_block_ids_to_zero（L1260-L1272）——
        # take_new_block_ids 每步排干；needs_kv_cache_zeroing=False → None
        sched = self.make_scheduler()  # uniform 单组 → False
        req = make_request("req-0", 32)
        sched.allocate_slots_for_waiting(req, 32, 0, None)
        assert sched._get_new_block_ids_to_zero() is None
        # 混合精度两组 → True：新块 id 过线给 worker 清零
        sched2 = self.make_scheduler(num_blocks=10, num_groups=2)
        req2 = make_request("req-0", 32)
        sched2.allocate_slots_for_waiting(req2, 32, 0, None)
        assert sorted(sched2._get_new_block_ids_to_zero()) == [1, 2, 3, 4]
        # 排干后为空 → None（L1272 `or None`）
        assert sched2._get_new_block_ids_to_zero() is None

    def test_free_blocks_end_of_life(self):
        # 第 12 站：_free_blocks（L2329-L2332）→ manager.free → 逆序归还
        sched = self.make_scheduler(num_blocks=5)
        req = make_request("req-0", 48)  # 3 块
        sched.requests["req-0"] = req
        sched.allocate_slots_for_waiting(req, 48, 0, None)
        assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == 1
        req.status = RequestStatus.FINISHED_STOPPED
        sched._free_blocks(req)
        # 块回池、请求销账
        assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == 4
        single = sched.kv_cache_manager.coordinator.single_type_managers[0]
        assert "req-0" not in single.req_to_blocks
        assert "req-0" not in sched.requests

    def test_free_reverse_order_tail_first(self):
        # m12："tail blocks are freed first"——逆序归还自由队列（single_type
        # L519-L527 reversed）→ 下一次批量分配按尾块优先拿到
        pool = BlockPool(
            num_gpu_blocks=11, enable_caching=False, hash_block_size=BLOCK_SIZE
        )
        allocated = pool.get_new_blocks(5)  # ids 1..5（0 被 null 占）
        pool.free_blocks(reversed(allocated))  # 终局逆序 free
        # 队尾是 [5,4,3,2,1]（尾块先驱逐语义）；队头仍是最旧未用块 6..
        queue = pool.free_block_queue.get_all_free_blocks()
        assert [b.block_id for b in queue[-5:]] == [5, 4, 3, 2, 1]
        # 用满后再取：先耗完 6..10，然后按 5,4,3,2,1（LRU 尾优先驱逐序）
        pool.get_new_blocks(4)  # 6,7,8,9
        nxt = pool.get_new_blocks(2)  # 10, 然后 5
        assert [b.block_id for b in nxt] == [10, 5]


# --------------------------------------------------------------------------- #
# H. worker 侧镜像（m7 第 7 站，gpu_model_runner.py:L1441-L1474）
# --------------------------------------------------------------------------- #


class TestWorkerMirror:
    def test_zero_new_blocks_then_extend_and_append_row(self):
        runner = make_runner(num_blocks=10)
        # 上一任主人留下的陈旧字节（块从自由队列回收，m8 的卫生制度所针对）
        runner.kv_caches[LAYER].view(torch.int32).fill_(7)
        # 第一步：新请求建档（block_ids 全量随首帧过线）+ 清零账到达
        req = make_request("req-0", 32)
        new_data = NewRequestData.from_request(req, ([1, 2],))
        output = SchedulerOutput(
            scheduled_new_reqs=[new_data],
            scheduled_cached_reqs=make_cached_data([], [], [], []),
            num_scheduled_tokens={"req-0": 32},
            total_num_scheduled_tokens=32,
            finished_req_ids=set(),
            new_block_ids_to_zero=[1, 2],
        )
        runner._update_states(output)
        # 清零：新块 1、2 的显存归零（m8：防陈旧 NaN/data）；块 0 未动仍是 7
        flat = runner.kv_caches[LAYER].view(torch.int32)
        for block_id in (1, 2):
            start = block_id * runner.page_size_el
            assert flat[start : start + runner.page_size_el].abs().sum() == 0
        assert flat[: runner.page_size_el].abs().sum() > 0
        # 建档：CachedRequestState.block_ids = 全量块表；页表行写入 [1,2]
        assert runner.requests["req-0"].block_ids == ([1, 2],)
        row_idx = runner.input_batch.req_id_to_index["req-0"]
        assert list(runner.input_batch.block_table.block_table.np[row_idx][:2]) == [1, 2]

        # 第二步：在跑请求增量 new_block_ids → 差量 extend + append_row 追加
        runner.kv_caches[LAYER].view(torch.int32).fill_(7)
        output2 = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=make_cached_data(
                ["req-0"], new_block_ids=[([3],)], computed=[48]
            ),
            num_scheduled_tokens={"req-0": 16},
            total_num_scheduled_tokens=16,
            finished_req_ids=set(),
            new_block_ids_to_zero=[3],
        )
        runner._update_states(output2)
        assert runner.requests["req-0"].block_ids == ([1, 2, 3],)
        row = runner.input_batch.block_table.block_table.np[row_idx]
        assert list(row[:3]) == [1, 2, 3]
        assert runner.input_batch.block_table.num_blocks_per_row[row_idx] == 3
        # 只清了新块 3；块 1、2 的陈旧字节不受第二次清零影响（已被本步前向
        # 覆写的语义由真实 kernel 承担，此处验证清零账只对增量生效）
        flat = runner.kv_caches[LAYER].view(torch.int32)
        start3 = 3 * runner.page_size_el
        assert flat[start3 : start3 + runner.page_size_el].abs().sum() == 0

    def test_resumed_from_preemption_replaces_whole_table(self):
        # SOURCE 行为：被抢占恢复的请求整表替换（L1447-L1452 assert req_index
        # is None + block_ids = new_block_ids）
        runner = make_runner()
        runner.requests["req-0"] = CachedRequestState(
            req_id="req-0",
            prompt_token_ids=[0],
            block_ids=([1, 2],),
            num_computed_tokens=32,
            output_token_ids=[],
        )
        output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=make_cached_data(
                ["req-0"],
                new_block_ids=[([9, 10],)],
                computed=[0],
                resumed={"req-0"},
            ),
            num_scheduled_tokens={"req-0": 33},
            total_num_scheduled_tokens=33,
            finished_req_ids=set(),
            new_block_ids_to_zero=[9, 10],
        )
        runner._update_states(output)
        assert runner.requests["req-0"].block_ids == ([9, 10],)


def make_cached_data(
    req_ids, new_block_ids, computed, resumed=(), num_output_tokens=None
):
    from implementation.output import CachedRequestData

    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(resumed),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=list(new_block_ids),
        num_computed_tokens=list(computed),
        num_output_tokens=num_output_tokens or [0] * len(req_ids),
    )


# --------------------------------------------------------------------------- #
# I. 槽位恒等式（m9，block_table.py:L379-L442）
# --------------------------------------------------------------------------- #


class TestSlotMapping:
    def make_bt(self, max_num_reqs=2, max_blocks_per_req=8, max_tokens=128):
        return BlockTable(
            block_size=BLOCK_SIZE,
            max_num_reqs=max_num_reqs,
            max_num_blocks_per_req=max_blocks_per_req,
            max_num_batched_tokens=max_tokens,
            pin_memory=False,  # HOST SEAM：CPU host 无 pinned memory
            device=torch.device("cpu"),
            kernel_block_size=BLOCK_SIZE,  # 分配块 = kernel 块（免细分直通）
        )

    def test_identity_slot_equals_block_times_size_plus_offset(self):
        # m9 恒等式：slot = block_table[req][pos // block_size] × block_size
        # + pos % block_size（L434-L440）——块表行 [3, 1, 7] 的换算
        bt = self.make_bt()
        bt.append_row([3, 1, 7], 0)
        num_tokens = 48
        query_start_loc = torch.tensor([0, num_tokens, num_tokens], dtype=torch.int32)
        positions = torch.arange(num_tokens, dtype=torch.int64)
        bt.compute_slot_mapping(1, query_start_loc, positions)
        slots = bt.slot_mapping.np[:num_tokens]
        expected = [
            bt.block_table.np[0][pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
            for pos in range(num_tokens)
        ]
        assert list(slots) == expected
        # 抽查三段：块 3 → 48..63；块 1 → 16..31；块 7 → 112..127
        assert slots[0] == 3 * 16 and slots[15] == 3 * 16 + 15
        assert slots[16] == 1 * 16 and slots[31] == 1 * 16 + 15
        assert slots[32] == 7 * 16 and slots[47] == 7 * 16 + 15

    def test_tail_padded_with_pad_slot_id(self):
        # SOURCE 行为：最后一个 program 专职把 [num_tokens, max_num_tokens)
        # 填 PAD——CUDA graph 捕获 max 形状、尾部每拍重填（L399-L408）
        bt = self.make_bt(max_tokens=64)
        bt.append_row([1], 0)
        num_tokens = 20
        query_start_loc = torch.tensor([0, num_tokens, num_tokens], dtype=torch.int32)
        positions = torch.arange(num_tokens, dtype=torch.int64)
        bt.compute_slot_mapping(1, query_start_loc, positions)
        tail = bt.slot_mapping.np[num_tokens:64]
        assert (tail == PAD_SLOT_ID).all()
        assert PAD_SLOT_ID == -1

    def test_two_requests_segments(self):
        # 每个 program 处理一请求的 token 区间（query_start_loc 切段）；
        # positions 是各请求自己的序列位置（两条 decode 各从 0 起）
        bt = self.make_bt(max_num_reqs=2, max_tokens=64)
        bt.append_row([2], 0)
        bt.append_row([5], 1)
        qs = torch.tensor([0, 16, 32, 32], dtype=torch.int32)
        positions = torch.cat(
            [
                torch.arange(16, dtype=torch.int64),
                torch.arange(16, dtype=torch.int64),
            ]
        )
        bt.compute_slot_mapping(2, qs, positions)
        assert bt.slot_mapping.np[0] == 2 * 16
        assert bt.slot_mapping.np[15] == 2 * 16 + 15
        assert bt.slot_mapping.np[16] == 5 * 16
        assert bt.slot_mapping.np[31] == 5 * 16 + 15

    def test_append_row_incremental_offset(self):
        # append_row 差量追加：行内 offset 由 num_blocks_per_row 记账
        # （L151-L154）
        bt = self.make_bt()
        bt.append_row([3], 0)
        bt.append_row([1, 7], 0)
        assert list(bt.block_table.np[0][:3]) == [3, 1, 7]
        assert bt.num_blocks_per_row[0] == 3
        # 空追加为 no-op（L143-L144）
        bt.append_row([], 0)
        assert bt.num_blocks_per_row[0] == 3

    def test_commit_copies_active_rows_only(self):
        # m15：commit_block_table 每拍只拷活跃行（L213-L214）——CPU/GPU 双镜像
        bt = self.make_bt(max_num_reqs=4)
        bt.append_row([3, 1], 0)
        bt.append_row([7], 2)  # 行 2 本拍不活跃
        bt.commit_block_table(1)
        assert bt.block_table.gpu[0][:2].tolist() == [3, 1]
        # 行 2 没被拷：GPU 镜像仍是初始零
        assert bt.block_table.gpu[2][0].item() == 0
        assert bt.block_table.cpu[2][0].item() == 7

    def test_get_device_tensor_slices_active_rows(self):
        # 读侧出口：块表张量交给 attention metadata builder（L250-L252）
        bt = self.make_bt(max_num_reqs=4)
        bt.append_row([3], 0)
        bt.append_row([1], 1)
        bt.commit_block_table(2)
        t = bt.get_device_tensor(1)
        assert t.shape[0] == 1
        assert t[0][0].item() == 3


# --------------------------------------------------------------------------- #
# J. 页的物理形状（m10/m13，kv_cache_interface.py:L184-L226 / cache.py:L43-L51）
# --------------------------------------------------------------------------- #


class TestPageShape:
    def test_real_page_size_bytes_formula(self):
        # 公式：2(K,V) × block_size × num_kv_heads × head_dim × dtype 字节
        spec = make_spec(block_size=16, num_kv_heads=8, head_size=128,
                         dtype=torch.float16)
        assert spec.real_page_size_bytes == 2 * 16 * 8 * 128 * 2  # 65536
        # page_size_bytes 无量化 padding 时与 real 相等
        assert spec.page_size_bytes == spec.real_page_size_bytes

    def test_per_token_kv_bytes_llama2_7b(self):
        # 理论卡口径：每 token KV = 2 × num_kv_heads × head_dim × dtype
        # （Llama-2-7B FP16: 2×32×128×2 = 16384 B/token/层）
        spec = make_spec(num_kv_heads=32, head_size=128, dtype=torch.float16)
        per_token = 2 * 32 * 128 * 2
        assert spec.real_page_size_bytes == BLOCK_SIZE * per_token

    def test_worker_num_blocks_from_bytes(self):
        # worker 侧换算：num_blocks = numel // page_size_bytes
        # （gpu_model_runner.py:L7406-L7407）——两侧同源同值
        spec = make_spec(block_size=16, num_kv_heads=8, head_size=128,
                         dtype=torch.float16)
        num_blocks = 10
        raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
        assert raw.numel() % spec.page_size_bytes == 0
        assert raw.numel() // spec.page_size_bytes == num_blocks
        # reshape 成标准全注意力视图 [num_blocks, 2(K,V), block_size, kv_heads,
        # head_dim]——page_size_bytes 里的 2× 就是 K 与 V 两半
        view = raw.view(torch.float16).view(num_blocks, 2, 16, 8, 128)
        assert view.shape == (10, 2, 16, 8, 128)

    def test_default_block_size_is_16(self):
        # m13：DEFAULT_BLOCK_SIZE=16（cache.py:L47）——分配/哈希/寻址最小粒度
        assert CacheConfig.DEFAULT_BLOCK_SIZE == 16
        cfg = CacheConfig()  # None → 默认 16
        assert cfg.block_size == 16

    def test_attention_spec_is_full_attention_spec_base(self):
        assert issubclass(FullAttentionSpec, AttentionSpec)


# --------------------------------------------------------------------------- #
# K. KVBlockZeroer（m8，worker/utils.py:L93-L213）
# --------------------------------------------------------------------------- #


class TestKVBlockZeroer:
    def build_zeroer(self, num_blocks=4):
        spec = make_spec(num_kv_heads=2, head_size=64)
        seg_el = spec.page_size_bytes // 8  # 每段（K 或 V 半页）的 int32 元素数
        # K/V 外层布局（block_dim=1）：一块缓冲装 K 与 V 两段
        # [2(K/V), num_blocks, 段内元素]——kernel docstring 的"two segments
        # per buffer"布局
        kv = torch.empty((2, num_blocks, seg_el), dtype=torch.int32)
        kv[0].fill_(3)  # K 段：陈旧字节
        kv[1].fill_(5)  # V 段：陈旧字节
        context = {"kv": _Ctx(kv)}

        class _Backend:
            # HOST 侧 duck type：block_dim=1（K/V 外层）→ 段 stride = 段元素数
            def get_kv_cache_block_dim(self, kernel_bs, kv_heads, head_size,
                                       cache_dtype_str):
                return 1

        class _Group:
            kv_cache_spec = spec
            backend = _Backend()
            layer_names = ["kv"]
            kv_cache_group_id = 0

        zeroer = KVBlockZeroer(
            device=torch.device("cpu"),
            attn_groups_iter=[_Group()],
            kernel_block_sizes=[16],
            cache_dtype="auto",
            static_forward_context=context,
        )
        return zeroer, kv, seg_el

    def test_zero_block_ids_clears_stale_bytes(self):
        # m8：块从自由队列回收，上一任主人留下的字节还躺在显存——
        # zero_block_ids 把指定块的内存清零（K 段 + V 段一并）
        zeroer, kv, seg_el = self.build_zeroer()
        assert int(kv.abs().sum()) > 0
        zeroer.zero_block_ids([1])
        # 块 1 的 K 段与 V 段都归零；块 0 未动
        assert kv[0, 1].abs().sum() == 0
        assert kv[1, 1].abs().sum() == 0
        assert kv[0, 0].abs().sum() > 0
        assert kv[1, 0].abs().sum() > 0

    def test_empty_ids_noop(self):
        zeroer, kv, _ = self.build_zeroer()
        zeroer.zero_block_ids([])
        assert int(kv.abs().sum()) > 0


class _Ctx:
    """static_forward_context[layer].kv_cache 的最小镜像。"""

    def __init__(self, kv_cache: torch.Tensor):
        self.kv_cache = kv_cache


# --------------------------------------------------------------------------- #
# L. 全链 e2e：一个请求的 KV 一生（站 1-12 串走）
# --------------------------------------------------------------------------- #


class TestRequestKvLife:
    def test_admission_growth_free_end_to_end(self):
        # 站 2→4 入场：数块 → 拿块 → 挂账；站 5-6 过线打包+清零账；
        # 站 7 worker 镜像；站 9 槽位换算；站 11 长大；站 12 终局还块
        num_blocks = 10
        sched_config = make_config(num_blocks=num_blocks, num_groups=2)
        sched = Scheduler(
            kv_cache_config=sched_config,
            max_model_len=4096,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            enable_caching=False,
        )
        req = make_request("req-0", 64)  # 两组各 cdiv(64,16)=4 → 共 8 块
        sched.requests["req-0"] = req
        blocks = sched.allocate_slots_for_waiting(req, 64, 0, None)
        assert blocks is not None
        ids = blocks.get_block_ids()
        assert ids == ([1, 2, 3, 4], [5, 6, 7, 8])  # 组0 拿 1-4，组1 拿 5-8
        # 过线打包：新请求全量块表 + 清零账
        new_data = sched.make_new_reqs_data([req], {req.request_id: blocks})
        assert new_data[0].block_ids == ids
        assert sorted(sched._get_new_block_ids_to_zero()) == [1, 2, 3, 4, 5, 6, 7, 8]
        # 终局还块：全回池
        req.status = RequestStatus.FINISHED_STOPPED
        sched._free_blocks(req)
        assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == num_blocks - 1

    def test_single_request_waste_under_one_block(self):
        # m1 尾部浪费上界：分页下单请求最多浪费 block_size-1 个 token 位
        # ——100 token 只占 7 块（112 槽，浪费 12 < 16）
        mgr = make_manager()
        req = make_request("req-0", 100)
        result = mgr.allocate_slots(req, 100)
        capacity = BLOCK_SIZE * len(result.get_block_ids()[0])
        assert capacity - 100 == 12
        assert 0 <= capacity - 100 < BLOCK_SIZE


# --------------------------------------------------------------------------- #
# M. KVCacheBlocks 契约面（kv_cache_manager.py:L32-L114）
# --------------------------------------------------------------------------- #


class TestKVCacheBlocksContract:
    def test_get_block_ids_allow_none(self):
        # SOURCE 行为：allow_none=True 且全组为空 → None（不占带宽）
        from implementation.kv_cache_manager import KVCacheBlocks

        empty = KVCacheBlocks(((),))
        assert empty.get_block_ids(allow_none=True) is None
        assert empty.get_block_ids() == ([],)
        nonempty = KVCacheBlocks(((KVCacheBlock(3), KVCacheBlock(1)),))
        assert nonempty.get_block_ids(allow_none=True) == ([3, 1],)

    def test_add_concatenates(self):
        from implementation.kv_cache_manager import KVCacheBlocks

        a = KVCacheBlocks(((KVCacheBlock(1),),))
        b = KVCacheBlocks(((KVCacheBlock(2),),))
        assert [blk.block_id for blk in (a + b).blocks[0]] == [1, 2]

    def test_empty_kv_cache_blocks_prewuilt(self):
        # WC2 物证：预构空对象避免 GC（kv_cache_manager.py:L180-L187）
        mgr = make_manager()
        assert mgr.empty_kv_cache_blocks.get_block_ids(allow_none=True) is None
        again = mgr.create_kv_cache_blocks(([],))
        assert again is mgr.empty_kv_cache_blocks  # 复用同一预构对象
