"""ch15 前缀缓存 —— 单元+契约测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测行为**
（锚点 = vllm/... 行号，基线 v0.27.1 现核，非 v2 资产的 v0.21.0 旧行号）。
本章精简版跑 enable_prefix_caching=True 支（默认开），对照 ch13 的 False 支。

行为清单（按 dossier.mechanisms 对账）：
- m1 链式哈希：hash_i = H(parent, tokens_i, extra_keys)、首块 parent=NONE_HASH
  种子、请求侧只算新满 hash_block_size 块、构造尾+append 增量
  （kv_cache_utils.py:L596-L623 / L691-L748 / request.py:L249-L265）
- m2 非 radix 平面表：BlockHashToBlockMap 单块→dict 退化路径（不去重 NOTE #1、
  union 省 GC NOTE #2）、get_cached_block 逐 group 任一 miss 整体 miss
  （block_pool.py:L33-L140 / L198-L223）
- m3 extra keys：mm/lora/cache_salt(仅首块) 拌进哈希——同 token 不同语义必不同
  哈希（kv_cache_utils.py:L430-L447 / L558-L593）
- m4 命中查找主路径：phase 1 沿链 miss 即停、get_computed_blocks 的
  max_cache_hit_length=num_tokens−1（全命中退一 token 拿 logits、块对齐回退整块）
  （single_type_kv_cache_manager.py:L681-L777 / kv_cache_manager.py:L229-L295）
- m5 touch 救回：ref_cnt+1、ref_cnt==0 时 O(1) remove 出 free queue；
  add_local_computed_blocks 挂块（block_pool.py:L702-L717 /
  single_type_kv_cache_manager.py:L232-L289）
- m6 满块写回：新满块 set_block_hash+insert；block_mask False 不入表
  （block_pool.py:L225-L342）
- m7 LRU 不变量一·逆序 free：尾块先挂回、排更靠驱逐端
  （single_type_kv_cache_manager.py:L519-L527）
- m8 LRU 不变量二·劈分：无哈希块 prepend_n 队头先驱逐、有哈希块 append_n
  LRU 尾；缓存关闭跳过劈分（block_pool.py:L719-L742）
- m9 惰性驱逐：free 不清哈希、get_new_blocks 复用才 _maybe_evict 摘哈希；
  反向索引清部分条目别名（block_pool.py:L647-L700 / L571-L590）
- m10 move_block_hashes：CoW 后条目重指私有拷贝（block_pool.py:L629-L645）
- m11 F2：抢占 free 不清哈希 → 重排回 waiting 重走准入 → 重命中自己的前缀 →
  touch 救回；最坏（块被复用）全量重 prefill
  （scheduler.py:L1274-L1315 / L744-L766）
- m12 粒度分离：BlockHashListWithBlockSize 惰性重串（链尾即前缀指纹）、
  resolve_block_hashes 细粒度保留原始列表
  （kv_cache_utils.py:L2245-L2315 / L2321-L2351）
- m13 块内 CoW 三件套：cache_partial_block 块内边界注册、find phase 2 块内
  自高向低探测、_apply_cow 换尾登记拷贝对
  （block_pool.py:L445-L544 / single_type:L741-L762 / L347-L357 / L405-L425）
- m14 CoW 拷贝过线：take_kv_cache_block_copies → SchedulerOutput.
  kv_cache_block_copies → copy_kv_cache_blocks_inplace；retained 步序栅栏
  （kv_cache_manager.py:L831-L846 / scheduler.py:L1181-L1190 /
  gpu_model_runner.py:L1219-L1228）
- m15 混合不动点：每类型接受或缩短、full 排首向下封闭、simple hybrid 一轮；
  SWA finder 右到左窗口连续段（kv_cache_coordinator.py:L685-L817）
- m16 Marconi 钉住：num_uncached_common_prefix_tokens=longest−reconciled →
  shared_prefix_boundary 写回 → reachable_boundaries 特赦 →
  _mamba_block_aligned_split 停点
- m17 retention 三态校验（kv_cache_coordinator.py:L30-L57）
- m18 开关面：enable_prefix_caching=False → NoPrefixCache 命中恒 0；
  skip_reading_prefix_cache 跳读
- m19 PrefixCacheStats 命中率口径（record 的 preempted 分账）
- m20 reset_prefix_cache：全空闲才清（block_pool.py:L763-L797）
"""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.block_pool import BlockHashToBlockMap, BlockPool  # noqa: E402
from implementation.cache import CacheConfig  # noqa: E402
from implementation.hashing import sha256  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from implementation.kv_cache_coordinator import (  # noqa: E402
    HybridKVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache,
    UnitaryKVCacheCoordinator,
    _validate_prefix_cache_retention_interval,
)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.kv_cache_utils import (  # noqa: E402
    BlockHashListWithBlockSize,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    make_block_hash_with_group_id,
    need_extra_keys,
    resolve_block_hashes,
)
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.stats import PrefixCacheStats  # noqa: E402


# --------------------------------------------------------------------------- #
# 构造辅助：真实装配的最小镜像（PYTHONHASHSEED 播种 → NONE_HASH 可复现）
# --------------------------------------------------------------------------- #

os.environ.setdefault("PYTHONHASHSEED", "0")
init_none_hash(sha256)
HASHER16 = get_request_block_hasher(16, sha256)
HASHER8 = get_request_block_hasher(8, sha256)


def full_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
    )


def swa_spec(block_size: int, window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        sliding_window=window,
    )


def mamba_spec(block_size: int, mode: str = "align") -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((8, 8),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
    )


def kv_config(specs, num_blocks: int = 64) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)
        ],
    )


def make_manager(specs, hash_block_size: int, num_blocks: int = 64,
                 enable_caching: bool = True, max_model_len: int = 512,
                 watermark: float = 0.0) -> KVCacheManager:
    scheduler_bs = math.lcm(*(s.block_size for s in specs))
    return KVCacheManager(
        kv_cache_config=kv_config(specs, num_blocks),
        max_model_len=max_model_len,
        scheduler_block_size=scheduler_bs,
        hash_block_size=hash_block_size,
        enable_caching=enable_caching,
    )


def make_request(rid: str, tokens, hasher=HASHER16, **kwargs) -> Request:
    kwargs.setdefault("block_hasher", hasher)
    return Request(rid, list(tokens), **kwargs)


def run_request(mgr: KVCacheManager, req: Request) -> None:
    """waiting 准入→分配→写回的最小闭环（num_computed_tokens 在
    update_from_output 之后才推进——分配时仍是旧值 0）。"""
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(
        req,
        req.num_tokens - num_hit,
        num_new_computed_tokens=num_hit,
        new_computed_blocks=blocks,
    )
    assert out is not None
    req.num_computed_tokens = req.num_tokens


def free_queue_ids(pool: BlockPool) -> list[int]:
    """按驱逐序（队头=先驱逐）走侵入式链表。"""
    q = pool.free_block_queue
    ids, cur = [], q.fake_free_list_head.next_free_block
    while cur is not None and cur is not q.fake_free_list_tail:
        ids.append(cur.block_id)
        cur = cur.next_free_block
    return ids


def cached_hashes(pool: BlockPool) -> set:
    return set(pool.cached_block_hash_to_block._cache.keys())


# --------------------------------------------------------------------------- #
# m1 链式哈希：算与攒
# --------------------------------------------------------------------------- #


class TestChainedHash:
    def test_hash_block_tokens_deterministic_and_chained(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L596-L623
        h0 = hash_block_tokens(sha256, None, list(range(16)))
        h0b = hash_block_tokens(sha256, None, list(range(16)))
        assert h0 == h0b
        assert isinstance(h0, bytes)
        # 首块 parent=None → NONE_HASH 种子
        from implementation.kv_cache_utils import NONE_HASH

        h0_seed = hash_block_tokens(sha256, NONE_HASH, list(range(16)))
        assert h0 == h0_seed
        # 链式：第二块依赖第一块的哈希（Merkle 性质）
        h1 = hash_block_tokens(sha256, h0, list(range(16, 32)))
        assert h1 == hash_block_tokens(sha256, h0, list(range(16, 32)))
        assert h1 != h0
        # 换 token → 换哈希；换 parent → 换哈希
        assert hash_block_tokens(sha256, None, list(range(1, 17))) != h0
        assert hash_block_tokens(sha256, h1, list(range(16, 32))) != h1

    def test_request_block_hasher_only_full_blocks(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L705-L748
        req = make_request("r", list(range(37)))  # 2 满块(16)+5 尾
        assert len(req.block_hashes) == 2
        # 增量：append 到跨过下一边界才多算一块
        req.append_output_token_ids(list(range(37, 43)))  # 43 token 仍 2 满块
        assert len(req.block_hashes) == 2
        req.append_output_token_ids(list(range(43, 50)))  # 50 → 3 满块
        assert len(req.block_hashes) == 3
        # 哈希与逐块手算一致（parent 链式）
        manual0 = hash_block_tokens(sha256, None, list(range(0, 16)))
        manual1 = hash_block_tokens(sha256, manual0, list(range(16, 32)))
        manual2 = hash_block_tokens(sha256, manual1, list(range(32, 48)))
        assert req.block_hashes == [manual0, manual1, manual2]

    def test_none_hash_seeded_by_pythonhashseed(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L99-L114
        # PYTHONHASHSEED=0 已在模块头播种：NONE_HASH = sha256("0")
        from implementation.kv_cache_utils import NONE_HASH

        assert NONE_HASH == sha256("0")

    def test_hasher_disabled_when_caching_off(self):
        # SOURCE: vllm/v1/engine/core.py:L220-L229：关缓存则不装 hasher
        from implementation.engine_core import assemble_block_hasher

        assert assemble_block_hasher(
            CacheConfig(enable_prefix_caching=True), hash_block_size=16
        ) is not None
        assert assemble_block_hasher(
            CacheConfig(enable_prefix_caching=False), hash_block_size=16
        ) is None
        req = make_request("r", list(range(32)), block_hasher=None)
        assert req.block_hashes == []


# --------------------------------------------------------------------------- #
# m3 extra keys：语义隔离
# --------------------------------------------------------------------------- #


class TestExtraKeys:
    def test_need_extra_keys_predicate(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L430-L447
        plain = make_request("p", [1] * 32)
        assert need_extra_keys(plain) is False
        salted = make_request("s", [1] * 32, cache_salt="tenant-a")
        assert need_extra_keys(salted) is True
        lora = make_request(
            "l", [1] * 32, lora_request=SimpleNamespace(lora_name="adapter-1")
        )
        assert need_extra_keys(lora) is True
        mm = make_request(
            "m",
            [1] * 32,
            mm_features=[
                SimpleNamespace(
                    identifier="img-1",
                    mm_position=SimpleNamespace(offset=0, length=4),
                )
            ],
        )
        assert need_extra_keys(mm) is True

    def test_cache_salt_only_first_block(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L558-L593（start_token_idx==0）
        req = make_request("s", list(range(64)), cache_salt="tenant-a")
        k0, mm0 = generate_block_hash_extra_keys(req, 0, 16, 0)
        k1, mm1 = generate_block_hash_extra_keys(req, 16, 32, mm0)
        assert k0 == ("tenant-a",)
        assert k1 is None  # 盐只拌进首块（extra_keys 不含盐）
        # 同 token 不同盐 → 不同哈希（跨语义不误命中）；且因链式传播，
        # 首块的盐差会沿 parent 链传给后续所有块的哈希
        other = make_request("o", list(range(64)), cache_salt="tenant-b")
        assert req.block_hashes[0] != other.block_hashes[0]
        assert req.block_hashes[1] != other.block_hashes[1]  # 链式传播

    def test_lora_name_in_every_block(self):
        lora_req = make_request(
            "l", list(range(32)), lora_request=SimpleNamespace(lora_name="a1")
        )
        k0, _ = generate_block_hash_extra_keys(lora_req, 0, 16, 0)
        k1, _ = generate_block_hash_extra_keys(lora_req, 16, 32, 0)
        assert k0 == ("a1",) and k1 == ("a1",)


# --------------------------------------------------------------------------- #
# m2/m4 平面表 + 命中查找主路径
# --------------------------------------------------------------------------- #


class TestFlatHashTable:
    def test_make_block_hash_with_group_id_roundtrip(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L57-L76
        bh = bytes(range(32))
        key = make_block_hash_with_group_id(bh, 3)
        assert get_block_hash(key) == bh
        assert get_group_id(key) == 3

    def test_map_union_no_dedup(self):
        # SOURCE: vllm/v1/core/block_pool.py:L33-L140（NOTE #1/#2）
        m = BlockHashToBlockMap()
        k = make_block_hash_with_group_id(b"h" * 8, 0)
        b1, b2 = KVCacheBlock(1), KVCacheBlock(2)
        m.insert(k, b1)
        assert m.get_one_block(k) is b1
        m.insert(k, b2)  # 同键第二块 → union dict
        assert m.contain(k, 1) and m.contain(k, 2)
        assert m.get_one_block(k) in (b1, b2)  # 任取一块
        assert m.pop(k, 1) is b1
        assert m.contain(k, 2) and not m.contain(k, 1)
        assert m.pop(k, 2) is b2
        assert m.pop(k, 2) is None

    def test_get_cached_block_any_group_miss(self):
        # SOURCE: vllm/v1/core/block_pool.py:L198-L223
        pool = BlockPool(8, enable_caching=True, hash_block_size=16)
        bh = hash_block_tokens(sha256, None, list(range(16)))
        blk = pool.get_new_blocks(1)[0]
        pool._insert_block_hash(make_block_hash_with_group_id(bh, 0), blk,
                                num_tokens=16)
        assert pool.get_cached_block(bh, [0]) == [blk]
        assert pool.get_cached_block(bh, [0, 1]) is None  # group 1 miss 整体 miss


class TestHitLookup:
    def test_phase1_miss_stops_and_full_hit_minus_one_token(self):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L731-L739
        #   + kv_cache_manager.py:L253-L259
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        pool = mgr.block_pool
        reqA = make_request("a", list(range(64)))  # 4 满块
        run_request(mgr, reqA)
        mgr.free(reqA)
        # 部分共享：reqB 前 32 token 同 A（2 块命中，第 3 块不同 → 断）
        reqB = make_request("b", list(range(32)) + list(range(100, 132)))
        blocks, hit, junction = mgr.get_computed_blocks(reqB)
        assert hit == 32 and junction == 0
        assert [b.block_id for b in blocks.blocks[0]] == [1, 2]
        # 全命中：prompt 与 A 完全一致 → max_cache_hit_length=63 → 只 3 块(48)
        reqC = make_request("c", list(range(64)))
        _, hit_c, _ = mgr.get_computed_blocks(reqC)
        assert hit_c == 48  # 退一 token + 块对齐回退整块

    def test_skip_reading_prefix_cache(self):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L246-L251 + request.py:L291-L302
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        reqA = make_request("a", list(range(64)))
        run_request(mgr, reqA)
        mgr.free(reqA)
        reqB = make_request("b", list(range(64)),
                            skip_reading_prefix_cache=True)
        blocks, hit, boundary = mgr.get_computed_blocks(reqB)
        assert hit == 0 and boundary == 0
        assert blocks is mgr.empty_kv_cache_blocks

    def test_shared_prefix_reused_across_requests(self):
        # NOTE #1 的可观察面：命中返回第一个物理块，块表 append-only
        mgr = make_manager([full_spec(16)], hash_block_size=16, num_blocks=32)
        reqA = make_request("a", list(range(32)))
        run_request(mgr, reqA)
        reqB = make_request("b", list(range(48)))  # 前 32 同 A
        run_request(mgr, reqB)  # B 命中 A 的两块（共享）
        st = mgr.coordinator.single_type_managers[0]
        blk1 = st.req_to_blocks["a"][0]
        assert st.req_to_blocks["b"][0] is blk1  # 同一物理块
        mgr.free(reqA)
        # A 的块被 B 引用（ref_cnt=1）不回队；B 自己 free 后回队
        assert blk1.ref_cnt == 1
        assert blk1.block_hash is not None  # 仍被 B 用着
        mgr.free(reqB)
        # 再来 C：命中任一物理块
        reqC = make_request("c", list(range(32)))
        blocks, hit, _ = mgr.get_computed_blocks(reqC)
        assert hit == 16


# --------------------------------------------------------------------------- #
# m5 touch 救回 + m6 满块写回
# --------------------------------------------------------------------------- #


class TestTouchAndCacheWriteback:
    def test_touch_rescues_from_free_queue(self):
        # SOURCE: vllm/v1/core/block_pool.py:L702-L717
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        pool = mgr.block_pool
        reqA = make_request("a", list(range(32)))
        run_request(mgr, reqA)
        mgr.free(reqA)
        blk = pool.blocks[1]
        assert blk.ref_cnt == 0 and blk.block_hash is not None
        assert 1 in free_queue_ids(pool)  # 在 free queue 当驱逐候选
        pool.touch([blk])
        assert blk.ref_cnt == 1
        assert 1 not in free_queue_ids(pool)  # O(1) remove 出队

    def test_shared_block_refcount(self):
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        reqA = make_request("a", list(range(32)))
        run_request(mgr, reqA)
        st = mgr.coordinator.single_type_managers[0]
        blk1 = st.req_to_blocks["a"][0]
        assert blk1.ref_cnt == 1
        reqB = make_request("b", list(range(48)))  # 前 32 同 A
        run_request(mgr, reqB)
        assert blk1.ref_cnt == 2  # 共享物理块的引用计数

    def test_cache_full_blocks_and_mask(self):
        # SOURCE: vllm/v1/core/block_pool.py:L259-L299 + single_type:L427-L477
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        pool = mgr.block_pool
        req = make_request("a", list(range(40)))  # 2 满块 + 8 尾
        run_request(mgr, req)
        blocks = mgr.coordinator.single_type_managers[0].req_to_blocks["a"]
        assert blocks[0].block_hash is not None
        assert blocks[1].block_hash is not None
        assert blocks[2].block_hash is None  # 尾块不满不入表（hash_bs==block_bs）
        assert len(cached_hashes(pool)) == 2

    def test_block_mask_false_not_entering_table(self):
        # SOURCE: vllm/v1/core/block_pool.py:L271-L276（block_mask 文档原文：
        #   永不可能服务命中的块不占哈希表）
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        pool = mgr.block_pool
        req = make_request("a", list(range(32)))
        # delay_cache_blocks=True 跳过自动写回（P/D 占位参数）——手工控表
        out = mgr.allocate_slots(req, 32, delay_cache_blocks=True)
        assert out is not None
        st = mgr.coordinator.single_type_managers[0]
        before = len(cached_hashes(pool))
        pool.cache_full_blocks(
            request=req,
            blocks=st.req_to_blocks["a"],
            num_cached_blocks=0,
            num_full_blocks=2,
            block_size=16,
            kv_cache_group_id=0,
            block_mask=[True, False],
        )
        assert st.req_to_blocks["a"][0].block_hash is not None
        assert st.req_to_blocks["a"][1].block_hash is None  # 被 mask 掉
        assert len(cached_hashes(pool)) == before + 1


# --------------------------------------------------------------------------- #
# m7/m8 LRU 双不变量
# --------------------------------------------------------------------------- #


class TestLRUInvariants:
    def test_free_reverse_order(self):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L519-L527
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        pool = mgr.block_pool
        req = make_request("a", list(range(48)))  # 3 满块 [1,2,3]
        run_request(mgr, req)
        mgr.free(req)
        # 逆序 + 全带哈希 → append 到 LRU 尾：尾序 [3,2,1]（3 是链尾、
        # 最长前缀要求、潜在复用者最少、排最靠驱逐端）
        assert free_queue_ids(pool)[-3:] == [3, 2, 1]

    def test_free_blocks_split_without_hash_first(self):
        # SOURCE: vllm/v1/core/block_pool.py:L719-L742（#42656）
        pool = BlockPool(8, enable_caching=True, hash_block_size=16)
        # b1：无哈希（从未入表）；b2：有哈希
        b1, b2 = pool.get_new_blocks(2)
        bh = hash_block_tokens(sha256, None, list(range(16)))
        pool._insert_block_hash(make_block_hash_with_group_id(bh, 0), b2, 16)
        # 先释放一个无关块垫底，再看劈分相对序
        b3 = pool.get_new_blocks(1)[0]
        pool.free_blocks([b3])  # 无哈希 → prepend 队头
        pool.free_blocks([b1, b2])  # b1 无哈希、b2 有哈希
        ids = free_queue_ids(pool)
        # b1（无哈希）与 b3（无哈希）都在驱逐端最前；b2（有哈希）在 LRU 尾
        assert ids[0] in (1, 3) and ids[1] in (1, 3)
        assert ids[-1] == 2  # 有哈希块排最可复用端

    def test_free_blocks_no_split_when_caching_off(self):
        # SOURCE: vllm/v1/core/block_pool.py:L732-L738（缓存关闭恒 append 保
        #   GPU 局部性——注释原话）
        pool = BlockPool(8, enable_caching=False, hash_block_size=16)
        blocks = pool.get_new_blocks(3)
        pool.free_blocks(reversed(blocks))  # 逆序传入（调用约定不变）
        assert free_queue_ids(pool)[-3:] == [3, 2, 1]  # 全部 append_n 保序


# --------------------------------------------------------------------------- #
# m9 惰性驱逐 + m10 move_block_hashes
# --------------------------------------------------------------------------- #


class TestLazyEviction:
    def test_free_keeps_hash_reuse_evicts(self):
        # SOURCE: vllm/v1/core/block_pool.py:L647-L700
        pool = BlockPool(2, enable_caching=True, hash_block_size=16)
        req = make_request("a", list(range(16)))
        blk = pool.get_new_blocks(1)[0]  # 唯一非 null 块
        pool.cache_full_blocks(
            request=req, blocks=[blk], num_cached_blocks=0,
            num_full_blocks=1, block_size=16, kv_cache_group_id=0,
        )
        pool.free_blocks([blk])  # 归零入队（带哈希 → LRU 尾）
        assert blk.block_hash is not None  # free 不清哈希（F2 物质基础）
        assert len(cached_hashes(pool)) == 1
        reused = pool.get_new_blocks(1)[0]  # 从队头取（正是这块）
        assert reused is blk
        assert reused.block_hash is None  # 复用才摘哈希（惰性驱逐）
        assert len(cached_hashes(pool)) == 0

    def test_remove_aliases_via_reverse_index(self):
        # SOURCE: vllm/v1/core/block_pool.py:L571-L590 + cached_block_hashes_by_block
        pool = BlockPool(8, enable_caching=True, hash_block_size=8)
        blk = pool.get_new_blocks(1)[0]
        k_main = make_block_hash_with_group_id(b"main", 0)
        k_alias = make_block_hash_with_group_id(b"alias", 0)
        pool._insert_block_hash(k_main, blk, num_tokens=16)
        pool._insert_block_hash(k_alias, blk, num_tokens=8)  # 第二条进反向索引
        assert blk.block_hash == k_main
        assert k_alias in pool.cached_block_hashes_by_block[blk.block_id]
        removed = pool._remove_cached_block_hashes(blk)
        assert set(removed) == {k_main, k_alias}  # 一次摘干净
        assert blk.block_hash is None
        assert not cached_hashes(pool)

    def test_move_block_hashes_repoints_to_cow(self):
        # SOURCE: vllm/v1/core/block_pool.py:L629-L645
        pool = BlockPool(8, enable_caching=True, hash_block_size=8)
        src, dst = pool.get_new_blocks(2)
        k = make_block_hash_with_group_id(b"cache-key", 0)
        pool._insert_block_hash(k, src, num_tokens=24)
        pool.move_block_hashes(src, dst)
        assert src.block_hash is None
        assert dst.block_hash == k
        assert dst.block_hash_num_tokens == 24  # num_tokens 只跟主哈希
        assert pool.cached_block_hash_to_block.get_one_block(k) is dst


# --------------------------------------------------------------------------- #
# m12 粒度分离与惰性重串
# --------------------------------------------------------------------------- #


class TestBlockHashView:
    def test_block_hash_list_with_block_size(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2245-L2315（docstring 图例）
        hashes = [b"A", b"B", b"C", b"D"]
        view = BlockHashListWithBlockSize(hashes, 16, 32)
        assert len(view) == 2
        assert view[0] == b"B"  # 16 粒度第 2 个哈希直接当 32 粒度用
        assert view[1] == b"D"
        assert list(view) == [b"B", b"D"]
        assert view[:] == [b"B", b"D"]

    def test_resolve_block_hashes(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2321-L2351
        hashes = [b"A", b"B", b"C", b"D"]
        assert resolve_block_hashes(hashes, 16, 16) is hashes  # 等粒度直用
        view = resolve_block_hashes(hashes, 16, 32)
        assert isinstance(view, BlockHashListWithBlockSize)
        # 细粒度查找保留原始列表（alignment < block_size 且整除）
        fine = resolve_block_hashes(
            hashes, 16, 32, supports_fine_grained_hash_lookup=True,
            alignment_tokens=16,
        )
        assert fine is hashes
        # 对齐粒度 ≥ 块尺寸 → 粗视图
        coarse = resolve_block_hashes(
            hashes, 16, 32, supports_fine_grained_hash_lookup=True,
            alignment_tokens=32,
        )
        assert coarse is not hashes


# --------------------------------------------------------------------------- #
# m13/m14 块内 CoW 部分命中 + 拷贝过线
# --------------------------------------------------------------------------- ##


def make_partial_hit_manager():
    """full(64) + mamba(64, align) 混合；hash_bs=16 → enable_partial_hash_hits。"""
    mgr = make_manager([full_spec(64), mamba_spec(64, "align")],
                       hash_block_size=16, num_blocks=32)
    coord = mgr.coordinator
    assert isinstance(coord, HybridKVCacheCoordinator)
    assert coord.enable_partial_hash_hits is True
    assert coord._cache_hit_alignment_tokens == 16
    return mgr


def register_mamba_boundary_entry(mgr: KVCacheManager, req: Request,
                                  num_tokens: int) -> None:
    """mamba 组的边界状态注册：真实由 MambaManager.cache_blocks 重写（其内部
    调同一 block_pool.cache_partial_block 原语，L1729-L1735）完成——精简版按
    dossier.delete 第 6 条删了 mamba align 内部，测试以同一原语补上 mamba 组
    （group 1）的条目，复现真实可观察行为。"""
    mamba_mgr = mgr.coordinator.single_type_managers[1]
    mgr.block_pool.cache_partial_block(
        request=req,
        block=mamba_mgr.req_to_blocks[req.request_id][0],
        num_tokens=num_tokens,
        kv_cache_group_id=1,
        block_size=64,
    )


class TestPartialHitCoW:
    def test_cache_partial_block_registers_interior_boundary(self):
        # SOURCE: vllm/v1/core/block_pool.py:L445-L557
        mgr = make_partial_hit_manager()
        pool = mgr.block_pool
        req = make_request("a", list(range(48)), hasher=HASHER16)  # 3 个 16 边界
        run_request(mgr, req)
        # 48 token < block 64：满块 0；prompt 尾 48//16*16=48 → 部分条目
        full_mgr = mgr.coordinator.single_type_managers[0]
        blk = full_mgr.req_to_blocks["a"][0]
        assert blk.block_hash is not None
        assert blk.block_hash_num_tokens == 48  # 部分条目覆盖 48 token
        assert blk.block_hash == make_block_hash_with_group_id(
            req.block_hashes[2], 0
        )

    def test_find_phase2_probes_high_to_low_and_cow(self):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L741-L762 +
        #   L347-L357 + L405-L425
        mgr = make_partial_hit_manager()
        reqA = make_request("a", list(range(48)), hasher=HASHER16)
        run_request(mgr, reqA)
        register_mamba_boundary_entry(mgr, reqA, 48)
        mgr.free(reqA)
        # reqB 共享前 48 token（prompt 80）
        reqB = make_request("b", list(range(48)) + list(range(200, 232)),
                            hasher=HASHER16)
        blocks, hit, junction = mgr.get_computed_blocks(reqB)
        assert hit == 48  # phase 2 探到 48 边界（fine_idx=2，自高向低）
        assert junction == 0
        full_mgr = mgr.coordinator.single_type_managers[0]
        out = mgr.allocate_slots(
            reqB, reqB.num_tokens - hit,
            num_new_computed_tokens=hit, new_computed_blocks=blocks,
        )
        assert out is not None
        # 部分命中在 add_local_computed_blocks 记账、allocate_new_blocks 消费
        # （pop 掉换尾）——消费证据 = 拷贝对存在 + cow 块进块表
        assert "b" not in full_mgr._partial_hit_reqs  # 已被 CoW 换尾消费
        blk_table = full_mgr.req_to_blocks["b"]
        copies = full_mgr.take_pending_cow_copies()
        assert len(copies) == 1
        src, cow = copies[0]
        assert cow in blk_table
        assert cow.ref_cnt == 2  # 请求 1 + CoW 保留 1
        assert src.ref_cnt >= 1

    def test_take_kv_cache_block_copies_pipeline(self):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L831-L846
        mgr = make_partial_hit_manager()
        reqA = make_request("a", list(range(48)), hasher=HASHER16)
        run_request(mgr, reqA)
        register_mamba_boundary_entry(mgr, reqA, 48)
        mgr.free(reqA)
        reqB = make_request("b", list(range(48)) + list(range(200, 232)),
                            hasher=HASHER16)
        blocks, hit, _ = mgr.get_computed_blocks(reqB)
        mgr.allocate_slots(reqB, reqB.num_tokens - hit,
                           num_new_computed_tokens=hit, new_computed_blocks=blocks)
        # full 与 mamba 两组都部分命中 → 各自 CoW 换尾
        copies, retained = mgr.take_kv_cache_block_copies()
        assert len(copies) == 2
        assert all(c.src_block_id != c.dst_block_id for c in copies)
        assert len(retained) == 4  # 两端块引用保留（2 对 × 2）
        # drain 后再取为空
        assert mgr.take_kv_cache_block_copies() == ([], [])


class TestCopyOverTheWire:
    def test_copy_kv_cache_blocks_inplace_on_storage(self):
        # SOURCE: vllm/v1/worker/utils.py:L528-L564 + gpu_model_runner.py:L1219-L1228
        from implementation.gpu_model_runner import apply_scheduler_output_side_effects
        from implementation.kv_cache_utils import KVCacheBlockCopy
        from implementation.output import SchedulerOutput

        num_blocks, page = 4, 64
        t = torch.arange(num_blocks * page, dtype=torch.uint8)
        src_bytes = t[1 * page:2 * page].clone()
        out = SchedulerOutput(
            kv_cache_block_copies=[
                KVCacheBlockCopy(src_block_id=1, dst_block_id=3)
            ]
        )
        apply_scheduler_output_side_effects(out, [t], num_blocks)
        assert torch.equal(t[3 * page:4 * page], src_bytes)

    def test_scheduler_packs_copies_with_fence(self):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1181-L1190 + L2356-L2380
        sched = Scheduler(
            kv_config([full_spec(16)]), max_model_len=512,
            scheduler_block_size=16, hash_block_size=16,
        )
        pool = sched.kv_cache_manager.block_pool
        src, dst = pool.get_new_blocks(2)
        sched.kv_cache_manager.coordinator.single_type_managers[
            0]._pending_cow_copies.append((src, dst))
        sched.defer_block_free = True
        sched.sched_step_seq = 5
        out = sched.pack_kv_cache_block_copies()
        assert out is not None and len(out.kv_cache_block_copies) == 1
        # retained 未立即还池（步序栅栏），fence 过后才真 free
        assert src.ref_cnt == 1 and dst.ref_cnt == 1
        sched.processed_step_seq = 6  # fence = 5+1 = 6 已处理
        sched._drain_deferred_frees()
        assert src.ref_cnt == 0 and dst.ref_cnt == 0


# --------------------------------------------------------------------------- #
# m15 混合不动点
# --------------------------------------------------------------------------- #


class TestHybridFixedPoint:
    def test_simple_hybrid_full_and_swa_reconcile(self):
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L685-L817
        # full(16) + swa(16, window=48=3 块)
        mgr = make_manager([full_spec(16), swa_spec(16, 48)],
                           hash_block_size=16, num_blocks=64)
        coord = mgr.coordinator
        assert isinstance(coord, HybridKVCacheCoordinator)
        # full 排首（attention_groups 分桶结果）
        assert isinstance(coord.attention_groups[0].spec, FullAttentionSpec)
        # 请求 A 6 满块：full 组 6 块全缓存；SWA 组也稠密缓存
        reqA = make_request("a", list(range(96)))
        run_request(mgr, reqA)
        swa_mgr = coord.single_type_managers[1]
        swa_blocks = list(swa_mgr.req_to_blocks["a"])
        mgr.free(reqA)
        # 摘掉 SWA 组第 4 块（hash[3]）→ SWA 右到左的窗口连续段被拦腰打断，
        # 只能退到 [0..2] 三个连续块——把候选长度从 80 缩到 48
        mgr.block_pool._remove_cached_block_hashes(swa_blocks[3])
        reqB = make_request("b", list(range(96)))
        blocks, hit, boundary = mgr.get_computed_blocks(reqB)
        assert hit == 48  # SWA 窗 3 连续块把候选缩到 48
        # full 组块表被裁到最终 hit_length（向下封闭：查一次后只裁剪）
        assert len(blocks.blocks[0]) == 48 // 16
        # SWA 组：窗口内 3 块都是真块（无 null 占位——命中在序列头）
        assert all(not b.is_null for b in blocks.blocks[1])
        # boundary − hit = longest(80) − reconciled(48) = 32：full 缓得更深
        # 但 SWA 不认——num_uncached_common_prefix_tokens 的写回形态
        assert boundary == 48 + 32

    def test_num_uncached_common_prefix_tokens(self):
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L810-L817
        # partial-hit 粒度配置（full(64)+mamba(64,align)、hash_bs=16）——差分
        # 电池证明该配置下 impl 的 mamba 边界替换与钉版真源码逐字节一致；
        # block_size==hash_block_size 配置（如 full(16)+mamba(16,align)、
        # hash_bs=16）impl 与 pin 有已注记边界（impl-notes.md）：真实 align 块
        # 表是 [NULL×4, 唯一状态块] 只登记最后一块（hash[4]@80），而
        # max_cache_hit_length=num_tokens−1 永探不到 → mamba 组恒 miss、不动点
        # 把整笔命中拖 0（钉版差分实测 hit==0/boundary==64），不在此断言。
        mgr = make_partial_hit_manager()
        reqA = make_request("a", list(range(48)), hasher=HASHER16)
        run_request(mgr, reqA)
        register_mamba_boundary_entry(mgr, reqA, 48)
        mamba_mgr = mgr.coordinator.single_type_managers[1]
        mamba_block = mamba_mgr.req_to_blocks["a"][0]  # 持 mamba 组 @48 条目
        mgr.free(reqA)
        mgr.new_step_starts()  # 调度步边界协议（真实每步调；驱动跨步准入）
        # 两组都持 @48 → reconciled==longest → uncached=0 → boundary 归零
        reqB = make_request("b", list(range(48)) + list(range(200, 232)),
                            hasher=HASHER16)  # 80 token 共享前 48
        _, hit, boundary = mgr.get_computed_blocks(reqB)
        assert hit == 48
        assert boundary == 0  # uncached=0 → boundary 归零（if num_uncached else 0）
        # 差值场景：摘掉 mamba 组 @48 条目 → full 组仍持 48（longest）、mamba
        # miss → reconciled 0、uncached=48（各组都认但稀疏组还没缓的共享前缀）
        mgr.block_pool._remove_cached_block_hashes(mamba_block)
        reqC = make_request("c", list(range(48)) + list(range(200, 232)),
                            hasher=HASHER16)
        blocks, hit_c, boundary_c = mgr.get_computed_blocks(reqC)
        assert hit_c == 0
        assert blocks is mgr.empty_kv_cache_blocks  # 不动点拖到全 miss
        assert boundary_c == 0 + 48  # uncached = longest(48) − reconciled(0)


# --------------------------------------------------------------------------- #
# m16 Marconi 钉住 / m17 retention
# --------------------------------------------------------------------------- #


class TestMarconiPinning:
    def test_mamba_reachable_block_mask_three_modes(self):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1358-L1414
        from implementation.single_type_kv_cache_manager import MambaManager

        spec = mamba_spec(16, "align")
        # None → 稠密（全缓存）
        assert MambaManager.reachable_block_mask(
            0, 8, alignment_tokens=16, kv_cache_spec=spec, use_eagle=False,
        ) is None
        # 0 → 只留 reachable_boundaries
        mask = MambaManager.reachable_block_mask(
            0, 8, alignment_tokens=16, kv_cache_spec=spec, use_eagle=False,
            retention_interval=0, reachable_boundaries=[79],
        )
        assert mask is not None
        assert sum(mask) == 1  # 只在边界那一个状态块
        assert mask[79 // 16 - 1] is True  # 边界 79 → 块 4（0-based 3）
        # 正数 → 每段一条 + 边界特赦
        mask32 = MambaManager.reachable_block_mask(
            0, 8, alignment_tokens=16, kv_cache_spec=spec, use_eagle=False,
            retention_interval=32, reachable_boundaries=[79],
        )
        seg_positions = {i for i, v in enumerate(mask32) if v}
        assert {1, 3, 5, 7}.issubset(seg_positions)  # 每 2 块一段的尾
        assert 3 in seg_positions  # replay 边界特赦

    def test_swa_reachable_mask_sparse_and_boundary(self):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L995-L1055
        from implementation.single_type_kv_cache_manager import SlidingWindowManager

        spec = swa_spec(16, 48)
        mask = SlidingWindowManager.reachable_block_mask(
            0, 8, alignment_tokens=16, kv_cache_spec=spec, use_eagle=False,
            retention_interval=0, reachable_boundaries=[79],
        )
        need = 3  # cdiv(48-1,16)=3 连续块
        end = 79 // 16 + 0  # shift=0
        for j in range(max(0, end - need), min(8, end)):
            assert mask[j] is True  # replay 边界的 need 块尾强制 True
        assert sum(mask) == need

    def test_shared_prefix_boundary_written_back(self):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L286-L295 + request.py:L190-L193
        # + scheduler.py:L760-L766（调度器写、cache_blocks/_mamba_block_aligned_
        # split 读的跨模块协议）
        # partial-hit 粒度配置（block_size > hash_block_size）——差分电池证明
        # impl 的 mamba 边界替换与钉版逐字节一致；block_size==hash_block_size
        # 配置的 impl≠pin 边界见 impl-notes.md（真实 mamba 恒 miss → 0/64，
        # 不在此断言）。写回控制流本身与钉版 L744-L766 逐字一致。
        mgr = make_partial_hit_manager()
        pool = mgr.block_pool
        reqA = make_request("a", list(range(48)), hasher=HASHER16)
        run_request(mgr, reqA)
        register_mamba_boundary_entry(mgr, reqA, 48)
        mamba_mgr = mgr.coordinator.single_type_managers[1]
        mamba_block = mamba_mgr.req_to_blocks["a"][0]  # 持 mamba 组 @48 条目
        mgr.free(reqA)
        mgr.new_step_starts()  # 调度步边界协议（真实每步调）
        # 摘掉 mamba 组 @48 边界条目 → B 查出 uncached>0 → boundary = hit+uncached
        pool._remove_cached_block_hashes(mamba_block)
        reqB = make_request("b", list(range(48)) + list(range(200, 232)),
                            hasher=HASHER16)  # 80 token 共享前 48
        _, hit, boundary = mgr.get_computed_blocks(reqB)
        assert hit == 0
        assert boundary == 0 + 48  # hit + uncached = 最长单组命中(48)
        sched = Scheduler(kv_config([full_spec(64), mamba_spec(64, "align")]),
                          max_model_len=512, scheduler_block_size=64,
                          hash_block_size=16)
        sched.kv_cache_manager = mgr
        sched.admission_lookup(reqB)
        assert reqB.shared_prefix_boundary == boundary  # 写回 Request

    def test_mamba_block_aligned_split_stops_at_junction(self):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L362-L437
        sched = Scheduler(kv_config([full_spec(32), mamba_spec(32, "align")]),
                          max_model_len=512, scheduler_block_size=32,
                          hash_block_size=16)
        req = make_request("r", list(range(200)), hasher=HASHER16)
        req.num_computed_tokens = 0
        req.shared_prefix_boundary = 64
        # chunk [0,100) → 停在 junction 64（块对齐下取整：0+(64-0)//32*32）
        n = sched._mamba_block_aligned_split(req, num_new_tokens=100)
        assert n == 64
        # junction 在区间外 → 不截
        n2 = sched._mamba_block_aligned_split(req, num_new_tokens=30)
        assert n2 == 30
        # 无 junction、partial-tail 边界（192）在 chunk [0,100) 外 → 也不截
        req2 = make_request("r2", list(range(200)), hasher=HASHER16)
        sched.mamba_partial_cache_hit = True
        n3 = sched._mamba_block_aligned_split(req2, num_new_tokens=100)
        assert n3 == 100


class TestRetentionValidation:
    def test_validate_three_states(self):
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L30-L57
        cfg_swa = kv_config([full_spec(16), swa_spec(16, 48)])
        cfg_plain = kv_config([full_spec(16)])
        _validate_prefix_cache_retention_interval(None, 16, cfg_plain)  # ok
        with pytest.raises(ValueError, match="no sliding-window or Mamba"):
            _validate_prefix_cache_retention_interval(16, 16, cfg_plain)
        with pytest.raises(ValueError, match="non-negative"):
            _validate_prefix_cache_retention_interval(-16, 16, cfg_swa)
        with pytest.raises(ValueError, match="multiple of scheduler_block_size"):
            _validate_prefix_cache_retention_interval(24, 16, cfg_swa)
        _validate_prefix_cache_retention_interval(32, 16, cfg_swa)  # ok


# --------------------------------------------------------------------------- #
# m11 F2：抢占恢复撞前缀缓存
# --------------------------------------------------------------------------- #


class TestPreemptionRecovery:
    def test_preempt_free_keeps_hash_and_rehit(self):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1274-L1315 + L744-L766
        sched = Scheduler(kv_config([full_spec(16)]), max_model_len=512,
                          scheduler_block_size=16, hash_block_size=16)
        mgr = sched.kv_cache_manager
        pool = mgr.block_pool
        req = make_request("a", list(range(64)))
        run_request(mgr, req)
        req.status = RequestStatus.RUNNING
        sched.waiting.append(req)
        sched._preempt_request(req)
        assert req.status is RequestStatus.PREEMPTED
        assert req.num_computed_tokens == 0
        assert sched.waiting[0] is req  # 回 waiting 队头
        assert req.num_preemptions == 1
        # 哈希保留：块全在表里、带哈希排在 LRU 尾
        assert len(cached_hashes(pool)) == 4
        # 被抢占请求重排回来重走准入：重命中自己的前缀（max 63 → 3 块 48）
        blocks, hit, _ = sched.admission_lookup(req)
        assert hit == 48
        # touch 救回：分配后块不在 free queue
        out = mgr.allocate_slots(req, 64 - hit, num_new_computed_tokens=hit,
                                 new_computed_blocks=blocks)
        assert out is not None
        st = mgr.coordinator.single_type_managers[0]
        used = {b.block_id for b in st.req_to_blocks["a"]}
        free_ids = set(free_queue_ids(pool))
        assert not (used & free_ids)

    def test_preempt_worst_case_full_recompute(self):
        # F2 最坏分支：被抢占期间块被取走复用（惰性驱逐）→ 全量重 prefill
        sched = Scheduler(kv_config([full_spec(16)], num_blocks=8),
                          max_model_len=512,
                          scheduler_block_size=16, hash_block_size=16)
        mgr = sched.kv_cache_manager
        pool = mgr.block_pool
        req = make_request("a", list(range(48)))
        run_request(mgr, req)
        req.status = RequestStatus.RUNNING
        sched._preempt_request(req)
        # 抢占期间池紧：别的请求把队头的 a 前缀块取走复用（摘哈希）
        while pool.get_num_free_blocks() > 0:
            pool.get_new_blocks(1)
        assert len(cached_hashes(pool)) == 0  # 全被惰性驱逐
        blocks, hit, _ = mgr.get_computed_blocks(req)
        assert hit == 0  # 前缀失效 → 退化为全量重 prefill


# --------------------------------------------------------------------------- #
# m18 开关面 / m19 观测 / m20 reset
# --------------------------------------------------------------------------- #


class TestSwitchAndStats:
    def test_caching_off_no_prefix_cache_coordinator(self):
        # SOURCE: vllm/v1/core/kv_cache_coordinator.py:L864-L876
        mgr = make_manager([full_spec(16)], hash_block_size=16,
                           enable_caching=False)
        assert isinstance(mgr.coordinator,
                          KVCacheCoordinatorNoPrefixCache)
        req = make_request("a", list(range(64)))
        blocks, hit, boundary = mgr.get_computed_blocks(req)
        assert hit == 0 and boundary == 0

    def test_unitary_coordinator_single_group(self):
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        assert isinstance(mgr.coordinator, UnitaryKVCacheCoordinator)

    def test_prefix_cache_stats_record(self):
        # SOURCE: vllm/v1/metrics/stats.py:L115-L142
        stats = PrefixCacheStats()
        stats.record(num_tokens=100, num_hits=80, preempted=False)
        stats.record(num_tokens=64, num_hits=48, preempted=True)
        assert stats.requests == 1 and stats.queries == 100 and stats.hits == 80
        assert stats.preempted_requests == 1
        assert stats.preempted_queries == 64 and stats.preempted_hits == 48

    def test_reset_prefix_cache_requires_all_free(self):
        # SOURCE: vllm/v1/core/block_pool.py:L763-L797
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        pool = mgr.block_pool
        req = make_request("a", list(range(32)))
        run_request(mgr, req)
        assert mgr.reset_prefix_cache() is False  # 有在用块 → 拒绝
        mgr.free(req)
        assert mgr.reset_prefix_cache() is True
        assert len(cached_hashes(pool)) == 0
        for blk in pool.blocks:
            assert blk.block_hash is None


class TestManagerFreeOrderIntact:
    def test_kv_cache_manager_free_reverse(self):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L567-L578（注释原话
        #   『tail blocks are evicted first』）
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        pool = mgr.block_pool
        req = make_request("a", list(range(48)))
        run_request(mgr, req)
        mgr.free(req)
        # 3 个带哈希块逆序挂回 LRU 尾：驱逐序尾段 [3,2,1]
        assert free_queue_ids(pool)[-3:] == [3, 2, 1]

    def test_full_hit_all_token_recompute_last(self):
        # theory：全命中也须重算最后一个 token 拿 logits；块对齐回退整块
        mgr = make_manager([full_spec(16)], hash_block_size=16)
        reqA = make_request("a", list(range(17)))  # 1 满块+1 尾
        run_request(mgr, reqA)
        mgr.free(reqA)
        reqB = make_request("b", list(range(17)))
        _, hit, _ = mgr.get_computed_blocks(reqB)
        # max=16 → 1 块=16；B 只需重算 1 个 token（17-16）
        assert hit == 16
