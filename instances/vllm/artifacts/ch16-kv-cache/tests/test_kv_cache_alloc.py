"""TDD 测试 —— 以真实 vLLM v1 KV 块分配与多注意力协调的可观察行为为准绳。

不验证精简版自洽，而是断言它复现真实 vllm/v1/core 的行为：
- allocate_slots 三阶段：释放 skipped 块 / 挂前缀命中 + external 命中 / 新建 new+lookahead；
- get_kv_cache_coordinator 三态工厂（不缓存 / 单组 Unitary / 多组 Hybrid）；
- SlidingWindow / ChunkedLocal 的 get_num_skipped_tokens 差异化；
- remove_skipped_blocks 逆序换 null + 归还 free queue（append-only 下标不变）；
- HybridKVCacheCoordinator.find_longest_cache_hit 不动点迭代收敛；
- get_num_blocks_to_allocate 的 skipped 折抵 + 可驱逐块计数 + admission cap；
- num_tokens_to_cache 封顶到 request.num_tokens（防草稿污染缓存）。
"""
import hashlib

from implementation.block_pool import BlockPool
from implementation.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache,
    UnitaryKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from implementation.kv_cache_manager import KVCacheManager
from implementation.kv_cache_utils import get_request_block_hasher
from implementation.request import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    Request,
    SlidingWindowSpec,
)
from implementation.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    FullAttentionManager,
    SlidingWindowManager,
    get_manager_for_kv_cache_spec,
    spec_manager_map,
)

BLOCK_SIZE = 4


def _hash_fn(x):
    return hashlib.sha256(repr(x).encode()).digest()


def make_request(rid, token_ids, block_size=BLOCK_SIZE, **kw):
    hasher = get_request_block_hasher(block_size, _hash_fn)
    return Request(rid, token_ids, block_hasher=hasher, **kw)


def _full_config(num_blocks, block_size=BLOCK_SIZE):
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[
            KVCacheGroupSpec(kv_cache_spec=FullAttentionSpec(block_size=block_size))
        ],
    )


def make_manager(num_blocks=64, block_size=BLOCK_SIZE, enable_caching=True,
                 max_model_len=1024, max_num_batched_tokens=1024):
    return KVCacheManager(
        kv_cache_config=_full_config(num_blocks, block_size),
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        hash_block_size=block_size,
        enable_caching=enable_caching,
    )


# --------------------- 三态工厂 get_kv_cache_coordinator ---------------------

def test_factory_no_prefix_cache_when_disabled():
    coord = get_kv_cache_coordinator(
        _full_config(16), max_model_len=128, max_num_batched_tokens=128,
        enable_caching=False, hash_block_size=BLOCK_SIZE,
    )
    assert isinstance(coord, KVCacheCoordinatorNoPrefixCache)


def test_factory_unitary_for_single_group():
    coord = get_kv_cache_coordinator(
        _full_config(16), max_model_len=128, max_num_batched_tokens=128,
        enable_caching=True, hash_block_size=BLOCK_SIZE,
    )
    assert isinstance(coord, UnitaryKVCacheCoordinator)


def test_factory_hybrid_for_multiple_groups():
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_groups=[
            KVCacheGroupSpec(kv_cache_spec=FullAttentionSpec(block_size=BLOCK_SIZE)),
            KVCacheGroupSpec(
                kv_cache_spec=SlidingWindowSpec(
                    block_size=BLOCK_SIZE, sliding_window=8
                )
            ),
        ],
    )
    coord = get_kv_cache_coordinator(
        config, max_model_len=128, max_num_batched_tokens=128,
        enable_caching=True, hash_block_size=BLOCK_SIZE,
    )
    assert isinstance(coord, HybridKVCacheCoordinator)


def test_no_prefix_cache_supports_zero_groups():
    # 关缓存协调器支持任意组数（含 0 组），命中恒空。
    config = KVCacheConfig(num_blocks=8, kv_cache_groups=[])
    coord = get_kv_cache_coordinator(
        config, max_model_len=64, max_num_batched_tokens=64,
        enable_caching=False, hash_block_size=BLOCK_SIZE,
    )
    blocks, length = coord.find_longest_cache_hit([], 16)
    assert blocks == () and length == 0


# --------------------- spec → manager 映射 + admission cap 注入 ---------------------

def test_spec_manager_map_dispatch():
    assert spec_manager_map[FullAttentionSpec] is FullAttentionManager
    assert spec_manager_map[SlidingWindowSpec] is SlidingWindowManager
    assert spec_manager_map[ChunkedLocalAttentionSpec] is ChunkedLocalAttentionManager


def test_admission_cap_injected_for_sliding_window():
    pool = BlockPool(64, True, BLOCK_SIZE)
    spec = SlidingWindowSpec(block_size=BLOCK_SIZE, sliding_window=8)
    mgr = get_manager_for_kv_cache_spec(
        spec, max_num_batched_tokens=16, max_model_len=1024,
        block_pool=pool, enable_caching=True, kv_cache_group_id=0,
    )
    # cdiv(8-1+16, 4)+1 = cdiv(23,4)+1 = 6+1 = 7
    assert mgr._max_admission_blocks_per_request == 7


def test_admission_cap_none_for_full_attention():
    pool = BlockPool(64, True, BLOCK_SIZE)
    spec = FullAttentionSpec(block_size=BLOCK_SIZE)
    mgr = get_manager_for_kv_cache_spec(
        spec, max_num_batched_tokens=16, max_model_len=1024,
        block_pool=pool, enable_caching=True, kv_cache_group_id=0,
    )
    assert mgr._max_admission_blocks_per_request is None


# --------------------- get_num_skipped_tokens 三态差异 ---------------------

def test_full_attention_never_skips():
    pool = BlockPool(64, True, BLOCK_SIZE)
    mgr = FullAttentionManager(
        FullAttentionSpec(block_size=BLOCK_SIZE), pool, True, 0
    )
    assert mgr.get_num_skipped_tokens(0) == 0
    assert mgr.get_num_skipped_tokens(1000) == 0


def test_sliding_window_skipped_tokens():
    pool = BlockPool(64, True, BLOCK_SIZE)
    spec = SlidingWindowSpec(block_size=BLOCK_SIZE, sliding_window=4)
    mgr = SlidingWindowManager(
        spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0
    )
    # max(0, num_computed - sliding_window + 1)
    assert mgr.get_num_skipped_tokens(3) == 0
    assert mgr.get_num_skipped_tokens(7) == 4
    assert mgr.get_num_skipped_tokens(0) == 0


def test_chunked_local_skipped_tokens_rounds_to_chunk():
    pool = BlockPool(64, True, BLOCK_SIZE)
    spec = ChunkedLocalAttentionSpec(block_size=BLOCK_SIZE, attention_chunk_size=8)
    mgr = ChunkedLocalAttentionManager(
        spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0
    )
    # (num_computed // chunk) * chunk
    assert mgr.get_num_skipped_tokens(13) == 8
    assert mgr.get_num_skipped_tokens(8) == 8
    assert mgr.get_num_skipped_tokens(7) == 0


# --------------------- remove_skipped_blocks 逆序换 null + 归还 ---------------------

def test_remove_skipped_blocks_recycles_window_outside():
    # sliding_window=8, block_size=4, num_computed=11 -> skip 4 -> 1 块换 null。
    pool = BlockPool(64, True, BLOCK_SIZE)
    spec = SlidingWindowSpec(block_size=BLOCK_SIZE, sliding_window=8)
    mgr = SlidingWindowManager(
        spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0
    )
    blocks = pool.get_new_blocks(3)  # B0,B1,B2
    mgr.req_to_blocks["r"] = list(blocks)
    free_before = pool.get_num_free_blocks()

    mgr.remove_skipped_blocks("r", total_computed_tokens=11)

    req_blocks = mgr.req_to_blocks["r"]
    # 下标不变（append-only），仅 B0 被换成 null。
    assert req_blocks[0] is pool.null_block
    assert req_blocks[1] is blocks[1]
    assert req_blocks[2] is blocks[2]
    # 真实块 B0 归还 free queue。
    assert pool.get_num_free_blocks() == free_before + 1


def test_remove_skipped_blocks_idempotent_early_stop():
    pool = BlockPool(64, True, BLOCK_SIZE)
    spec = SlidingWindowSpec(block_size=BLOCK_SIZE, sliding_window=8)
    mgr = SlidingWindowManager(
        spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0
    )
    blocks = pool.get_new_blocks(3)
    mgr.req_to_blocks["r"] = list(blocks)
    mgr.remove_skipped_blocks("r", 11)
    free_after_first = pool.get_num_free_blocks()
    # 再次调用同样窗口：遇已 null 早停，不重复归还。
    mgr.remove_skipped_blocks("r", 11)
    assert pool.get_num_free_blocks() == free_after_first


def test_full_attention_remove_skipped_is_noop():
    pool = BlockPool(64, True, BLOCK_SIZE)
    mgr = FullAttentionManager(
        FullAttentionSpec(block_size=BLOCK_SIZE), pool, True, 0
    )
    blocks = pool.get_new_blocks(3)
    mgr.req_to_blocks["r"] = list(blocks)
    free_before = pool.get_num_free_blocks()
    mgr.remove_skipped_blocks("r", 1000)
    assert mgr.req_to_blocks["r"] == list(blocks)
    assert pool.get_num_free_blocks() == free_before


# --------------------- allocate_slots 端到端（全注意力主路径，承接 ch15）---------------------

def test_allocate_slots_basic_full_attention():
    mgr = make_manager(num_blocks=64)
    req = make_request("r", list(range(20)))  # 5 块
    blocks = mgr.allocate_slots(req, num_new_tokens=20)
    assert blocks is not None
    assert len(blocks.blocks[0]) == 5


def test_allocate_slots_returns_none_when_insufficient():
    mgr = make_manager(num_blocks=4)  # 池小（null 占 1，可用 3）
    req = make_request("r", list(range(40)))  # 需 10 块
    assert mgr.allocate_slots(req, num_new_tokens=40) is None


def test_allocate_slots_raises_when_no_tokens_at_all():
    mgr = make_manager()
    req = make_request("r", list(range(8)))
    try:
        mgr.allocate_slots(req, num_new_tokens=0, num_external_computed_tokens=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no new and no external tokens")


# --------------------- num_tokens_to_cache 封顶到 request.num_tokens ---------------------

def test_lookahead_tokens_not_cached():
    # lookahead（投机草稿）预留槽位，但 num_tokens_to_cache 封顶到 request.num_tokens，
    # 草稿块不应被写入前缀缓存供他人命中。
    mgr = make_manager(num_blocks=64)
    req = make_request("r", list(range(8)))  # 8 token = 2 满块
    mgr.allocate_slots(req, num_new_tokens=8, num_lookahead_tokens=8)
    coord = mgr.coordinator
    single = coord.single_type_managers[0]
    # 只缓存定稿的 2 块，未把 lookahead 槽位算进缓存满块。
    assert single.num_cached_block["r"] == 2


def test_lookahead_reserves_slots():
    mgr = make_manager(num_blocks=64)
    req = make_request("r", list(range(8)))
    blocks = mgr.allocate_slots(req, num_new_tokens=8, num_lookahead_tokens=8)
    # 8 main + 8 lookahead = 16 token = 4 块槽位。
    assert len(blocks.blocks[0]) == 4


# --------------------- external computed tokens 分配真实块 ---------------------

def test_external_computed_tokens_allocate_real_blocks():
    # connector 命中的 external token 不是 vLLM 前缀命中块，需 get_new_blocks 分配真实块。
    mgr = make_manager(num_blocks=64)
    req = make_request("r", list(range(16)))
    free_before = mgr.block_pool.get_num_free_blocks()
    # 8 external token（2 块）已由 connector 算好，0 个本地命中块。
    blocks = mgr.allocate_slots(
        req, num_new_tokens=8, num_external_computed_tokens=8,
    )
    assert blocks is not None
    single = mgr.coordinator.single_type_managers[0]
    # external 的 2 块 + new 8 token 的 2 块 = 4 块。
    assert len(single.req_to_blocks["r"]) == 4
    assert mgr.block_pool.get_num_free_blocks() == free_before - 4


# --------------------- get_num_blocks_to_allocate：可驱逐命中块计入预算 ---------------------

def test_get_num_blocks_counts_evictable_hit_blocks():
    # 先让 r1 跑完并释放，使其块成为 ref_cnt==0 的驱逐候选（仍在 cache map 中）。
    mgr = make_manager(num_blocks=64)
    r1 = make_request("r1", list(range(16)))
    mgr.allocate_slots(r1, num_new_tokens=16)
    mgr.free(r1)

    # r2 同前缀，命中这些可驱逐块。
    r2 = make_request("r2", list(range(16)))
    computed, n_hit = mgr.get_computed_blocks(r2)
    assert n_hit > 0
    coord = mgr.coordinator
    single = coord.single_type_managers[0]
    n = single.get_num_blocks_to_allocate(
        "r2",
        num_tokens=16,
        new_computed_blocks=computed.blocks[0],
        total_computed_tokens=n_hit,
        num_tokens_main_model=16,
    )
    # num_required=cdiv(16,4)=4，命中 3 块 -> 新建 1 块；3 个命中块都是 free queue 中
    # 的驱逐候选，touch 时占容量须计入 -> 1 + 3 = 4。
    assert len(computed.blocks[0]) == 3
    assert n == 1 + 3


def test_admission_cap_clamps_required_blocks():
    pool = BlockPool(64, True, BLOCK_SIZE)
    spec = SlidingWindowSpec(block_size=BLOCK_SIZE, sliding_window=8)
    mgr = get_manager_for_kv_cache_spec(
        spec, max_num_batched_tokens=16, max_model_len=1024,
        block_pool=pool, enable_caching=True, kv_cache_group_id=0,
    )
    # 不开 cap：需块数 = cdiv(400,4) = 100。
    n_uncapped = mgr.get_num_blocks_to_allocate(
        "r", num_tokens=400, new_computed_blocks=[],
        total_computed_tokens=0, num_tokens_main_model=400,
    )
    # 开 cap：夹到 max_admission_blocks_per_request = 7（且无 skipped/命中时即为该值）。
    n_capped = mgr.get_num_blocks_to_allocate(
        "r", num_tokens=400, new_computed_blocks=[],
        total_computed_tokens=0, num_tokens_main_model=400,
        apply_admission_cap=True,
    )
    assert n_uncapped == 100
    assert n_capped == 7


# --------------------- Unitary 命中委托唯一 manager ---------------------

def test_unitary_find_longest_cache_hit_delegates():
    mgr = make_manager(num_blocks=64)
    r1 = make_request("r1", list(range(16)))
    mgr.allocate_slots(r1, num_new_tokens=16)
    r2 = make_request("r2", list(range(16)))
    computed, n_hit = mgr.get_computed_blocks(r2)
    # 16 token，max_cache_hit_length = num_tokens-1 = 15 -> 3 满块 = 12 token。
    assert n_hit == 12
    assert len(computed.blocks[0]) == 3


# --------------------- Hybrid 不动点迭代收敛（full + sliding window）---------------------

def _hybrid_coord(num_blocks=64, block_size=BLOCK_SIZE, sliding_window=8):
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[
            KVCacheGroupSpec(kv_cache_spec=FullAttentionSpec(block_size=block_size)),
            KVCacheGroupSpec(
                kv_cache_spec=SlidingWindowSpec(
                    block_size=block_size, sliding_window=sliding_window
                )
            ),
        ],
    )
    return get_kv_cache_coordinator(
        config, max_model_len=1024, max_num_batched_tokens=1024,
        enable_caching=True, hash_block_size=block_size,
    )


def test_hybrid_verify_and_split_full_attn_first():
    coord = _hybrid_coord()
    # full attention 排在分桶首位（更紧初始上界）。
    first_spec = coord.attention_groups[0][0]
    assert isinstance(first_spec, FullAttentionSpec)
    # 两类 block_size 相同 -> lcm == block_size。
    assert coord.lcm_block_size == BLOCK_SIZE


def test_hybrid_simple_hybrid_flag():
    coord = _hybrid_coord()
    # 1 full + 1 other -> simple hybrid（一轮早停）。
    assert len(coord.attention_groups) == 2
    is_simple = len(coord.attention_groups) == 2 and isinstance(
        coord.attention_groups[0][0], FullAttentionSpec
    )
    assert is_simple


def test_hybrid_find_longest_cache_hit_converges_to_min():
    # 让 full 与 sliding window 两组都缓存同一前缀，命中长度取两类一致的最短。
    coord = _hybrid_coord(sliding_window=8)
    # 构造一个请求并把它的满块分别缓存进两个 group。
    req = make_request("r", list(range(24)))  # 6 满块
    block_hashes = req.block_hashes
    pool = coord.block_pool

    # 为每个 group 各分配并缓存 6 块（full 全缓存；sliding window 也全缓存）。
    for gid, mgr in enumerate(coord.single_type_managers):
        blocks = pool.get_new_blocks(6)
        mgr.req_to_blocks[req.request_id] = list(blocks)
        mgr.cache_blocks(req, num_tokens=24)

    # 不动点迭代：两组都命中 6 块（24 token），但 max_cache_hit_length = 23 -> 上界 20。
    hit_blocks, hit_len = coord.find_longest_cache_hit(block_hashes, 23)
    # 收敛到的命中长度是两类一致接受的长度，且为 lcm 的倍数、不超过上界。
    assert hit_len % coord.lcm_block_size == 0
    assert hit_len <= 20
    assert hit_len > 0
    # full 组的块数与命中长度一致。
    full_gid = coord.attention_groups[0][1][0]
    assert len(hit_blocks[full_gid]) == hit_len // BLOCK_SIZE


def test_hybrid_no_hit_returns_zero():
    coord = _hybrid_coord()
    req = make_request("r", list(range(24)))
    # 未缓存任何块 -> 命中为 0。
    hit_blocks, hit_len = coord.find_longest_cache_hit(req.block_hashes, 23)
    assert hit_len == 0
