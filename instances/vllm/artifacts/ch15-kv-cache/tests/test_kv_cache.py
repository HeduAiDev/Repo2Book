"""TDD 测试 —— 以真实 vLLM v1 分页 KV 缓存的可观察行为为准绳。

这些测试不是验证精简版自洽，而是断言它复现真实 vllm/v1/core 的行为：
双向链表 LRU 顺序、链式块哈希前缀一致性、extra keys 语义隔离、引用计数
touch/free、前缀命中复用、抢占重算的前缀缓存缓解（f11）。
"""
import hashlib

from implementation.block_pool import BlockHashToBlockMap, BlockPool
from implementation.kv_cache_manager import KVCacheManager
from implementation.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_request_block_hasher,
    hash_block_tokens,
    make_block_hash_with_group_id,
    need_extra_keys,
)
from implementation.request import FullAttentionSpec, Request

BLOCK_SIZE = 4


def _hash_fn(x):
    return hashlib.sha256(repr(x).encode()).digest()


def make_request(rid, token_ids, block_size=BLOCK_SIZE, **kw):
    hasher = get_request_block_hasher(block_size, _hash_fn)
    return Request(rid, token_ids, block_hasher=hasher, **kw)


# ----------------------- FreeKVCacheBlockQueue (双向链表 LRU) -----------------------

def test_queue_init_order_is_by_block_id():
    blocks = [KVCacheBlock(i) for i in range(4)]
    q = FreeKVCacheBlockQueue(blocks)
    assert q.num_free_blocks == 4
    assert [b.block_id for b in q.get_all_free_blocks()] == [0, 1, 2, 3]


def test_popleft_takes_front_lru():
    blocks = [KVCacheBlock(i) for i in range(4)]
    q = FreeKVCacheBlockQueue(blocks)
    first = q.popleft()
    assert first.block_id == 0
    assert q.num_free_blocks == 3
    # popped block detached from list
    assert first.prev_free_block is None and first.next_free_block is None
    assert [b.block_id for b in q.get_all_free_blocks()] == [1, 2, 3]


def test_popleft_empty_raises():
    q = FreeKVCacheBlockQueue([])
    try:
        q.popleft()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_popleft_n():
    blocks = [KVCacheBlock(i) for i in range(5)]
    q = FreeKVCacheBlockQueue(blocks)
    got = q.popleft_n(3)
    assert [b.block_id for b in got] == [0, 1, 2]
    assert q.num_free_blocks == 2
    assert [b.block_id for b in q.get_all_free_blocks()] == [3, 4]


def test_remove_middle_is_o1_and_keeps_others():
    blocks = [KVCacheBlock(i) for i in range(4)]
    q = FreeKVCacheBlockQueue(blocks)
    q.remove(blocks[1])
    assert q.num_free_blocks == 3
    assert [b.block_id for b in q.get_all_free_blocks()] == [0, 2, 3]


def test_append_n_goes_to_tail():
    blocks = [KVCacheBlock(i) for i in range(3)]
    q = FreeKVCacheBlockQueue(blocks)
    taken = q.popleft_n(3)
    # free in reversed order (tail of request first) -> tail-most ends up front
    q.append_n(list(reversed(taken)))
    assert [b.block_id for b in q.get_all_free_blocks()] == [2, 1, 0]


# ----------------------- 链式块哈希 -----------------------

def test_block_hash_is_chained_with_parent():
    toks = [1, 2, 3, 4]
    h0 = hash_block_tokens(_hash_fn, None, toks)
    h1_same_parent = hash_block_tokens(_hash_fn, h0, [5, 6, 7, 8])
    # 同样的 token 但父哈希不同 -> 不同哈希（前缀一致才同 hash）
    h1_diff_parent = hash_block_tokens(_hash_fn, _hash_fn(b"other"), [5, 6, 7, 8])
    assert h1_same_parent != h1_diff_parent


def test_request_block_hasher_only_hashes_full_blocks():
    # 10 tokens, block_size 4 -> 2 full blocks (8 tokens), 2 leftover
    req = make_request("r", list(range(10)))
    assert len(req.block_hashes) == 2


def test_request_block_hasher_chain_matches_manual():
    req = make_request("r", list(range(8)))
    h0 = hash_block_tokens(_hash_fn, None, [0, 1, 2, 3])
    h1 = hash_block_tokens(_hash_fn, h0, [4, 5, 6, 7])
    assert req.block_hashes == [h0, h1]


def test_append_extends_block_hashes():
    req = make_request("r", list(range(4)))
    assert len(req.block_hashes) == 1
    req.append_output_token_ids([4, 5, 6, 7])
    assert len(req.block_hashes) == 2


# ----------------------- extra keys (语义隔离) -----------------------

def test_need_extra_keys_plain_request_false():
    req = make_request("r", list(range(8)))
    assert need_extra_keys(req) is False


def test_cache_salt_only_first_block():
    req = make_request("r", list(range(8)), cache_salt="tenantA")
    assert need_extra_keys(req) is True
    # first block (start=0) gets salt
    keys0, _ = generate_block_hash_extra_keys(req, 0, 4, 0)
    assert keys0 == ("tenantA",)
    # later block (start>0) does not
    keys1, _ = generate_block_hash_extra_keys(req, 4, 8, 0)
    assert keys1 is None


def test_cache_salt_changes_block_hash():
    plain = make_request("a", list(range(8)))
    salted = make_request("b", list(range(8)), cache_salt="ns")
    # first block hash differs due to salt; rest of prefix-cache isolation follows
    assert plain.block_hashes[0] != salted.block_hashes[0]


class _LoRA:
    def __init__(self, name):
        self.lora_name = name


def test_lora_name_in_extra_keys():
    req = make_request("r", list(range(8)), lora_request=_LoRA("adapterX"))
    assert need_extra_keys(req) is True
    keys, _ = generate_block_hash_extra_keys(req, 0, 4, 0)
    assert keys == ("adapterX",)


# ----------------------- BlockHashToBlockMap -----------------------

def test_map_single_then_dict_on_duplicate():
    m = BlockHashToBlockMap()
    key = make_block_hash_with_group_id(b"h", 0)
    b1, b2 = KVCacheBlock(1), KVCacheBlock(2)
    m.insert(key, b1)
    assert m.get_one_block(key) is b1
    m.insert(key, b2)  # same hash, different block -> union becomes dict
    got = m.get_one_block(key)
    assert got in (b1, b2)
    # pop one, the other remains
    popped = m.pop(key, 1)
    assert popped is b1
    assert m.get_one_block(key) is b2


# ----------------------- BlockPool 引用计数 / touch / free -----------------------

def make_pool(num_blocks=8, enable_caching=True):
    return BlockPool(num_blocks, enable_caching, BLOCK_SIZE)


def test_null_block_is_block_id_zero_and_skipped():
    pool = make_pool()
    assert pool.null_block.block_id == 0
    assert pool.null_block.is_null is True
    # null block removed from free queue
    assert pool.get_num_free_blocks() == 7


def test_get_new_blocks_increments_ref_cnt():
    pool = make_pool()
    blocks = pool.get_new_blocks(2)
    assert all(b.ref_cnt == 1 for b in blocks)
    assert pool.get_num_free_blocks() == 5


def test_free_blocks_returns_to_queue_when_ref_zero():
    pool = make_pool()
    blocks = pool.get_new_blocks(2)
    free_before = pool.get_num_free_blocks()
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == free_before + 2
    assert all(b.ref_cnt == 0 for b in blocks)


def test_touch_rescues_block_from_free_queue():
    pool = make_pool()
    blk = pool.get_new_blocks(1)[0]
    # cache it then free -> ref_cnt 0, still in queue as eviction candidate
    key = make_block_hash_with_group_id(b"hh", 0)
    blk.block_hash = key
    pool.cached_block_hash_to_block.insert(key, blk)
    pool.free_blocks([blk])
    assert blk.ref_cnt == 0
    free_after_free = pool.get_num_free_blocks()
    # touch -> ref_cnt 1, removed from free queue
    pool.touch([blk])
    assert blk.ref_cnt == 1
    assert pool.get_num_free_blocks() == free_after_free - 1


def test_maybe_evict_clears_hash_when_reused():
    pool = make_pool()
    blk = pool.get_new_blocks(1)[0]
    key = make_block_hash_with_group_id(b"k", 0)
    blk.block_hash = key
    pool.cached_block_hash_to_block.insert(key, blk)
    pool.free_blocks([blk])
    # freed block sits at the tail (LRU front is the never-used blocks); drain
    # the whole free queue so the freed+cached block is eventually re-popped,
    # which must trigger eviction of its old hash.
    reused = pool.get_new_blocks(pool.get_num_free_blocks())
    assert blk in reused
    assert blk.block_hash is None
    assert pool.get_cached_block(key[:-4], [0]) is None


# ----------------------- 端到端：前缀缓存命中 -----------------------

def make_manager(num_blocks=32):
    spec = FullAttentionSpec(block_size=BLOCK_SIZE)
    return KVCacheManager(
        kv_cache_spec=spec,
        num_blocks=num_blocks,
        max_model_len=1024,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
    )


def test_first_request_no_hit_then_caches():
    mgr = make_manager()
    req = make_request("r1", list(range(12)))  # 3 full blocks
    computed, n_hit = mgr.get_computed_blocks(req)
    assert n_hit == 0
    assert computed.blocks == ((),)
    out = mgr.allocate_slots(req, num_new_tokens=req.num_tokens)
    assert out is not None
    # full blocks cached
    block_ids = out.get_block_ids()[0]
    assert len(block_ids) == 3


def test_second_request_hits_shared_prefix():
    mgr = make_manager()
    # first request, full prefix of 12 tokens
    req1 = make_request("r1", list(range(12)))
    mgr.allocate_slots(req1, num_new_tokens=req1.num_tokens)

    # second request shares first 8 tokens then diverges
    req2 = make_request("r2", list(range(8)) + [100, 101, 102, 103])
    computed, n_hit = mgr.get_computed_blocks(req2)
    # max_cache_hit_length = num_tokens - 1 = 11 -> 2 full blocks of shared prefix
    assert n_hit == 8
    hit_ids = computed.get_block_ids()[0]
    assert len(hit_ids) == 2
    # the hit blocks are the SAME physical blocks as req1's first two
    req1_ids = mgr.coordinator.single_type_managers[0].req_to_blocks["r1"]
    assert hit_ids == [req1_ids[0].block_id, req1_ids[1].block_id]


def test_hit_blocks_are_touched_not_reallocated():
    mgr = make_manager()
    req1 = make_request("r1", list(range(12)))
    mgr.allocate_slots(req1, num_new_tokens=req1.num_tokens)
    shared_block = mgr.coordinator.single_type_managers[0].req_to_blocks["r1"][0]
    assert shared_block.ref_cnt == 1

    req2 = make_request("r2", list(range(8)) + [9, 9, 9, 9])
    computed, n_hit = mgr.get_computed_blocks(req2)
    mgr.allocate_slots(
        req2,
        num_new_tokens=req2.num_tokens - n_hit,
        num_new_computed_tokens=n_hit,
        new_computed_blocks=computed,
    )
    # shared block now referenced by both requests
    assert shared_block.ref_cnt == 2


def test_allocate_slots_returns_none_when_insufficient():
    mgr = make_manager(num_blocks=4)  # 3 usable after null block
    req = make_request("r", list(range(40)))  # needs 10 blocks
    out = mgr.allocate_slots(req, num_new_tokens=req.num_tokens)
    assert out is None


# ----------------------- f11: 抢占重算靠前缀缓存命中缓解 -----------------------

def test_preempted_request_reuses_undropped_prefix_blocks():
    mgr = make_manager()
    req = make_request("r1", list(range(12)))
    computed, _ = mgr.get_computed_blocks(req)
    mgr.allocate_slots(req, num_new_tokens=req.num_tokens)
    orig_block_ids = [
        b.block_id
        for b in mgr.coordinator.single_type_managers[0].req_to_blocks["r1"]
    ]

    # 抢占：释放块（ref_cnt->0），但 hash 保留在 map，块留在 free queue 作驱逐候选
    mgr.free(req)
    assert all(
        mgr.block_pool.blocks[bid].block_hash is not None for bid in orig_block_ids
    )

    # 重排回来从 0 重 prefill：前缀块未被取走复用 -> 直接命中
    req.num_computed_tokens = 0
    req.num_preemptions += 1
    computed2, n_hit2 = mgr.get_computed_blocks(req)
    hit_ids = computed2.get_block_ids()[0]
    # 命中之前算过的前缀块（同 block_id），省去重算
    assert n_hit2 > 0
    assert set(hit_ids).issubset(set(orig_block_ids))
