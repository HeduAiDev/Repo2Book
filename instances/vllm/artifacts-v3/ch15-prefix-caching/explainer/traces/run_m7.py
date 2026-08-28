# ch15 m7 LRU 不变量一·逆序 free 驱动：manager.free 把块逆序传入 free_blocks
# （single_type_kv_cache_manager.py:L519-L527）——尾块（更长前缀、更苛刻的复用
# 条件）先挂回、排更靠驱逐端。反事实对照：同一原语正序传入 = 链头先驱逐
# = 最长可复用前缀被拦腰斩断。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec  # noqa: E402
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.request import Request  # noqa: E402
import torch  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)


def make_manager():
    cfg = KVCacheConfig(num_blocks=64, kv_cache_tensors=[],
                        kv_cache_groups=[KVCacheGroupSpec(
                            ["l.0"], FullAttentionSpec(
                                block_size=16, num_kv_heads=2, head_size=8,
                                dtype=torch.float16))])
    return KVCacheManager(kv_cache_config=cfg, max_model_len=512,
                          scheduler_block_size=16, hash_block_size=16)


def run_request(mgr, req):
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, req.num_tokens - num_hit,
                             num_new_computed_tokens=num_hit,
                             new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


def free_queue_ids(pool):
    q = pool.free_block_queue
    ids, cur = [], q.fake_free_list_head.next_free_block
    while cur is not None and cur is not q.fake_free_list_tail:
        ids.append(cur.block_id)
        cur = cur.next_free_block
    return ids


# --- 场景一（真实路径）：manager.free 逆序传入 ---
mgr = make_manager()
pool = mgr.block_pool
req = Request("a", list(range(48)), block_hasher=HASHER16)  # 3 满块 [1,2,3]
run_request(mgr, req)
alloc_order = [b.block_id for b in
               mgr.coordinator.single_type_managers[0].req_to_blocks["a"]]
mgr.free(req)
tail3 = free_queue_ids(pool)[-3:]
out = {
    "params": {
        "block_size": 16, "hash_block_size": 16, "prompt_tokens": 48,
        "num_blocks_allocated": 3, "allocation_order": alloc_order,
        "chain_head_block": alloc_order[0], "chain_tail_block": alloc_order[-1],
    },
    "case_reverse": {
        "input_to_free_blocks": "reversed([1, 2, 3]) = [3, 2, 1]",
        "free_queue_tail3": tail3,
        "first_evicted_among_three": tail3[0],
        "note": "尾块 3 = 覆盖 48 token 的链尾（复用条件最苛刻）排最靠驱逐端",
    },
}

# --- 场景二（反事实对照）：同一 free_blocks 原语、正序传入 ---
mgr2 = make_manager()
pool2 = mgr2.block_pool
req2 = Request("a2", list(range(48)), block_hasher=HASHER16)
run_request(mgr2, req2)
blks = mgr2.coordinator.single_type_managers[0].pop_blocks_for_free("a2")
pool2.free_blocks(blks)  # 正序 [1,2,3]（违反调用约定——仅为展示顺序即策略）
tail3_fwd = free_queue_ids(pool2)[-3:]
out["case_forward_counterfactual"] = {
    "input_to_free_blocks": "[1, 2, 3]（正序，违反约定）",
    "free_queue_tail3": tail3_fwd,
    "first_evicted_among_three": tail3_fwd[0],
    "note": "链头块 1（任何 >=16 token 的共享前缀都用得上它）反而最先被驱逐",
}

# --- 场景三：驱逐序的后果量化——小池（null+3 块，无新块垫队头），池紧取 1 块 ---
def small_manager():
    cfg = KVCacheConfig(num_blocks=4, kv_cache_tensors=[],
                        kv_cache_groups=[KVCacheGroupSpec(
                            ["l.0"], FullAttentionSpec(
                                block_size=16, num_kv_heads=2, head_size=8,
                                dtype=torch.float16))])
    return KVCacheManager(kv_cache_config=cfg, max_model_len=512,
                          scheduler_block_size=16, hash_block_size=16)


# 3a. 逆序约定：free 后队头即链尾 3 → 取走的恰是块 3 → 前 32 token 仍可命中
mgr3 = small_manager()
pool3 = mgr3.block_pool
req3 = Request("a3", list(range(48)), block_hasher=HASHER16)
run_request(mgr3, req3)
mgr3.free(req3)  # 逆序 → 队 = [3,2,1]
evicted_rev = pool3.get_new_blocks(1)[0]
reqP = Request("p", list(range(48)), block_hasher=HASHER16)  # 与 a3 同前缀的来者
_, hit_rev, _ = mgr3.get_computed_blocks(reqP)

# 3b. 正序反事实：同一池、同一请求，正序 free → 队头即链头 1 → 取走块 1 → 整条报废
mgr4 = small_manager()
pool4 = mgr4.block_pool
req4 = Request("a4", list(range(48)), block_hasher=HASHER16)
run_request(mgr4, req4)
blks4 = mgr4.coordinator.single_type_managers[0].pop_blocks_for_free("a4")
pool4.free_blocks(blks4)  # 正序 → 队 = [1,2,3]
evicted_fwd = pool4.get_new_blocks(1)[0]
reqQ = Request("q", list(range(48)), block_hasher=HASHER16)
_, hit_fwd, _ = mgr4.get_computed_blocks(reqQ)

out["case_evict_consequence"] = {
    "pool_num_blocks": 4,
    "reverse_evicted_block_id": evicted_rev.block_id,
    "reverse_surviving_hit_tokens": hit_rev,
    "forward_evicted_block_id": evicted_fwd.block_id,
    "forward_surviving_hit_tokens": hit_fwd,
    "note": "同样池紧取 1 块：逆序约定取走链尾 3 → 来者仍命中 32 token；正序取走链头 1 → 命中 0、整条前缀报废",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m7.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
