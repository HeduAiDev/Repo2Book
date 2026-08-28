# ch15 m5 touch 救回驱动：命中块 ref_cnt+1、ref_cnt==0 时 O(1) remove 出 free queue
# （block_pool.py:L702-L717）+ add_local_computed_blocks 挂块（single_type:L232-L289）
# ——多请求共享同一物理块的引用计数基础。
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


mgr = make_manager()
pool = mgr.block_pool
snapshots = []


def snap(label, block_ids, note=""):
    snapshots.append({
        "时点": label,
        "ref_cnt": {i: pool.blocks[i].ref_cnt for i in block_ids},
        "in_free_queue": {i: (i in free_queue_ids(pool)) for i in block_ids},
        "has_hash": {i: (pool.blocks[i].block_hash is not None)
                     for i in block_ids},
        "note": note,
    })


blk_ids = [1, 2]
reqA = Request("a", list(range(32)), block_hasher=HASHER16)
run_request(mgr, reqA)
snap("A 在跑（32 token）", blk_ids, "块 1、2 各被 A 独占")

mgr.free(reqA)
snap("A 完成 free", blk_ids, "ref_cnt 归零但哈希仍在——块回队当驱逐候选")

reqB = Request("b", list(range(48)), block_hasher=HASHER16)  # 前 32 同 A
blocks, hit, _ = mgr.get_computed_blocks(reqB)
out_b = {"get_computed_blocks": {"hit_tokens": hit,
                                 "hit_block_ids": [b.block_id for b in blocks.blocks[0]]}}
run_request(mgr, reqB)
snap("B 准入命中+分配（前 32 命中 A 的块）", blk_ids, "touch：ref_cnt 0→1、O(1) 出队救回")

reqC = Request("c", list(range(48)), block_hasher=HASHER16)  # 也共享前 32
run_request(mgr, reqC)
snap("C 也进场（共享同两块）", blk_ids, "同一物理块 ref_cnt=2——共享前缀的物理基础")

mgr.free(reqB)
snap("B 完成 free", blk_ids, "C 还引用 → ref_cnt 2→1、不回队")

mgr.free(reqC)
snap("C 完成 free", blk_ids, "最后引用者放手续 1→0、回 LRU 尾等下一个命中")

out = {
    "params": {"block_size": 16, "hash_block_size": 16,
               "A_tokens": 32, "B_tokens": 48, "C_tokens": 48,
               "shared_prefix_tokens": 32, "watched_block_ids": blk_ids},
    "B_admission": out_b,
    "snapshots": snapshots,
    "final_free_queue_tail": free_queue_ids(pool)[-4:],
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m5.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
