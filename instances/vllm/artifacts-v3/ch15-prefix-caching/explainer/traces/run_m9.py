# ch15 m9 惰性驱逐驱动：free 只挂回队列不动哈希；块被 get_new_blocks 复用那刻
# 才 _maybe_evict_cached_block 摘哈希（block_pool.py:L647-L700）；
# _remove_cached_block_hashes 靠反向索引 cached_block_hashes_by_block
# 把部分条目别名一次摘干净（L571-L590）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.block_pool import BlockPool  # noqa: E402
from implementation.request import Request  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)


def map_size(pool):
    return len(pool.cached_block_hash_to_block._cache)


out = {"params": {"pool_blocks_case1": 2, "pool_blocks_case2": 8,
                  "hash_block_size": 16}}

# --- 场景一：free 不清哈希 → 复用才摘 ---
pool = BlockPool(2, enable_caching=True, hash_block_size=16)
req = Request("a", list(range(16)), block_hasher=HASHER16)
blk = pool.get_new_blocks(1)[0]           # 唯一非 null 块（id 1）
pool.cache_full_blocks(
    request=req, blocks=[blk], num_cached_blocks=0, num_full_blocks=1,
    block_size=16, kv_cache_group_id=0)
steps = []
steps.append({"step": 1, "action": "cache_full_blocks（满块入表）",
              "blk_has_hash": blk.block_hash is not None,
              "map_size": map_size(pool), "blk_ref_cnt": blk.ref_cnt})
pool.free_blocks([blk])
steps.append({"step": 2, "action": "free_blocks（ref_cnt 归零回队）",
              "blk_has_hash": blk.block_hash is not None,
              "map_size": map_size(pool), "blk_ref_cnt": blk.ref_cnt,
              "note": "哈希原样留在 map——『缓存』与『空闲』是同一队列的两端"})
reused = pool.get_new_blocks(1)[0]
steps.append({"step": 3, "action": "get_new_blocks 复用该块（池紧）",
              "reused_is_same_block": reused is blk,
              "blk_has_hash": blk.block_hash is not None,
              "map_size": map_size(pool), "blk_ref_cnt": blk.ref_cnt,
              "note": "_maybe_evict_cached_block 这一刻才摘哈希——驱逐是惰性隐式的"})
out["case_lazy_evict"] = {"steps": steps}

# --- 场景二：别名一次摘干净（主哈希 + 反向索引） ---
pool2 = BlockPool(8, enable_caching=True, hash_block_size=8)
blk2 = pool2.get_new_blocks(1)[0]
k_main = kcu.make_block_hash_with_group_id(b"main", 0)
k_alias = kcu.make_block_hash_with_group_id(b"alias", 0)
pool2._insert_block_hash(k_main, blk2, num_tokens=16)
pool2._insert_block_hash(k_alias, blk2, num_tokens=8)  # 第二条进反向索引
steps2 = []
steps2.append({"step": 1, "action": "主哈希 + 部分条目别名都指向块",
               "map_size": map_size(pool2),
               "aliases_in_reverse_index": len(
                   pool2.cached_block_hashes_by_block[blk2.block_id]),
               "primary_is_main": blk2.block_hash == k_main})
removed = pool2._remove_cached_block_hashes(blk2)
steps2.append({"step": 2, "action": "_remove_cached_block_hashes（驱逐/晋升/重指的第一步）",
               "num_removed": len(removed), "map_size": map_size(pool2),
               "blk_has_hash": blk2.block_hash is not None,
               "reverse_index_empty": blk2.block_id
               not in pool2.cached_block_hashes_by_block,
               "note": "主哈希+别名一次摘干净——不留悬空键"})
out["case_alias_cleanup"] = {"steps": steps2}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m9.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
