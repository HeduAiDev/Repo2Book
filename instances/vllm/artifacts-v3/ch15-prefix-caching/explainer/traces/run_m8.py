# ch15 m8 LRU 不变量二·free_blocks 劈分驱动（block_pool.py:L719-L742，#42656）：
# 归零块劈两半——无哈希块（never match APC）prepend_n 到队头先驱逐、
# 有哈希块 append_n 到 LRU 尾；缓存关闭跳过劈分保 GPU 局部性（#48017）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.block_pool import BlockPool  # noqa: E402

kcu.init_none_hash(sha256)


def free_queue_ids(pool):
    q = pool.free_block_queue
    ids, cur = [], q.fake_free_list_head.next_free_block
    while cur is not None and cur is not q.fake_free_list_tail:
        ids.append(cur.block_id)
        cur = cur.next_free_block
    return ids


out = {"params": {
    "pool_blocks": 8,
    "null_block_id": 0,
    "b1": "无哈希（从未入表——例：解码尾块未满、或 mask 掉的块）",
    "b2": "有哈希（已入表的满块）",
}}

# --- 场景一：劈分（缓存开） ---
pool = BlockPool(8, enable_caching=True, hash_block_size=16)
b1, b2 = pool.get_new_blocks(2)          # b1=id1 无哈希；b2=id2 给哈希
bh = kcu.hash_block_tokens(sha256, None, list(range(16)))
pool._insert_block_hash(
    kcu.make_block_hash_with_group_id(bh, 0), b2, num_tokens=16)
b3 = pool.get_new_blocks(1)[0]           # b3=id3 无哈希
pool.free_blocks([b3])                   # 先释放一个无哈希块垫底
pool.free_blocks([b1, b2])               # 同一次 free：无哈希 + 有哈希混合
ids_on = free_queue_ids(pool)
out["case_split_on"] = {
    "queue_head_to_tail": ids_on,
    "first_evicted": ids_on[0],
    "second_evicted": ids_on[1],
    "last_evicted": ids_on[-1],
    "b1_id": b1.block_id, "b2_id": b2.block_id, "b3_id": b3.block_id,
    "b1_has_hash": b1.block_hash is not None,
    "b2_has_hash": b2.block_hash is not None,
    "note": "无哈希块 1、3 抢占队头先驱逐；有哈希块 2 沉到 LRU 尾最可复用端",
}

# --- 场景二：缓存关——跳过劈分恒 append（GPU 局部性） ---
pool_off = BlockPool(8, enable_caching=False, hash_block_size=16)
blocks = pool_off.get_new_blocks(3)      # id 1,2,3 全无哈希
pool_off.free_blocks(reversed(blocks))   # 逆序传入（调用约定不变）
ids_off = free_queue_ids(pool_off)
out["case_split_off"] = {
    "queue_head_to_tail": ids_off,
    "tail3": ids_off[-3:],
    "first_evicted_among_three": ids_off[-3],
    "note": "enable_caching=False 时无哈希块也走 append 分——刚用过的块沉队尾，下个请求复用同块保 GPU 局部性",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m8.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
