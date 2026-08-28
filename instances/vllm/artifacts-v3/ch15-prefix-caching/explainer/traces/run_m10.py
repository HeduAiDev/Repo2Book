# ch15 m10 move_block_hashes 驱动：CoW 后把 src 块全部前缀缓存条目重指到 dst
# 私有拷贝（block_pool.py:L629-L645）——条目活着不摘、请求块表 append-only 由它兜住。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.block_pool import BlockPool  # noqa: E402

kcu.init_none_hash(sha256)


def map_size(pool):
    return len(pool.cached_block_hash_to_block._cache)


pool = BlockPool(8, enable_caching=True, hash_block_size=8)
src, dst = pool.get_new_blocks(2)   # id 1、id 2
k = kcu.make_block_hash_with_group_id(b"cache-key", 0)
pool._insert_block_hash(k, src, num_tokens=24)

steps = []
steps.append({"step": 1, "action": "src 持有条目（主哈希、num_tokens 24）",
              "src_has_hash": src.block_hash is not None,
              "dst_has_hash": dst.block_hash is not None,
              "map_points_to": src.block_id, "map_size": map_size(pool)})
pool.move_block_hashes(src, dst)
steps.append({"step": 2, "action": "move_block_hashes(src, dst)",
              "src_has_hash": src.block_hash is not None,
              "dst_has_hash": dst.block_hash is not None,
              "map_points_to": pool.cached_block_hash_to_block.get_one_block(
                  k).block_id,
              "map_size": map_size(pool),
              "dst_hash_num_tokens": dst.block_hash_num_tokens,
              "note": "条目不摘、只重指——同键查表改拿私有拷贝 dst；num_tokens 只跟主哈希走"})
out = {
    "params": {"pool_blocks": 8, "hash_block_size": 8,
               "src_block_id": src.block_id, "dst_block_id": dst.block_id,
               "entry_num_tokens": 24},
    "steps": steps,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m10.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
