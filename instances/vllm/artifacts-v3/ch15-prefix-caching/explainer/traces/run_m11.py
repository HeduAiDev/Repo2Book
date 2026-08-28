# ch15 m11 F2 回收驱动：抢占 free 不清哈希（scheduler.py:L1274-L1315）→
# 被抢占请求重排回 waiting 队头 → 重走准入查询 get_computed_blocks（L744-L766）
# 重命中自己的前缀 → touch 救回——『重算』变『重载元数据+补算』；
# 最坏分支：被抢占期间块被取走复用（惰性驱逐）→ 全量重 prefill。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec)
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
import torch  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)


def kv_config(specs, num_blocks=64):
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                         kv_cache_groups=[KVCacheGroupSpec([f"l.{i}"], s)
                                          for i, s in enumerate(specs)])


def full_spec(bs):
    return FullAttentionSpec(block_size=bs, num_kv_heads=2, head_size=8,
                             dtype=torch.float16)


def make_sched(num_blocks):
    return Scheduler(kv_config([full_spec(16)], num_blocks),
                     max_model_len=512, scheduler_block_size=16,
                     hash_block_size=16)


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


def map_size(pool):
    return len(pool.cached_block_hash_to_block._cache)


out = {"params": {"block_size": 16, "hash_block_size": 16,
                  "prompt_tokens": 64, "cached_blocks_before_preempt": 4,
                  "pool_blocks_worst_case": 8}}

# --- 主线：free 不清哈希 → 重排回 waiting → 重命中 → touch 救回 ---
sched = make_sched(64)
mgr = sched.kv_cache_manager
pool = mgr.block_pool
req = Request("a", list(range(64)), block_hasher=HASHER16)
run_request(mgr, req)
req.status = RequestStatus.RUNNING
sched.waiting.append(req)
steps = [{"phase": "抢占前", "status": "RUNNING",
          "num_computed_tokens": req.num_computed_tokens,
          "map_size": map_size(pool), "num_preemptions": req.num_preemptions}]
sched._preempt_request(req)
steps.append({"phase": "_preempt_request", "status": "PREEMPTED",
              "num_computed_tokens": req.num_computed_tokens,
              "map_size": map_size(pool),
              "num_preemptions": req.num_preemptions,
              "waiting_head_is_req": sched.waiting[0] is req,
              "note": "free 全部块但哈希留在 map——重算的成本上限已经改写"})
blocks, hit, junction = sched.admission_lookup(req)
steps.append({"phase": "重排回来重走准入", "num_new_local_computed_tokens": hit,
              "hit_block_ids": [b.block_id for b in blocks.blocks[0]],
              "junction": junction,
              "note": "对自己的前缀全量重命中（max_cache_hit_length=63 → 3 块）"})
out_blocks = mgr.allocate_slots(req, 64 - hit, num_new_computed_tokens=hit,
                                new_computed_blocks=blocks)
assert out_blocks is not None
st = mgr.coordinator.single_type_managers[0]
used = {b.block_id for b in st.req_to_blocks["a"]}
steps.append({"phase": "allocate_slots（touch 救回+补算）",
              "rescued_not_in_free_queue":
                  not (used & set(free_queue_ids(pool))),
              "recompute_tokens": 64 - hit,
              "note": "『重算』变『重载元数据+补算』：只补 16 token 而非 64"})
out["main_path"] = {"steps": steps, "recompute_tokens": 64 - hit,
                    "full_recompute_tokens_would_be": 64}

# --- 最坏分支：被抢占期间池紧、块被取走复用 → 前缀失效 → 全量重 prefill ---
sched2 = make_sched(8)
mgr2 = sched2.kv_cache_manager
pool2 = mgr2.block_pool
req2 = Request("a", list(range(48)), block_hasher=HASHER16)
run_request(mgr2, req2)
req2.status = RequestStatus.RUNNING
sched2._preempt_request(req2)
worst = [{"phase": "抢占后（8 块小池）", "map_size": map_size(pool2),
          "free_blocks": pool2.get_num_free_blocks()}]
while pool2.get_num_free_blocks() > 0:
    pool2.get_new_blocks(1)   # 别的请求把队头块逐个取走复用（惰性驱逐摘哈希）
worst.append({"phase": "抢占期间池被抽干", "map_size": map_size(pool2),
              "free_blocks": pool2.get_num_free_blocks()})
_, hit2, _ = mgr2.get_computed_blocks(req2)
worst.append({"phase": "重排回来重走准入", "hit": hit2,
              "recompute_tokens": 48 - hit2,
              "note": "前缀失效 → 退化为全量重 prefill——F2 的上界"})
out["worst_case"] = {"steps": worst}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m11.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
