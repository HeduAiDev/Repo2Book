# ch15 m21 调用全景驱动：一次调度 tick 里「算/查/挂/写」四操作的真实调用编排。
# 场景：A(64 token) 先跑完并 free（4 满块留表）；B(80 token、前 32 与 A 相同) 进
# waiting，跑一拍 admission_lookup + allocate_slots——用只观察不改行为的方法包装器
# 记录完整调用栈序列（谁调谁、关键实参、锚点即 pin v0.27.1 真源码行号）。
# 读者反馈①③：正文各节只见局部实现，看不出四操作分别被谁在哪调用——本 trace 给全景。
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec)
from implementation.request import Request  # noqa: E402
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


EVENTS = []          # 只观察不改行为的调用账（op, detail）
_seq = [0]


def ev(op, detail):
    _seq[0] += 1
    EVENTS.append({"seq": _seq[0], "op": op, "detail": detail})


def wrap_instance(obj, name, op, argfn=None):
    """实例属性 shadow 一个普通 bound method，记录后调原方法。"""
    orig = getattr(obj, name)

    def wrapper(*a, **k):
        ev(op, argfn(a, k) if argfn else "")
        return orig(*a, **k)
    setattr(obj, name, wrapper)


def run_request(mgr, req):
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, req.num_tokens - num_hit,
                             num_new_computed_tokens=num_hit,
                             new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


out = {"params": {"block_size": 16, "hash_block_size": 16,
                  "A": "64 token（4 满块），跑完 free——前缀留表",
                  "B": "80 token、前 32 与 A 相同——本拍被调度",
                  "observability": "方法包装器只记录调用与实参，不改行为（同 m15 口径）"}}

# --- A 的入场（setup，B 的 tick 之前的世界）：构造即算 + 首拍四操作 + free ---
sched = make_sched(64)
mgr = sched.kv_cache_manager
pool = mgr.block_pool
coord = mgr.coordinator
man = coord.single_type_managers[0]
ManCls = type(man)

req_a = Request("a", list(range(64)), block_hasher=HASHER16)
out["A_setup"] = {
    "hashes_at_construction": len(req_a.block_hashes),
    "note": "Request 构造尾 update_block_hashes 即算满块哈希（request.py:L208-L209）",
}
run_request(mgr, req_a)
out["A_setup"]["map_size_after_run"] = len(
    pool.cached_block_hash_to_block._cache)
out["A_setup"]["a_blocks_cached"] = len(man.req_to_blocks["a"])
mgr.free(req_a)
out["A_setup"]["map_size_after_free"] = len(
    pool.cached_block_hash_to_block._cache)
out["A_setup"]["note_free"] = "free 后哈希留表（m9）：B 的查表有东西可命中"

# --- B 的一拍：包装全部关键方法，记录调用编排 ---
req_b = Request("b", list(range(32)) + [1000 + i for i in range(48)],
                block_hasher=HASHER16)
out["B_setup"] = {
    "hashes_at_construction": len(req_b.block_hashes),
    "shared_prefix_with_A": 32,
    "note": "构造即算 5 个满块哈希（80//16）；h0/h1 与 A 逐字节相同、h2 起分叉",
}

EVENTS.clear()
_seq[0] = 0

# 查：scheduler.admission_lookup → mgr.get_computed_blocks →
#     coord.find_longest_cache_hit → FullAttentionManager.find → pool.get_cached_block
wrap_instance(mgr, "get_computed_blocks", "查·KVCacheManager.get_computed_blocks",
              lambda a, k: f"request={a[0].request_id}")
CoordCls = type(coord)
_orig_coord_find = CoordCls.find_longest_cache_hit


def _coord_find_wrapper(*a, **k):
    ev("查·Unitary.find_longest_cache_hit（委托唯一管家）",
       f"max_cache_hit_length={a[1] if len(a) > 1 else k.get('max_cache_hit_length')}")
    return _orig_coord_find(coord, *a, **k)
coord.find_longest_cache_hit = _coord_find_wrapper


def _mgr_find_wrapper(*a, **k):
    ev("查·FullAttentionManager.find_longest_cache_hit（phase 1 miss 即断）",
       f"max_length={k.get('max_length')}")
    return ManCls.find_longest_cache_hit(*a, **k)
man.find_longest_cache_hit = _mgr_find_wrapper


_orig_get_cached = pool.get_cached_block


def _get_cached_wrapper(block_hash, group_ids):
    hit = _orig_get_cached(block_hash, group_ids)
    ev("查·BlockPool.get_cached_block（平面 dict 一次查）",
       f"hit={'块 ' + str(hit[0].block_id) if hit else 'miss'}")
    return hit
pool.get_cached_block = _get_cached_wrapper

# 挂：allocate_slots → coord.allocate_new_computed_blocks →
#     manager.add_local_computed_blocks → pool.touch
wrap_instance(coord, "allocate_new_computed_blocks",
              "挂·coordinator.allocate_new_computed_blocks",
              lambda a, k: f"num_local_computed_tokens={k.get('num_local_computed_tokens')}")
wrap_instance(man, "add_local_computed_blocks",
              "挂·manager.add_local_computed_blocks（内部调 touch）",
              lambda a, k: f"request={a[0]} "
                           f"new_computed_blocks={[b.block_id for b in a[1]]}")
_orig_touch = pool.touch


def _touch_wrapper(blocks):
    ev("挂·BlockPool.touch（ref_cnt+1、O(1) 摘出自由队列）",
       f"blocks={[b.block_id for b in blocks]}")
    return _orig_touch(blocks)
pool.touch = _touch_wrapper

# 写（新块）：allocate_slots → coord.allocate_new_blocks →
#     manager.allocate_new_blocks → pool.get_new_blocks
wrap_instance(coord, "allocate_new_blocks", "写·新块·coordinator.allocate_new_blocks",
              lambda a, k: f"num_tokens(需槽位数)={a[1]}")
wrap_instance(man, "allocate_new_blocks", "写·新块·manager.allocate_new_blocks",
              lambda a, k: f"num_tokens={a[1]}")
_orig_get_new = pool.get_new_blocks


def _get_new_wrapper(num_blocks):
    ret = _orig_get_new(num_blocks)
    ev("写·新块·BlockPool.get_new_blocks（popleft 队头、惰性摘哈希）",
       f"num_blocks={num_blocks} → {[b.block_id for b in ret]}")
    return ret
pool.get_new_blocks = _get_new_wrapper

# 写回（满块入表）：allocate_slots 尾 → coord.cache_blocks →
#     manager.cache_blocks → pool.cache_full_blocks
wrap_instance(coord, "cache_blocks", "写回·coordinator.cache_blocks",
              lambda a, k: f"num_computed_tokens={a[1]}")
wrap_instance(man, "cache_blocks", "写回·manager.cache_blocks（幂等闸+进度账）",
              lambda a, k: f"num_tokens={a[1]}")
_orig_cfb = pool.cache_full_blocks


def _cfb_wrapper(**k):
    before = len(pool.cached_block_hash_to_block._cache)
    _orig_cfb(**k)
    after = len(pool.cached_block_hash_to_block._cache)
    ev("写回·BlockPool.cache_full_blocks（满块入表）",
       f"num_cached_blocks={k['num_cached_blocks']} num_full_blocks={k['num_full_blocks']}"
       f" → map {before}→{after}")
    return None
pool.cache_full_blocks = _cfb_wrapper

# --- 跑 B 的一拍（真实调度侧入口：scheduler.admission_lookup → allocate_slots）---
ev("拍·Scheduler.admission_lookup（waiting 准入那一步）", "num_computed_tokens==0 才查")
blocks_b, hit_b, junction_b = sched.admission_lookup(req_b)
out["B_tick"] = {"hit_tokens": hit_b, "junction": junction_b,
                 "hit_block_ids": [b.block_id for b in blocks_b.blocks[0]]}
ev("拍·Scheduler.schedule→KVCacheManager.allocate_slots（同一拍内的分配）",
   f"num_new_tokens={req_b.num_tokens - hit_b}（80−{hit_b}）")
out_b = mgr.allocate_slots(req_b, req_b.num_tokens - hit_b,
                           num_new_computed_tokens=hit_b,
                           new_computed_blocks=blocks_b)
assert out_b is not None
req_b.num_computed_tokens = req_b.num_tokens

out["tick_events"] = EVENTS
out["tick_summary"] = {
    "tick_event_count": len(EVENTS),
    "get_cached_block_calls": sum(1 for e in EVENTS
                                  if "get_cached_block" in e["op"]),
    "touch_blocks": sum(len(json.dumps(e["detail"]).split(",")) - 1 for e in EVENTS
                        if "touch" in e["op"] and "blocks=" in e["detail"]),
    "num_new_block_ids": len(out_b.blocks[0]) if out_b else 0,
    "b_block_table_len": len(man.req_to_blocks["b"]),
    "b_num_cached_block_ledger": man.num_cached_block["b"],
    "map_size_after_tick": len(pool.cached_block_hash_to_block._cache),
    "note": "四操作在同一拍内的顺序：查(admission_lookup) → 挂+写新块+写回(allocate_slots 三段)",
}
# 调用点的 pin v0.27.1 行号锚（表「锚点」列的出处，全部现核于 instances/vllm/source）
out["pin_anchors"] = {
    "算·构造即算": "vllm/v1/request.py:L208-L209（__init__ 尾 update_block_hashes）",
    "算·每拍续算": "vllm/v1/request.py:L249-L265（append_output_token_ids → "
                  "update_block_hashes；scheduler.py:L2094-L2111 每拍回填时调）",
    "查·调度器侧入口": "vllm/v1/core/sched/scheduler.py:L744-L766（waiting 准入；"
                      "被抢占者 num_computed_tokens 归零后也从这再进）；running 拍 "
                      "L576-L582 只分配不查",
    "查·manager 门面": "vllm/v1/core/kv_cache_manager.py:L229-L295"
                     "（max_cache_hit_length=num_tokens−1 在 L259）",
    "查·单组委托": "vllm/v1/core/kv_cache_coordinator.py:L486-L504（Unitary 直接"
                 "委托唯一管家）",
    "查·管家 phase 1": "vllm/v1/core/single_type_kv_cache_manager.py:L682-L739",
    "查·平面 dict": "vllm/v1/core/block_pool.py:L198-L217（get_cached_block）",
    "挂": "kv_cache_manager.py:L535-L540（allocate_slots 内 "
        "coordinator.allocate_new_computed_blocks）→ single_type_kv_cache_manager.py:"
        "L232-L289（add_local_computed_blocks）→ block_pool.py:L702-L717（touch）",
    "写·新块": "kv_cache_manager.py:L542-L547（allocate_new_blocks）→ "
             "single_type_kv_cache_manager.py:L330-L369 → block_pool.py:L647-L661"
             "（get_new_blocks）",
    "写回·满块": "kv_cache_manager.py:L559-L563（allocate_slots 尾，"
              "num_tokens_to_cache=min(total+new, num_tokens)）→ "
              "kv_cache_coordinator.py:L652-L683（Hybrid 对齐；基类 L273-L288）→ "
              "single_type_kv_cache_manager.py:L427-L477（幂等闸 L448）→ "
              "block_pool.py:L225-L342（cache_full_blocks）",
    "写回·异步模式补拍": "vllm/v1/core/sched/async_scheduler.py:L65-L69"
                     "（update_from_output 里再 cache_blocks 一次）",
    "写回·KV connector 载入完成": "scheduler.py:L2650/L2669（远端 KV 到货后补登记）",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m21.json"),
          "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
