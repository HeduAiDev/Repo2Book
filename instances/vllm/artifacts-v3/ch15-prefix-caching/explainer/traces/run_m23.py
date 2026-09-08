# ch15 m23 组化与组账本视图驱动：读者反馈②「kv cache 为什么要分 group、
# 为什么需要有 block view 这种东西」。
# 两个配置对照实跑：
#   uniform（单组 full 16）——Unitary coordinator、1 个管家、1 张块表；
#   hybrid（full 16 + swa 16/窗 48）——Hybrid coordinator、2 个管家各持组号与
#     各自的 req_to_blocks 账本、但 block_pool 是同一个对象（id 相同）；
#     同一前缀哈希拼上组号后在两组各查各的物理块（m2 键构成在这里落地）。
# 「block view」两层的实证：①组账本视图——同一请求在两组各有一张块表
# （SWA 组窗外块以 null 占位）；②粒度视图——BlockHashListWithBlockSize 的
# 零成本重串（m12 已跑，这里记录配置事实：两组 block_size 不同时 hash_block_size
# 取 GCD）。本驱动用同块大小聚焦①；②的数字见 m12。
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
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec)
from implementation.request import Request  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
import torch  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)


def kv_config(specs, num_blocks=64):
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                         kv_cache_groups=[KVCacheGroupSpec([f"l.{i}"], s)
                                          for i, s in enumerate(specs)])


def full_spec():
    return FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8,
                             dtype=torch.float16)


def swa_spec(window=48):
    return SlidingWindowSpec(block_size=16, num_kv_heads=2, head_size=8,
                             dtype=torch.float16, sliding_window=window)


def run_request(sched, req):
    blocks, hit, _ = sched.admission_lookup(req)
    out = sched.kv_cache_manager.allocate_slots(
        req, req.num_tokens - hit, num_new_computed_tokens=hit,
        new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


def block_view(blks):
    return ["NULL" if b.is_null else b.block_id for b in blks]


out = {"params": {"hash_block_size": 16,
                  "note": "两个配置同 hash 粒度；hybrid 两组同块大小（16）时 "
                          "GCD=16——块大小不同时的粒度视图重串见 m12"}}

# --- 配置一：uniform（单组 full）——不分组的情形 ---
sched_u = Scheduler(kv_config([full_spec()], 64), max_model_len=512,
                    scheduler_block_size=16, hash_block_size=16)
cu = sched_u.kv_cache_manager.coordinator
out["uniform"] = {
    "coordinator_class": type(cu).__name__,
    "num_groups": len(sched_u.kv_cache_manager.kv_cache_config.kv_cache_groups),
    "num_managers": len(cu.single_type_managers),
    "manager_group_ids": [m.kv_cache_group_id for m in cu.single_type_managers],
    "attention_groups": [{"spec": type(g.spec).__name__,
                          "group_ids": list(g.group_ids)}
                         for g in getattr(cu, "attention_groups", [])],
    "note": "全部层同一个 spec——一张块表一个管家，不需要调和",
}

# --- 配置二：hybrid（full + swa 窗 48）——分组的情形 ---
sched_h = Scheduler(kv_config([full_spec(), swa_spec(48)], 64),
                    max_model_len=512, scheduler_block_size=16,
                    hash_block_size=16)
mgr_h = sched_h.kv_cache_manager
ch = mgr_h.coordinator
managers = ch.single_type_managers
out["hybrid_structure"] = {
    "coordinator_class": type(ch).__name__,
    "num_groups": len(mgr_h.kv_cache_config.kv_cache_groups),
    "managers": [
        {"class": type(m).__name__, "kv_cache_group_id": m.kv_cache_group_id,
         "block_size": m.block_size}
        for m in managers
    ],
    "one_shared_block_pool": all(m.block_pool is ch.block_pool for m in managers),
    "block_pool_is_manager_pool": mgr_h.block_pool is ch.block_pool,
    "separate_ledgers": (id(managers[0].req_to_blocks)
                         != id(managers[1].req_to_blocks)),
    "attention_groups_full_first": [
        {"spec": type(g.spec).__name__, "group_ids": list(g.group_ids)}
        for g in ch.attention_groups
    ],
    "note": "一个共享 BlockPool + 每 group 一个管家各持账本（req_to_blocks）；"
            "attention_groups 按 spec 类型归并、full 排首（给不动点最紧上界）",
}

# --- 同一请求在两组的块表形态（组账本视图）---
req = Request("r", list(range(64)), block_hasher=HASHER16)
run_request(sched_h, req)
tables_after_prefill = {
    "full_group_block_table": block_view(managers[0].req_to_blocks["r"]),
    "swa_group_block_table": block_view(managers[1].req_to_blocks["r"]),
}
# 再跑一拍 decode：allocate_slots 开头 remove_skipped_blocks 把 SWA 窗外块
# 以 null 换位回收（processed-token 基准）——这就是两组块表形态分叉的机制。
req.append_output_token_ids(999)
alloc = sched_h.kv_cache_manager.allocate_slots(req, 1)
assert alloc is not None
req.num_computed_tokens += 1
out["group_ledger_views"] = {
    "request": "64 token prompt（4 满块）+ 1 个 decode token",
    "after_prefill": tables_after_prefill,
    "after_one_decode_tick": {
        "full_group_block_table": block_view(managers[0].req_to_blocks["r"]),
        "swa_group_block_table": block_view(managers[1].req_to_blocks["r"]),
    },
    "swa_window": 48, "swa_window_blocks": 48 // 16,
    "note": "同一请求两张块表：prefill 后两组各持 4 个物理块（页统一：等大块）；"
            "下一拍 allocate_slots 开头 remove_skipped_blocks 只在 SWA 组把窗外块"
            "（64−48=16 token=1 块）以 null 换位回收——full 组纹丝不动。两组的"
            "『保留多少历史』由各自 spec 说了算，这正是分组的全部意义",
}

# --- 同一哈希、两个键：组号进键的实证 ---
bh = req.block_hashes[0]
key_full = kcu.make_block_hash_with_group_id(bh, 0)
key_swa = kcu.make_block_hash_with_group_id(bh, 1)
out["group_id_in_key"] = {
    "same_hash_prefix": key_full[:-4] == key_swa[:-4],
    "key_full_tail_bytes": list(key_full[-4:]),
    "key_swa_tail_bytes": list(key_swa[-4:]),
    "note": "32 字节哈希拼 4 字节组号（big-endian）——同一前缀在每组各查各的"
            "物理块，组号进键才不串门（kv_cache_utils.py:L57-L66）",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m23.json"),
          "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
