# ch15 m15 混合不动点驱动（kv_cache_coordinator.py:L685-L817）：
# 场景 A（simple hybrid：full+SWA 一轮即停）与场景 B（三组 full+SWA48+SWA32：
# 真正的重启第二轮）——finder 调用计数/实参/返回以类方法包装器记录（不改行为）。
# SWA finder 右到左找窗口连续段（single_type:L896-L993）；full 排首向下封闭
# （第二轮只 min 裁剪、不重查）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.request import Request  # noqa: E402
from implementation.single_type_kv_cache_manager import (  # noqa: E402
    FullAttentionManager, SlidingWindowManager)
import torch  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)

CALL_LOG = []


def instrument(manager_cls, label):
    orig = manager_cls.__dict__["find_longest_cache_hit"]

    def wrapped(cls, *, block_hashes, max_length, kv_cache_group_ids,
                block_pool, kv_cache_spec, drop_eagle_block, alignment_tokens):
        blocks, hit = orig.__func__(cls, block_hashes=block_hashes,
                                    max_length=max_length,
                                    kv_cache_group_ids=kv_cache_group_ids,
                                    block_pool=block_pool,
                                    kv_cache_spec=kv_cache_spec,
                                    drop_eagle_block=drop_eagle_block,
                                    alignment_tokens=alignment_tokens)
        CALL_LOG.append({
            "finder": label,
            "window": getattr(kv_cache_spec, "sliding_window", None),
            "max_length_in": max_length,
            "hit_out": hit,
            "num_hit_blocks": len(blocks[0]),
            "null_padded": sum(1 for b in blocks[0] if b.is_null),
        })
        return blocks, hit

    manager_cls.find_longest_cache_hit = classmethod(wrapped)


instrument(FullAttentionManager, "full")
instrument(SlidingWindowManager, "swa")


def full_spec(bs):
    return FullAttentionSpec(block_size=bs, num_kv_heads=2, head_size=8,
                             dtype=torch.float16)


def swa_spec(bs, window):
    return SlidingWindowSpec(block_size=bs, num_kv_heads=2, head_size=8,
                             dtype=torch.float16, sliding_window=window)


def make_manager(specs):
    cfg = KVCacheConfig(num_blocks=64, kv_cache_tensors=[],
                        kv_cache_groups=[KVCacheGroupSpec([f"l.{i}"], s)
                                         for i, s in enumerate(specs)])
    return KVCacheManager(kv_cache_config=cfg, max_model_len=512,
                          scheduler_block_size=16, hash_block_size=16)


def run_request(mgr, req):
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, req.num_tokens - num_hit,
                             num_new_computed_tokens=num_hit,
                             new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


out = {"params": {"prompt_tokens": 96, "block_size": 16,
                  "hash_block_size": 16, "max_cache_hit_length": 95}}

# --- 场景 A：simple hybrid（full + SWA 窗 48）一轮即停 ---
mgrA = make_manager([full_spec(16), swa_spec(16, 48)])
coordA = mgrA.coordinator
reqA = Request("a", list(range(96)), block_hasher=HASHER16)
run_request(mgrA, reqA)
swa_blocks_A = list(coordA.single_type_managers[1].req_to_blocks["a"])
mgrA.free(reqA)
# 摘掉 SWA 组第 4 块（hash[3]@64）的哈希：窗口连续段被打断
mgrA.block_pool._remove_cached_block_hashes(swa_blocks_A[3])
reqB = Request("b", list(range(96)), block_hasher=HASHER16)
CALL_LOG.clear()
blocks_B, hit_B, boundary_B = mgrA.get_computed_blocks(reqB)
out["case_simple_hybrid"] = {
    "groups": "full(16) + swa(16, window=48)",
    "swa_contiguous_blocks_needed": 3,
    "removed_swa_hash_at_block_idx": 3,
    "finder_calls": list(CALL_LOG),
    "num_while_passes": 1,
    "reconciled_hit": hit_B,
    "full_group_blocks_kept": len(blocks_B.blocks[0]),
    "swa_group_blocks_kept": len(blocks_B.blocks[1]),
    "swa_null_padded": sum(1 for b in blocks_B.blocks[1] if b.is_null),
    "longest_hit_length": 80,
    "num_uncached_common_prefix_tokens": 80 - hit_B,
    "shared_prefix_boundary": boundary_B,
}

# --- 场景 B：三组（full + SWA48 + SWA32）——真正的第二轮 ---
mgrC = make_manager([full_spec(16), swa_spec(16, 48), swa_spec(16, 32)])
coordC = mgrC.coordinator
reqP = Request("p", list(range(96)), block_hasher=HASHER16)
run_request(mgrC, reqP)
swa48_blocks = list(coordC.single_type_managers[1].req_to_blocks["p"])
swa32_blocks = list(coordC.single_type_managers[2].req_to_blocks["p"])
mgrC.free(reqP)
# SWA48 组摘 hash[3]（窗 3 连续块被打到 48）；SWA32 组摘 hash[4]（首探即断）
mgrC.block_pool._remove_cached_block_hashes(swa48_blocks[3])
mgrC.block_pool._remove_cached_block_hashes(swa32_blocks[4])
reqQ = Request("q", list(range(96)), block_hasher=HASHER16)
CALL_LOG.clear()
blocks_Q, hit_Q, boundary_Q = mgrC.get_computed_blocks(reqQ)


def split_passes(calls):
    """按 attention_groups 的规范序（full→swa48→swa32）切轮：位置回退即新轮。
    full 第二轮缺席（向下封闭只 min 裁剪）正是要展示的事实。"""
    order = {"full": 0, "swa48": 1, "swa32": 2}
    key = lambda c: ("full" if c["finder"] == "full"
                     else f"swa{c['window']}")
    passes, cur, prev_pos = [], [], None
    for c in calls:
        pos = order[key(c)]
        if prev_pos is not None and pos <= prev_pos:
            passes.append(cur)
            cur = []
        cur.append(c)
        prev_pos = pos
    if cur:
        passes.append(cur)
    return passes


passes = split_passes(CALL_LOG)
out["case_three_groups"] = {
    "groups": "full(16) + swa(16, window=48) + swa(16, window=32)",
    "swa48_contiguous_blocks_needed": 3,
    "swa32_contiguous_blocks_needed": 2,
    "removed_swa48_hash_at_block_idx": 3,
    "removed_swa32_hash_at_block_idx": 4,
    "finder_calls": list(CALL_LOG),
    "num_while_passes": len(passes),
    "pass1_full_hit": 80,
    "pass1_swa48_hit": 48,
    "pass1_swa32_hit": 48,
    "pass1_swa32_max_length_in": 48,
    "pass1_swa32_null_padded": 1,
    "reconciled_hit": hit_Q,
    "longest_hit_length": 80,
    "num_uncached_common_prefix_tokens": 80 - hit_Q,
    "shared_prefix_boundary": boundary_Q,
    "full_group_blocks_kept_final": len(blocks_Q.blocks[0]),
    "note_pass1": "候选长度在一轮内逐组传递：full 80 → SWA48 缩到 48 → SWA32 拿到的已是 48（其右到左窗口连续段 [NULL,b1,b2] 也只值 48）",
    "note_pass2": "第二轮 full 不重查（向下封闭：只 min 裁剪——5 次调用里没有它），两个 SWA 以 48 复验通过 → 收敛",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m15.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
