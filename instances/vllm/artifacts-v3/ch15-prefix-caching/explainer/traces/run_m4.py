# ch15 m4 命中查找主路径驱动：phase 1 沿满块链逐块查、第一个 miss 即断
# （single_type_kv_cache_manager.py:L731-L739）+ get_computed_blocks 的
# max_cache_hit_length=num_tokens−1（kv_cache_manager.py:L253-L259）。
# 探测内景以同一原语 block_pool.get_cached_block 逐块复现（phase 1 循环体逐字同款）。
import json
import math
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


def full_spec(bs):
    return FullAttentionSpec(block_size=bs, num_kv_heads=2, head_size=8,
                             dtype=torch.float16)


def make_manager(hash_bs, num_blocks=64):
    cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                        kv_cache_groups=[KVCacheGroupSpec(["l.0"], full_spec(16))])
    return KVCacheManager(kv_cache_config=cfg, max_model_len=512,
                          scheduler_block_size=16, hash_block_size=hash_bs)


def run_request(mgr, req):
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, req.num_tokens - num_hit,
                             num_new_computed_tokens=num_hit,
                             new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


mgr = make_manager(16)
pool = mgr.block_pool

# A 先跑：64 token = 4 满块（块 id 1-4），完成后 free——前缀留在表里
reqA = Request("a", list(range(64)), block_hasher=HASHER16)
run_request(mgr, reqA)
mgr.free(reqA)

out = {"params": {
    "block_size": 16,
    "hash_block_size": 16,
    "prompt_A_tokens": 64,
    "A_cached_blocks": 4,
    "A_block_ids": [1, 2, 3, 4],
}}

# --- 场景 B：前 32 token 同 A、后面分叉 → 链上走到第 3 块 miss 即断 ---
reqB = Request("b", list(range(32)) + list(range(100, 132)), block_hasher=HASHER16)
probes = []
for i, bh in enumerate(reqB.block_hashes):
    cached = pool.get_cached_block(bh, [0])
    probes.append({"block_idx": i, "cover_tokens": f"{i * 16}-{i * 16 + 15}",
                   "hit": cached is not None,
                   "block_id": cached[0].block_id if cached else None,
                   "probed": i <= 2})  # phase 1 在 i=2 miss 即 break，i=3 不再探
    if cached is None:
        break
blocks_b, hit_b, junction_b = mgr.get_computed_blocks(reqB)
out["case_B"] = {
    "prompt_tokens": reqB.num_tokens,
    "max_cache_hit_length": reqB.num_tokens - 1,
    "probes": probes,
    "first_miss_block_idx": 2,
    "hit_tokens": hit_b,
    "hit_block_ids": [b.block_id for b in blocks_b.blocks[0]],
    "junction": junction_b,
    "recompute_tokens": reqB.num_tokens - hit_b,
}

# --- 场景 C：与 A 完全一致 → 全命中也退一 token（要 logits）+ 块对齐再回退整块 ---
reqC = Request("c", list(range(64)), block_hasher=HASHER16)
blocks_c, hit_c, _ = mgr.get_computed_blocks(reqC)
out["case_C"] = {
    "prompt_tokens": reqC.num_tokens,
    "max_cache_hit_length": reqC.num_tokens - 1,
    "probe_budget_blocks": (reqC.num_tokens - 1) // 16,
    "all_blocks_in_table": True,
    "hit_tokens": hit_c,
    "hit_block_ids": [b.block_id for b in blocks_c.blocks[0]],
    "last_token_recompute": 1,
    "block_align_extra_backoff": 64 - 1 - hit_c,
    "recompute_tokens": reqC.num_tokens - hit_c,
}

# --- 场景 D：17 token prompt（1 满块+1 尾）——最小回退例 ---
mgr2 = make_manager(16)
reqA2 = Request("a2", list(range(17)), block_hasher=HASHER16)
run_request(mgr2, reqA2)
mgr2.free(reqA2)
reqD = Request("d", list(range(17)), block_hasher=HASHER16)
_, hit_d, _ = mgr2.get_computed_blocks(reqD)
out["case_D"] = {
    "prompt_tokens": 17,
    "max_cache_hit_length": 17 - 1,
    "hit_tokens": hit_d,
    "recompute_tokens": 17 - hit_d,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
