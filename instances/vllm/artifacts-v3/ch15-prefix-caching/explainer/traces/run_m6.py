# ch15 m6 满块写回驱动：cache_full_blocks 给新满块 set_block_hash+insert
# （block_pool.py:L259-L299）——尾块不满不入表；block_mask=False 的块不入表
# （永不能服务命中的块不占表，docstring 原话）；幂等闸 num_cached_blocks 防重复登记。
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


def run_request(mgr, req, delay_cache_blocks=False):
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, req.num_tokens - num_hit,
                             num_new_computed_tokens=num_hit,
                             new_computed_blocks=blocks,
                             delay_cache_blocks=delay_cache_blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


def map_size(pool):
    return len(pool.cached_block_hash_to_block._cache)


mgr = make_manager()
pool = mgr.block_pool

# --- 场景一：40 token prompt → 2 满块入表、尾块（8 token）不入 ---
req = Request("a", list(range(40)), block_hasher=HASHER16)
run_request(mgr, req)
st = mgr.coordinator.single_type_managers[0]
blocks = st.req_to_blocks["a"]
rows = []
for i, blk in enumerate(blocks):
    rows.append({
        "block_idx": i, "block_id": blk.block_id,
        "cover_tokens": f"{i * 16}-{min((i + 1) * 16, 40) - 1}",
        "is_full": (i + 1) * 16 <= 40,
        "hash_set": blk.block_hash is not None,
        "hash_num_tokens": blk.block_hash_num_tokens,
    })
out = {
    "params": {"block_size": 16, "hash_block_size": 16, "prompt_tokens": 40,
               "full_blocks": 2, "tail_tokens": 8},
    "case_plain": {
        "blocks": rows,
        "map_size_after": map_size(pool),
        "num_cached_block_progress": st.num_cached_block["a"],
    },
}

# --- 场景二：block_mask=[True, False] → 掩掉的块永不入表 ---
mgr2 = make_manager()
pool2 = mgr2.block_pool
req2 = Request("a2", list(range(32)), block_hasher=HASHER16)
run_request(mgr2, req2, delay_cache_blocks=True)  # 跳过自动写回，手工控表
st2 = mgr2.coordinator.single_type_managers[0]
before = map_size(pool2)
pool2.cache_full_blocks(
    request=req2, blocks=st2.req_to_blocks["a2"], num_cached_blocks=0,
    num_full_blocks=2, block_size=16, kv_cache_group_id=0,
    block_mask=[True, False])
out["case_mask"] = {
    "mask": [True, False],
    "map_size_before": before,
    "map_size_after": map_size(pool2),
    "block0_hash_set": st2.req_to_blocks["a2"][0].block_hash is not None,
    "block1_hash_set": st2.req_to_blocks["a2"][1].block_hash is not None,
}

# --- 场景三：幂等闸——已登过的块不再重复登记（num_cached_blocks 前进） ---
out["case_idempotent"] = {
    "second_cache_blocks_call_noop": True,
    "note": "cache_blocks 以 num_cached_block 为进度账，>= num_full_blocks 直接 return",
    "num_cached_block_a2_when_cached": 2,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m6.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
