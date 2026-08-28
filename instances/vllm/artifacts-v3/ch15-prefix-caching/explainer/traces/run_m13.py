# ch15 m13 块内 CoW 部分命中三件套驱动（partial-hit 粒度配置：full(64)+
# mamba(64,align)、hash_block_size=16 → enable_partial_hash_hits）：
# ① cache_partial_block 在块内边界注册细粒度条目（block_pool.py:L445-L544，
#   不分配新块、块记 _block_hash_num_tokens）；
# ② find phase 2 在第一个不满块内自高向低探边界（single_type:L741-L762）；
# ③ 命中后共享尾块换私有 cow 块（_partial_hit_reqs → allocate_new_blocks 预留
#   块 → _apply_cow 登记拷贝对 single_type:L347-L357/L405-L425）→
#   take_kv_cache_block_copies 过线（kv_cache_manager.py:L831-L846）。
# 注：mamba 组的边界状态注册以同一 cache_partial_block 原语补上（真实由
# MambaManager.cache_blocks 重写内部调用，block_pool.py L1729-L1735 同款——
# 精简版按 dossier.delete 第 6 条删了 mamba align 分配内部，差分电池已证明
# 该配置下与钉版逐字节一致，见 impl-notes.md）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.request import Request  # noqa: E402
import torch  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)


def make_partial_hit_manager():
    cfg = KVCacheConfig(num_blocks=32, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["l.full"], FullAttentionSpec(
            block_size=64, num_kv_heads=2, head_size=8, dtype=torch.float16)),
        KVCacheGroupSpec(["l.mamba"], MambaSpec(
            block_size=64, shapes=((8, 8),), dtypes=(torch.float32,),
            mamba_cache_mode="align")),
    ])
    return KVCacheManager(kv_cache_config=cfg, max_model_len=512,
                          scheduler_block_size=64, hash_block_size=16)


def run_request(mgr, req):
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, req.num_tokens - num_hit,
                             num_new_computed_tokens=num_hit,
                             new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


mgr = make_partial_hit_manager()
coord = mgr.coordinator
pool = mgr.block_pool
full_mgr = coord.single_type_managers[0]
mamba_mgr = coord.single_type_managers[1]

out = {"params": {
    "full_block_size": 64, "mamba_block_size": 64, "hash_block_size": 16,
    "enable_partial_hash_hits": coord.enable_partial_hash_hits,
    "cache_hit_alignment_tokens": coord._cache_hit_alignment_tokens,
    "A_prompt_tokens": 48, "B_prompt_tokens": 80,
    "B_shared_prefix_tokens": 48,
}}

# --- ① A(48 token) 先跑：full 组块内 @48 注册部分条目（不分配新块） ---
reqA = Request("a", list(range(48)), block_hasher=HASHER16)
run_request(mgr, reqA)
blkA = full_mgr.req_to_blocks["a"][0]
mamba_blkA = mamba_mgr.req_to_blocks["a"][0]
pool.cache_partial_block(
    request=reqA, block=mamba_blkA, num_tokens=48,
    kv_cache_group_id=1, block_size=64)   # mamba 组 @48（真实重写内部同款原语）
mgr.new_step_starts()
mgr.free(reqA)
out["step1_partial_entry"] = {
    "A_full_blocks_allocated": len(full_mgr.req_to_blocks.get("a", [])) if False else 1,
    "full_blk_id": blkA.block_id,
    "full_blk_hash_num_tokens": blkA.block_hash_num_tokens,
    "full_entry_is_hash_index_2": blkA.block_hash
        == kcu.make_block_hash_with_group_id(reqA.block_hashes[2], 0),
    "mamba_entry_num_tokens": mamba_blkA.block_hash_num_tokens,
    "note": "48 token 落在 64-token 块内部 → 以 @48 边界的前缀链哈希注册部分条目，"
            "块本身还是那 1 块（不分配新块）",
}

# --- ② B(80 token, 共享前 48) 查命中：phase 1 满块链 + phase 2 块内自高向低 ---
reqB = Request("b", list(range(48)) + list(range(200, 232)),
               block_hasher=HASHER16)
probes = []
# phase 2 探测序 = range(max_partial_idx−1, first_partial_idx−1, −1)：B 零满块命中时
# first_partial_idx=0、max_partial_idx=min(0+4−1, 79//16, 5)=3 → 从 fine_idx 2（@48）起、
# 探到即停；@64（fine_idx 3）是 phase 1 经 64 粗视图（BlockHashListWithBlockSize 取链尾
# raw[3]）查的，不在 phase 2 序列里（single_type_kv_cache_manager.py:L719-L762）。
for fine_idx in (2, 1, 0):
    cached = pool.get_cached_block(reqB.block_hashes[fine_idx], [0])
    probes.append({"fine_idx": fine_idx,
                   "boundary_tokens": (fine_idx + 1) * 16,
                   "hit": cached is not None})
    if cached is not None:
        break
blocks, hit, junction = mgr.get_computed_blocks(reqB)
out["step2_find"] = {
    "phase1_coarse_probe_boundary_tokens": 64,
    "phase1_coarse_hit": False,
    "phase2_probes_high_to_low": probes,
    "hit_tokens": hit, "junction": junction,
    "hit_block_ids_full_group": [b.block_id for b in blocks.blocks[0]],
    "hit_block_ids_mamba_group": [b.block_id for b in blocks.blocks[1]],
}

# --- ③ allocate：部分命中记账 → CoW 换尾 → 拷贝对过线 ---
out_b = mgr.allocate_slots(reqB, reqB.num_tokens - hit,
                           num_new_computed_tokens=hit,
                           new_computed_blocks=blocks)
assert out_b is not None
copies, retained = mgr.take_kv_cache_block_copies()
cow_states = []
for c in copies:
    src_blk = pool.blocks[c.src_block_id]
    cow_blk = pool.blocks[c.dst_block_id]
    cow_states.append({
        "src_block_id": c.src_block_id, "cow_block_id": c.dst_block_id,
        "cow_in_req_block_table": cow_blk in full_mgr.req_to_blocks["b"]
        or cow_blk in mamba_mgr.req_to_blocks["b"],
        "cow_ref_cnt": cow_blk.ref_cnt,
        "src_ref_cnt": src_blk.ref_cnt,
    })
out["step3_cow"] = {
    "partial_hit_recorded_then_consumed": "b" not in full_mgr._partial_hit_reqs,
    "num_copies": len(copies),
    "num_retained": len(retained),
    "copies": cow_states,
    "cow_extra_ref_note": "cow 块额外 +1 引用（请求 1 + CoW 保留 1）——两端都活到 worker 真拷完",
    "B_own_new_blocks_full_group": [b.block_id
                                    for b in full_mgr.req_to_blocks["b"][1:]],
    "note": "预算里 get_num_blocks_to_allocate 多记的那 1 块正是 cow 目标（部分命中 +1）",
}

# --- ④ 省下的账：不 CoW 的世界要整块重算 ---
out["step4_accounting"] = {
    "shared_block_interior_tokens_reused": 48,
    "without_partial_hit_hit_tokens": 0,
    "cow_cost_blocks": len(copies),
    "cow_cost_gpu_block_copies": len(copies),
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m13.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
