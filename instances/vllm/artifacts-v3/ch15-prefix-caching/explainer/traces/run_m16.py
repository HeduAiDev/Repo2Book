# ch15 m16 Marconi 钉住驱动：不动点产出 num_uncached_common_prefix_tokens
# =longest−reconciled（kv_cache_coordinator.py:L810-L817）→ get_computed_blocks
# 折进 shared_prefix_boundary 写回 Request（kv_cache_manager.py:L286-L295 +
# scheduler.py:L760-L766）→ reachable_boundaries 特赦（single_type:L1404-L1412）
# → _mamba_block_aligned_split 把 chunk 停在 junction（scheduler.py:L424-L437）。
# 配置 = partial-hit 粒度（full(64)+mamba(64,align)、hash_bs=16——impl 与钉版
# 差分一致的配置，见 impl-notes.md）；mamba 组边界条目以同一 cache_partial_block
# 原语注册（真实由 MambaManager.cache_blocks 重写内部 L1729-L1735 同款调用）。
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
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.single_type_kv_cache_manager import MambaManager  # noqa: E402
import torch  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)


def partial_hit_specs(bs):
    return [FullAttentionSpec(block_size=bs, num_kv_heads=2, head_size=8,
                              dtype=torch.float16),
            MambaSpec(block_size=bs, shapes=((8, 8),), dtypes=(torch.float32,),
                      mamba_cache_mode="align")]


def kv_config(specs, num_blocks=32):
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                         kv_cache_groups=[KVCacheGroupSpec([f"l.{i}"], s)
                                          for i, s in enumerate(specs)])


def make_mgr(num_blocks=32):
    return KVCacheManager(kv_cache_config=kv_config(partial_hit_specs(64), num_blocks),
                          max_model_len=512, scheduler_block_size=64,
                          hash_block_size=16)


def run_request(mgr, req):
    blocks, num_hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, req.num_tokens - num_hit,
                             num_new_computed_tokens=num_hit,
                             new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens = req.num_tokens


out = {"params": {"full_block_size": 64, "mamba_block_size": 64,
                  "hash_block_size": 16,
                  "A_tokens": 48, "B_tokens": 80, "shared_prefix_tokens": 48,
                  "scheduler_block_size": 64}}

# --- ① junction 的产出：full 组缓了 @48、mamba 组没缓 → 不动点拖到 0、差值 48 ---
mgr = make_mgr()
pool = mgr.block_pool
mamba_mgr = mgr.coordinator.single_type_managers[1]
reqA = Request("a", list(range(48)), block_hasher=HASHER16)
run_request(mgr, reqA)
mamba_blkA = mamba_mgr.req_to_blocks["a"][0]
pool.cache_partial_block(request=reqA, block=mamba_blkA, num_tokens=48,
                         kv_cache_group_id=1, block_size=64)  # mamba @48
mgr.new_step_starts()
mgr.free(reqA)
# 对照 1：两组都持 @48 → reconciled==longest → uncached=0 → boundary 归零
reqB0 = Request("b0", list(range(48)) + list(range(200, 232)),
                block_hasher=HASHER16)
_, hit_both, boundary_both = mgr.get_computed_blocks(reqB0)
# 差值场景：摘掉 mamba 组 @48 条目 → full 独持 48（longest）、mamba miss
pool._remove_cached_block_hashes(mamba_blkA)
reqB = Request("b", list(range(48)) + list(range(200, 232)),
               block_hasher=HASHER16)
blocks_B, hit_B, boundary_B = mgr.get_computed_blocks(reqB)
out["step1_junction"] = {
    "both_groups_cached_hit": hit_both,
    "both_groups_cached_boundary": boundary_both,
    "mamba_entry_removed_hit": hit_B,
    "longest_hit_length": 48,
    "reconciled_hit": hit_B,
    "num_uncached_common_prefix_tokens": 48 - hit_B,
    "shared_prefix_boundary_formula": "hit + uncached = 0 + 48",
    "boundary": boundary_B,
    "blocks_empty_after_reconcile": blocks_B is mgr.empty_kv_cache_blocks,
    "note": "full 组认 48（longest）但稀疏组没缓 → 差值 48 就是『各组都认但"
            "稀疏组还没缓的共享前缀』——Marconi junction 的原料",
}

# --- ② 写回 Request：调度器准入时写、cache/mask/split 读 ---
sched = Scheduler(kv_config(partial_hit_specs(64)), max_model_len=512,
                  scheduler_block_size=64, hash_block_size=16)
sched.kv_cache_manager = mgr
reqB.num_computed_tokens = 0
sched.admission_lookup(reqB)
out["step2_writeback"] = {
    "request_shared_prefix_boundary": reqB.shared_prefix_boundary,
    "written_by": "scheduler.py admission_lookup（L760-L766）",
    "read_by": "cache_blocks reachable_boundaries + _mamba_block_aligned_split",
}

# --- ③ 特赦段：稀疏驻留下 junction 的边界状态永远保留 ---
mask_amnesty = MambaManager.reachable_block_mask(
    0, 8, alignment_tokens=64,
    kv_cache_spec=MambaSpec(block_size=64, shapes=((8, 8),),
                            dtypes=(torch.float32,), mamba_cache_mode="align"),
    use_eagle=False, retention_interval=0,
    reachable_boundaries=[159, 112])   # replay 边界(B 80token-1 对齐 64→128)、junction 112→64
out["step3_mask_amnesty"] = {
    "retention_interval": 0,
    "replay_boundary_tokens": 159, "replay_boundary_aligned": 128,
    "junction_boundary_tokens": 112, "junction_boundary_aligned": 64,
    "mask_true_block_positions": [i for i, v in enumerate(mask_amnesty) if v],
    "mask_len": len(mask_amnesty),
    "num_true": sum(mask_amnesty),
    "note": "retention=0 只留 reachable_boundaries 的状态块——replay 边界与 "
            "junction 各留一块（块内子边界按对齐下取整归属块）",
}

# --- ④ 调度配合：chunk 停在 junction（块对齐下取整），边界状态才真被算出来 ---
sched2 = Scheduler(kv_config(partial_hit_specs(32)), max_model_len=512,
                   scheduler_block_size=32, hash_block_size=16)
reqL = Request("l", list(range(200)), block_hasher=HASHER16)
reqL.shared_prefix_boundary = 64
n_junction = sched2._mamba_block_aligned_split(reqL, num_new_tokens=100)
reqL2 = Request("l2", list(range(200)), block_hasher=HASHER16)
n_no_junction = sched2._mamba_block_aligned_split(reqL2, num_new_tokens=100)
reqL3 = Request("l3", list(range(210)), block_hasher=HASHER16)
sched2.mamba_partial_cache_hit = True   # hash_bs(16) < block(32) → partial-tail 停点开
n_tail = sched2._mamba_block_aligned_split(reqL3, num_new_tokens=210)
out["step4_split"] = {
    "block_size": 32, "prompt_tokens": 200,
    "junction": 64, "chunk_tokens": 100,
    "chunk_with_junction_stops_at": n_junction,
    "chunk_without_junction": n_no_junction,
    "partial_tail_case_prompt_tokens": 210,
    "partial_tail_boundary": 208,
    "partial_tail_case_full_chunk": n_tail,
    "note": "junction 64 落在 chunk [0,100) 内 → 停在 64（块对齐下取整）；"
            "无 junction 不截；prompt 210 的首 chunk 停在 last_cache_position 192，"
            "下一个 chunk [192,210) 再收在 partial-tail 哈希边界 208——边界状态才真被算出来",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m16.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
