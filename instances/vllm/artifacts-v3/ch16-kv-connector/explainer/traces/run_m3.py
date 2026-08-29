# ch16 m3 双命中仲裁：本地命中呈 block_aligned_local 给 connector；远端严格
# 更长 → truncate 砍本地子块尾（免 CoW）；否则保尾不加载；混合发散且无外部
# 命中 → 回退全组一致边界（scheduler.py:L791-L821 / kv_cache_manager.py:L297-L342）。
# 配置：full(16)+mamba-align(16) 两组、hash 8（ch15 m15/m16 的 partial-hit 粒度）
# ——块内边界命中（partial_tail≠0）只在此形态出生。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (  # noqa: E402
    HASHER8,
    blk_by_hash,
    dump,
    hybrid_scheduler,
    make_request,
    run_and_cache_prefix,
)

trace = {
    "mechanism": "m3 双命中仲裁（子块尾）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "block_size": 16,
        "hash_block_size": 8,
        "groups": "full(16) + mamba-align(16)",
        "seed_prompt_tokens": 40,          # 种子前缀：0..39
        "probe_prompt_tokens": 56,         # b = 前 40 共享 + 16 新
    },
    "caseA": {},
    "caseB": {},
}

SHARED = list(range(40))
NEW = list(range(100, 116))


def probe_groups(s, tokens):
    """逐组命中探测（find_longest_cache_hit_per_group 只读调用）。"""
    probe = make_request("probe", tokens, hasher=HASHER8)
    computed, per_group_hits = (
        s.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
            probe.block_hashes, probe.num_tokens - 1
        )
    )
    return {"per_group_hits_tokens": per_group_hits, "num_tokens": probe.num_tokens}


# ---------------- 场景 A：远端严格更长（16 > partial_tail 8）→ 砍本地尾 ----------------
sA = hybrid_scheduler([])
run_and_cache_prefix(sA, "seed", SHARED, hasher=HASHER8)
trace["caseA"]["probe_before_schedule"] = probe_groups(sA, SHARED + NEW)

sA.connector.ext_answer = [(16, False)]
b = make_request("b", SHARED + NEW, hasher=HASHER8)
sA.add_request(b)
out = sA.schedule()
pool = sA.kv_cache_manager.block_pool

local_hit = trace["caseA"]["probe_before_schedule"]["per_group_hits_tokens"][0]
partial_tail = local_hit % 16
block_aligned = local_hit - partial_tail
b_ids = sA.kv_cache_manager.get_block_ids("b")[0]
seed_b0 = blk_by_hash(pool, b.block_hashes[1]).block_id   # 16-token 边界
seed_b1 = blk_by_hash(pool, b.block_hashes[3]).block_id   # 32-token 边界
seed_partial = blk_by_hash(pool, b.block_hashes[4])       # 40-token 子块尾条目

trace["caseA"].update({
    "ext_answer": 16,
    "local_hit_tokens": local_hit,             # full 组块内边界命中 40
    "mamba_group_hit_tokens": trace["caseA"]["probe_before_schedule"]["per_group_hits_tokens"][1],
    "partial_tail": partial_tail,              # 40 % 16 = 8
    "block_aligned_local_presented_to_connector": block_aligned,  # 呈 32 不呈 40
    "query_seen_by_connector": [
        e for e in sA.connector.events if e[0] == "query"
    ][-1][2],
    "remote_strictly_longer": 16 > partial_tail,
    "local_after_truncate": block_aligned,     # 砍尾后本地 = 32
    "external_adopted": 16,
    "num_computed_tokens_admission": block_aligned + 16,   # 48
    "scheduled_tokens": out.num_scheduled_tokens["b"],     # 56 − 48 = 8
    "b_block_table_len": len(b_ids),
    "b_first_two_blocks_are_seed_shared": b_ids[:2] == [seed_b0, seed_b1],
    "seed_shared_block_refcnt": pool.blocks[seed_b0].ref_cnt,   # b 采用 → 1
    "subtail_block_id": seed_partial.block_id,
    "subtail_block_in_b_table": seed_partial.block_id in b_ids,  # False
    "subtail_block_refcnt": seed_partial.ref_cnt,                # 0 → 免 CoW
    "subtail_still_in_hash_table": seed_partial.block_hash is not None,
})

# ---------------- 场景 B：远端不严格更长（8 不> 8）→ 保尾外部 0；发散回退 ----------------
sB = hybrid_scheduler([])
run_and_cache_prefix(sB, "seed", SHARED, hasher=HASHER8)
trace["caseB"]["probe_before_schedule"] = probe_groups(sB, SHARED + NEW)

sB.connector.ext_answer = [(8, False)]
b2 = make_request("b", SHARED + NEW, hasher=HASHER8)
sB.add_request(b2)
out2 = sB.schedule()

trace["caseB"].update({
    "ext_answer": 8,
    "local_hit_tokens": trace["caseB"]["probe_before_schedule"]["per_group_hits_tokens"][0],  # 40
    "partial_tail": 40 % 16,
    "remote_strictly_longer": 8 > 8,
    "external_adopted": 0,
    "hit_diverged_no_external_fallback": True,   # mamba 32 < full 40 → 回退
    "reconciled_boundary_tokens": 32,            # 全组一致边界
    "scheduled_tokens": out2.num_scheduled_tokens["b"],  # 56 − 32 = 24
    "update_state_after_alloc_ext": [
        e for e in sB.connector.events if e[0] == "update_state_after_alloc"
    ][-1][2],                                    # 0
})

print(dump("m3", trace))
