# ch16 m10 失败回滚：invalid_block_ids → _handle_invalid_blocks 双策——
# fail（默认）整请求 FINISHED_ERROR；recompute 按第一个坏块截断
# num_computed_tokens 到最长有效前缀（块对齐）、共享坏块只重算一次、
# 重算区补登记清零 record_blocks_for_zeroing（块对齐——半有效块清零会抹掉
# 有效前缀）（scheduler.py:L2743-L2914 / kv_cache_manager.py:L817-L829）。
# 场景1：fail 策略——第 3 块坏 → 整请求 FINISHED_ERROR。
# 场景2：recompute 异步失败——64 token、ext 48（3 块）、第 3 块（idx 2）坏
#   → 截断到 32 → 补缓存 2 块 + 补登记清零 → 重算 32。
# 场景3：共享坏块（sync）——seed/b 共享 3 块、中间块坏：seed 截到 16、b 按
#   已标记处理回退自己的 cached 计数 48——同一块只重算一次。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (  # noqa: E402
    dump,
    free_count,
    full_free,
    make_request,
    script_scheduler,
    worker_output,
)
from implementation.request import RequestStatus  # noqa: E402

trace = {
    "mechanism": "m10 失败回滚（第一个坏块截断 + 双策 + 补登记清零）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "block_size": 16,
        "场景1": "fail 策略：ext 48 同步、坏块 idx 2",
        "场景2": "recompute + 异步：ext 48、坏块 idx 2",
        "场景3": "共享坏块（sync recompute）",
    },
}

# ---------------- 场景1：fail（默认）→ FINISHED_ERROR ----------------
s1 = script_scheduler([(48, False)])            # 默认 failure_policy="fail"
r = make_request("r1", range(64))
s1.add_request(r)
out1 = s1.schedule()
bad1 = s1.kv_cache_manager.get_block_ids("r1")[0][2]   # 第 3 块（idx 2）坏
s1.update_from_output(out1, worker_output(invalid={bad1}))
trace["case1_fail"] = {
    "policy": "fail（默认）",
    "prompt_tokens": 64,
    "external_tokens": 48,
    "bad_block_index": 2,
    "bad_block_id": bad1,
    "r1_status": r.status.name,                 # FINISHED_ERROR
    "r1_removed_from_requests": "r1" not in s1.requests,
    "free_blocks_after": free_count(s1),
    "pool_free_baseline": full_free(s1),
}

# ---------------- 场景2：recompute + 异步失败 → 截断 + 补缓存 + 补登记清零 ----------------
s2 = script_scheduler([(48, True)], failure_policy="recompute")
r2 = make_request("r1", range(64))
s2.add_request(r2)
outA = s2.schedule()
mgr2 = s2.kv_cache_manager.coordinator.single_type_managers[0]
ids2 = s2.kv_cache_manager.get_block_ids("r1")[0]
bad2 = ids2[2]                                  # 第 3 块（idx 2）坏

s2.update_from_output(outA, worker_output(invalid={bad2}))
after_invalid = {
    "bad_block_index": 2,
    "bad_block_id": bad2,
    "num_computed_tokens_truncated_to": r2.num_computed_tokens,   # 2*16=32
    "in_failed_recving_set": "r1" in s2.failed_recving_kv_req_ids,
}

s2.update_from_output(outA, worker_output(finished_recving={"r1"}))
mgr2.new_block_ids.clear()                      # 只看失败分支补登记的清零账
outB = s2.schedule()
after_promote = {
    "cached_blocks_valid_prefix": None,         # 补缓存后（下面读）
    "zeroing_re_registered_block_ids": list(mgr2.new_block_ids),  # 重算区 2 块
    "recompute_tokens_scheduled": outB.num_scheduled_tokens["r1"],  # 64−32=32
    "cached_blocks_after_recompute": mgr2.num_cached_block.get("r1"),  # 4
}
# 补缓存的 2 块（有效前缀）在 promote 时入表——从块哈希账读回
after_promote["cached_blocks_valid_prefix"] = mgr2.num_cached_block.get("r1") - 2

trace["case2_recompute_async"] = {
    "policy": "recompute",
    "prompt_tokens": 64,
    "external_tokens_async": 48,
    "blocks_allocated": len(ids2),              # 3
    "after_invalid_blocks": after_invalid,
    "after_finished_recving_promote": after_promote,
    "needs_kv_cache_zeroing": s2.needs_kv_cache_zeroing,   # False：单组 fp16 派生关
    "zeroing注": "本场景单组 fp16 → needs_kv_cache_zeroing=False，失败分支跳过补登记"
                 "（zeroing_re_registered_block_ids 恒空）。开关打开（mamba/混合精度）时："
                 "失败加载把截断区留在『未写』态且其清零曾被跳过——record_blocks_for_"
                 "zeroing 从块对齐的 32 起补登记清零，重算前必须干净（zeroing a "
                 "partially-valid block would wipe its valid prefix）——见 case2b。",
}

# ---------------- 场景3：共享坏块只重算一次（sync 路径） ----------------
s3 = script_scheduler([], failure_policy="recompute")
seed = make_request("seed", range(48))
s3.add_request(seed)
s3.schedule()                                   # seed 算完 48、3 块入哈希表
b = make_request("b", list(range(48)))
s3.add_request(b)
s3.schedule()                                   # b 命中 seed 的 32 + 新算 16
ids_seed = s3.kv_cache_manager.get_block_ids("seed")[0]
ids_b = s3.kv_cache_manager.get_block_ids("b")[0]
bad3 = ids_b[1]                                 # 中间块，seed 与 b 共享
failed = s3._handle_invalid_blocks({bad3}, {})
trace["case3_shared_bad_block"] = {
    "seed_computed_before": 48,
    "b_computed_before": 48,                    # 32 命中 + 16 新算
    "shared_bad_block_id": bad3,
    "shared_bad_in_both_tables": bad3 in ids_seed and bad3 in ids_b,
    "affected_req_ids": sorted(failed),         # ["b", "seed"]
    "seed_truncated_to": seed.num_computed_tokens,   # 1*16=16（第一个坏块截断）
    "b_falls_back_to_cached": b.num_computed_tokens, # 3*16=48（坏块已被 seed 标记重算）
    "语义": "seed 重算坏块；b 把该块当『将被 seed 重算』——同一块只重算一次",
}

# ---------------- 场景2b：重算区补登记清零（record_blocks_for_zeroing） ----------------
# 说明：异步失败 + 补登记清零的端到端组合需要 needs_kv_cache_zeroing=True
# （混合精度派生）——而 pin 源码对混合组的 invalid 扫描是 TODO
# （scheduler.py:L2781『add support for hybrid memory allocator』，单组为准），
# 故此处按单测同款直接驱动 record_blocks_for_zeroing（kv_cache_manager.py:L817-L829）。
s2b = script_scheduler([], needs_zeroing=True, enable_caching=False)
rb = make_request("r1", range(64))
s2b.add_request(rb)
s2b.schedule()                                  # NoPrefixCache 支路：分配 4 块
mgr2b = s2b.kv_cache_manager.coordinator.single_type_managers[0]
mgr2b.new_block_ids.clear()
ids2b = s2b.kv_cache_manager.get_block_ids("r1")[0]
non_aligned_rejected = False
try:
    s2b.kv_cache_manager.record_blocks_for_zeroing("r1", 17)   # 非块对齐 → 断言拒绝
except AssertionError:
    non_aligned_rejected = True
s2b.kv_cache_manager.record_blocks_for_zeroing("r1", 32)       # 从块对齐 32 起补登记
trace["case2b_zeroing_re_register"] = {
    "needs_kv_cache_zeroing": s2b.needs_kv_cache_zeroing,      # True（混合精度派生）
    "blocks_on_table": len(ids2b),                             # 4
    "non_block_aligned_attempted_token": 17,                   # 17 % 16 != 0
    "non_block_aligned_rejected": non_aligned_rejected,        # → AssertionError
    "re_registered_from_token": 32,                            # 块对齐
    "zeroing_re_registered_block_ids": sorted(mgr2b.new_block_ids),
    "expected_recompute_region_blocks": ids2b[2:],             # 块 2、3（token 32..63）
    "why_block_aligned": "zeroing a partially-valid block would wipe its valid prefix"
                         "——半有效块清零会抹掉有效前缀（kv_cache_manager.py:L823-L825）",
    "端到端注": "异步失败+清零的组合走混合精度配置；pin 对混合组的 invalid 扫描是"
               "TODO（scheduler.py:L2781），本场景按单测同款直接调用补登记入口。",
}
print(dump("m10", trace))
