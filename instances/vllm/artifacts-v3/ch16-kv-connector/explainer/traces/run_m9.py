# ch16 m9 完成回收与全命中退一 token：worker get_finished 报 finished_recving
# → _update_from_kv_xfer_finished 入集 → 下拍 _try_promote 调
# _update_waiting_for_remote_kv：补缓存（延迟入哈希表到此刻）+ 全命中退一
# token（要 logits）→ 按 num_preemptions 回 WAITING/PREEMPTED
# （scheduler.py:L2714-L2741 + L2635-L2693）。
# 场景1：64-token prompt、ext 48 异步 → 完成后同拍续算 16。
# 场景2：64-token prompt、ext 64 异步（全命中）→ 退一 token → 本拍补算 1。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import dump, make_request, script_scheduler, worker_output  # noqa: E402

trace = {
    "mechanism": "m9 完成回收（补缓存 + 全命中退一 token + 回队）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {"block_size": 16, "场景1": "ext 48 异步", "场景2": "ext 64 异步（全命中）"},
    "case1_partial_hit": {},
    "case2_full_hit": {},
}

# ---------------- 场景1：部分命中（48/64）→ 补缓存 3 块 + 同拍续算 16 ----------------
s = script_scheduler([(48, True)])
req = make_request("r1", range(64))
s.add_request(req)
out1 = s.schedule()
mgr = s.kv_cache_manager.coordinator.single_type_managers[0]

# 包一层记录提升前后的 num_computed_tokens（真实函数照跑）
promote_log = []
orig = s._update_waiting_for_remote_kv
def wrapped(r):
    before = r.num_computed_tokens
    orig(r)
    promote_log.append({"req": r.request_id, "before": before,
                        "after": r.num_computed_tokens})
s._update_waiting_for_remote_kv = wrapped

s.update_from_output(out1, worker_output(finished_recving={"r1"}))
trace["case1_partial_hit"].update({
    "step": 1,
    "status_after_async": req.status.name,                     # WAITING_FOR_REMOTE_KVS
    "num_computed_tokens_set_ahead": req.num_computed_tokens,  # 48（先行）
    "cached_blocks_in_window": mgr.num_cached_block.get("r1", 0),  # 0（窗口期未缓存）
    "in_finished_recving_set": "r1" in s.finished_recving_kv_req_ids,
})
out2 = s.schedule()
trace["case1_partial_hit"].update({
    "step": 2,
    "promote_log": promote_log,
    "status_after_promote": req.status.name,                   # RUNNING（同拍续算）
    "cached_blocks_after_promote": mgr.num_cached_block.get("r1"),  # 4（3 补缓存+1 新算）
    "scheduled_tokens_after_promote": out2.num_scheduled_tokens["r1"],  # 64−48=16
})

# ---------------- 场景2：全命中（64/64）→ 退一 token 补算 logits ----------------
s2 = script_scheduler([(64, True)])
req2 = make_request("r1", range(64))
s2.add_request(req2)
outA = s2.schedule()
promote_log2 = []
orig2 = s2._update_waiting_for_remote_kv
def wrapped2(r):
    before = r.num_computed_tokens
    orig2(r)
    promote_log2.append({"req": r.request_id, "before": before,
                         "after": r.num_computed_tokens})
s2._update_waiting_for_remote_kv = wrapped2

s2.update_from_output(outA, worker_output(finished_recving={"r1"}))
outB = s2.schedule()
trace["case2_full_hit"].update({
    "num_computed_tokens_set_ahead": 64,                       # 先行记账=全命中
    "promote_log": promote_log2,                               # 64 → 63（退一 token）
    "scheduled_tokens_after_promote": outB.num_scheduled_tokens["r1"],  # 1
    "why": "要 logits 必须重算最后一个 token——与本地前缀缓存同一条契约（kv_cache_manager.py:L253-L258 同源）",
})
print(dump("m9", trace))
