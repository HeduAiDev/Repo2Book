# ch16 m5 异步准入护轨 reserved_blocks：async load 全程占块、零前向进度、
# 不可抢占 → 只许 fits in (free − 其余在途 prefill 预约 − 水位)
# （scheduler.py:L965-L985 + _request_remaining_blocks L2614-L2633）。
# 两道门的精确账（kv_cache_manager.py:L472-L527，同一预测器 get_num_blocks_
# to_allocate）：full-ISL 门 required_full ≤ free（不扣预约）；护轨门
# required_this_step ≤ free − reserved。本脚本直接调预测器取两门实数。
# 场景（16 块池，null 外 15 可用）：
#   r1 = 128 token、ext 64 异步；r2 = 144 token、ext 128 异步。
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


def predict(s, rid, num_tokens, total_computed, main_model, cap):
    """read-only：与两道门同一预测器（apply_admission_cap 同 full-ISL 门）。"""
    return s.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
        request_id=rid,
        num_tokens=num_tokens,
        new_computed_blocks=s.kv_cache_manager.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=total_computed,
        num_local_computed_tokens=0,
        num_tokens_main_model=main_model,
        apply_admission_cap=cap,
    )


steps = []
s = script_scheduler([], num_blocks=16)

# 步1：r1 异步准入（ext 64 → 本拍占 4 块；全程 8 块）
r1 = make_request("r1", range(128))
s.connector.ext_answer = [(64, True)]
s.add_request(r1)
free_before1 = free_count(s)
reserved_before1 = s._inflight_prefill_reserved_blocks()
r1_full = predict(s, "r1", 128, 64, 128, cap=True)     # full-ISL 门的需求
r1_step = predict(s, "r1", 64, 64, 64, cap=False)       # 护轨门的需求（本拍 ext）
out1 = s.schedule()
steps.append({
    "step": 1,
    "event": "r1 异步准入（ext 64）",
    "r1_status": r1.status.name,
    "r1_ext_blocks_held": len(s.kv_cache_manager.get_block_ids("r1")[0]),
    "free_at_admission": free_before1,            # 15
    "reserved_at_admission": reserved_before1,    # 0（首个 async 无他人预约）
    "full_isl_gate": {"required_full_blocks": r1_full, "free": free_before1,
                      "pass": r1_full <= free_before1},          # 8 ≤ 15 ✓
    "guardrail_gate": {"required_this_step_blocks": r1_step,
                       "available": free_before1 - reserved_before1,
                       "pass": r1_step <= free_before1 - reserved_before1},  # 4 ≤ 15 ✓
    "free_blocks_after": free_count(s),           # 15 − 4 = 11
    "inflight_reserved_after": s._inflight_prefill_reserved_blocks(),  # r1 尚需 4
})

# 步2：r2 求入 → full-ISL 门过、护轨门拒（8 > 11−4=7）
r2 = make_request("r2", list(range(100, 244)))   # 144 token
s.connector.ext_answer = [(128, True)]
s.add_request(r2)
r2_full = predict(s, "r2", 144, 128, 144, cap=True)
r2_step = predict(s, "r2", 128, 128, 128, cap=False)
out2 = s.schedule()
free2 = free_count(s)
reserved2 = s._inflight_prefill_reserved_blocks()
steps.append({
    "step": 2,
    "event": "r2 求入（ext 128，本拍需 8 块、全程 9 块）",
    "free_blocks": free2,                         # 11 不变（r2 没占块）
    "inflight_reserved_blocks": reserved2,        # 4（r1 的预约）
    "full_isl_gate": {"required_full_blocks": r2_full, "free": free2,
                      "pass": r2_full <= free2},                 # 9 ≤ 11 ✓ 过
    "guardrail_gate": {"required_this_step_blocks": r2_step,
                       "available": free2 - reserved2,
                       "pass": r2_step <= free2 - reserved2},    # 8 > 7 ✗ 拒
    "r2_status": r2.status.name,                  # 留 WAITING
    "r2_in_skipped_waiting": r2 in list(s.skipped_waiting),   # False：不占 skipped
    "r2_scheduled_tokens": out2.num_scheduled_tokens.get("r2", 0),
    "r2_blocks_held": 0,
})

# 步3：r1 传输完成 → 提升回 WAITING、同拍续算 64 → 又占 4 块；r2 再问：
# 这次是 full-ISL 门拒（9 > free 7）——容量真不够，不是预约的锅
s.update_from_output(out1, worker_output(finished_recving={"r1"}))
out3 = s.schedule()
free3 = free_count(s)
reserved3 = s._inflight_prefill_reserved_blocks()
steps.append({
    "step": 3,
    "event": "r1 传输完成→提升→同拍续算 64",
    "r1_scheduled_tokens": out3.num_scheduled_tokens.get("r1", 0),   # 64
    "r1_status": r1.status.name,
    "free_blocks": free3,                         # 11 − 4 = 7
    "inflight_reserved_blocks": reserved3,        # 0（r1 已算满 128）
    "full_isl_gate": {"required_full_blocks": r2_full, "free": free3,
                      "pass": r2_full <= free3},                 # 9 > 7 ✗ 拒
    "r2_status": r2.status.name,
})

# 步4：r1 完成 → 释放 8 块 → r2 准入（full-ISL 9 ≤ 15；护轨 8 ≤ 15−0）
r1.status = RequestStatus.FINISHED_STOPPED
s._free_request(r1)
free4 = free_count(s)
reserved4 = s._inflight_prefill_reserved_blocks()
s.connector.ext_answer = [(128, True)]
out4 = s.schedule()
steps.append({
    "step": 4,
    "event": "r1 完成→释放 8 块；r2 再问",
    "free_at_admission": free4,                   # 7 + 8 = 15
    "reserved_at_admission": reserved4,           # 0（r1 已出 _inflight_prefills）
    "full_isl_gate": {"required_full_blocks": r2_full, "free": free4,
                      "pass": r2_full <= free4},                 # 9 ≤ 15 ✓
    "guardrail_gate": {"required_this_step_blocks": r2_step,
                       "available": free4 - reserved4,
                       "pass": r2_step <= free4 - reserved4},    # 8 ≤ 15 ✓
    "r2_status": r2.status.name,                  # WAITING_FOR_REMOTE_KVS
    "r2_ext_blocks_held": len(s.kv_cache_manager.get_block_ids("r2")[0]),  # 8
    "free_blocks_after": free_count(s),           # 15 − 8 = 7
    "inflight_reserved_after": s._inflight_prefill_reserved_blocks(),  # 1（r2 尾 16 token）
    "r2_scheduled_tokens": out4.num_scheduled_tokens.get("r2", 0),     # 0（async 零前向）
})

trace = {
    "mechanism": "m5 异步准入护轨 reserved_blocks",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "pool_blocks": 16,
        "free_baseline": full_free(s),            # 15（null 块恒占 1）
        "r1": "128 token，ext 64 异步",
        "r2": "144 token，ext 128 异步",
        "watermark": 0.0,
        "门序注": "两道门同用 get_num_blocks_to_allocate 预测器（apply_admission_cap"
                 " 同 full-ISL 门）：full-ISL 门先查（required_full ≤ free，不扣"
                 "预约），护轨门后查（required_this_step ≤ free − reserved）。",
    },
    "steps": steps,
    "死锁反例": "若无护轨：r1 占 4 块等远端、r2 占 8 块等远端 → free=3，两者都"
               "零前向且不可抢占 → 互相等对方释放 = 死锁（预约正是防这一幕）。",
}
print(dump("m5", trace))
