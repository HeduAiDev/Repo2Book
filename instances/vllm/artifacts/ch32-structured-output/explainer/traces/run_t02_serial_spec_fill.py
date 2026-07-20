"""ch32 explainer driver t02 - m05 串行填充+投机展开, m12 装配前 -1 补齐与统计扣减.

阶段 A(m12,装配前):Scheduler.update_draft_token_ids_in_output
    validate_tokens 过滤不合语法草稿 -> -1 补齐到原长度 -> num_invalid_spec_tokens
阶段 B(m05,装配中):StructuredOutputManager.grammar_bitmask 串行分支
    每个草稿位置各填一行 -> accept_tokens 试探推进 -> 末尾 rollback(state_advancements)
阶段 C(m12,消费点):Scheduler.make_spec_decoding_stats 把 num_invalid 从分母里扣掉

两个请求:rA 草稿全合法(2/2);rB 第二个草稿越界(1/2 合法 -> 补一个 -1)。
输出 JSON 存 t02_serial_spec_fill.json。
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CH, "implementation"))
sys.path.insert(0, os.path.join(CH, "tests"))

import torch  # noqa: E402

from conftest import FakeBackend, FakeGrammar, make_request  # noqa: E402
from output import DraftTokenIds, SchedulerOutput  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from structured_output_manager import StructuredOutputManager  # noqa: E402

VOCAB = 96
MAX_NUM_SEQS = 4
MAX_NUM_SPEC = 2
COLS = -(-VOCAB // 32)

out = {"params": {"vocab_size": VOCAB, "max_num_seqs": MAX_NUM_SEQS,
                  "max_num_spec_tokens": MAX_NUM_SPEC,
                  "bitmask_cols": COLS,
                  "bitmask_buffer_rows": MAX_NUM_SEQS * (1 + MAX_NUM_SPEC),
                  "rows_per_request": 1 + MAX_NUM_SPEC}}

# 语法:位置 0 允许 {5,7};位置 1 允许 {9};位置 2 允许 {11,13};位置 3 允许 {15}
ALLOWED = [{5, 7}, {9}, {11, 13}, {15}]
mgr = StructuredOutputManager(max_num_seqs=MAX_NUM_SEQS,
                              max_num_spec_tokens=MAX_NUM_SPEC)
mgr.backend = FakeBackend(VOCAB)
sched = Scheduler(mgr, num_spec_tokens=MAX_NUM_SPEC)

gA = FakeGrammar([set(s) for s in ALLOWED])
gB = FakeGrammar([set(s) for s in ALLOWED])
rA = make_request("rA", gA)
rB = make_request("rB", gB)
sched.requests["rA"] = rA
sched.requests["rB"] = rB

so = SchedulerOutput(num_scheduled_tokens={"rA": 3, "rB": 3},
                     scheduled_spec_decode_tokens={"rA": [0, 0], "rB": [0, 0]})
so.has_structured_output_requests = True

# ---- 阶段 A:装配前的草稿过滤 + -1 补齐 -----------------------------------------
# 草稿:rA=[5,9] 全合法;rB=[5,8] 第二个越界(位置 1 只允许 9)
draft = DraftTokenIds(req_ids=["rA", "rB"], draft_token_ids=[[5, 9], [5, 8]])
before = {k: list(v) for k, v in zip(draft.req_ids, draft.draft_token_ids)}
sched.update_draft_token_ids_in_output(draft, so)
out["stageA_spec_prefilter"] = {
    "orig_num_spec_tokens_per_req": 2,
    "draft_before_filter": before,
    "validated_and_padded": {k: list(v)
                             for k, v in so.scheduled_spec_decode_tokens.items()},
    "num_invalid_spec_tokens": dict(so.num_invalid_spec_tokens or {}),
    "rows_per_request_after_padding": 1 + MAX_NUM_SPEC,
}

# ---- 阶段 B:串行分支逐行填充(手工展开与真实调用交叉验证) ---------------------
# 掩码缓冲的分配与真实 grammar_bitmask 首次调用时一致(见阶段 B 交叉验证)
mgr._grammar_bitmask = mgr.backend.allocate_token_bitmask(
    MAX_NUM_SEQS * (1 + MAX_NUM_SPEC))
serial_rows = []
cumulative_index = 0
for req_id in ["rA", "rB"]:
    request = sched.requests[req_id]
    grammar = request.structured_output_request.grammar
    apply_bitmask = mgr.should_fill_bitmask(request)
    state_advancements = 0
    req_tokens = so.scheduled_spec_decode_tokens.get(req_id, ())
    for token in itertools.chain(req_tokens, (-1,)):
        pos_before = grammar.position
        apply_before = bool(apply_bitmask)
        mgr._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
        row = mgr._grammar_bitmask[cumulative_index]
        allowed_now = sorted(
            t for t in range(VOCAB)
            if (int(row[t // 32].item()) >> (t % 32)) & 1)
        full_mask_row = all(int(row[c].item()) == -1 for c in range(COLS))
        if token == -1:
            apply_bitmask = False
        accepted_here = False
        if apply_bitmask and not grammar.is_terminated():
            accepted = grammar.accept_tokens(req_id, [token])
            assert accepted, (token, req_id)
            accepted_here = True
            state_advancements += 1
        serial_rows.append({
            "req_id": req_id,
            "row_index": cumulative_index,
            "token_at_this_position": token,
            "grammar_position_before_fill": pos_before,
            "apply_bitmask_before": apply_before,
            "row_is_full_mask_all_minus_1": full_mask_row,
            "allowed_tokens_written": allowed_now if not full_mask_row else "ALL",
            "accept_tokens_called": accepted_here,
            "grammar_position_after": grammar.position,
            "state_advancements_so_far": state_advancements,
        })
        cumulative_index += 1
    pos_before_rollback = grammar.position
    if state_advancements > 0:
        grammar.rollback(state_advancements)
    serial_rows.append({
        "req_id": req_id,
        "row_index": None,
        "token_at_this_position": None,
        "grammar_position_before_fill": pos_before_rollback,
        "apply_bitmask_before": None,
        "row_is_full_mask_all_minus_1": None,
        "allowed_tokens_written": "rollback(%d)" % state_advancements,
        "accept_tokens_called": False,
        "grammar_position_after": grammar.position,
        "state_advancements_so_far": state_advancements,
    })
out["stageB_serial_fill"] = {
    "rows": serial_rows,
    "cumulative_index_final": cumulative_index,
    "buffer_rows_allocated": MAX_NUM_SEQS * (1 + MAX_NUM_SPEC),
    "rows_after_trim": cumulative_index,
    "gA_position_after_rollback": gA.position,
    "gB_position_after_rollback": gB.position,
    "gA_rollback_log": gA.rollback_log,
    "gB_rollback_log": gB.rollback_log,
}

# 交叉验证:同一场景直接调真实 grammar_bitmask(重建干净状态)
mgr2 = StructuredOutputManager(max_num_seqs=MAX_NUM_SEQS,
                               max_num_spec_tokens=MAX_NUM_SPEC)
mgr2.backend = FakeBackend(VOCAB)
gA2 = FakeGrammar([set(s) for s in ALLOWED])
gB2 = FakeGrammar([set(s) for s in ALLOWED])
rA2 = make_request("rA", gA2)
rB2 = make_request("rB", gB2)
bm = mgr2.grammar_bitmask({"rA": rA2, "rB": rB2}, ["rA", "rB"],
                          {k: list(v) for k, v in so.scheduled_spec_decode_tokens.items()})
out["stageB_crosscheck_real_call"] = {
    "bitmask_rows": int(bm.shape[0]),
    "bitmask_cols": int(bm.shape[1]),
    "gA_fill_log_rows": gA2.fill_log,
    "gB_fill_log_rows": gB2.fill_log,
    "gA_accept_log": [list(x) for x in gA2.accept_log],
    "gB_accept_log": [list(x) for x in gB2.accept_log],
    "gA_rollback_log": gA2.rollback_log,
    "gB_rollback_log": gB2.rollback_log,
    "gA_position_final": gA2.position,
    "gB_position_final": gB2.position,
    "per_row_allowed_tokens": [
        sorted(t for t in range(VOCAB)
               if (int(bm[r][t // 32]) >> (t % 32)) & 1)
        if not all(int(bm[r][c]) == -1 for c in range(COLS)) else "ALL"
        for r in range(bm.shape[0])],
}

# ---- 阶段 C:num_invalid_spec_tokens 的消费点 -----------------------------------
stats = None
consume = []
for req_id in ["rA", "rB"]:
    n_draft = 2
    n_acc = 2 if req_id == "rA" else 1
    stats = sched.make_spec_decoding_stats(
        stats, num_draft_tokens=n_draft, num_accepted_tokens=n_acc,
        num_invalid_spec_tokens=so.num_invalid_spec_tokens, request_id=req_id)
    consume.append({
        "req_id": req_id,
        "num_draft_tokens_raw": n_draft,
        "num_invalid_for_req": (so.num_invalid_spec_tokens or {}).get(req_id, 0),
        "num_draft_tokens_counted": n_draft - (so.num_invalid_spec_tokens or {}).get(req_id, 0),
        "num_accepted_tokens": n_acc,
        "cum_num_draft_tokens": stats.num_draft_tokens,
        "cum_num_accepted_tokens": stats.num_accepted_tokens,
    })
out["stageC_stats_consumption"] = {
    "rows": consume,
    "acceptance_rate_with_deduction": round(
        stats.num_accepted_tokens / stats.num_draft_tokens, 4),
    "acceptance_rate_without_deduction": round(
        stats.num_accepted_tokens / 4, 4),
}

path = os.path.join(HERE, "t02_serial_spec_fill.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
