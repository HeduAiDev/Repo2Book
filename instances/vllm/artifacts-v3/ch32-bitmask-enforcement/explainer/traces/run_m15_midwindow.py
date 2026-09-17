# ch32 m15 驱动：掩码装配窗口内的思考结束检测（#42452 修复现场）。
# 真 StructuredOutputManager.grammar_bitmask 串行分支（精简版逐字）+ 真 xgrammar
# grammar 'root ::= "a" "b" "c"' + FakeReasoner（结束标记 = token 999）。
# 草稿窗口 [r3, MARKER, a, b]：标记落在窗口中段——标记行不推进、其后行翻转受约束、
# bonus 行双触发（should_fill_bitmask 仍 False 的时序坑）、翻转后容忍草稿被拒。
import sys

import numpy as np

from _ch32_common import (TOK_A, TOK_B, TOK_C, VOCAB, FakeRequest,
                          RealBackend, allowed_ids, dump, get_tokenizer,
                          make_vllm_config, new_grammar, row_full_allow)
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
from vllm.v1.structured_output import StructuredOutputManager

EBNF = 'root ::= "a" "b" "c"'
tok = get_tokenizer()
R3, MARKER = 7, 999  # r3=思考内容草稿；999=结束标记（替身）
out = {"env": {"grammar": EBNF, "token_ids": {"a": TOK_A, "b": TOK_B, "c": TOK_C,
                                              "r3_draft": R3, "marker": MARKER},
               "vocab": VOCAB}}


class FakeReasoner:
    def __init__(self, tokenizer=None, **kwargs):
        pass

    def is_reasoning_end(self, prompt_token_ids):
        return False

    def is_reasoning_end_streaming(self, all_token_ids, delta_ids):
        # 流式判定：新到的这个 token 是结束标记 → 思考结束
        return list(delta_ids)[-1] == MARKER


def make_mgr(num_spec, with_reasoner=True):
    m = StructuredOutputManager(make_vllm_config(
        max_num_seqs=8, num_speculative_tokens=num_spec))
    m.backend = RealBackend(VOCAB)
    if with_reasoner:
        m.reasoner_cls = FakeReasoner
        m.tokenizer = None
    return m


def snap(mask):
    rows = []
    for i in range(mask.shape[0]):
        if row_full_allow(mask[i]):
            rows.append({"row": i, "full_allow": True, "allowed_count": VOCAB})
        else:
            a = allowed_ids(mask[i], VOCAB)
            rows.append({"row": i, "full_allow": False, "allowed": a})
    return rows


# ── 场景一：草稿 [r3, MARKER, a, b]（4 spec 位 + bonus = 5 行）────────────────
mgr = make_mgr(num_spec=4)
g = new_grammar(EBNF)
# 历史：prompt [1,2,3] + 已生成思考 [4,5]（reasoning_ended=None → 先对 prompt 判定）
req = FakeRequest("r1", grammar=g, prompt_token_ids=[1, 2, 3],
                  all_token_ids=[1, 2, 3, 4, 5])
drafts = [R3, MARKER, TOK_A, TOK_B]
mask = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": drafts})
out["scene1_flip"] = {
    "history_len": 5,
    "drafts": drafts,
    "mask_rows": int(mask.shape[0]),
    "rows": snap(mask),
    "accept_ledger": [ts for _, ts in g.accepts],
    "rollback_ledger": g.rollbacks,
    "reasoning_ended_after_call": req.structured_output_request.reasoning_ended,
    "note": "行 0（r3）与行 1（MARKER）整行 -1——标记行填行发生在翻转之前且标记是思考"
            "内容不推进；行 2 起 apply_bitmask 翻 True：受约束且试探推进（a、b 各一次）；"
            "bonus 行双触发：should_fill_bitmask 此刻仍 False——reasoning_ended 只在 "
            "bonus_apply 求值时被 prompt 级判定缓存成 False（窗口内的翻转不持久化，"
            "要等 ⑤ 拍 should_advance），靠 or apply_bitmask(已翻 True) 兜住受约束；"
            "末尾 rollback(2) 复位。reasoning_ended_after_call=False 即时序坑物证",
}

# ── 场景二：翻转后草稿非法——容忍拒绝（post_reasoning_end_in_window）──────────
mgr = make_mgr(num_spec=4)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g, prompt_token_ids=[1, 2, 3],
                  all_token_ids=[1, 2, 3, 4, 5])
drafts_bad = [R3, MARKER, 888, TOK_B]  # 888 非法（语法在位置 0 只收 a 系）
mask = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": drafts_bad})
out["scene2_tolerate_rejection"] = {
    "drafts": drafts_bad,
    "mask_rows": int(mask.shape[0]),
    "rows": snap(mask),
    "accept_ledger": [ts for _, ts in g.accepts],
    "accept_outcomes": [not any(t in (888,) for t in ts) or ts == [TOK_B] for ts in
                        [ts for _, ts in g.accepts]],
    "rollback_ledger": g.rollbacks,
    "no_assertion_raised": True,
    "note": "标记后的草稿 888 先于掩码存在、不保证合法：accept(888) 被真语法拒收——"
            "post_reasoning_end_in_window=True 时容忍（不抛 AssertionError）；后续 b "
            "也被拒（FSM 停在位置 0 等 a）——行内容如实反映『翻转时刻的语法状态』",
}

# ── 场景三（对照组）：无 reasoner → should_fill_bitmask 恒 True → 全窗口受约束 ─
mgr = make_mgr(num_spec=3, with_reasoner=False)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g, prompt_token_ids=[1, 2, 3],
                  all_token_ids=[1, 2, 3, 4, 5])
mask = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [TOK_A, TOK_B, TOK_C]})
out["scene3_no_reasoner"] = {
    "drafts": [TOK_A, TOK_B, TOK_C],
    "mask_rows": int(mask.shape[0]),
    "rows": snap(mask),
    "any_full_allow_row": any(row_full_allow(mask[i]) for i in range(mask.shape[0])),
    "accept_ledger": [ts for _, ts in g.accepts],
    "rollback_ledger": g.rollbacks,
    "note": "无 reasoner 时 should_fill_bitmask 恒 True 且 detect_reasoning_end=False："
            "整窗口（含 bonus）从行 0 起全部受语法约束（合法草稿 a/b/c 的正常路径，"
            "与 m07 场景一同款）——对照场景一：仅 reasoner 存在与否，行内容从"
            "『标记前放行』变为『全程约束』",
}

# ── 场景四（语义断言物证）：无 reasoner + 未过 validate 的思考型草稿 → 当场炸 ──
mgr = make_mgr(num_spec=4, with_reasoner=False)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g, prompt_token_ids=[1, 2, 3],
                  all_token_ids=[1, 2, 3, 4, 5])
err = None
try:
    mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": drafts})
except AssertionError as e:
    err = repr(e)
out["scene4_unvalidated_draft_asserts"] = {
    "drafts": drafts,
    "assertion_error": err,
    "note": "无 reasoner 的窗口要求草稿已过 validate_tokens（m17 的过滤+(-1) 补齐）；"
            "直接塞思考型草稿 r3=7 → 语法拒收 → AssertionError 当场炸——语义断言"
            "守护的是『调度与语法状态不失配』，只有 post_reasoning_end 场景容忍拒绝",
}

dump("trace_m15_midwindow.json", out)
print("s1 rows:", [(r["row"], r["full_allow"], r.get("allowed")) for r in out["scene1_flip"]["rows"]])
print("s1 accepts:", out["scene1_flip"]["accept_ledger"], "rollback:", out["scene1_flip"]["rollback_ledger"],
      "reasoning_ended:", out["scene1_flip"]["reasoning_ended_after_call"])
print("s2 accepts:", out["scene2_tolerate_rejection"]["accept_ledger"], "rows:",
      [(r["row"], r["full_allow"], r.get("allowed")) for r in out["scene2_tolerate_rejection"]["rows"]])
