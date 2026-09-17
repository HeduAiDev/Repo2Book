# ch32 m07/m08 驱动：串行填充·spec 窗口展开 + 整行 -1 语义。
# 真 StructuredOutputManager.grammar_bitmask（精简版）+ 真 xgrammar matcher
# （grammar 'root ::= "a" "b" "c"'，gpt2：a=64 b=65 c=66 EOS=50256）——
# 第 j 行掩码真实反映『接受 d_1..d_{j-1} 后第 j 位哪些 token 合法』。
import sys

import numpy as np

from _ch32_common import (EOS, TOK_A, TOK_B, TOK_C, VOCAB, FakeRequest,
                          RealBackend, allowed_ids, dump, get_tokenizer,
                          make_vllm_config, new_grammar, row_full_allow)
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
from vllm.v1.structured_output import StructuredOutputManager

EBNF = 'root ::= "a" "b" "c"'
tok = get_tokenizer()
out = {"env": {"grammar": EBNF, "token_ids": {"a": TOK_A, "b": TOK_B, "c": TOK_C, "eos": EOS},
               "vocab": VOCAB,
               "grammar_adapter": "XgrGrammar=pin backend_xgrammar.py 六方法适配（编译侧归 ch31 注入）"}}


def make_mgr(num_spec, is_diffusion=False):
    m = StructuredOutputManager(make_vllm_config(
        max_num_seqs=8, num_speculative_tokens=num_spec, is_diffusion=is_diffusion))
    m.backend = RealBackend(VOCAB)
    return m


def rows_snapshot(mask):
    rows = []
    for i in range(mask.shape[0]):
        if row_full_allow(mask[i]):
            rows.append({"row": i, "full_allow": True, "allowed_count": VOCAB})
        else:
            a = allowed_ids(mask[i], VOCAB)
            rows.append({"row": i, "full_allow": False, "allowed": a,
                         "allowed_tokens": [repr(tok.decode([t])) for t in a]})
    return rows


# ── 场景一：3 个合法草稿 [a,b,c] → 4 行（3 spec + bonus）+ 试探推进 3 次 + rollback(3)
mgr = make_mgr(num_spec=3)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g)
mask = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [TOK_A, TOK_B, TOK_C]})
out["scene1_valid_drafts"] = {
    "drafts": [TOK_A, TOK_B, TOK_C],
    "mask_rows": int(mask.shape[0]),
    "rows": rows_snapshot(mask),
    "accept_ledger": [ts for _, ts in g.accepts],
    "rollback_ledger": g.rollbacks,
    "num_processed_after": g.num_processed_tokens,
    "is_terminated": g.is_terminated(),
    "note": "行 0 允许 a（位置 0）；行 1 允许 b（试探接受 a 后的位置 1）；行 2 允许 c；"
            "bonus 行允许 EOS——每行都是『接受前序草稿后』的真实下一状态；"
            "末尾 rollback(3) 复位到本步开始处（真推进只留给 update_from_output）",
}
# rollback 后 FSM 已复位：再填一行应回到位置 0（允许 a）
g2check = torch.zeros((1, -(-VOCAB // 32)), dtype=torch.int32)
g.fill_bitmask(g2check, 0)
out["scene1_valid_drafts"]["post_rollback_refill_allowed"] = allowed_ids(g2check[0], VOCAB)

# ── 场景二：-1 哨兵 [a, -1, c]：哨兵行按旧标志填、当行即停推进、次行起放行──
mgr = make_mgr(num_spec=3)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g)
mask = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [TOK_A, -1, TOK_C]})
out["scene2_sentinel"] = {
    "drafts": [TOK_A, -1, TOK_C],
    "mask_rows": int(mask.shape[0]),
    "rows": rows_snapshot(mask),
    "accept_ledger": [ts for _, ts in g.accepts],
    "rollback_ledger": g.rollbacks,
    "note": "行 0（t=a）：受约束、试探接受 a；行 1（t=-1 哨兵）：填行发生在标志翻转"
            "之前→仍按旧标志受约束（允许 b——已接受 a 后的真实状态）、当行即停推进、"
            "apply_bitmask 翻 False；行 2（t=c）：整行 -1 放行、不推进；bonus 行："
            "should_fill_bitmask(无 reasoner)=True→再受约束（仍允许 b——a 之后没再推进）",
}

# ── 场景三：非法草稿触发 AssertionError（非 post_reasoning_end 场景不容忍拒绝）──
mgr = make_mgr(num_spec=1)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g)
err = None
try:
    mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [999]})
except AssertionError as e:
    err = repr(e)
out["scene3_invalid_draft"] = {
    "drafts": [999],
    "assertion_error": err,
    "note": "调度器排进的草稿必须已被 validate_tokens 过滤（m17）——这里绕过过滤"
            "直接塞非法草稿，串行分支的语义断言当场炸：调度与语法状态失配是 bug 不是运行态",
}

# ── 场景四（m08 甲）：diffusion 无 bonus 行（有 spec 行时跳过）────────────────
mgr = make_mgr(num_spec=2, is_diffusion=True)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g)
mask = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [TOK_A, TOK_B]})
out["scene4_diffusion"] = {
    "drafts": [TOK_A, TOK_B],
    "mask_rows": int(mask.shape[0]),
    "rows": rows_snapshot(mask),
    "note": "diffusion 画布步不采 AR bonus token → 无 bonus 行（L333-L335）",
}

# ── 场景五（m08 乙）：整行 -1 = 32 位全 1 = 全允许；跨步复用残留清理是正确性必需──
mgr = make_mgr(num_spec=0)
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g)
mask1 = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
raw_step1 = mgr._grammar_bitmask[0].numpy().tolist()
out["scene5_full_mask_semantics"] = {
    "step1_request_constrained": {
        "row0_raw_int32": raw_step1,
        "row0_allowed": allowed_ids(mgr._grammar_bitmask[0], VOCAB),
    },
}
# step2：注入 reasoner、思考未结束 → should_fill_bitmask=False → 该行必须显式
# fill(-1) 重置，否则 step1 残留位会误杀本拍合法 token
class FakeReasoner:
    def __init__(self, tokenizer=None, **kw):
        pass

    def is_reasoning_end(self, prompt_token_ids):
        return False

    def is_reasoning_end_streaming(self, all_token_ids, delta_ids):
        return False


mgr.reasoner_cls = FakeReasoner
mgr.tokenizer = None
req.structured_output_request.reasoning_ended = False
mask2 = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
raw_step2 = mgr._grammar_bitmask[0].numpy().tolist()
u32 = raw_step2[0] & 0xFFFFFFFF
out["scene5_full_mask_semantics"]["step2_thinking_not_ended"] = {
    "row0_raw_int32": raw_step2,
    "row0_full_allow": row_full_allow(mask2[0]),
    "bits_of_first_int32": format(int(u32), "032b"),
    "ones_in_first_int32": format(int(u32), "032b").count("1"),
    "zeros_in_first_int32": format(int(u32), "032b").count("0"),
    "note": "step1 的行 0 写过位表（仅 a 位=1）；step2 不受约束→整行 fill_(-1)："
            "int32 -1 = 补码全 1 = 32 位全允许——不是全禁；跨步复用缓冲下这行 fill "
            "不是清洁而是正确性必需（残留位会误杀本拍合法 token）",
}
# 反事实验证：若 step2 不重置（直接读残留），行 0 仍是 step1 的位表
out["scene5_full_mask_semantics"]["counterfactual_residue"] = {
    "what_if_not_reset": "残留行只允许 a——思考段的任何其他 token 都会被打成 -inf",
    "residue_row_allows": allowed_ids(torch.tensor(raw_step1, dtype=torch.int32), VOCAB),
}

# ── 场景六：预算分配（m04 顺带）：max_seqs*(1+num_spec) 行、只分配一次、裁剪──
mgr = make_mgr(num_spec=3)
assert mgr.backend.alloc_calls == []
g = new_grammar(EBNF)
req = FakeRequest("r1", grammar=g)
m1 = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [TOK_A, TOK_B, TOK_C]})
m2 = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [TOK_A, TOK_B, TOK_C]})
out["scene6_budget"] = {
    "max_num_seqs": 8,
    "num_spec": 3,
    "alloc_calls": mgr.backend.alloc_calls,
    "buffer_rows_allocated": int(mgr._grammar_bitmask.shape[0]),
    "int32_per_row": int(mgr._grammar_bitmask.shape[1]),
    "mask_rows_returned_step1": int(m1.shape[0]),
    "mask_rows_returned_step2": int(m2.shape[0]),
    "note": "8*(1+3)=32 行预算、跨步只分配一次；每步裁到 cumulative_index=4 行只传活跃前缀",
}

dump("trace_m07_m08_serial_window.json", out)
for k in ("scene1_valid_drafts", "scene2_sentinel", "scene4_diffusion", "scene6_budget"):
    v = out[k]
    print(k, "rows:", v["mask_rows"] if "mask_rows" in v else None,
          "| accepts:", v.get("accept_ledger"), "| rollback:", v.get("rollback_ledger"))
print("sentinel rows:", [(r["row"], r["full_allow"], r["allowed"]) for r in out["scene2_sentinel"]["rows"]])
