"""ch32 explainer driver t05 - m16 推理模型耦合:推理段不填掩码、不推进 FSM.

真实 StructuredOutputManager.should_fill_bitmask / should_advance / _get_reasoner
配一个透明的 FakeReasoner(token 99 = </think>,ch31 已讲后端契约,这里只需要
「推理是否已结束」这一个判据),逐步观察:
  step1-2 推理进行中 -> should_fill_bitmask=False -> 掩码整行 -1(全允许)
  step3   本步吐出 </think> -> should_advance 侦测到并置 reasoning_ended=True,
          但本步仍返回 False(「刚结束,下一步再推进」)
  step4   reasoning_ended=True -> 填真实语法掩码,FSM 开始推进
最后一组对照:enable_in_reasoning=True 时第一步就受约束。
输出 JSON 存 t05_reasoning_gate.json。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CH, "implementation"))
sys.path.insert(0, os.path.join(CH, "tests"))

from conftest import FakeBackend, FakeGrammar, make_request  # noqa: E402
from structured_output_manager import StructuredOutputManager  # noqa: E402

VOCAB = 96
COLS = -(-VOCAB // 32)
END = 99  # 扮演 </think> 的 token id


class FakeReasoner:
    """只实现 should_fill_bitmask / should_advance 用到的两个判据。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def is_reasoning_end(self, token_ids):
        return END in list(token_ids)

    def is_reasoning_end_streaming(self, all_token_ids, delta_token_ids):
        return END in list(delta_token_ids)


def row_summary(bm, r):
    if all(int(bm[r][c]) == -1 for c in range(COLS)):
        return {"row_is_full_mask_all_minus_1": True, "allowed_tokens": "ALL",
                "num_allowed": VOCAB}
    allowed = [t for t in range(VOCAB) if (int(bm[r][t // 32]) >> (t % 32)) & 1]
    return {"row_is_full_mask_all_minus_1": False, "allowed_tokens": allowed,
            "num_allowed": len(allowed)}


FULL_MASK_VALUE = None


def run(enable_in_reasoning):
    global FULL_MASK_VALUE
    mgr = StructuredOutputManager(max_num_seqs=4, max_num_spec_tokens=0)
    mgr.backend = FakeBackend(VOCAB)
    mgr.reasoner_cls = FakeReasoner
    mgr.enable_in_reasoning = enable_in_reasoning

    FULL_MASK_VALUE = int(mgr._full_mask.item())  # _full_mask = torch.tensor(-1)
    g = FakeGrammar([{5, 7}, {9}, {11, 13}])
    req = make_request("rA", g)
    req.prompt_token_ids = [1, 2, 3]
    req._all_token_ids = [1, 2, 3]
    req.all_token_ids = req._all_token_ids
    req.num_computed_tokens = 3

    # 每步:(本步之后 all_token_ids 追加的新 token, 步末 num_computed_tokens)
    emissions = [50, 51, END, 5]
    rows = []
    for step, tok in enumerate(emissions, start=1):
        ended_before = req.structured_output_request.reasoning_ended
        fill = mgr.should_fill_bitmask(req)
        bm = mgr.grammar_bitmask({"rA": req}, ["rA"], {})
        rowinfo = row_summary(bm, 0)
        # 本步产出 tok,写回请求状态(模拟 update_from_output 之前的状态推进)
        req._all_token_ids.append(tok)
        req.all_token_ids = req._all_token_ids
        req.num_computed_tokens = len(req._all_token_ids) - 1
        adv = mgr.should_advance(req)
        if adv:
            g.accept_tokens("rA", [tok]) if tok in g.allowed_at[
                min(g.position, len(g.allowed_at) - 1)] else None
        rows.append({
            "step": step,
            "emitted_token": tok,
            "reasoning_ended_before": ended_before,
            "should_fill_bitmask": bool(fill),
            "bitmask_row": rowinfo,
            "should_advance": bool(adv),
            "reasoning_ended_after": req.structured_output_request.reasoning_ended,
            "grammar_position": g.position,
        })
    return rows


_default = run(False)
_override = run(True)
out = {
    "params": {"vocab_size": VOCAB, "reasoning_end_token_id": END,
               "grammar_allowed_at_pos0": [5, 7],
               "full_mask_int32_value": FULL_MASK_VALUE},
    "default_enable_in_reasoning_false": _default,
    "override_enable_in_reasoning_true": _override,
}

path = os.path.join(HERE, "t05_reasoning_gate.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
