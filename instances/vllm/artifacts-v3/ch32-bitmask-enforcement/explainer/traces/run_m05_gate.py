# ch32 m05 驱动：调度侧门控 has_structured_output_requests 置位（use_structured_
# output 且非 prefill chunk）+ get_grammar_bitmask 行序账本。真 Scheduler
# _update_after_schedule / get_grammar_bitmask（精简版）+ 真 manager.grammar_bitmask。
# 两步走：step1 prefill 中段请求 P 被排除；step2 P 完成 prefill 进账本。
import sys
import types

import numpy as np

from _ch32_common import (FakeRequest, FakeSchedulerOutput, dump, make_manager)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))

V = 64


class FakeGrammar:
    """六方法契约替身（每行允许一个可区分 token，行内容按请求 id 定）。"""

    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.fill_calls = []

    def fill_bitmask(self, bitmask, index):
        import torch
        row = np.zeros(bitmask.shape[1], dtype=np.uint32)
        for tok in self.allowed:
            row[tok // 32] |= np.uint32(1 << (tok % 32))
        bitmask[index] = torch.from_numpy(row.view(np.int32))
        self.fill_calls.append(index)

    def is_terminated(self):
        return False

    def accept_tokens(self, request_id, tokens):
        return True

    def validate_tokens(self, tokens):
        return list(tokens)

    def rollback(self, num_tokens):
        pass

    def reset(self):
        pass


from vllm.v1.core.sched.scheduler import Scheduler

mgr = make_manager(V, max_num_seqs=8)
sched = Scheduler.__new__(Scheduler)
sched.requests = {}
sched.structured_output_manager = mgr
sched._inflight_prefills = set()

# P：结构化输出、prompt 100 token 的 chunked prefill 中段（已算 32）；
# A：结构化输出、decode 中（1 token/步）；N：无结构化输出、decode 中。
gP, gA = FakeGrammar({17}), FakeGrammar({5})
P = FakeRequest("P", grammar=gP, num_tokens=100, num_computed_tokens=0)
A = FakeRequest("A", grammar=gA, num_tokens=8, num_computed_tokens=8)
N = FakeRequest("N", use_structured_output=False, grammar=None,
                num_tokens=8, num_computed_tokens=8)
sched.requests.update({"P": P, "A": A, "N": N})

out = {"env": {"vocab_toy": V, "P_prompt_tokens": 100},
       "requests": {
           "P": {"use_structured_output": True, "num_tokens": 100,
                 "note": "chunked prefill：每步排 32 token，is_prefill_chunk=真 直到算完"},
           "A": {"use_structured_output": True, "num_tokens": 8, "note": "decode"},
           "N": {"use_structured_output": False, "num_tokens": 8, "note": "decode 无语法"}}}

# ── step 1：P 排 32（prefill 中段 32<100 → is_prefill_chunk=True 被排除）──────
so1 = FakeSchedulerOutput(num_scheduled_tokens={"P": 32, "A": 1, "N": 1})
sched._update_after_schedule(so1)
go1 = sched.get_grammar_bitmask(so1)
out["step1"] = {
    "num_scheduled_tokens": {"P": 32, "A": 1, "N": 1},
    "P_is_prefill_chunk": P.is_prefill_chunk,
    "P_num_computed_tokens": P.num_computed_tokens,
    "A_is_prefill_chunk": A.is_prefill_chunk,
    "has_structured_output_requests": so1.has_structured_output_requests,
    "grammar_output_is_None": go1 is None,
    "grammar_output_ids": go1.structured_output_request_ids if go1 else None,
    "mask_rows": 0 if go1 is None else int(go1.grammar_bitmask.shape[0]),
    "P_fill_called": len(gP.fill_calls),
    "A_fill_rows": list(gA.fill_calls),
}

# ── step 2：P 再排 32（64<100 仍中段）→ 仍被排除（第二拍核验）────────────────
so2 = FakeSchedulerOutput(num_scheduled_tokens={"P": 32, "A": 1, "N": 1})
sched._update_after_schedule(so2)
go2 = sched.get_grammar_bitmask(so2)
out["step2"] = {
    "num_scheduled_tokens": {"P": 32, "A": 1, "N": 1},
    "P_is_prefill_chunk": P.is_prefill_chunk,
    "P_num_computed_tokens": P.num_computed_tokens,
    "grammar_output_ids": go2.structured_output_request_ids,
    "mask_rows": int(go2.grammar_bitmask.shape[0]),
    "P_fill_called_total": len(gP.fill_calls),
}

# ── step 3：P 排最后 36（100 算完 → is_prefill_chunk=False → 进账本）──────────
so3 = FakeSchedulerOutput(num_scheduled_tokens={"P": 36, "A": 1, "N": 1})
sched._update_after_schedule(so3)
go3 = sched.get_grammar_bitmask(so3)
out["step3"] = {
    "num_scheduled_tokens": {"P": 36, "A": 1, "N": 1},
    "P_is_prefill_chunk": P.is_prefill_chunk,
    "P_num_computed_tokens": P.num_computed_tokens,
    "grammar_output_ids": go3.structured_output_request_ids,
    "mask_rows": int(go3.grammar_bitmask.shape[0]),
    "P_fill_rows": list(gP.fill_calls),
    "note": "行序=num_scheduled_tokens 迭代序（P 先排 → P 行 0、A 行 1）；"
            "N 永不进 ids（use_structured_output=False）",
}

# ── 对照：全部无结构化输出 → has_structured_output_requests 恒 False 快返 ─────
sched2 = Scheduler.__new__(Scheduler)
sched2.requests = {"N1": FakeRequest("N1", use_structured_output=False),
                   "N2": FakeRequest("N2", use_structured_output=False)}
sched2.structured_output_manager = mgr
sched2._inflight_prefills = set()
so4 = FakeSchedulerOutput(num_scheduled_tokens={"N1": 1, "N2": 1})
sched2._update_after_schedule(so4)
out["contrast_all_plain"] = {
    "has_structured_output_requests": so4.has_structured_output_requests,
    "get_grammar_bitmask_returns": "None（L1649-L1650 快返，manager 不被触）",
}

dump("trace_m05_gate.json", out)
print("step1 ids:", out["step1"]["grammar_output_ids"], "rows:", out["step1"]["mask_rows"])
print("step3 ids:", out["step3"]["grammar_output_ids"], "rows:", out["step3"]["mask_rows"])
print("P fill rows:", out["step3"]["P_fill_rows"])
