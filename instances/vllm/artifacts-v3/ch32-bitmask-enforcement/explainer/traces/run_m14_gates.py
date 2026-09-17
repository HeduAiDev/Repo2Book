# ch32 m14 驱动：思考门控三件套——should_fill_bitmask（填不填/prompt 级缓存）、
# should_advance（推不推进：v0.27 new_token_ids 精确窗口 vs 旧占位数推导 #43388）、
# trim_reasoning_for_advance（一步内思考+语法混块 #44006）——全部走真
# StructuredOutputManager/真 Scheduler.update_from_output（精简版）。
import sys

import numpy as np
import torch

from _ch32_common import (FakeRequest, dump, make_manager)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

V = 64
MARKER = 50  # 思考结束标记 token id（替身词表内）
P1, P2, P3 = 11, 12, 13          # prompt
T1, T2 = 21, 22                  # 思考内容
G1, G2 = 31, 32                  # 语法内容
out = {"env": {"vocab_toy": V, "marker_token": MARKER,
               "ids": {"p1": P1, "p2": P2, "p3": P3, "t1": T1, "t2": T2,
                        "g1": G1, "g2": G2}}}


class FakeReasoner:
    def __init__(self, tokenizer=None, **kwargs):
        self.prompt_end_calls = []
        self.streaming_calls = []

    def is_reasoning_end(self, prompt_token_ids):
        self.prompt_end_calls.append(list(prompt_token_ids))
        return False  # prompt 全程在思考内

    def is_reasoning_end_streaming(self, all_token_ids, delta_ids):
        delta = list(delta_ids)
        self.streaming_calls.append((len(list(all_token_ids)), delta))
        return MARKER in delta  # 流式判定：delta 窗口里出现结束标记


from vllm.v1.structured_output.backend_types import StructuredOutputGrammar


class FG(StructuredOutputGrammar):
    """六方法替身：accept 可编程拒收（reject 集合）；继承真实 ABC——
    update_from_output L1823 的 isinstance 语义断言要求替身履约。"""

    def __init__(self, reject=()):
        self.reject = set(reject)
        self.accepts = []

    def fill_bitmask(self, bitmask, index):
        pass

    def is_terminated(self):
        return False

    def accept_tokens(self, request_id, tokens):
        self.accepts.append((request_id, list(tokens)))
        return not any(t in self.reject for t in tokens)

    def validate_tokens(self, tokens):
        return list(tokens)

    def rollback(self, num_tokens):
        pass

    def reset(self):
        pass


mgr = make_manager(V, max_num_seqs=8)
mgr.reasoner_cls = FakeReasoner
mgr.tokenizer = None

# ── 甲：should_fill_bitmask——首次对 prompt 判一次并缓存进请求 ─────────────────
g = FG()
req = FakeRequest("r1", grammar=g, prompt_token_ids=[P1, P2, P3])
f1 = mgr.should_fill_bitmask(req)
f2 = mgr.should_fill_bitmask(req)
reasoner = req.structured_output_request.reasoner
out["should_fill_bitmask"] = {
    "prompt_end_calls_total": len(reasoner.prompt_end_calls),
    "first_call_returns": f1,
    "second_call_returns": f2,
    "reasoning_ended_cached": req.structured_output_request.reasoning_ended,
    "note": "首次调用对 prompt 判一次（is_reasoning_end→False：整个 prompt 都在思考"
            "内）并缓存进请求；第二次调用不再触发 prompt 级判定（1 次调用记录不变）",
}

# ── 乙：should_advance——#43388 两窗口对照 ─────────────────────────────────────
# 场景：all = [p1,p2,p3, t1,t2, MARKER, g1]（本步 new=[t1,t2,MARKER,g1]，标记在 idx 5）
ALL = [P1, P2, P3, T1, T2, MARKER, G1]
NEW = [T1, T2, MARKER, G1]

# 路径一（v0.27 新）：new_token_ids 精确窗口
req2 = FakeRequest("r2", grammar=FG(), prompt_token_ids=[P1, P2, P3],
                   all_token_ids=ALL, num_computed_tokens=0,
                   num_output_placeholders=0)
adv_new = mgr.should_advance(req2, new_token_ids=NEW)
out["should_advance_new_window"] = {
    "all_token_ids_len": len(ALL),
    "new_token_ids_len": len(NEW),
    "window_start_computed": len(ALL) - len(NEW),
    "delta_window": NEW,
    "returns": adv_new,
    "reasoning_ended_persisted": req2.structured_output_request.reasoning_ended,
    "reasoning_end_token_index": req2.structured_output_request.reasoning_end_token_index,
    "note": "delta 窗口从 len(all)-len(new)=3 起恰覆盖 [t1,t2,MARKER,g1]——标记在窗口内"
            "→ 检出结束、持久化 reasoning_ended=True、边界定位 idx=5（标记本身）",
}

# 路径二（旧占位数推导，#43388 的坑）：async+spec 草稿被拒后占位数残留 >0
req3 = FakeRequest("r3", grammar=FG(), prompt_token_ids=[P1, P2, P3],
                   all_token_ids=ALL, num_computed_tokens=8,
                   num_output_placeholders=2)
adv_fallback = mgr.should_advance(req3)
out["should_advance_fallback_window"] = {
    "num_computed_tokens": 8,
    "num_output_placeholders": 2,
    "delta_from": 8 - 2,
    "window_start_computed": 6,
    "delta_window": ALL[6:],
    "returns": adv_fallback,
    "reasoning_ended_stays": req3.structured_output_request.reasoning_ended,
    "note": "占位数推导 start = 8-2 = 6 > 标记位置 5 → delta=[g1] 窗口错过标记 → "
            "返回 False、reasoning_ended 不置位——语法永远不生效（#43388 的病灶）；"
            "同一请求走 new_token_ids 路径则正确检出（见上）",
}

# ── 丙：trim_reasoning_for_advance（#44006）+ update_from_output 真推进 ────────
sched = Scheduler.__new__(Scheduler)
sched.requests = {}
sched.structured_output_manager = mgr
sched.num_sampled_tokens_per_step = 1
sched.log_stats = False
sched.num_spec_tokens = 0
sched._inflight_prefills = set()

# 命中修复：reasoning_end_token_index=5 → trim 掉 [t1,t2,MARKER] 只喂 [g1]
gA = FG(reject={T1, T2, MARKER})   # 思考 token 全部不合语法（真实场景的抽象）
reqA = FakeRequest("A", grammar=gA, prompt_token_ids=[P1, P2, P3], all_token_ids=ALL)
reqA.structured_output_request.reasoning_ended = True
reqA.structured_output_request.reasoning_end_token_index = 5
sched.requests["A"] = reqA
so = type("SO", (), {})()
so.num_scheduled_tokens = {"A": 4}
so.scheduled_spec_decode_tokens = {}
mro = ModelRunnerOutput(req_ids=["A"], req_id_to_index={"A": 0},
                        sampled_token_ids=[NEW])
sched.update_from_output(so, mro)
out["trim_and_advance"] = {
    "new_token_ids_fed": NEW,
    "first_idx": len(ALL) - len(NEW),
    "num_reasoning_trimmed": 5 + 1 - 3,
    "accept_ledger_with_trim": [ts for _, ts in gA.accepts],
    "status_after": str(reqA.status),
    "note": "end_idx=5、first_idx=3 → num_reasoning=6-3=3：裁掉 [t1,t2,MARKER] 只喂 "
            "accept([g1])——成功推进、请求存活",
}

# 对照（修复前的行为）：边界没记录（None）→ 整块直喂 → 语法拒收 → FINISHED_ERROR
gB = FG(reject={T1, T2, MARKER})
reqB = FakeRequest("B", grammar=gB, prompt_token_ids=[P1, P2, P3], all_token_ids=ALL)
reqB.structured_output_request.reasoning_ended = True
reqB.structured_output_request.reasoning_end_token_index = None  # 无边界 → trim 直通
sched.requests = {"B": reqB}
so2 = type("SO", (), {})()
so2.num_scheduled_tokens = {"B": 4}
so2.scheduled_spec_decode_tokens = {}
mro2 = ModelRunnerOutput(req_ids=["B"], req_id_to_index={"B": 0},
                         sampled_token_ids=[NEW])
sched.update_from_output(so2, mro2)
out["no_trim_kills_request"] = {
    "accept_ledger_no_trim": [ts for _, ts in gB.accepts],
    "accept_returns": False,
    "status_after": str(reqB.status),
    "status_is_finished_error": reqB.status == RequestStatus.FINISHED_ERROR,
    "resumable": reqB.resumable,
    "note": "同一混块不裁思考 token 直接喂 accept_tokens([t1,t2,MARKER,g1]) → 语法"
            "拒收（思考内容不合语法）→ update_from_output 判 FINISHED_ERROR 杀请求"
            "（#44006 修复前的真实死法）",
}

dump("trace_m14_gates.json", out)
print("fill:", out["should_fill_bitmask"])
print("new window:", {k: v for k, v in out["should_advance_new_window"].items() if k != "note"})
print("fallback:", {k: v for k, v in out["should_advance_fallback_window"].items() if k != "note"})
print("trim:", out["trim_and_advance"]["accept_ledger_with_trim"], "status:", out["trim_and_advance"]["status_after"])
print("no-trim:", out["no_trim_kills_request"]["accept_ledger_no_trim"], "status:", out["no_trim_kills_request"]["status_after"])
