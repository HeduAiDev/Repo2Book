# ch32 m17 驱动：spec 草稿回传链。甲=真 Scheduler.update_draft_token_ids_in_output
# （草稿截断→should_advance 门控→validate_tokens 过滤→-1 补齐→num_invalid 记账），
# 真 xgrammar validate（'a b c' 文法：前缀 a,b 合法、999 非法）；乙=真
# DraftTokensHandler（CUDA copy_stream D2H 往返 + has_structured_output_reqs 门控）。
import sys

import numpy as np
import torch

from _ch32_common import (TOK_A, TOK_B, VOCAB, FakeRequest,
                          FakeSchedulerOutput, dump, make_manager, new_grammar)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.structured_output import StructuredOutputManager
from _ch32_common import RealBackend, make_vllm_config

EBNF = 'root ::= "a" "b" "c"'
out = {"env": {"grammar": EBNF, "vocab": VOCAB,
               "cuda_device": torch.cuda.get_device_name(0)}}


class NoReasonerMgr(StructuredOutputManager):
    pass


def fresh_sched():
    m = StructuredOutputManager(make_vllm_config(max_num_seqs=8))
    m.backend = RealBackend(VOCAB)
    s = Scheduler.__new__(Scheduler)
    s.requests = {}
    s.structured_output_manager = m
    s.log_stats = False
    s.num_sampled_tokens_per_step = 1
    s.num_spec_tokens = 3
    s._inflight_prefills = set()
    return s, m


# ── 甲段·round 1：语法已生效（should_advance=True）→ validate 过滤 + -1 补齐 ──
sched, mgr = fresh_sched()
g1 = new_grammar(EBNF)
r1 = FakeRequest("r1", grammar=g1, prompt_token_ids=[1], all_token_ids=[1, TOK_A],
                 num_computed_tokens=2, num_output_placeholders=0)
sched.requests["r1"] = r1
so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 4},
                         scheduled_spec_decode_tokens={"r1": [TOK_A, TOK_B, 999, 77777]})
# worker 回传的草稿（可能长于已排数——截断到 4 内演示：回传 5 条）
returned = [TOK_A, TOK_B, 999, 77777, TOK_A]
drafts = DraftTokenIds(req_ids=["r1"], draft_token_ids=[list(returned)])
sched.update_draft_token_ids_in_output(drafts, so)
out["round1_filtered"] = {
    "returned_drafts": returned,
    "scheduled_placeholder_len": 4,
    "should_advance": mgr.should_advance(r1),
    "validate_called": [ts for ts in g1.validates],
    "sched_spec_after": so.scheduled_spec_decode_tokens["r1"],
    "num_invalid_spec_tokens": so.num_invalid_spec_tokens,
    "note": "截断到已排 4 条 → 真 validate_tokens 过滤：a,b 合法保留、999 起断链"
            "（含其后的 77777）→ [a,b] → -1 补齐回 4 条；num_invalid 记账 {r1: 2}"
            "进 scheduler_output（update_from_output 里抵扣接受率统计的分母）",
}

# ── 甲段·round 2：思考未结束（should_advance=False）→ 不过滤、原样保留 ────────
sched2, mgr2 = fresh_sched()


class NeverEndReasoner:
    def __init__(self, tokenizer=None, **kw):
        pass

    def is_reasoning_end(self, prompt_token_ids):
        return False

    def is_reasoning_end_streaming(self, all_token_ids, delta_ids):
        return False


mgr2.reasoner_cls = NeverEndReasoner
mgr2.tokenizer = None
g2 = new_grammar(EBNF)
r2 = FakeRequest("r2", grammar=g2, prompt_token_ids=[1, 2], all_token_ids=[1, 2, 3])
sched2.requests["r2"] = r2
so2 = FakeSchedulerOutput(num_scheduled_tokens={"r2": 3},
                          scheduled_spec_decode_tokens={"r2": [0, 0, 0]})
drafts2 = DraftTokenIds(req_ids=["r2"], draft_token_ids=[[999, 888, 777]])
sched2.update_draft_token_ids_in_output(drafts2, so2)
out["round2_thinking_not_filtered"] = {
    "drafts": [999, 888, 777],
    "should_advance": mgr2.should_advance(r2),
    "validate_called": [ts for ts in g2.validates],
    "sched_spec_after": so2.scheduled_spec_decode_tokens["r2"],
    "num_invalid_spec_tokens": so2.num_invalid_spec_tokens,
    "note": "思考段内语法不该生效：should_advance=False → validate 不调用、草稿原样"
            "保留（思考内容本来就不受语法约束，过滤反而错）",
}

# ── 乙段：真 DraftTokensHandler（CUDA）——门控 + copy_stream D2H 往返 ───────────
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


class FakeInputBatch:
    def __init__(self, req_ids, has_structured):
        self.req_ids = list(req_ids)
        self.has_structured_output_reqs = has_structured


dev = torch.device("cuda")
handler = DraftTokensHandler(dev)

# 门控关：无结构化请求 → 整批跳过回传（get 返回 -1 占位）
ib_off = FakeInputBatch(["q1", "q2"], has_structured=False)
draft_gpu = torch.tensor([[10, 11, 12], [20, 21, 22]], device=dev, dtype=torch.int64)
handler.set_draft_tokens(ib_off, draft_gpu)
got_off = handler.get_draft_tokens()
out["handler_gate_off"] = {
    "has_structured_output_reqs": False,
    "draft_tokens_np_is_none": handler.draft_tokens_np is None,
    "returned": got_off.draft_token_ids,
    "req_ids_passthrough": got_off.req_ids,
    "note": "门控关：D2H 回传整体跳过，get_draft_tokens 返回 [-1]*3 占位（该占位"
            "同时服务 async 关闭路径——填表侧把 -1 当哨兵处理）",
}

# 门控开：有结构化请求 → copy_stream 上 async D2H → event 同步 → 值往返一致
ib_on = FakeInputBatch(["q1", "q2"], has_structured=True)
draft_gpu2 = torch.tensor([[64, 65, 999], [1, 2, 3]], device=dev, dtype=torch.int64)
handler.set_draft_tokens(ib_on, draft_gpu2)
got_on = handler.get_draft_tokens()
out["handler_gate_on"] = {
    "has_structured_output_reqs": True,
    "draft_tokens_np_is_none": handler.draft_tokens_np is None,
    "gpu_drafts": draft_gpu2.cpu().tolist(),
    "returned": got_on.draft_token_ids,
    "roundtrip_equal": got_on.draft_token_ids == draft_gpu2.cpu().tolist(),
    "returned_is_python_list": isinstance(got_on.draft_token_ids[0], list),
    "copy_event_is_blocking": True,
    "note": "门控开：独立 copy_stream 上 async_copy_to_np 把 GPU 草稿 D2H；"
            "copy_event(blocking=True) 防忙轮询 CUDA 驱动锁；record_stream 防缓存"
            "分配器提前复用 draft_tokens 的显存——get 时 synchronize 后 tolist 交纯"
            " Python 列表，EngineCore deferred 链第一步拿的正是它",
}

dump("trace_m17_draft_return.json", out)
print("r1 after:", out["round1_filtered"]["sched_spec_after"], out["round1_filtered"]["num_invalid_spec_tokens"])
print("r2 after:", out["round2_thinking_not_filtered"]["sched_spec_after"], out["round2_thinking_not_filtered"]["num_invalid_spec_tokens"])
print("gate off:", out["handler_gate_off"]["returned"])
print("gate on roundtrip:", out["handler_gate_on"]["roundtrip_equal"])
