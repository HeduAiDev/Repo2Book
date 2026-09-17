# ch32 m16 驱动：异步调度下的延后采样。甲=真 AsyncScheduler._update_after_schedule
# （pending_structured_output_tokens 置位 + 占位记账 + -1 占位数组）两拍走表；
# 乙=真 EngineCore.step_with_batch_queue（精简版）+ spy executor/scheduler 三拍
# 事件账（fill→deferred→兑现链 take_draft→update_draft→bitmask→sample_tokens）。
import sys
import types

from _ch32_common import (FakeRequest, FakeSchedulerOutput, dump,
                          make_manager, make_vllm_config)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

out = {"env": {"note": "AsyncScheduler._update_after_schedule 为真实现代码"
                       "（精简版逐字）；EngineCore 驱动用 spy executor/scheduler 承载"
                       "消费面（真实装配归 ch09/ch12）"}}

# ── 甲段：pending 信号两拍 ─────────────────────────────────────────────────────
mgr = make_manager(64, max_num_seqs=8)
cfg = make_vllm_config(max_num_seqs=8, num_speculative_tokens=2)
sched = AsyncScheduler(cfg, None)
sched._inflight_prefills = set()
req = FakeRequest("r1", grammar=None, use_structured_output=True,
                  num_tokens=10, num_computed_tokens=10,
                  num_output_placeholders=0, spec_token_ids=[])
sched.requests = {"r1": req}
sched.structured_output_manager = mgr

# 第 1 拍：placeholders=0 → pending 不置位；随后 +1+2=3。
# 注意调度账：num_computed 10→11、tokens+placeholders=13 → 下一拍的真实调度量
# 是 deficit=2（num_tokens_with_spec + placeholders - computed，scheduler.py:L516-
# L520 的公式）——第 2 拍排 2 个 token 才不会把请求判成 prefill chunk。
so1 = FakeSchedulerOutput(num_scheduled_tokens={"r1": 1},
                          scheduled_spec_decode_tokens={"r1": [5, 6]},
                          num_spec_tokens_to_schedule=2)
sched._update_after_schedule(so1)
step1 = {
    "num_output_placeholders_before": 0,
    "num_scheduled_tokens": 1,
    "pending_structured_output_tokens": so1.pending_structured_output_tokens,
    "placeholders_after": req.num_output_placeholders,
    "spec_token_ids": req.spec_token_ids,
    "deficit_after": (req.num_tokens + req.num_output_placeholders
                      - req.num_computed_tokens),
}
# 第 2 拍：排 deficit=2（真实调度公式）；进门时 placeholders=3>0 → pending 置位
so2 = FakeSchedulerOutput(num_scheduled_tokens={"r1": 2},
                          scheduled_spec_decode_tokens={"r1": [7, 8]},
                          num_spec_tokens_to_schedule=2)
sched._update_after_schedule(so2)
step2 = {
    "num_output_placeholders_before": 3,
    "num_scheduled_tokens": 2,
    "pending_structured_output_tokens": so2.pending_structured_output_tokens,
    "placeholders_after": req.num_output_placeholders,
}
out["pending_signal"] = {
    "step1": step1,
    "step2": step2,
    "num_sampled_tokens_per_step": sched.num_sampled_tokens_per_step,
    "cur_num_spec_tokens_step1": 2,
    "note": "判定在加账之前：step1 进门时 placeholders=0 → 不置位；加账 +1（AR "
            "bonus）+2（spec）=3；step2 进门时 3>0 → pending=True——本拍掩码要吃"
            "『上拍真实 token』而它们还没回来，必须延后采样。step2 排 2 个 token "
            "是真实调度公式的 deficit 回填（num_tokens_with_spec + placeholders - "
            "num_computed_tokens）——排不够会把请求误判回 prefill chunk",
}

# ── 乙段：step_with_batch_queue 三拍事件账 ────────────────────────────────────
from vllm.v1.engine.core import EngineCore

EVENTS = []


class SpySched:
    def __init__(self):
        self.calls = 0
        self.pending_flags = [False, True, True]  # 拍 1 立即采样；拍 2 起延后

    def has_requests(self):
        return self.calls < 3  # 前 3 拍有请求可排（拍 3 排队收尾）

    def schedule(self, throttle=False):
        i = self.calls
        self.calls += 1
        EVENTS.append(f"schedule#{i + 1}")
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1}, total_num_scheduled_tokens=1,
            has_structured_output_requests=True,
            pending_structured_output_tokens=self.pending_flags[i])
        so.tag = f"batch#{i + 1}"
        return so

    def get_grammar_bitmask(self, scheduler_output):
        tag = "deferred" if scheduler_output.pending_structured_output_tokens else "immediate"
        EVENTS.append(f"bitmask({tag})")
        return object()

    def update_from_output(self, scheduler_output, model_output):
        EVENTS.append(f"update_from_output({getattr(scheduler_output, 'tag', '?')})")
        return {}

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        EVENTS.append(f"update_draft_in_output({draft_token_ids.draft_token_ids})")


class SpyExec:
    def execute_model(self, scheduler_output, non_block=False):
        from concurrent.futures import Future
        EVENTS.append("dispatch(non_block)")
        fut = Future()
        fut.set_result(None)
        return fut

    def sample_tokens(self, grammar_output, non_block=False):
        from concurrent.futures import Future
        EVENTS.append(f"sample_tokens(non_block={non_block})")
        fut = Future()
        fut.set_result(("mro", None))
        return fut

    def take_draft_token_ids(self):
        EVENTS.append("take_draft_token_ids")
        from vllm.v1.outputs import DraftTokenIds
        return DraftTokenIds(req_ids=["r1"], draft_token_ids=[[64, 65, 999]])


class Cfg:
    speculative_config = None
    max_concurrent_batches = 2
    model_config = types.SimpleNamespace(is_diffusion=False)
    scheduler_config = types.SimpleNamespace(async_scheduling=True)


core = EngineCore.__new__(EngineCore)
core.scheduler = SpySched()
core.model_executor = SpyExec()
core.batch_queue_size = 2
from collections import deque
core.batch_queue = deque(maxlen=2)
core.use_spec_decode = True
core.check_for_draft_tokens = True
core.async_scheduling = True

call1 = core.step_with_batch_queue()
events_after_call1 = list(EVENTS)
call2 = core.step_with_batch_queue()
events_after_call2 = list(EVENTS)
call3 = core.step_with_batch_queue()
events_after_call3 = list(EVENTS)

out["three_calls"] = {
    "batch_queue_size": 2,
    "call1": {"returns": [call1[0] is None, call1[1]],
              "events": events_after_call1,
              "note": "拍 1：pending=False → 立即 bitmask+sample_tokens(non_block=True) "
                      "→ appendleft 入队 → 队未满（1<2）且有请求 → 早退（不收输出）"},
    "call2": {"returns": [call2[0], call2[1]],
              "new_events": events_after_call2[len(events_after_call1):],
              "note": "拍 2：pending=True → 本拍挂起 deferred → pop 拍 1 收输出 "
                      "update_from_output → 兑现链：take_draft → update_draft_in_"
                      "output(过滤+补齐) → bitmask(deferred) → sample_tokens → 重新入队"},
    "call3": {"returns": [call3[0], call3[1]],
              "new_events": events_after_call3[len(events_after_call2):],
              "note": "拍 3：无新请求（队里有拍 2 的 deferred 批）→ pop 拍 2 批收输出"},
}

dump("trace_m16_deferred.json", out)
print("call1:", out["three_calls"]["call1"]["events"])
print("call2 new:", out["three_calls"]["call2"]["new_events"])
print("call3 new:", out["three_calls"]["call3"]["new_events"])
