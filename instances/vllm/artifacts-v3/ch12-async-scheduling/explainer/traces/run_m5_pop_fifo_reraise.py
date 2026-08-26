# ch12 m5/m21 驱动：下半段 pop FIFO 顺序 + None ⇒ exec_future 重抛真异常。
# ScriptedExecutor 替身（控制 future 结果与失败注入；worker 链的内景归 m4 e2e）。
# 真源锚点：core.py:L689-L739（下半段）、L207-L212（三元组 deque）、L681/L696。
import json
import os
import sys
from concurrent.futures import Future

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.core import EngineCore  # noqa: E402
from implementation.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402


class ScriptedExecutor:
    def __init__(self):
        self.sampled_rows = []
        self.execute_calls = 0
        self.sample_calls = 0
        self.fail_execute_with = None

    def execute_model(self, scheduler_output, non_block=False):
        self.execute_calls += 1
        fut = Future()
        if self.fail_execute_with is not None:
            fut.set_exception(self.fail_execute_with)
        else:
            fut.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
        return fut

    def sample_tokens(self, grammar_output, non_block=False):
        self.sample_calls += 1
        fut = Future()
        if self.sampled_rows:
            row = self.sampled_rows.pop(0)
            fut.set_result(ModelRunnerOutput(
                req_ids=["req-0"], req_id_to_index={"req-0": 0},
                sampled_token_ids=[list(row)],
            ))
        else:
            fut.set_result(None)  # sample_tokens 拿不到输出（execute 已失败的信号）
        return fut

    def take_draft_token_ids(self):
        return None


def make_engine(executor):
    cfg = VllmConfig(scheduler_config=SchedulerConfig(), max_model_len=64)
    cfg.check_and_set_default_async_scheduling()
    engine = EngineCore(cfg, model_executor=executor)
    req = Request(request_id="req-0", prompt_token_ids=[1, 2],
                  sampling_params=SamplingParams(max_tokens=8))
    engine.scheduler.add_request(req)
    return engine, req


trace = {
    "mechanism": "m5 pop FIFO + None⇒重抛 / m21 deque 三元组",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/engine/core.py:L689-L739, L207-L212, L681, L696",
    "part1_fifo": {
        "setup": "max_tokens=8、脚本采样行 [7]/[8]/[9] 逐拍交货",
        "steps": [],
    },
    "part2_failure": {},
}

# ------------------------------------------------ 段一：FIFO 保序（正常路）
executor = ScriptedExecutor()
executor.sampled_rows = [[7], [8], [9]]
engine, req = make_engine(executor)

# 拍 1：批 A 入队（上半段填管道优先 → return None）
out1, ex1 = engine.step_with_batch_queue()
engine.post_step(model_executed=ex1)
trace["part1_fifo"]["steps"].append({
    "call": 1, "queue_len": len(engine.batch_queue),
    "returned": "None" if out1 is None else "outputs",
    "returned_executed": ex1,
    "note": "上半段：盲调度批A→execute→sample→appendleft 三元组→队未满且有活→return (None, True) 不等结果",
    "future_type_of_batchA": type(engine.batch_queue[-1][0]).__name__,
    "triple_arity": 3,
    "triple_fields": "(future, scheduler_output, exec_future)",
})
# 拍 2：先照常调度批 B → 队满 [B,A] → pop 最老批 A（FIFO）
out2, ex2 = engine.step_with_batch_queue()
engine.post_step(model_executed=ex2)
eco2 = out2[0].outputs[0]
trace["part1_fifo"]["steps"].append({
    "call": 2, "queue_len": len(engine.batch_queue),
    "returned": "outputs", "returned_executed": ex2,
    "note": "先照常调度批B入队（队满[B,A]）→同一次调用下半段 pop 最老批A收结果",
    "popped_tokens": list(eco2.new_token_ids),
    "req_output_token_ids": list(req.output_token_ids),
    "fifo_note": "appendleft 进（新批在 index0）/pop 出（最老批在末尾）——先调度的批先取结果",
})
# 拍 3：pop 批 B（t8 到账）；批 C 在飞
out3, ex3 = engine.step_with_batch_queue()
engine.post_step(model_executed=ex3)
eco3 = out3[0].outputs[0]
trace["part1_fifo"]["steps"].append({
    "call": 3, "queue_len": len(engine.batch_queue),
    "returned": "outputs", "returned_executed": ex3,
    "note": "pop 批B：t8 到账；批C 在飞（下一拍 pop 到 [9]）",
    "popped_tokens": list(eco3.new_token_ids),
    "req_output_token_ids": list(req.output_token_ids),
})
trace["part1_fifo"]["terminal"] = {
    "queue_len": len(engine.batch_queue),
    "token_arrival_order": list(req.output_token_ids),
    "order_note": "t7 先于 t8 到账——FIFO 保序（每批 token 到账顺序与调度顺序一致）",
}

# ------------------------------------------------ 段二：None ⇒ 重抛真异常
executor2 = ScriptedExecutor()
executor2.sampled_rows = []  # sample_tokens → None
executor2.fail_execute_with = RuntimeError("real worker failure")
engine2, req2 = make_engine(executor2)
trace["part2_failure"]["setup"] = {
    "sample_returns": "None",
    "execute_fails_with": "RuntimeError('real worker failure')",
    "contract": "core.py:L701-L706：None from sample_tokens() implies that the original execute_model() call failed - raise that exception",
}
out1, ex1 = engine2.step_with_batch_queue()  # 批 A 入队（sample future=None 已定）
engine2.post_step(model_executed=ex1)
trace["part2_failure"]["step1"] = {
    "queue_len": len(engine2.batch_queue),
    "returned": "None" if out1 is None else "outputs",
    "returned_executed": ex1,
    "note": "批A 入队：sample future 的结果是 None（execute 已失败的信号先躺在 exec_future 里）",
}
try:
    engine2.step_with_batch_queue()  # 调度批 B → 队满 → pop 批 A → 重抛
    trace["part2_failure"]["step2"] = "未抛异常（异常）"
except RuntimeError as e:
    trace["part2_failure"]["step2"] = {
        "raised": "RuntimeError", "message": str(e),
        "note": "future.result() 得 None → exec_model_fut.result() 重抛 execute 的真异常（不是吞成 unexpected error）",
    }

out = os.path.join(os.path.dirname(__file__), "m5_pop_fifo_reraise.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
