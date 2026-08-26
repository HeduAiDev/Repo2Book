# ch12 m14 驱动：deferred sampling——structured+占位在途时本拍不采样，
# pop 上拍结果、update_from_output 之后才补 bitmask+sample_tokens，
# 批重新 appendleft 入队（复用同一 exec_future）。
# 真实链（EngineCore → UniProcExecutor → worker）；HOST SEAM 同 m4。
# 真源锚点：async_scheduler.py:L31-L33（置位）、output.py:L235-L241（标志对）、
# core.py:L665-L677（分流）、L719-L737（补采）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.core import EngineCore  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402


def one_hot_row(token_id, vocab=16):
    row = [0.0] * vocab
    row[token_id] = 1.0
    return row


cfg = VllmConfig(scheduler_config=SchedulerConfig(), max_model_len=64)
cfg.check_and_set_default_async_scheduling()
engine = EngineCore(cfg)
req = Request(request_id="req-0", prompt_token_ids=[1, 2],
              sampling_params=SamplingParams(max_tokens=8),
              structured_output_request=object())  # use_structured_output=True
engine.scheduler.add_request(req)
runner = engine.model_executor.driver_worker.model_runner
sched = engine.scheduler

runner.enqueue_logits([
    {"req-0": one_hot_row(7)}, {"req-0": one_hot_row(8)},
    {"req-0": one_hot_row(9)}, {"req-0": one_hot_row(10)},
])

trace = {
    "mechanism": "m14 deferred sampling（structured + async）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/core/sched/async_scheduler.py:L31-L33; vllm/v1/core/sched/output.py:L235-L241; vllm/v1/engine/core.py:L665-L677, L719-L737",
    "params": "prompt=2、max_tokens=8、use_structured_output=True（grammar_bitmask 由 ch30 seam 恒全 1，数值不变）",
    "beats": [],
}

order = []
_upd = sched.update_from_output
sched.update_from_output = lambda so, mo: (order.append("update"), _upd(so, mo))[1]
_spl = engine.model_executor.sample_tokens
engine.model_executor.sample_tokens = lambda g, non_block=False: (
    order.append("sample"), _spl(g, non_block))[1]


def beat(tag):
    runner.release_async_copies()
    outputs, executed = engine.step_fn()
    engine.post_step(model_executed=executed)
    rec = {
        "beat_tag": tag,
        "order_this_call": list(order),
        "queue_len_after": len(engine.batch_queue),
        "executed": executed,
        "req_ph": req.num_output_placeholders,
        "req_output_token_ids": list(req.output_token_ids),
    }
    if outputs:
        rec["delivered_tokens"] = [
            t for eco in outputs.values() for o in eco.outputs
            for t in o.new_token_ids
        ]
    trace["beats"].append(rec)
    order.clear()
    return outputs


# 拍1：prefill，ph=0 → pending 不置位 → 立即采样（order=[sample]）
so_flags_1 = {"pending": None}
out1 = beat("拍1 prefill（ph=0 → 不 pending → 立即采样）")
# 拍2：decode，ph=1>0 → pending 置位 → 上半段不采样；下半段 pop 批A、update 之后补采
out2 = beat("拍2 decode（ph=1 → pending → deferred：pop+update 后才补采）")
# 拍3：pop deferred 批——拍2 采出的 token 到账
out3 = beat("拍3 pop deferred 批（拍2 补采的 token 到账）")

trace["deferred_mechanism"] = {
    "beat2_order": trace["beats"][1]["order_this_call"],
    "beat2_order_note": (
        "同一次 step_with_batch_queue 调用内：上半段看到 pending_structured_output_tokens"
        "=True → 不采样（批先入队、前向已发起不浪费）；下半段 pop 批A → update_from_output"
        "（此刻 token 齐了）→ 补 get_grammar_bitmask + sample_tokens → appendleft 重新入队"
        "（复用同一 exec_future，不重跑前向）——顺序必须是 update → sample"
    ),
    "beat3_delivery": trace["beats"][2].get("delivered_tokens"),
    "queue_note": "拍2 结束队列长度 1 = deferred 批已重新入队待后续轮次 pop",
    "why_defer": (
        "async 下调度新批时上一拍输出 token 可能还在 D2H 路上；结构化输出的 grammar"
        " bitmask 要基于本批将采样的位置（含 spec 草稿）计算——缺 token 就算不出正确"
        " 掩码，立即采样会采出违反语法的 token（WC3）"
    ),
}

out = os.path.join(os.path.dirname(__file__), "m14_deferred.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
