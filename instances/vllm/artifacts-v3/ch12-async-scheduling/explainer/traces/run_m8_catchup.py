# ch12 m8 驱动：追赶公式的占位项——num_new_tokens = num_tokens_with_spec
# + num_output_placeholders − num_computed_tokens（同步版占位恒 0，async 灌上值）。
# 对照：同一请求在同步 Scheduler 下的同一拍（公式相同、占位项=0 → 无 token 可排）。
# 真源锚点：vllm/v1/core/sched/scheduler.py:L516-L520。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402


def make_cfg(async_scheduling):
    cfg = VllmConfig(scheduler_config=SchedulerConfig(async_scheduling=async_scheduling),
                     max_model_len=64)
    if async_scheduling is None:
        cfg.check_and_set_default_async_scheduling()
    return cfg


def walk(async_scheduling):
    cfg = make_cfg(async_scheduling)
    sched = cfg.scheduler_config.get_scheduler_cls()(
        vllm_config=cfg, log_stats=False, num_gpu_blocks=64)
    req = Request(request_id="req-0", prompt_token_ids=[1, 2],
                  sampling_params=SamplingParams(max_tokens=8))
    sched.add_request(req)
    rows = []
    for beat in (1, 2):
        before = {
            "tws": req.num_tokens_with_spec,
            "ph": req.num_output_placeholders,
            "computed": req.num_computed_tokens,
        }
        so = sched.schedule()
        rows.append({
            "beat": beat,
            "formula_inputs_before": before,
            "formula_value": (before["tws"] + before["ph"] - before["computed"]),
            "scheduled": dict(so.num_scheduled_tokens),
            "after": {
                "ph": req.num_output_placeholders,
                "computed": req.num_computed_tokens,
                "in_flight": req.num_in_flight_tokens,
            },
        })
    return type(sched).__name__, rows


trace = {
    "mechanism": "m8 追赶公式的占位项（scheduler.py:L516-L520）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/core/sched/scheduler.py:L516-L520",
    "formula": "num_new_tokens = num_tokens_with_spec + num_output_placeholders − num_computed_tokens",
    "params": "prompt=2、max_tokens=8、无 spec；两拍连调 schedule()（中间不交货——模拟批 A 还在 GPU 上）",
}
cls_a, rows_a = walk(None)
trace["async_AsyncScheduler"] = {"scheduler_cls": cls_a, "rows": rows_a}
trace["async_note"] = (
    "拍2 公式 = tws(2) + ph(1) − computed(2) = 1——此刻 t 的采样还没回来，"
    "调度器凭占位数盲排『下一个位置』；排入后 ph=2、computed=3（乐观计入在飞）"
)
cls_s, rows_s = walk(False)
trace["sync_Scheduler_control"] = {"scheduler_cls": cls_s, "rows": rows_s}
trace["sync_note"] = (
    "对照（同步版）：占位恒 0 → 拍2 公式 = 2+0−2 = 0——num_new_tokens==0 走 continue"
    "（scheduler.py:L557 起 0-token 分支第 (2) 因：同步引擎必须等 update_from_output"
    "把真 token 追加进 tws 才有下一拍可排——这正是重叠版要拆掉的依赖）"
)

out = os.path.join(os.path.dirname(__file__), "m8_catchup.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
