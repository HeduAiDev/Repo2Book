# ch12 m6 驱动：AsyncScheduler._update_after_schedule 占位 +1——非 prefill-chunk
# 请求 ph += num_sampled_tokens_per_step + spec 数；prefill-chunk 不占位。
# 真源锚点：vllm/v1/core/sched/async_scheduler.py:L19-L49。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402


def make_sched(max_num_scheduled_tokens=2048):
    cfg = VllmConfig(scheduler_config=SchedulerConfig(), max_model_len=64)
    cfg.check_and_set_default_async_scheduling()
    cls = cfg.scheduler_config.get_scheduler_cls()
    sched = cls(vllm_config=cfg, log_stats=False, num_gpu_blocks=64)
    sched.scheduler_config.max_num_scheduled_tokens = max_num_scheduled_tokens
    sched.max_num_scheduled_tokens = max_num_scheduled_tokens
    return sched


trace = {
    "mechanism": "m6 占位 +1（AsyncScheduler._update_after_schedule）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/core/sched/async_scheduler.py:L19-L49",
    "num_sampled_tokens_per_step": 1,
    "scenario_a_full_prefill": {},
    "scenario_b_chunked_prefill": {},
    "spec_note": (
        "无 spec 配置（speculative_config=None）时 cur_num_spec_tokens=0 → 占位=1；"
        "spec 场景 ph += num_sampled(1)+spec数(k)，且 spec_token_ids 被整体换成"
        " [-1]*num_spec_tokens_to_schedule 占位列表（真 token 由 worker 原地替换，"
        "async_scheduler.py:L42-L44 注释）——本章精简版无 spec 登记分支（归 ch33），"
        "spec 占位对账见 m15 trace"
    ),
}

# -------- 场景 A：全量 prefill（prompt=2，一次排完）→ 立即占位
sched = make_sched()
req = Request(request_id="req-a", prompt_token_ids=[1, 2],
              sampling_params=SamplingParams(max_tokens=8))
sched.add_request(req)
rows = []
for beat in (1, 2):
    so = sched.schedule()
    rows.append({
        "beat": beat,
        "scheduled": dict(so.num_scheduled_tokens),
        "is_prefill_chunk": bool(req.is_prefill_chunk),
        "ph": req.num_output_placeholders,
        "computed": req.num_computed_tokens,
        "in_flight": req.num_in_flight_tokens,
        "spec_token_ids": list(req.spec_token_ids),
    })
trace["scenario_a_full_prefill"] = {
    "params": "prompt=2、max_tokens=8、预算 2048（全量一次排完）",
    "rows": rows,
    "note": "拍1 全量 prefill：is_prefill_chunk=False（computed=2 不再 < num_tokens+ph）→ ph=0+1=1；拍2 盲排 1 个位置 → ph=1+1=2",
}

# -------- 场景 B：chunked prefill（prompt=6、预算 4 → 分两拍）→ chunk 期间不占位
sched2 = make_sched(max_num_scheduled_tokens=4)
req2 = Request(request_id="req-b", prompt_token_ids=[1, 2, 3, 4, 5, 6],
               sampling_params=SamplingParams(max_tokens=8))
sched2.add_request(req2)
rows2 = []
for beat in (1, 2, 3):
    so = sched2.schedule()
    rows2.append({
        "beat": beat,
        "scheduled": dict(so.num_scheduled_tokens),
        "is_prefill_chunk": bool(req2.is_prefill_chunk),
        "ph": req2.num_output_placeholders,
        "computed": req2.num_computed_tokens,
        "num_tokens": req2.num_tokens,
    })
trace["scenario_b_chunked_prefill"] = {
    "params": "prompt=6、max_tokens=8、max_num_scheduled_tokens=4（prefill 分两块）",
    "rows": rows2,
    "note": "拍1 chunk（computed=4 < num_tokens=6）→ is_prefill_chunk=True → continue 不占位（ph=0）；拍2 排完余下 2 token → 非 chunk → ph=1；拍3 盲排 1 → ph=2",
    "why": "prefill chunk 的最后 token 还没算出 logits，谈不上『下一拍的采样位置』——占位只加在完整 decode 步上",
}

out = os.path.join(os.path.dirname(__file__), "m6_placeholder_plus.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
