# ch12 m9 驱动：early-stop 剪枝（async 专属）——
# computed + 2 − ph ≥ prompt + max_tokens 就不排多余一步。
# 两个对照案例：确信到顶（剪枝）vs 远没到顶（照排）。
# 真源锚点：vllm/v1/core/sched/scheduler.py:L488-L502。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.outputs import ModelRunnerOutput  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402


def make_sched():
    cfg = VllmConfig(scheduler_config=SchedulerConfig(), max_model_len=64)
    cfg.check_and_set_default_async_scheduling()
    cls = cfg.scheduler_config.get_scheduler_cls()
    return cls(vllm_config=cfg, log_stats=False, num_gpu_blocks=64)


def mro(req_id, tokens):
    return ModelRunnerOutput(req_ids=[req_id], req_id_to_index={req_id: 0},
                             sampled_token_ids=[list(tokens)])


trace = {
    "mechanism": "m9 early-stop 剪枝（async 专属）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/core/sched/scheduler.py:L488-L502",
    "guard": "request.num_output_placeholders > 0 and computed + 2 − ph ≥ prompt + max_tokens → req_index += 1; continue",
    "why_plus_2_minus_ph": (
        "注释原文：(computed+1) − (ph−1)——占位也计入 computed，减掉 (ph−1) 移除 draft"
        " token 的影响，保证即使 draft 全拒也不再需要多一步"
    ),
    "uniform_note": "不排 partial draft tokens——'this prevents uniform decode optimizations'（保统一 decode 的 CUDA graph 优化）",
}


def run_case(label, max_tokens):
    sched = make_sched()
    req = Request(request_id="req-0", prompt_token_ids=[1, 2],
                  sampling_params=SamplingParams(max_tokens=max_tokens))
    sched.add_request(req)
    so1 = sched.schedule()  # 拍1：prefill → computed=2、ph=1
    so2 = sched.schedule()  # 拍2：盲排 1 → computed=3、ph=2
    sched.update_from_output(so1, mro(req.request_id, [7]))  # pop 批A：t7 到账 → ph=1、tws=3
    # 拍3 判定现场（schedule 之前的账本快照）
    computed_at_check = req.num_computed_tokens
    ph_at_check = req.num_output_placeholders
    tws_at_check = req.num_tokens_with_spec
    lhs = computed_at_check + 2 - ph_at_check
    rhs = req.num_prompt_tokens + req.max_tokens
    so3 = sched.schedule()
    return {
        "case": label,
        "max_tokens": max_tokens,
        "prompt_tokens": req.num_prompt_tokens,
        "state_at_beat3_check": {
            "computed": computed_at_check,
            "ph": ph_at_check,
            "tws": tws_at_check,
            "lhs_computed_plus_2_minus_ph": lhs,
            "rhs_prompt_plus_max_tokens": rhs,
            "pruned": lhs >= rhs,
        },
        "beat3_scheduled": dict(so3.num_scheduled_tokens),
        "beat3_total": so3.total_num_scheduled_tokens,
        "req_after": {
            "computed": req.num_computed_tokens,
            "ph": req.num_output_placeholders,
            "output_token_ids": list(req.output_token_ids),
        },
    }


trace["case_prune"] = run_case("确信上拍已达 max_tokens（max_tokens=2）", 2)
trace["case_prune_note"] = (
    "拍3 判定：computed(3)+2−ph(1)=4 ≥ prompt(2)+max_tokens(2)=4 → 剪枝——"
    "t9 即便还在 D2H 路上，位置 3 已确定不再需要前向（t9 是位置 2 的采样输出，"
    "max_tokens 已到）。若不剪：追赶公式 3+1−3=1 会白算一拍"
)
trace["case_not_prune"] = run_case("远没到顶（max_tokens=8）", 8)
trace["case_not_prune_note"] = (
    "拍3 判定：computed(3)+2−ph(1)=4 < prompt(2)+max_tokens(8)=10 → 不剪，"
    "照排追赶公式 tws(3)+ph(1)−computed(3)=1 个位置"
)

out = os.path.join(os.path.dirname(__file__), "m9_early_stop.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
