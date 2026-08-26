# ch12 m15 驱动：spec 拒绝回扣——num_rejected 同步回退 num_computed_tokens 与
# num_output_placeholders（stale 不回扣）。机制本体归 ch33，此处只对账回扣算术。
# 人灌模式声明：本章精简版无 spec 登记分支（dossier.delete 第 6 条），spec 状态
# 按 tests/test_spec_rejection_rolls_back_computed_and_placeholders 的同一模式
# 人工注入对账（impl-notes『账本口径备注』：ph = 1 bonus + 3 spec = 4）。
# 真源锚点：vllm/v1/core/sched/scheduler.py:L1769-L1784。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.outputs import ModelRunnerOutput  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402

cfg = VllmConfig(scheduler_config=SchedulerConfig(), max_model_len=64)
cfg.check_and_set_default_async_scheduling()
sched = cfg.scheduler_config.get_scheduler_cls()(
    vllm_config=cfg, log_stats=False, num_gpu_blocks=64)

# prompt=5：prefill 后 computed=5，回扣 3 后仍为正、可心算
req = Request(request_id="req-0", prompt_token_ids=[1, 2, 3, 4, 5],
              sampling_params=SamplingParams(max_tokens=8))
sched.add_request(req)
so = sched.schedule()
trace = {
    "mechanism": "m15 spec 拒绝回扣（scheduler.py:L1769-L1784）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/core/sched/scheduler.py:L1769-L1784",
    "params": "prompt=5、max_tokens=8、num_sampled_tokens_per_step=1、draft k=3",
    "manual_injection_note": (
        "spec 深水归 ch33：本章精简版 schedule 不登记草稿（dossier.delete 第 6 条），"
        "此处按 tests 同一模式人灌 spec 状态对账——ph 人灌为 4 = 1 bonus + 3 spec"
        "（与 _update_after_schedule 的 num_sampled + spec 数 灌值口径一致）"
    ),
    "step1_prefill": {
        "computed": req.num_computed_tokens,
        "ph_impl": req.num_output_placeholders,
    },
}

# 人灌：本批带 3 个草稿 token、占位 1 bonus + 3 spec = 4
so.scheduled_spec_decode_tokens[req.request_id] = [11, 12, 13]
req.spec_token_ids = [11, 12, 13]
req.num_output_placeholders = 4
trace["manual_spec_injection"] = {
    "scheduled_spec_decode_tokens": list(so.scheduled_spec_decode_tokens[req.request_id]),
    "ph_manual": req.num_output_placeholders,
    "computed_before": req.num_computed_tokens,
}

# 采样结果：只有 bonus token [7]，0 个草稿被接受
mro = ModelRunnerOutput(req_ids=[req.request_id],
                        req_id_to_index={req.request_id: 0},
                        sampled_token_ids=[[7]])
computed_before = req.num_computed_tokens
ph_before = req.num_output_placeholders
# 观测回扣后的中间态：包一层 _update_request_with_output（回扣在其之前已完成）
ph_at_delivery = [None]
_orig_urwo = sched._update_request_with_output


def urwo_probe(request, new_token_ids, is_stale=False):
    ph_at_delivery[0] = request.num_output_placeholders
    return _orig_urwo(request, new_token_ids, is_stale)


sched._update_request_with_output = urwo_probe
sched.update_from_output(so, mro)

num_draft = 3
num_sampled = 1
num_accepted = max(len([7]) - num_sampled, 0)
num_rejected = num_draft - num_accepted
trace["rejection_arithmetic"] = {
    "num_draft_tokens": num_draft,
    "num_sampled": num_sampled,
    "num_accepted_max_len_minus_num_sampled": num_accepted,
    "num_rejected": num_rejected,
    "computed_before": computed_before,
    "computed_after_rollback": req.num_computed_tokens,
    "ph_before": ph_before,
    "ph_after_rollback_before_delivery": ph_at_delivery[0],
    "ph_after_delivery": req.num_output_placeholders,
    "output_token_ids": list(req.output_token_ids),
    "cache_blocks_calls": [list(c) for c in sched.kv_cache_manager.cache_blocks_calls],
}
trace["rollback_note"] = (
    "注释原话：'Rejections roll back num_computed_tokens (and, under async scheduling,"
    " num_output_placeholders, which covers the spec tokens)'——占位覆盖 spec token，"
    "拒绝回扣必须双回退：computed 5−3=2、ph 4−3=1，再经 delivery [7] 扣 1 → ph=0；"
    "cache_blocks 参数 = computed(2)−ph(0) = 2（真实已算）。A stale rejection count"
    " predates the preemption rollback and must not apply（stale 不回扣——m7 段二同防线）"
)

out = os.path.join(os.path.dirname(__file__), "m15_spec_rollback.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
