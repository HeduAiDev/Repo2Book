# ch12 m7 驱动：占位 -1 + 乐观块转正——真 token 到达 ph -= len(new_token_ids)
# （assert ≥0、stale 不扣防 underflow）+ cache_blocks(computed−ph) 转正。
# 真源锚点：vllm/v1/core/sched/async_scheduler.py:L51-L70。
import json
import os
import sys
import time

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
    "mechanism": "m7 占位 -1 + 块转正（AsyncScheduler._update_request_with_output）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/core/sched/async_scheduler.py:L51-L70",
    "invariant": "num_computed_tokens − num_output_placeholders = 真实已算（cache_blocks 的参数就是它）",
    "host_seam_note": "KVCacheManager 是契约面（块 id 合成整数、cache_blocks 只记账不写哈希）；块池内景归 ch13/ch15",
}

# -------- 段一：交货扣减 + 块转正
sched = make_sched()
req = Request(request_id="req-0", prompt_token_ids=[1, 2],
              sampling_params=SamplingParams(max_tokens=8))
sched.add_request(req)
so1 = sched.schedule()  # 拍1：prefill，computed=2、ph=1
so2 = sched.schedule()  # 拍2：盲排 1，computed=3、ph=2
state_before = {
    "computed": req.num_computed_tokens, "ph": req.num_output_placeholders,
    "in_flight": req.num_in_flight_tokens, "tws": req.num_tokens_with_spec,
    "blocks_held": len(sched.kv_cache_manager.req_to_blocks[req.request_id]),
}
# 下半段 pop 批 A（= so1，prefill 批）交货 t7
outs = sched.update_from_output(so1, mro(req.request_id, [7]))
state_after = {
    "computed": req.num_computed_tokens, "ph": req.num_output_placeholders,
    "in_flight": req.num_in_flight_tokens, "tws": req.num_tokens_with_spec,
    "output_token_ids": list(req.output_token_ids),
    "real_computed_minus_ph": req.num_computed_tokens - req.num_output_placeholders,
}
trace["delivery"] = {
    "params": "prompt=2、max_tokens=8；拍1 prefill(ph=1)、拍2 盲排(ph=2, computed=3)；对批A(so1) 交货 [7]",
    "state_before_pop": state_before,
    "state_after_pop": state_after,
    "cache_blocks_calls": [list(c) for c in sched.kv_cache_manager.cache_blocks_calls],
    "note": "ph: 2−1=1（扣 len(new_token_ids)=1）；cache_blocks 参数 = computed(3)−ph(1) = 2 ——真实已算的 2 个 prompt 位转正（tws 2→3：t7 已入账本）",
}

# -------- 段二：stale 送达不扣占位（抢占时占位已清零，扣了就 underflow）
sched2 = make_sched()
req2 = Request(request_id="req-1", prompt_token_ids=[1, 2],
               sampling_params=SamplingParams(max_tokens=8))
sched2.add_request(req2)
sched2.schedule()
sched2.schedule()  # ph=2、computed=3
sched2._preempt_request(req2, time.monotonic())  # ph=0、stale=在飞数
preempt_state = {
    "ph_after_preempt": req2.num_output_placeholders,
    "num_stale_output_tokens": req2.num_stale_output_tokens,
    "computed_after_preempt": req2.num_computed_tokens,
}
# stale 送达：token 照收、占位不动（扣了就 underflow）
new_tokens, stopped = sched2._update_request_with_output(req2, [7], is_stale=True)
trace["stale_delivery"] = {
    "preempt_state": preempt_state,
    "stale_delivery": {
        "output_token_ids": list(req2.output_token_ids),
        "ph_unchanged": req2.num_output_placeholders,
        "stopped": stopped,
    },
    "note": "抢占把占位清零（scheduler.py:L1306）；stale 输出照送（丢会扰动 spec-decode acceptance）但不得回扣已清零的计数器——async_scheduler.py:L59-L63 的防线（#48245 类 underflow 的修复现场）",
}

out = os.path.join(os.path.dirname(__file__), "m7_delivery_convert.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
