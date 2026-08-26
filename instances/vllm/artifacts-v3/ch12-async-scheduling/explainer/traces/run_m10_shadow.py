# ch12 m10 驱动：worker 影子①②——采样 token 留 GPU（prev_sampled_token_ids 缓存
# 同一张量）+ token_ids_cpu 行只写 -1 占位（is_token_ids=True、num_tokens_no_spec
# 照常推进）+ prev_req_id_to_index 槽位表（discard 行不进表）。
# 真源锚点：gpu_model_runner.py:L3797-L3813 / L3815-L3842、gpu_input_batch.py:L309-L316。
# HOST SEAM：CPU 张量代 GPU 张量——『留 GPU』的证据是 prev_sampled_token_ids 与
# 采样输出是同一个张量对象（is 同一性），未做任何 D2H/tolist。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch  # noqa: E402

from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402


def make_runner(async_on=True):
    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(async_scheduling=async_on if async_on else False),
        max_model_len=64,
    )
    if async_on is None:
        cfg.check_and_set_default_async_scheduling()
    return GPUModelRunner(cfg, max_num_reqs=8, max_model_len=64, vocab_size=16)


class _SO:
    def __init__(self, sampled):
        self.sampled_token_ids = sampled
        self.logprobs_tensors = None


def fill(runner, reqs):
    for rid, prompt in reqs.items():
        runner.input_batch.add_request(rid, prompt)
        runner.requests[rid] = type(
            "ReqState", (), {"num_tokens": len(prompt), "all_token_ids": list(prompt)}
        )()


trace = {
    "mechanism": "m10 worker 影子①②：采样 token 不落 CPU",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/worker/gpu_model_runner.py:L3797-L3813, L3815-L3842; gpu_input_batch.py:L309-L316",
    "host_seam_note": (
        "HOST SEAM：CPU 张量代 GPU 张量、PIN_MEMORY=False。『留 GPU』的可观测证据 ="
        " prev_sampled_token_ids 与采样输出张量 is 同一（未 _to_list/未 .cpu()）——"
        "真实语义见注释 'Cache the sampled tokens on the GPU and avoid CPU sync'"
        "（gpu_model_runner.py:L3802）"
    ),
    "shadow_case_single": {},
    "shadow_case_discard": {},
    "sync_control": {},
}

# -------- 场景一：单请求 async 写回
runner = make_runner()
fill(runner, {"req-0": [1, 2, 3]})
runner.input_batch.num_tokens_no_spec[0] = 3
sampled = torch.tensor([[9]], dtype=torch.int64)
runner._bookkeeping_sync(scheduler_output=None, sampler_output=_SO(sampled),
                         logits=None, hidden_states=None, num_scheduled_tokens=1)
trace["shadow_case_single"] = {
    "params": "req-0 prompt=[1,2,3]（num_tokens_no_spec=3）、采样张量 [[9]]",
    "prev_sampled_token_ids_is_same_object": runner.input_batch.prev_sampled_token_ids is sampled,
    "prev_sampled_token_ids": runner.input_batch.prev_sampled_token_ids.tolist(),
    "prev_req_id_to_index": dict(runner.input_batch.prev_req_id_to_index),
    "token_ids_cpu_row": [int(v) for v in runner.input_batch.token_ids_cpu[0][:5]],
    "is_token_ids_row": [bool(v) for v in runner.input_batch.is_token_ids[0][:5]],
    "num_tokens_no_spec": int(runner.input_batch.num_tokens_no_spec[0]),
    "note": "位置 3 写 -1 占位（真 token 9 在 GPU 张量里）；is_token_ids=True、num_tokens_no_spec 3→4 账本照走——CPU 侧账本在走、真 token 在 GPU",
}

# -------- 场景二：3 请求、中间行被 discard（不该采样的请求）
runner2 = make_runner()
fill(runner2, {"req-0": [1], "req-1": [2], "req-2": [3]})
for i in range(3):
    runner2.input_batch.num_tokens_no_spec[i] = 1
sampled2 = torch.tensor([[7], [8], [9]], dtype=torch.int64)
runner2.input_batch.prev_sampled_token_ids = None
runner2.discard_request_mask.np[1] = True  # req-1 被标丢弃（乐观纠错的下游）
runner2._bookkeeping_sync(scheduler_output=None, sampler_output=_SO(sampled2),
                          logits=None, hidden_states=None, num_scheduled_tokens=3)
trace["shadow_case_discard"] = {
    "params": "3 请求各 1 prompt token；req-1 行 discard_request_mask=True（m18：optimistic_seq_lens < num_tokens 的行不该采样）",
    "prev_req_id_to_index": dict(runner2.input_batch.prev_req_id_to_index),
    "row0_after": [int(v) for v in runner2.input_batch.token_ids_cpu[0, :3]],
    "row1_after": [int(v) for v in runner2.input_batch.token_ids_cpu[1, :3]],
    "row2_after": [int(v) for v in runner2.input_batch.token_ids_cpu[2, :3]],
    "num_tokens_no_spec_rows": [int(v) for v in runner2.input_batch.num_tokens_no_spec[:3]],
    "note": "row1 保持 0 未动（invalid 行不写占位：sampled_ids=None → continue）；有效行 0/2 位置 1 写 -1、行长 1→2",
}

# -------- 对照：同步分支把 token 真写进 token_ids_cpu
runner3 = make_runner(async_on=False)
fill(runner3, {"req-0": [1, 2, 3]})
runner3.input_batch.num_tokens_no_spec[0] = 3
sampled3 = torch.tensor([[9]], dtype=torch.int64)
runner3._bookkeeping_sync(scheduler_output=None, sampler_output=_SO(sampled3),
                          logits=None, hidden_states=None, num_scheduled_tokens=1)
trace["sync_control"] = {
    "prev_sampled_token_ids": (
        "None" if runner3.input_batch.prev_sampled_token_ids is None else "非 None（异常）"
    ),
    "token_ids_cpu_row": [int(v) for v in runner3.input_batch.token_ids_cpu[0][:5]],
    "note": "同步分支 _to_list 把 9 真写进 token_ids_cpu 位置 3——async 与 sync 的分叉点就在 L3796/L3822 的 use_async_scheduling 判定",
}

out = os.path.join(os.path.dirname(__file__), "m10_shadow.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
