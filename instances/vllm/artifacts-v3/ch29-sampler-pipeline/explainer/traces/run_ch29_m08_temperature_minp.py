# ch29 m08 温度缩放与 min_p 尾部截断 — 驱动脚本（host）。
# 机制：温度 logits/T（非 all_random 时 temp<1e-5 替 1.0 防除零，in-place div_）；
# 正数缩放保序 → 温度不改变 argmax（greedy 快路径敢整条跳过它的数学依据）；
# min_p 阈值 = min_p×max_prob 砍尾、最高位恒留（builtin.py:L47-L49 声明的证明）。
# 行为基准：vllm/v1/sample/sampler.py:L227-L241、logits_processor/builtin.py:L102-L116
# （真实 v0.27.1 行号）。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
from vllm.v1.sample.sampler import _SAMPLING_EPS, Sampler

R = lambda x: round(float(x), 4)
out = {"sampling_eps": 1e-05}

# ── A. 温度：正缩放保序，argmax 不动 ────────────────────────────────
logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1]])
t2 = Sampler.apply_temperature(logits.clone(), torch.tensor([2.0]), all_random=True)
t05 = Sampler.apply_temperature(logits.clone(), torch.tensor([0.5]), all_random=True)
out["temperature_order_preserving"] = {
    "logits": [3.0, 2.0, 1.0, 0.5, 0.1],
    "argmax": 0,
    "T=2.0(钝化)": [R(v) for v in t2[0]],
    "T=0.5(锐化)": [R(v) for v in t05[0]],
    "argmax_T2": int(t2.argmax()), "argmax_T05": int(t05.argmax()),
    "claim": "除以正数 T 只缩放不改序：argmax 恒 0——温度放随机路径而 greedy 快路径敢跳过它的依据",
}

# ── B. 防除零：temp<eps 替 1.0（L228-L237）─────────────────────────
# 3 行批（每行同一组 logits），temp=[0.0, 0.7, 1e-06]——行 0/2 是 greedy 行。
batch3 = logits.repeat(3, 1)
guard = torch.where(torch.tensor([0.0, 0.7, 0.000001]) < _SAMPLING_EPS,
                    1.0, torch.tensor([0.0, 0.7, 0.000001]))
guarded = Sampler.apply_temperature(batch3.clone(), torch.tensor([0.0, 0.7, 1e-06]), all_random=False)
unguarded = logits.clone()
unguarded.div_(torch.tensor([[0.0]]))   # 反面演示：无保护直接除 0
out["divide_by_zero_guard"] = {
    "eps": 1e-05,
    "temp": [0.0, 0.7, 1e-06],
    "cond_temp_lt_eps": [True, False, True],
    "temp_after_guard": guard.tolist(),
    "logits_after_guard_row0": [R(v) for v in guarded[0]],
    "logits_after_guard_row1_T0.7": [R(v) for v in guarded[1]],
    "logits_after_guard_row2": [R(v) for v in guarded[2]],
    "unguarded_div0_row0": [str(v) for v in unguarded[0]],
    "claim": "temp=0 行除 1.0 → 原样返回；若不替换直接 div_(0) → inf（softmax 将 nan）——这就是 L231 替换的意义",
}

# ── C. min_p 砍尾：阈值 = min_p × max_prob，最高位恒留 ──────────────
def min_p_proc(min_p_value: float) -> MinPLogitsProcessor:
    cfg = type("C", (), {"scheduler_config": type("S", (), {"max_num_seqs": 4})})()
    mp = MinPLogitsProcessor(cfg, torch.device("cpu"), False)
    mp.min_p_cpu[0] = min_p_value
    mp.min_p_count = 1
    mp.min_p = mp.min_p_device[:1].unsqueeze(1).clone()
    return mp


probs = torch.softmax(logits[0], dim=-1)
out["min_p_tail_cut"] = {
    "logits(T=1.0)": [3.0, 2.0, 1.0, 0.5, 0.1],
    "probs": [R(p) for p in probs],
    "max_prob": R(probs.max()),
}
for mpv in (0.3, 0.5):
    mp = min_p_proc(mpv)
    masked = mp.apply(logits.clone())
    survivors = [i for i in range(5) if masked[0, i].item() != float("-inf")]
    out["min_p_tail_cut"][f"min_p={mpv}"] = {
        "threshold": R(mpv * probs.max()),
        "survivors": survivors,
        "probs_of_survivors": [R(probs[i]) for i in survivors],
        "logits_after": [("-inf" if masked[0, i].item() == float("-inf") else str(masked[0, i].item())) for i in range(5)],
        "argmax_before": 0, "argmax_after": 0,
    }
out["min_p_tail_cut"]["claim"] = (
    "阈值 = min_p×max_prob < max_prob 恒成立 → 最高位必留 → argmax 不变"
    "（builtin.py:L47-L49『Min-p never impacts greedy sampling』的两行证明）"
)

out["table_rows_echo"] = [
    ["温度 T=2.0(钝化)", "logits/T=[1.5, 1.0, 0.5, 0.25, 0.05]", "分布变平", "argmax 0 不变", "in-place div_"],
    ["温度 T=0.5(锐化)", "logits/T=[6.0, 4.0, 2.0, 1.0, 0.2]", "分布变尖", "argmax 0 不变", "正缩放保序是数学依据"],
    ["防除零(L231)", "temp=[0.0, 0.7, 0.000001]", "temp<1e-5 替 1.0 → [1.0, 0.7, 1.0]", "不替换则 div_(0)→inf", "temp=0 行除 1.0 原样返回"],
    ["min_p=0.3 砍尾", f"probs={[R(p) for p in probs]}", "阈值=0.3×0.6096=0.1829", "survivors=[0, 1](0.2243≥0.1829)", "argmax 前后都 0"],
    ["min_p=0.5 砍尾", "阈值=0.5×0.6096=0.3048", "0.2243<0.3048 也被砍", "survivors=[0]", "最高位恒留→argmax 不变"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m08_temperature_minp.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
