# ch29 m03 greedy 快路径与 torch.where 混批合并 — 驱动脚本（host）。
# 机制：all_greedy 直接 argmax 早退（L261-L271）整条跳过温度/不变列/截断；
# 混批两路先各算一遍（greedy_sample 全批 + 随机路全批），再按逐请求
# temp<1e-5 用 torch.where 逐行选（L296-L302，out= 复用张量）。
# 行为基准：vllm/v1/sample/sampler.py:L227-L241/L243-L302（真实 v0.27.1 行号）。
# 间谍法：包 sampler.greedy_sample / topk_topp_sampler 记录中间值，控制流全走真代码。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS, Sampler

R = lambda x: round(float(x), 4)


def make_metadata(batch: int, **over) -> SamplingMetadata:
    fields = dict(
        temperature=torch.full((batch,), 1.0),
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch),
        presence_penalties=torch.zeros(batch),
        repetition_penalties=torch.ones(batch),
        output_token_ids=[[] for _ in range(batch)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    fields.update(over)
    return SamplingMetadata(**fields)


def ensure_native_binding(sampler: Sampler) -> None:
    t = sampler.topk_topp_sampler
    if "forward" not in t.__dict__:
        t.forward = t.forward_native


out = {"sampling_eps": 1e-05}

# ── A. all_greedy 早退：温度没跑、logits 没被改写 ────────────────────
torch.manual_seed(0)
logits_a = torch.tensor([[1.0, 5.0, 2.0, 0.0, -1.0], [-3.0, 0.5, 4.0, 1.0, 2.0]])
before_a = logits_a.clone()
md_a = make_metadata(2, all_greedy=True, temperature=None)
ran_temperature = {"yes": False}

s_a = Sampler()
real_apply_temp = Sampler.apply_temperature


def spy_temp(logits, temp, all_random):
    ran_temperature["yes"] = True
    return real_apply_temp(logits, temp, all_random)


Sampler.apply_temperature = staticmethod(spy_temp)
out_a = s_a(logits_a, md_a)
Sampler.apply_temperature = staticmethod(real_apply_temp)
out["all_greedy_early_return"] = {
    "logits": [[1.0, 5.0, 2.0, 0.0, -1.0], [-3.0, 0.5, 4.0, 1.0, 2.0]],
    "all_greedy": True,
    "temperature": None,
    "argmax": [1, 2],
    "sampled": out_a.sampled_token_ids.flatten().tolist(),
    "temperature_ran": ran_temperature["yes"],
    "caller_logits_untouched": bool(torch.equal(logits_a, before_a)),
    "logprobs_tensors": "None(max_num_logprobs=None 默认路径)",
}

# ── B. 混批：两路先各算一遍，torch.where 逐行选 ─────────────────────
torch.manual_seed(0)
logits_b = torch.tensor([
    [0.1, 6.0, 1.0, 0.5, 0.2],   # row0 greedy(temp=0)
    [2.0, 1.9, 1.8, 0.0, 0.0],   # row1 random(temp=1)
    [0.0, 0.1, 0.2, 3.0, 0.3],   # row2 greedy(temp=0)
])
md_b = make_metadata(
    3, temperature=torch.tensor([0.0, 1.0, 0.0]),
)
s_b = Sampler()
ensure_native_binding(s_b)

# 间谍：记录真 sample() 内部两路各自的产出（控制流不改动）
captured = {}
real_greedy = Sampler.greedy_sample


def spy_greedy(logits):
    r = real_greedy(logits)
    captured["greedy_sampled"] = r.tolist()
    captured["greedy_logits_snapshot"] = [[R(v) for v in row] for row in logits]
    return r


s_b.greedy_sample = spy_greedy
real_topk = s_b.topk_topp_sampler.forward


def spy_topk(logits, generators, k, p):
    toks, extra = real_topk(logits, generators, k, p)
    captured["random_sampled"] = toks.tolist()
    captured["random_logits_snapshot_after_temp"] = [[R(v) for v in row] for row in logits]
    return toks, extra


s_b.topk_topp_sampler.forward = spy_topk
out_b = s_b(logits_b.clone(), md_b)
final_b = out_b.sampled_token_ids.flatten().tolist()
temp_replaced = torch.where(
    torch.tensor([0.0, 1.0, 0.0]) < _SAMPLING_EPS, 1.0, torch.tensor([0.0, 1.0, 0.0])
)
out["mixed_batch_where_merge"] = {
    "temperature": [0.0, 1.0, 0.0],
    "temp_after_where_guard": temp_replaced.tolist(),  # 防 0 除：替 1.0
    "greedy_sampled_all_rows": captured["greedy_sampled"],
    "random_sampled_all_rows": captured["random_sampled"],
    "random_path_logits_after_temperature": captured["random_logits_snapshot_after_temp"],
    "where_cond_temp_lt_eps": [True, False, True],
    "merged": final_b,
    "note": "row0/row2 取 greedy 路(精确 argmax)；row1 取随机路掷骰结果",
}

# ── C. 哨兵对照：随机路换成可区分替身，钉死 where 逐行选择语义 ───────
torch.manual_seed(0)
s_c = Sampler()
sentinel = torch.tensor([4, 4, 4], dtype=torch.int64)


class _StubTopKTopP(torch.nn.Module):
    def forward(self, logits, generators, k, p):
        return sentinel.clone(), None


s_c.topk_topp_sampler = _StubTopKTopP()
out_c = s_c(logits_b.clone(), md_b)
out["sentinel_where_selection"] = {
    "greedy_sampled_all_rows": captured["greedy_sampled"],
    "sentinel_random_all_rows": sentinel.tolist(),
    "merged": out_c.sampled_token_ids.flatten().tolist(),
    "claim": "torch.where(temp<eps, greedy, random) 逐行选：[1, 4, 3]——greedy 行取 1/3、随机行取哨兵 4",
}

# ── D. 全随机批：greedy_sampled=None，只跑随机路 ─────────────────────
torch.manual_seed(5)
md_d = make_metadata(2, all_random=True, all_greedy=False,
                     temperature=torch.tensor([1.0, 1.0]))
s_d = Sampler()
ensure_native_binding(s_d)
greedy_called = {"yes": False}


def spy_greedy2(logits):
    greedy_called["yes"] = True
    return real_greedy(logits)


s_d.greedy_sample = spy_greedy2
out_d = s_d(torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]), md_d)
out["all_random_skips_greedy"] = {
    "all_random": True,
    "greedy_sample_called": greedy_called["yes"],
    "sampled": out_d.sampled_token_ids.flatten().tolist(),
}

g = captured["greedy_sampled"]
rnd = captured["random_sampled"]
out["table_rows_echo"] = [
    ["A: all_greedy 早退(L261-L271)", "temperature=None", "argmax=[1, 2]", "温度没跑/调用方 logits 未被改写", "早退返回 [1, 2]"],
    ["B: greedy 路全批先算", "temp=[0.0, 1.0, 0.0]", f"greedy_sampled={g}", "argmax 对全批 3 行都算", "row0/row2 将被选用"],
    ["B: 随机路全批也算", "temp<eps 替 1.0 → [1.0, 1.0, 1.0]", f"随机掷骰={rnd}", "温度/截断/掷骰对全批跑", "row1 将被选用"],
    ["B: torch.where 逐行选(L296-L301)", "cond=temp<1e-5=[True, False, True]", f"where(greedy={g}, random={rnd})", "out=greedy_sampled 复用张量", f"合并 {final_b}"],
    ["C: 哨兵对照", f"随机路替身恒返 {sentinel.tolist()}", f"greedy={g}", "同批同参数", f"合并 {out_c.sampled_token_ids.flatten().tolist()}——逐行选择实锤"],
    ["D: all_random 批", "greedy_sample 不被调用", "只跑随机路", f"掷骰 {out_d.sampled_token_ids.flatten().tolist()}", "无 where 合并(greedy_sampled=None)"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m03_greedy_fastpath_where.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
