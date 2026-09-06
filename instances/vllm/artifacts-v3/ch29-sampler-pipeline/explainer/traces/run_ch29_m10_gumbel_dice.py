# ch29 m10 Gumbel/exp 掷骰替代 multinomial — 驱动脚本（host）。
# 机制：random_sample（topk_topp_sampler.py:L450-L472）——q~Exp(1)、
# sample_with_exponential_noise 取 argmax(p/q)（L441-L447；同 dtype 走
# probs.div_(q) 原地支路、fp64 走 reciprocal+mul 支路），与按 p 抽样同分布
# （Gumbel-max 定理的 exp 形式）且全程无 CPU-GPU 同步——docstring 原话
# "We use this function instead of torch.multinomial because torch.multinomial
# causes CPU-GPU synchronization"（L455-L458）。
# 行为基准：vllm/v1/sample/ops/topk_topp_sampler.py:L434-L472（真实 v0.27.1 行号）。
import json
import os
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK 控制台打印数学符号免疫

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm.v1.sample.ops.topk_topp_sampler import (
    random_sample,
    sample_with_exponential_noise,
)

R = lambda x: round(float(x), 4)
out = {}

# ── A. 手算轮次：q 除进概率、argmax 出 token ────────────────────────
probs3 = torch.tensor([0.6, 0.3, 0.1])
q_manual = torch.tensor([
    [0.5, 2.0, 0.3],    # p/q=[1.2, 0.15, 0.3333] → 0
    [0.8, 0.2, 1.5],    # p/q=[0.75, 1.5, 0.0667] → 1
    [0.4, 0.5, 0.09],   # p/q=[1.5, 0.6, 1.1111] → 0
    [0.9, 0.25, 0.05],  # p/q=[0.6667, 1.2, 2.0] → 2
])
scores_rows, argmax_rows = [], []
for q_row in q_manual:
    # 注意：同 dtype 支路 probs.div_(q) 原地改写 probs——真调用方每步喂新 softmax
    toks = sample_with_exponential_noise(probs3.clone().unsqueeze(0), q_row.unsqueeze(0))
    scores_rows.append([R(v) for v in (probs3 / q_row)])
    argmax_rows.append(int(toks[0]))
out["manual_rounds"] = {
    "probs": [0.6, 0.3, 0.1],
    "rounds": [
        {"q": q_manual[i].tolist(), "p_div_q": scores_rows[i], "argmax": argmax_rows[i]}
        for i in range(4)
    ],
    "claim": "q~Exp(1) 越小、p/q 被放大越多——小概率 token 偶尔靠一次小 q 翻盘，翻盘频率恰为 p",
}

# ── B. 统计等价：20 万掷 vs multinomial vs 理论 ─────────────────────
N = 200000
torch.manual_seed(1234)
q = torch.empty(N, 3).exponential_()
gumbel_tokens = sample_with_exponential_noise(probs3.repeat(N, 1), q)
g_freq = torch.bincount(gumbel_tokens, minlength=3).float() / N
torch.manual_seed(1234)
m_tokens = torch.multinomial(probs3, N, replacement=True)
m_freq = torch.bincount(m_tokens, minlength=3).float() / N
out["statistical_equivalence"] = {
    "N": N,
    "theory": [0.6, 0.3, 0.1],
    "gumbel_p_div_q_freq": [R(v) for v in g_freq],
    "multinomial_freq": [R(v) for v in m_freq],
    "max_abs_dev_gumbel": R(max(abs(g_freq[i] - [0.6, 0.3, 0.1][i]) for i in range(3))),
    "max_abs_dev_multinomial": R(max(abs(m_freq[i] - [0.6, 0.3, 0.1][i]) for i in range(3))),
    "claim": "argmax(p/q) 与 torch.multinomial 同分布：20 万掷频率双双贴住 [0.6, 0.3, 0.1]（差距为采样噪声量级）",
}

# ── C. 三行证明（数值验证 P(i wins) = ∫e^{-q}·exp(−q(1−p_i)/p_i) dq = p_i）──
# P(win_i) = ∫_0^∞ e^{-q} · ∏_{j≠i} exp(−q·p_j/p_i) dq = ∫_0^∞ exp(−q/p_i) dq = p_i
import math
out["integral_check"] = {
    "formula": "P(win_i) = ∫_0^∞ e^{-q}·exp(−q(1−p_i)/p_i) dq = ∫_0^∞ exp(−q/p_i) dq = p_i",
    "i=0 p=0.6": R(0.6),
    "monte_carlo_p0": R(g_freq[0]),
    "i=1 p=0.3": R(0.3),
    "monte_carlo_p1": R(g_freq[1]),
    "i=2 p=0.1": R(0.1),
    "monte_carlo_p2": R(g_freq[2]),
    "claim": "q_i 独立 Exp(1)：i 胜出 ⟺ q_j ≥ q_i·p_j/p_i 对一切 j≠i；代入 Exp 尾概率 e^{-x} 逐项相乘后积分恰好塌缩成 p_i",
}

# ── D. 无 seed 批量 + 有 seed 逐请求覆写（L461-L469）────────────────
# 情形 1：全批无 seed（len(generators)=0 ≠ batch）→ 一次批量 q.exponential_()
torch.manual_seed(100)
t1 = random_sample(probs3.repeat(2, 1), {})
torch.manual_seed(100)
t2 = random_sample(probs3.repeat(2, 1), {})
# 情形 2：逐请求 seed —— 同种子重建 generator 两次，行 0 可复现、无 seed 行漂移。
# 取 5-token 均匀分布 + 6 行（行 0 有 seed、行 1-5 无）：漂移概率 1-(1/5)^5，避开巧合分支。
uni = torch.full((5,), 0.2)
g0a = torch.Generator().manual_seed(42)
g0b = torch.Generator().manual_seed(42)
torch.manual_seed(200)
seeded_a = random_sample(uni.repeat(6, 1), {0: g0a}).tolist()
torch.manual_seed(999)
seeded_b = random_sample(uni.repeat(6, 1), {0: g0b}).tolist()
drift_rows = [i for i in range(1, 6) if seeded_a[i] != seeded_b[i]]
# 情形 3：批内每行都有 seed（len(generators)==batch）→ 跳过批量填充、逐行覆写
ga = torch.Generator().manual_seed(42)
gb = torch.Generator().manual_seed(7)
ga2 = torch.Generator().manual_seed(42)
gb2 = torch.Generator().manual_seed(7)
both_a = random_sample(probs3.repeat(2, 1), {0: ga, 1: gb}).tolist()
both_b = random_sample(probs3.repeat(2, 1), {0: ga2, 1: gb2}).tolist()
out["seed_semantics"] = {
    "case_no_seed_reproducible_with_global_seed": {
        "run1": t1.tolist(), "run2_same_manual_seed": t2.tolist(),
        "equal": bool(torch.equal(t1, t2)),
    },
    "case_one_generator_overwrites_row0": {
        "probs": "5-token 均匀 [0.2]*5，批 6 行，generators={0: seed42}",
        "call1_global_seed_200": seeded_a,
        "call2_global_seed_999_fresh_generator_same_seed": seeded_b,
        "row0_stable": seeded_a[0] == seeded_b[0],
        "unseeded_rows_1_to_5_drift": drift_rows,
        "len_generators_vs_batch": "1 != 6 → 先批量 q.exponential_() 再用 generator 覆写第 0 行（L461-L469 TODO(woosuk): can be slow）",
    },
    "case_every_row_seeded": {
        "call1": both_a, "call2": both_b,
        "both_rows_reproducible": both_a == both_b,
        "len_generators_vs_batch": "2 == 2 → 跳过批量填充（L461 条件不成立），逐行 generator 覆写整批",
    },
}

# ── E. fp64 支路：q.dtype != probs.dtype → reciprocal+mul ──────────
torch.manual_seed(31)
q64 = torch.empty(4, 3).double().exponential_()
q64_snapshot = q64.clone()  # 分支 reciprocal_ 会原地改写 q64，先快照供直除对照
probs32 = probs3.repeat(4, 1)
toks_branch = sample_with_exponential_noise(probs32.clone(), q64)
toks_direct = (probs32.double() / q64_snapshot).argmax(dim=-1).view(-1)
out["fp64_branch"] = {
    "probs_dtype": "float32", "q_dtype": "float64",
    "branch": "q.dtype != probs.dtype → scores=q.reciprocal_(); scores.mul_(probs)（全程 fp64 避免中途降精度，L443-L446；q64 被 reciprocal_ 原地改写）",
    "branch_tokens": toks_branch.tolist(),
    "float64_direct_tokens": toks_direct.tolist(),
    "equal": bool(torch.equal(toks_branch, toks_direct)),
    "inplace_note": "同 dtype 支路 probs.div_(q) 会原地改写 probs——真调用方每步喂新 softmax（forward_native L141）；fp32 张量对 fp64 q 不能 div_ 原地升精度，故走 reciprocal+mul（两次舍入 vs 直除一次，仅极近平手时可能差一位）",
}

out["table_rows_echo"] = [
    ["手算轮 1", "q=[0.5, 2.0, 0.3]", "p/q=[1.2, 0.15, 0.3333]", "argmax=0", "概率最大者常规胜出"],
    ["手算轮 2", "q=[0.8, 0.2, 1.5]", "p/q=[0.75, 1.5, 0.0667]", "argmax=1", "token1 抽中一次小 q 翻盘"],
    ["手算轮 3", "q=[0.4, 0.5, 0.09]", "p/q=[1.5, 0.6, 1.1111]", "argmax=0", "头名稳定"],
    ["手算轮 4", "q=[0.9, 0.25, 0.05]", "p/q=[0.6667, 1.2, 2.0]", "argmax=2", "10% 概率的 token2 翻盘"],
    ["统计等价(20 万掷)", f"gumbel 频率={[R(v) for v in g_freq]}", f"multinomial 频率={[R(v) for v in m_freq]}", "理论=[0.6, 0.3, 0.1]", "同分布、差距为采样噪声"],
    ["三行证明", "P(win_i)=∫exp(−q/p_i)dq", "蒙特卡洛 p0=0.6→" + str(R(g_freq[0])), "p1=0.3→" + str(R(g_freq[1])) + " / p2=0.1→" + str(R(g_freq[2])), "Exp 尾概率代入后积分塌缩成 p_i"],
    ["seed 覆写(行 0 有 seed)", f"两次重建同 seed 行 0 恒 {seeded_a[0]}（全局 seed 200 vs 999）", f"无 seed 行 1-5 漂移行号={drift_rows}", "len(gen)=1≠6 先批量再覆写第 0 行", "TODO(woosuk): 逐请求循环 can be slow"],
    ["每行都有 seed", f"len(gen)=2==batch 跳过批量填充", f"{both_a} == {both_b}", "整批可复现", "FlashInfer 与逐请求 generator 不兼容须回退 native（→m13）"],
    ["fp64 支路", "q float64 / probs float32", f"reciprocal+mul 与 fp64 直除逐行一致={bool(torch.equal(toks_branch, toks_direct))}", "use_fp64_gumbel 全链选项", "spec decode 的 uniform 须 fp64（→ch32）"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m10_gumbel_dice.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
