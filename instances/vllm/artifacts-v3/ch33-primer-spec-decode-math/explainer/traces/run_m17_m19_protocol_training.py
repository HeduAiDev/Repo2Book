"""ch33 m17+m19 驱动脚本（greedy one-hot 退化 + 训练三件套示教轨迹）。

跑法：python explainer/traces/run_m17_m19_protocol_training.py
产出：explainer/traces/run_m17_m19_protocol_training.json

素材来源：implementation/spec_decode.py（verify_block_greedy / generate_stream /
acceptance_rate_sum_min）、implementation/dspark.py（sample_sequential）、
implementation/training.py（三件套 + train_toy_drafter）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import spec_decode as sd  # noqa: E402
import training as tr  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731
V4 = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

# ══ m17 · draft 恒 greedy 为何无损 + probabilistic 选项 ═════════════════════
P("== ch33 m17 · draft 恒 greedy：one-hot q 的退化 + probabilistic 的显式 q ==")

p = np.array([0.40, 0.30, 0.20, 0.10])
q = np.array([0.35, 0.25, 0.25, 0.15])
onehot = np.array([1.0, 0.0, 0.0, 0.0])
P(f"[m17·参数] 4 词表玩具：p_t = {V4(p)}、draft 全分布 q = {V4(q)}、greedy draft 的 one-hot = (1, 0, 0, 0)")

P("")
P("-- [m17·接受率对比] greedy 接受率低、但不损分布 --")
P(f"    Σmin(p_t, q)        = 0.35+0.25+0.2+0.1 = {F(sd.acceptance_rate_sum_min(p, q))}（probabilistic）")
P(f"    Σmin(p_t, one-hot)  = 0.4+0+0+0         = {F(sd.acceptance_rate_sum_min(p, onehot))}（greedy）")

P("")
P("-- [m17·情形一：draft greedy（q=one-hot）、target 温度采样 --")
P(f"    draft 恒提 x=0（argmax q），接受概率 = min(1, p(0)/q(0)) = min(1, 0.4/1) = {F(min(1.0, p[0] / 1.0))}")
P(f"    x=1..3 的 q(x)=0——永不被提出；kernel 里 draft_prob=0 直接拒（vLLM L828 除零守卫）")
res = sd.residual_distribution(p, onehot)
P(f"    拒绝质量 = Σmax(0,q−p) = {F(np.maximum(onehot - p, 0).sum())}；残差 = norm(max(0, p−q)) "
      f"= {V4(res / res.sum())}（0 位已扣干净）")
P(f"    P(y=0) = 1*0.4 + 0.6*0 = 0.4 = p(0) ✓（无损不靠 draft 准，靠残差兜底）")

P("")
P("-- [m17·情形二：target greedy（温度 0，vLLM V2 greedy 分支 L564-L586 的形态）--")
p_onehot = np.array([1.0, 0.0, 0.0, 0.0])  # target 分布退化为 argmax 的 one-hot
P(f"    p_t 退化为 one-hot(0) 后：draft 提 0 -> p(0)/q(0) 恒 >= 1 -> 接受；")
P(f"    draft 提 x!=0 -> p(x)=0 -> 恒拒 -> 恢复残差 norm(max(0,p−q)) 质量全在 0 位 -> 输出 0")
P(f"    两种情况合起来：接受 ⟺ draft == target argmax，输出 ≡ target greedy rollout（见下实跑）")

P("")
P("-- [m17·greedy 整流流] 与 target greedy rollout 逐位相同（bigram target、400 token）--")
def target_cond(prev):
    rows = {
        0: [0.60, 0.25, 0.10, 0.05],
        1: [0.10, 0.55, 0.25, 0.10],
        2: [0.05, 0.20, 0.60, 0.15],
        3: [0.25, 0.10, 0.15, 0.50],
    }
    return np.array(rows[prev])


def greedy_drafter(prev):
    g = int(np.argmax(target_cond(prev)))
    return [g] * 3, None


def target_greedy_rollout(anchor, n):
    out, prev = [], anchor
    for _ in range(n):
        t = int(np.argmax(target_cond(prev)))
        out.append(t)
        prev = t
    return out


stream, stats = sd.generate_stream(target_cond, greedy_drafter, anchor=0,
                                   n_tokens=400, rng=np.random.default_rng(3), greedy=True)
ref = target_greedy_rollout(0, 400)
P(f"    逐位相同 = {'YES' if stream == ref else 'NO'}——greedy 只影响接受率 "
      f"（mean_accepted/cycle = {F(stats['mean_accepted_per_cycle'])}），不影响输出分布")

P("")
P("-- [m17·probabilistic 的 q(x)] 序列采样逐位写 q 行（验证概率比测试用）--")
import dspark  # noqa: E402
rng_mk = np.random.default_rng(4)
markov = dspark.MarkovHead(rng_mk.normal(0, 0.5, (6, 4)), rng_mk.normal(0, 0.5, (4, 6)))
U = np.array([[2.0, 1.0, 0.5, 0.0, -0.5, -1.0],
              [0.5, 2.0, 1.0, 0.0, -0.5, -1.0]])
toks, dists, _ = dspark.sample_sequential(U, 1, markov, rng=np.random.default_rng(9), greedy=False)
for k, p_k in enumerate(dists):
    P(f"    位 {k}: q(x) = {V4(p_k)}（= vLLM probabilistic 模式写进 draft_logits[:,{k}] 的行）")

# ══ m19 · 训练三件套 ════════════════════════════════════════════════════════
P("")
P("== ch33 m19 · 训练三件套 L_ce/L_tv/L_conf + w_k=exp(−(k−1)/γ)（Eq.9-12）==")

GAMMA_T, V_T = 2, 4
target_probs = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.5, 0.3, 0.1]])
w = tr.position_weights(GAMMA_T)
P(f"[m19·参数] γ={GAMMA_T}、V={V_T}、target 冻结 = 行0 {V4(target_probs[0])} / 行1 {V4(target_probs[1])}")
P(f"    位置权重 w_k = exp(−(k−1)/γ) = {V4(w)}（位 1 权重 1，位 2 衰减到 {F(w[1])}）")
P(f"    总目标权重 α = (0.1, 0.9, 1.0)（L = 0.1*L_ce + 0.9*L_tv + 1.0*L_conf）")

P("")
P("-- [m19·损失轨迹] 有限差分梯度下降 150 步（lr=0.5、seed=0、每步从 target 采一个块）--")
P("    （注：L_ce 的 ground-truth 块每步重采，单步 loss 有抖动——看初态 vs 末态的总趋势）")
trace = tr.train_toy_drafter(target_probs, n_steps=151, lr=0.5,
                             rng=np.random.default_rng(0))
for step in (0, 30, 60, 90, 150):
    row = trace[step]
    P(f"    step {step:>3}: loss={F(row['loss']):>8}  L_tv={F(row['tv']):>8}  "
          f"接受率={V4(row['accept_rates'])}  E[tau]={F(row['expected_tau'])}")
P(f"    E[tau] 上限（γ=2 全对齐）：1+1+1 = 3.0；step150 已到 {F(trace[150]['expected_tau'])}——"
      f"L_tv 从 {F(trace[0]['tv'])} 降到 {F(trace[150]['tv'])}、E[tau] 从 {F(trace[0]['expected_tau'])} 升到 "
      f"{F(trace[150]['expected_tau'])}：最小化 L_tv 直接最大化期望接受率（Eq.10 下文断言的可运行版）")

P("")
P("-- [m19·为什么 α_tv=0.9 >> α_ce=0.1] 接受率只看 TV 距离 --")
P(f"    初始接受率 {V4(trace[0]['accept_rates'])}（draft 随机初始化 vs target）")
P(f"    → 末态接受率 {V4(trace[150]['accept_rates'])}：L_tv 把 draft 分布拉向 target，"
      f"『猜对答案』（L_ce）只是辅助")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {
        "m17": {"p_t": V4(p), "q": V4(q), "n_stream": 400, "seed": 3},
        "m19": {"gamma": GAMMA_T, "V": V_T, "n_steps": 151, "lr": 0.5, "seed": 0,
                "alphas": [0.1, 0.9, 1.0]},
    },
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
