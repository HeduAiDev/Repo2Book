"""ch33 m11+m12 驱动脚本（置信度头 Eq.(7)(8) + STS 校准）。

跑法：python explainer/traces/run_m11_m12_confidence.py
产出：explainer/traces/run_m11_m12_confidence.json

素材来源：implementation/dspark.py（ConfidenceHead / analytical_confidence_label）、
implementation/spec_decode.py（acceptance_rate_sum_min / analytical_acceptance_rate /
accepted / sample_recovered）、implementation/scheduler.py（ECE / 温度缩放 / STS）。
三方闭环：m02 手算接受率 ↔ Eq.(8) 解析标签 ↔ 经验接受率（同一组 (0.7,0.3)/(0.5,0.5)）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import dspark  # noqa: E402
import spec_decode as sd  # noqa: E402
import scheduler as sch  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731
V4 = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

# ══ m11 · 置信度头 ══════════════════════════════════════════════════════════
P("== ch33 m11 · 置信度头 c_k=σ(w·[h_k;W_1[x_(k-1)]])，监督=解析接受率 c*=1−½‖p_d−p_t‖_1 ==")

p_t = np.array([0.7, 0.3])
p_d = np.array([0.5, 0.5])

P("")
P("-- [m11·三方闭环] (A,B) 组：TV 恒等式两种形态 + 经验接受率 --")
sum_min = sd.acceptance_rate_sum_min(p_t, p_d)
tv_form = sd.analytical_acceptance_rate(p_t, p_d)
P(f"    Σmin(p_t,p_d) = 0.5+0.3 = {F(sum_min)}")
P(f"    1−½‖p_t−p_d‖_1 = 1−½*(0.2+0.2) = {F(tv_form)}（同一数字——TV 恒等式）")
rng = np.random.default_rng(11)
n_acc, n_tot = 0, 200000
for _ in range(n_tot):
    x = int(rng.choice(2, p=p_d))
    u = float(rng.random())
    if sd.accepted(x, p_t, p_d, u):
        n_acc += 1
P(f"    经验接受率（n={n_tot}，seed=11）= {n_acc}/{n_tot} = {F(n_acc / n_tot)}"
      f"（≈{F(sum_min)}：Eq.(8) 的监督标签就是 §2.1 接受准则的解析期望）")

P("")
P("-- [m11·三词表例] c* 再算一组（p_t=(0.5,0.4,0.1) vs p_d=(0.2,0.1,0.7)）--")
c_star3 = dspark.analytical_confidence_label(np.array([0.2, 0.1, 0.7]), np.array([0.5, 0.4, 0.1]))
P(f"    c* = 1−½*(0.3+0.3+0.6) = {F(c_star3)}（= Σmin 同值）")

P("")
P("-- [m11·头本体] 玩具置信头（d=8、r=4、seed=5）：一个 sigmoid 门 --")
rng_ch = np.random.default_rng(5)
h_k = rng_ch.normal(0, 1.0, 8)
emb_prev = rng_ch.normal(0, 1.0, 4)
w = rng_ch.normal(0, 0.5, 12)
head = dspark.ConfidenceHead(w)
z = float(np.concatenate([h_k, emb_prev]) @ w)
c_k = head.confidence(h_k, emb_prev)
P(f"    z = w·[h_k; W_1[x_(k-1)]] = {F(z)}（12 维内积：8 骨干 + 4 Markov 嵌入）")
P(f"    c_k = σ(z) = {F(c_k)}——c_k 建模的是『前缀全被接受时，位 k 存活』的条件概率")

# ══ m12 · STS 校准 ══════════════════════════════════════════════════════════
P("")
P("== ch33 m12 · STS：逐位温度缩放，让累积乘积 ∏c_i 对齐经验接受率 ==")

N, GAMMA = 4000, 3
TRUE_RATES = np.array([0.8, 0.75, 0.7])
rng = np.random.default_rng(23)
# 过自信的置信 logit：z = logit(rate) + 1.4 + 噪声（幅度≈排序正确、绝对值偏高）
conf_logits = np.log(TRUE_RATES / (1 - TRUE_RATES)) + 1.4 + rng.normal(0, 0.1, (N, GAMMA))
# 前缀存活结果（真接受过程模拟）
prefix_outcomes = np.ones((N, GAMMA), dtype=bool)
alive = np.ones(N, dtype=bool)
for k in range(GAMMA):
    acc_k = alive & (rng.random(N) < TRUE_RATES[k])
    prefix_outcomes[:, k] = acc_k
    alive = acc_k
raw_conf = 1.0 / (1.0 + np.exp(-conf_logits))
raw_cum = np.cumprod(raw_conf, axis=1)
true_a = np.cumprod(TRUE_RATES)
P(f"[m12·参数] N={N}、γ={GAMMA}、真实条件接受率 {V4(TRUE_RATES)}"
      f"（真实前缀存活 a = {V4(true_a)}）、置信偏置 +1.4（过自信）")
P("")
P("-- [m12·校准前] 原始置信度过自信：累积乘积远高于真实存活 --")
for k in range(GAMMA):
    ece_k = sch.expected_calibration_error(raw_cum[:, k], prefix_outcomes[:, k])
    P(f"    位 {k + 1}: mean ∏c = {F(raw_cum[:, k].mean())} vs 真实 a = {F(true_a[k])}"
          f"，ECE = {F(ece_k)}")

P("")
P("-- [m12·STS] 左到右逐位 1D 网格搜温度（grid 0.25..5.00 步 0.25）--")
res = sch.sequential_temperature_scaling(conf_logits, prefix_outcomes)
P(f"    逐位最优温度 T = {V4(res['temperatures'])}（T>1 = 把 sigmoid 摊平降温）")
cal_cum = res["cumulative"]
for k in range(GAMMA):
    ece_k = sch.expected_calibration_error(cal_cum[:, k], prefix_outcomes[:, k])
    P(f"    位 {k + 1}: 校准后 mean ∏c = {F(cal_cum[:, k].mean())} vs 真实 a = {F(true_a[k])}"
          f"，ECE = {F(ece_k)}（前：{F(sch.expected_calibration_error(raw_cum[:, k], prefix_outcomes[:, k]))}）")

P("")
P("-- [m12·保序] 温度缩放不扰动排序 --")
for k in range(GAMMA):
    order_raw = np.argsort(-raw_conf[:, k])
    order_cal = np.argsort(-(res["calibrated"][:, k]))
    P(f"    位 {k + 1}: argsort(原始) == argsort(校准后) : {bool((order_raw == order_cal).all())}"
          f"（σ(z/T) 对 z 单调 ⇒ 排序逐位不变）")

P("")
P("-- [m12·为什么必须校准] 调度器要绝对幅度，不是排序 --")
P(f"    未校准 ∏c 位 3 = {F(raw_cum[:, 2].mean())} 会被当成存活 {F(raw_cum[:, 2].mean())}"
      f" 去估 tau；真实只有 {F(true_a[2])}——Θ=τ·SPS(B) 的 τ 直接被高估 "
      f"{F(raw_cum[:, 2].mean() / true_a[2])} 倍")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {"m11": {"p_t": [0.7, 0.3], "p_d": [0.5, 0.5], "n_emp": n_tot, "seed": 11},
               "m12": {"N": N, "gamma": GAMMA, "true_rates": V4(TRUE_RATES),
                       "overconf_bias": 1.4, "seed": 23}},
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
