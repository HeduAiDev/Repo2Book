"""ch33 m13+m14+m15 驱动脚本（前缀存活链 + 硬件感知调度 + 非前瞻反例）。

跑法：python explainer/traces/run_m13_m14_m15_scheduler.py
产出：explainer/traces/run_m13_m14_m15_scheduler.json

素材来源：implementation/scheduler.py（hardware_aware_prefix_scheduler /
retrospective_global_search / retrospective_output_distribution）、
implementation/acceptance_length.py、implementation/spec_decode.py。
论文数字：§3.2.2 Algorithm 1 与 Appendix A（SPS(1)=1.0/SPS(2)=0.5/SPS(3)=0.45、
a_1=0.8、c_2 高 0.9 / 低 0、P(Y=A)=0.85）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import scheduler as sch  # noqa: E402
import acceptance_length as al  # noqa: E402
import spec_decode as sd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731

# ══ m13 · 前缀存活链式法则 ══════════════════════════════════════════════════
P("== ch33 m13 · 前缀存活 a_{r,j}=∏_{i<=j}c_{r,i}：条件⇄无条件互推 ==")
c = [0.8, 0.9, 0.5]
a = al.conditional_to_unconditional_rates(c)
back = al.unconditional_to_conditional_rates(a)
P(f"    条件置信 c = {c}")
P(f"    前缀存活 a = cumprod = {[F(v) for v in a]}（0.8、0.8*0.9=0.72、0.72*0.5=0.36）")
P(f"    逆向 c_i = p_i/p_(i-1)（p_0=1）= {[F(v) for v in back]}（vLLM synthetic 模式同构）")
mono = all(a[i] >= a[i + 1] - 1e-12 for i in range(len(a) - 1))
P(f"    a 单调不增：{mono}——全局按 a 降序排序天然满足『扩展位 j 前必含位 j-1』（前缀依赖）")

# ══ m14 · 硬件感知前缀调度器 ════════════════════════════════════════════════
P("")
P("== ch33 m14 · Algorithm 1：Θ=τ*·SPS(B) 贪心准入 + 早停 ==")

P("")
P("-- [m14·A] Appendix A 场景（R=1、γ=2、SPS: 1→1.0 / 2→0.5 / 3→0.45）--")
sps_a = {1: 1.0, 2: 0.5, 3: 0.45}
res_a = sch.hardware_aware_prefix_scheduler([[0.8, 0.9]], lambda b: sps_a[int(b)])
for row in res_a["trace"]:
    if row["event"] == "init":
        P(f"    init: B={row['B']}  tau*={F(row['tau'])}  Θ={F(row['theta'])}（= R·SPS(R) = 1*1.0）")
    elif row["event"] == "admit":
        P(f"    admit (req{row['req']}, 位{row['pos']}): a={F(row['a'])}  B={row['B']}  "
              f"tau*={F(row['tau'])}  Θ={F(row['theta'])}（= {F(row['tau'])}*{sps_a[row['B']]}）")
    else:
        P(f"    break：Θ 不再超过 Θ_best -> 早停（非前瞻）")
P(f"    返回 ℓ* = {res_a['ell_star']}，Θ_best = {F(res_a['theta_best'])}"
      f"——本例最佳是『一个 draft 都不验』（重载侧：验证挤占批容量不如不验）")

P("")
P("-- [m14·B] 双请求场景（R=2、γ=2、SPS: 2→8.0 / 3→6.0 / 4→4.8 / 5→4.0 / 6→3.5）--")
sps_b = {2: 8.0, 3: 6.0, 4: 4.8, 5: 4.0, 6: 3.5}
conf_b = [[0.9, 0.7], [0.5, 0.3]]
res_b = sch.hardware_aware_prefix_scheduler(conf_b, lambda b: sps_b[int(b)])
for row in res_b["trace"]:
    if row["event"] == "init":
        P(f"    init: B={row['B']}  tau*={F(row['tau'])}  Θ={F(row['theta'])}（= 2*8.0）")
    elif row["event"] == "admit":
        P(f"    admit (req{row['req']}, 位{row['pos']}): a={F(row['a'])}  B={row['B']}  "
              f"tau*={F(row['tau'])}  Θ={F(row['theta'])}（= {F(row['tau'])}*{sps_b[row['B']]}）")
    else:
        P(f"    break（Θ 回落）")
P(f"    返回 ℓ* = {res_b['ell_star']}，Θ_best = {F(res_b['theta_best'])}")
P(f"    无早停继续评估（离线全枚举对照）：")
# 早停发生在 admit(r0,位2) 之后（该步 Θ 已回落）；继续收 (r1,位1)、(r1,位2)：
tau2, B2 = 2.9 + 0.63, 4
for (a_v, nb) in ((0.5, 5), (0.15, 6)):
    tau2, B2 = tau2 + a_v, nb
    P(f"      admit 后 B={B2} tau*={F(tau2)} Θ={F(tau2 * sps_b[B2])}")
P(f"    全局最大仍是 Θ={F(res_b['theta_best'])}——Θ 单峰时早停=全局最优 ✓")

P("")
P("-- [m14·C] 锯齿 SPS：早停被悬崖坑掉的例子（§5.2 生产版去早停的动机）--")
sps_c = {2: 8.0, 3: 6.0, 4: 4.8, 5: 4.6, 6: 4.5}
res_c = sch.hardware_aware_prefix_scheduler(conf_b, lambda b: sps_c[int(b)])
for row in res_c["trace"]:
    if row["event"] == "init":
        P(f"    init: B={row['B']}  tau*={F(row['tau'])}  Θ={F(row['theta'])}")
    elif row["event"] == "admit":
        P(f"    admit (req{row['req']}, 位{row['pos']}): a={F(row['a'])}  B={row['B']}  "
              f"tau*={F(row['tau'])}  Θ={F(row['theta'])}（= {F(row['tau'])}*{sps_c[row['B']]}）")
    else:
        P(f"    break（Θ 回落即停）")
P(f"    早停版 ℓ* = {res_c['ell_star']}，Θ_best = {F(res_c['theta_best'])}")
tau2, B2 = 2.9 + 0.63, 4
for (a_v, bb) in ((0.5, 5), (0.15, 6)):
    tau2, B2 = tau2 + a_v, bb
    P(f"    不早停继续：admit 后 B={B2} tau*={F(tau2)} Θ={F(tau2 * sps_c[B2])}")
P(f"    全局搜索找到 Θ={F((2.9 + 0.63 + 0.5 + 0.15) * sps_c[6])} > 早停版 {F(res_c['theta_best'])}"
      f"——悬崖后 SPS 回稳，早停在局部最优被坑（生产异步版靠两步前预测做无约束搜索）")

# ══ m15 · 非前瞻反例 ════════════════════════════════════════════════════════
P("")
P("== ch33 m15 · 回顾式调度的选择偏差（Appendix A：0.85/0.15 ≠ 0.7/0.3）==")

p_t, p_d = np.array([0.7, 0.3]), np.array([0.5, 0.5])
a1 = sd.acceptance_rate_sum_min(p_t, p_d)
C2_HIGH, C2_LOW = 0.9, 0.0
P(f"    a_1 = Σmin(p_t,p_d) = {F(a1)}（论文假设值 0.8 与 TV 恒等式算出的值一致）")

P("")
P("-- [m15·无早停全局搜索] ℓ 的选择依赖 x_1 自己的实现 --")
ell_high, thetas_high = sch.retrospective_global_search(a1, C2_HIGH, lambda b: sps_a[int(b)])
P(f"    x_1=A（高延续置信 c_2={C2_HIGH}）：Θ = [1.0*1.0, 1.8*0.5, {F(1 + a1 + a1 * C2_HIGH)}*0.45] "
      f"= {[F(t) for t in thetas_high]} -> ℓ={ell_high}（准入 x_1）")
ell_low, thetas_low = sch.retrospective_global_search(a1, C2_LOW, lambda b: sps_a[int(b)])
P(f"    x_1=B（低延续置信 c_2={C2_LOW}）：Θ = [1.0*1.0, 1.8*0.5, {F(1 + a1 + a1 * C2_LOW)}*0.45] "
      f"= {[F(t) for t in thetas_low]} -> ℓ={ell_low}（不准入 x_1）")
P(f"    x_1 的准入取决于 x_1 自己——Markov 置信头算 a_2 需要已实例化的 x_1，回顾式搜索把 x_1 泄露进决策")

P("")
P("-- [m15·分布被改写] P(Y=A) = P(x_1=A)*1 + P(x_1=B)*p_t(A) = 0.5 + 0.5*0.7 = 0.85 --")
emp_bias = sch.retrospective_output_distribution(p_t, p_d, C2_HIGH, C2_LOW,
                                                 lambda b: sps_a[int(b)], 200000,
                                                 np.random.default_rng(13))
P(f"    经验模拟（n=200000，seed=13）：P(Y=A) = {F(emp_bias[0])}，P(Y=B) = {F(emp_bias[1])}"
      f"（≠ p_t = (0.7, 0.3)：不无损）")

P("")
P("-- [m15·早停版对照] Algorithm 1 在同场景：Θ_1=0.9 < Θ_0=1.0 -> ℓ=0 恒成立 --")
rng_c = np.random.default_rng(13)
counts = np.zeros(2)
for _ in range(200000):
    y = int(rng_c.choice(2, p=p_t))  # ℓ=0：target 直接重采
    counts[y] += 1
P(f"    输出分布 = p_t 本身：P(Y=A) = {F(counts[0] / 200000)}，P(Y=B) = {F(counts[1] / 200000)}"
      f"——早停把截断决策限制在截至当前步的前缀上，准入事件与未来 token 隔离（保无损）")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {"m13": {"c": c},
               "m14": {"sps_a": sps_a, "conf_a": [[0.8, 0.9]],
                       "sps_b": sps_b, "conf_b": conf_b, "sps_c": sps_c},
               "m15": {"p_t": [0.7, 0.3], "p_d": [0.5, 0.5], "a1": F(a1),
                       "c2_high": C2_HIGH, "c2_low": C2_LOW, "n": 200000, "seed": 13}},
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
