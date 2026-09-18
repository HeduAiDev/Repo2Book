"""ch33 m04 驱动脚本（期望接受长度 τ：前缀存活的链式几何账）。

跑法：python explainer/traces/run_m04_tau.py
产出：explainer/traces/run_m04_tau.json

素材来源：implementation/acceptance_length.py（prefix_survival_probs /
expected_accepted_length(_constant) / unconditional_to_conditional_rates /
simulate_accepted_length）。论文口径：§2.1 Eq.(1) 分母、§4.2 Table 1+脚注 4（含
bonus）、§4.3.1 Fig.2 的逐位条件接受率端点（Chat：EAGLE3 0.53→0.74 回升、
DFlash 0.72→0.63 衰减——中间位线性插值是本脚本的示教构造，端点为论文数）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import acceptance_length as al  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731

P("== ch33 m04 · 期望接受长度 tau：E[tau]=1+Σ_k ∏_{i<=k} α_i（§4.2 fn.4 含 bonus 口径）==")

# ── A) 恒 α 闭式 ────────────────────────────────────────────────────────────
P("")
P("-- [m04·A] 恒定 α=0.8、γ=3：链式账 vs 闭式 vs 蒙特卡洛 --")
ALPHA, GAMMA = 0.8, 3
cond = [ALPHA] * GAMMA
a = al.prefix_survival_probs(cond)
P(f"    条件率 α = {cond}")
P(f"    前缀存活 a_k = ∏α = {[F(v) for v in a]}（0.8, 0.8*0.8, 0.8*0.8*0.8）")
e_chain = al.expected_accepted_length(cond)
e_closed = al.expected_accepted_length_constant(ALPHA, GAMMA)
P(f"    E[tau] = 1 + 0.8 + 0.64 + 0.512 = {F(e_chain)}")
P(f"    闭式 (1-α^4)/(1-α) = (1-0.4096)/0.2 = {F(e_closed)}（两式相等 ✓）")
mc = al.simulate_accepted_length(cond, 30000, np.random.default_rng(5))
P(f"    Monte Carlo 30000 周期（seed=5）：mean tau = {F(mc)}")
e_one = al.expected_accepted_length_constant(1.0, GAMMA)
P(f"    对照 α=1（draft≡target）：E[tau] = γ+1 = {F(e_one)}（全收+bonus 上限）")

# ── B) 谱系杠杆：Fig.2 端点构成的两种 α 曲线 ────────────────────────────────
P("")
P("-- [m04·B] 谱系杠杆（γ=4）：EAGLE3 Chat 回升曲线 vs DFlash Chat 衰减曲线 --")
P("    （端点取论文 §4.3.1 Fig.2：EAGLE3 Chat 0.53→0.74、DFlash Chat 0.72→0.63；中间位线性插值）")
eagle = [0.53, 0.60, 0.67, 0.74]
dflash = [0.72, 0.69, 0.66, 0.63]
for name, prof in (("EAGLE3（浅自回归·回升）", eagle), ("DFlash（深并行·衰减）", dflash)):
    a = al.prefix_survival_probs(prof)
    e = al.expected_accepted_length(prof)
    P(f"    {name}: α = {prof}")
    P(f"        a = {[F(v) for v in a]}，E[tau] = 1 + Σa = {F(e)}")
eagle_swap = [0.72, 0.60, 0.67, 0.74]  # 只把位 1 换成 DFlash 的 0.72
a_sw = al.prefix_survival_probs(eagle_swap)
e_sw = al.expected_accepted_length(eagle_swap)
P(f"    位 1 杠杆单独量：[0.53,...]→[0.72,...]（其余三位不动）")
P(f"        E[tau] {F(al.expected_accepted_length(eagle))} -> {F(e_sw)}"
      f"（+{F(e_sw - al.expected_accepted_length(eagle))}——首位每 +0.19 的接受率，"
      f"通过前缀生存被后面每一位复利放大）")

# ── C) 条件率 ⇄ 无条件率互推 ────────────────────────────────────────────────
P("")
P("-- [m04·C] 条件率⇄无条件率互推（vLLM unconditional_to_conditional_rates 同构）--")
cond_rates = [0.8, 0.9, 0.5]
uncond = al.conditional_to_unconditional_rates(cond_rates)
back = al.unconditional_to_conditional_rates(uncond)
P(f"    条件率 c = {cond_rates}")
P(f"    前缀存活 a = cumprod = {[F(v) for v in uncond]}（0.8, 0.8*0.9, 0.72*0.5）")
P(f"    逆向 c_i = p_i/p_(i-1)（p_0=1）= {[F(v) for v in back]}（往返恒等 ✓）")
P(f"    a 单调不增：{'YES' if all(uncond[i] >= uncond[i+1] - 1e-12 for i in range(2)) else 'NO'}"
      f"——全局按 a 排序天然尊重块内前缀依赖")
guard_in = [0.5, 0.0, 0.3]
guard_out = al.unconditional_to_conditional_rates(guard_in)
P(f"    除零守卫：a = {guard_in} -> c = {[F(v) for v in guard_out]}（前位 0 -> 该位 0，"
      f"vLLM utils.py 同款）")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {"alpha": ALPHA, "gamma": GAMMA, "mc_cycles": 30000, "mc_seed": 5,
               "eagle_chat_profile": eagle, "dflash_chat_profile": dflash,
               "profile_note": "端点=论文 Fig.2（§4.3.1），中间位线性插值为示教构造"},
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
