"""ch33 m08 驱动脚本（Markov 头低秩转移偏置 B=W_1W_2 的参数账）。

跑法：python explainer/traces/run_m08_markov_account.py
产出：explainer/traces/run_m08_markov_account.json

素材来源：implementation/dspark.py（MarkovHead / markov_head_param_account）。
论文数字：V≈129280（DeepSeek 词表）、r=256 默认（§3.1 Eq.5）、序列循环延迟开销
0.2%~1.3%（§4.3.2，γ 4→16、batch=128、上下文 {512,1024,2048,4096} 平均）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import dspark  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731
V6 = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

P("== ch33 m08 · Markov 头低秩账：全表 V×V vs 低秩 B=W_1W_2（r=256）==")

# ── A) 真实规模参数账 ────────────────────────────────────────────────────────
P("")
P("-- [m08·A] V=129280、r=256（DeepSeek 词表 / 论文默认秩）--")
V, R = 129280, 256
acc = dspark.markov_head_param_account(V, R)
P(f"    全表 B∈R^(V×V)：V^2 = {acc['full']:.0f} 项（≈{F(acc['full'] / 1e10)}×10^10，fp16 也要 33 GB 级）")
P(f"    低秩 W_1（V×r）+ W_2（r×V）：各 V*r = {F(V * R)} 项，共 2Vr = {acc['low_rank']:.0f}（≈{F(acc['low_rank'] / 1e6)}×10^6）")
P(f"    省法 V/(2r) = 129280/512 = {F(acc['saving_factor'])}（论文口径 ≈253 倍，取整差异）")
P(f"    每步代价 = O(r) 查表 + O(rV) GEMV = {R} + {V * R} = {V * R + R} FLOPs（≈{F((V * R + R) / 1e6)}×10^6）")

# ── B) 玩具 MarkovHead：两步调用 ────────────────────────────────────────────
P("")
P("-- [m08·B] 玩具头（V=6、r=2、seed=4）：embed / bias 两步就是串行循环的全部 --")
rng = np.random.default_rng(4)
w1 = rng.normal(0, 0.5, (6, 2))
w2 = rng.normal(0, 0.5, (2, 6))
head = dspark.MarkovHead(w1, w2)
emb3 = head.embed([3])
P(f"    W_1 形状 {list(w1.shape)}、W_2 形状 {list(w2.shape)}")
P(f"    step1 embed(prev=3) = {V6(emb3)}（[B]->[B,r] 查表）")
P(f"    step2 bias(embed)   = {V6(head.bias(emb3))}（[B,r]->[B,V] = W_1[x]W_2 行向量）")
bias5 = head.bias(head.embed([5]))
P(f"    换 prev=5：bias = {V6(bias5)}——每换一个前驱 token 只换一行查表，W_2 不动")
P(f"    手工复算：bias(3)[0] = w1[3]·w2 列 0 = {F(w1[3] @ w2[:, 0])} ✓（与上行第 0 位对上）")

# ── C) 低秩不是免费——V<2r 时反而更贵（r 的选择有账） ────────────────────────
P("")
P("-- [m08·C] 反例意识：秩选大了不省钱（低秩的省法在 V>2r 才开始） --")
toy = dspark.markov_head_param_account(6, 4)
P(f"    V=6、r=4：全表 {toy['full']} 项 vs 低秩 2Vr = {toy['low_rank']} 项（V/(2r) = {F(toy['saving_factor'])} < 1，反而更贵）")
toy2 = dspark.markov_head_param_account(6, 2)
P(f"    V=6、r=2：全表 {toy2['full']} 项 vs 低秩 {toy2['low_rank']} 项（V/(2r) = {F(toy2['saving_factor'])} > 1，开始省）")
P(f"    省法在 V>2r={2 * R} 才开始（真实 V=129280 >> 512，省 {F(129280 / 512)} 倍）")

# ── D) 序列循环的延迟账（论文 §4.3.2 实测口径，echo 供图注引用） ────────────
P("")
P("-- [m08·D] 序列循环延迟开销（论文 §4.3.2 右图口径，非本脚本实测）--")
P("    batch=128、γ 从 4 到 16：序列循环给整轮延迟加 0.2%~1.3%（相对 DFlash 基线）")
P("    对照收益：γ=7 时接受长度 +16%（math）/+15%（code）/+18%（chat），γ=15 时 +30%/+26%/+22%")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {"V": V, "r": R, "toy_V": 6, "toy_r": 2, "seed": 4},
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
