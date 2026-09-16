"""ch28 m10（RoPE 只作用最后 64 维 + 输出侧按 −i 反旋）/ m11（attention sink）驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m10_m11_position.py
产出：explainer/traces/run_m10_m11_position.json（params + raw_stdout）。

素材来源：implementation/windowed_attention.py（论文忠实的小型参考实现）。
  A) m10：正向 RoPE 与反向 RoPE 只差两个符号；反旋后逐位回到原值；压缩条目带**块级位置** i·m。
     θ 口径：真实 rope_dim = 64（论文：RoPE 只作用最后 64 维）、theta = 10000；玩具用 rope_dim = 4
     让读者能自己按 θ_j = θ^{-2j/rope_dim} 算频率。
  B) m11：Exp(z'_h) 只进 softmax 分母 ⇒ 权重和 < 1（sink 拿走的份额）；sink 很负时总质量趋近 0。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import windowed_attention as wa  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

P("== ch28 m10/m11 · 位置细节（逆 RoPE）与 attention sink ==")

# ── A) m10：正向 / 反向 RoPE ──────────────────────────────────────────────
P("")
P("── A) m10：RoPE 只动最后 rope_dim 维；输出侧按 position −i 反旋 ──")
ROPE_DIM, THETA, POS = 4, 10000.0, 3
x = np.array([9.0, 9.0, 1.0, 2.0, 3.0, 4.0])   # 前 2 维 = NoPE 段（真实是前 448 维），后 4 维 = RoPE 段
freq = wa.rope_inv_freq(ROPE_DIM, THETA)
P(f"    rope_dim = {ROPE_DIM}（真实 64）、theta = {THETA}、position = {POS}；输入 x = {R(x)}")
P(f"    频率表 θ_j = theta^(-2j/rope_dim), j=0..{ROPE_DIM // 2 - 1} = {R(freq)}（θ_0 = 1、θ_1 = {round(float(freq[1]), 6)}）")
cos_v, sin_v = np.cos(POS * freq), np.sin(POS * freq)
P(f"    角度 = position × θ_j = {R(POS * freq)}；cos = {R(cos_v)}；sin = {R(sin_v)}")
fwd = wa.rope_forward(x, POS, ROPE_DIM, THETA)
P(f"    正向（偶数位 x·cos − partner·sin、奇数位 x·cos + partner·sin）x' = {R(fwd)}")
P(f"    前 2 维（NoPE 段）原样不动：{R(fwd[:2])} == {R(x[:2])} -> {bool(np.allclose(fwd[:2], x[:2]))}")
back = wa.rope_inverse(fwd, POS, ROPE_DIM, THETA)
P(f"    反向（偶数位 x·cos + partner·sin、奇数位 x·cos − partner·sin）x'' = {R(back)}")
P(f"    反旋后逐位回到原值：max|x'' − x| = {np.abs(back - x).max():.8f}")
P(f"    单看一对（位置 {2}、{3} 维）：x = [{x[2]}, {x[3]}] -> 正向 {R(fwd[2:4])} -> 反向 {R(back[2:4])}"
  f"（只差 partner 项的符号：正向 −sin/+sin、反向 +sin/−sin）")
P(f"    若不做反旋：压缩条目带的是**块级位置**（条目代表整块，位置取 i×m），query 带自己的位置——")
P(f"    两个位置对不上就对不齐；反旋把『条目的位置』换成『query 的位置』再进输出侧投影。")
P(f"    条目的块级位置（m=4、前 3 条）= {wa.compressed_entry_positions(3, 4).tolist()}"
  f"（pin 注释口径 (positions // compress_ratio) * compress_ratio）")
P(f"    对照：query 在前 12 个 token 里的位置 = {list(range(12))}；条目位置只有 {wa.compressed_entry_positions(3, 4).tolist()} 三个值")

# ── B) m11：attention sink ────────────────────────────────────────────────
P("")
P("── B) m11：Exp(z'_h) 只加到 softmax 的分母上（§2.3.3） ──")
logits = np.array([2.0, 1.0])
for sink in (None, 1.5, 0.0, -2.0, 10.0, 20.0):
    probs, mass = wa.softmax_with_sink(logits, sink)
    sink_s = "None(-inf)" if sink is None else sink
    P(f"    logits = {R(logits)}、sink = {sink_s}: 权重 = {R(probs)}、权重和(总质量) = {round(float(mass), 6)}"
      f"、sink 拿走 = {round(1 - float(mass), 6)}")
P(f"    方向记牢（最容易记反的一格）：sink 越大 ⇒ 总质量越小。sink 远大于 logits 时总质量趋近 0：")
for sink in (10.0, 20.0):
    probs, mass = wa.softmax_with_sink(logits, sink)
    P(f"      sink = {sink}: 总质量 = {mass:.8f}、权重 = {[f'{v:.8f}' for v in probs]}（这一头几乎不看任何东西）")
P(f"    sink → −inf 的那一端是另一个极限：总质量 → 1（= 没有 sink 的普通 softmax）——sink = -20 时总质量 = "
  f"{wa.softmax_with_sink(logits, -20.0)[1]:.8f}")
P(f"    分母 = Σ_j exp(z_j − m) + exp(sink − m)：分子里**没有** sink 对应的 value 项 ⇒ 权重和 < 1")
P(f"    ⇒ 每个头可以把『这台 query 投入多少注意力』调成不等于 1、甚至接近 0（论文原话的算术形态）")
P(f"    sink = None 或 −inf ⇒ exp(−inf) = 0 ⇒ 退化成普通 softmax（权重和 = 1.0）——pin 的 attn_sink 初值是 −inf，"
  f"等价『无 sink』；参考实现初值是 0，等价『有 sink』。有没有真值取决于检查点权重，不是语义。")
P("")
P("    与普通 softmax 的同台对照（同一个 logits、只有 sink 不同）：")
for sink in (None, 1.5):
    probs, mass = wa.softmax_with_sink(logits, sink)
    P(f"      sink={'None' if sink is None else sink}: 相对比例 = {R(probs / probs.sum())}（比例不变）、"
      f"绝对份额 = {R(probs)}（sink 一进来就整体缩水到 {round(float(mass), 6)}）")
kv = np.array([[1.0, 0.0], [0.0, 1.0]])
q1 = np.array([[1.0, 0.0]])
for sink in (None, 0.5):
    o, w = wa.core_attention_mqa(q1, kv, sink=sink)
    P(f"      核内落法（CoreAttn 带 sink={sink}）：权重 = {R(w)}、o = {R(o)} ⇒ 输出的**方向**由比例决定，"
      f"**幅度**被总质量缩放")
P(f"    attn_sink 是每层一个 fp32 参数（pin: 按 padded_heads 初始化 −inf、装载只填前 n_local_heads 槽）")

out = {
    "script": "run_m10_m11_position.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"rope_dim_toy": ROPE_DIM, "rope_dim_real": 64, "theta": THETA, "position": POS, "m": 4},
    "results": {
        "freq": freq.tolist(),
        "x": x.tolist(),
        "x_forward": fwd.tolist(),
        "x_inverse": back.tolist(),
        "roundtrip_maxdiff": float(np.abs(back - x).max()),
        "entry_positions": wa.compressed_entry_positions(3, 4).tolist(),
        "sink_table": [
            {"sink": s, "probs": wa.softmax_with_sink(logits, s)[0].tolist(),
             "mass": wa.softmax_with_sink(logits, s)[1]}
            for s in (None, 1.5, 0.0, -2.0, 10.0, 20.0, -20.0)
        ],
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
