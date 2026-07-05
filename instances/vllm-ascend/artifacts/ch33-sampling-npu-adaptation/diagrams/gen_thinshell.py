#!/usr/bin/env python3
"""fig28-1: 薄壳子类化 —— 三个采样器只覆写碰 NPU 同步 / 能上 Triton 的几处，其余继承基类不动。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


text(W / 2, 42, "薄壳子类化：只覆写几处热点，其余继承基类不动", 24, "middle", "#0f172a", "bold")
text(W / 2, 70, "红 = 昇腾覆写（碰 NPU 同步 / 能上 Triton）  灰 = 继承基类、一行不改", 14, "middle", "#64748b")

# 列标
text(330, 104, "vLLM 基类", 16, "middle", "#475569", "bold")
text(980, 104, "昇腾子类（薄壳）", 16, "middle", "#b91c1c", "bold")

rows = [
    ("Sampler", "AscendSampler",
     [("sample()  温度派发→greedy/topk-topp→where", False),
      ("apply_temperature / compute_logprobs", False),
      ("gather_logprobs / forward", False),
      ("greedy_sample()", True),
      ("apply_penalties()", True),
      ("__init__  装配 AscendTopKTopPSampler", True)]),
    ("TopKTopPSampler", "AscendTopKTopPSampler",
     [("forward()  调用约定", False),
      ("forward_native()  top-k/top-p→softmax→random_sample", True)]),
    ("RejectionSampler", "AscendRejectionSampler",
     [("parse_output / compute_probs", False),
      ("apply_penalties()  /  forward()", True),
      ("prepare_sampling()  /  __init__()", True)]),
]

y = 124
for base, asc, methods in rows:
    n = len(methods)
    bh = 40 + n * 30
    # 基类框
    box(60, y, 540, bh, "#f8fafc", "#cbd5e1", 12, 1.8, dash="6 5")
    text(80, y + 28, "class " + base, 15, "start", "#475569", "bold", mono=True)
    # 子类框
    box(720, y, 540, bh, "#fff7f7", "#ef4444", 12, 1.8)
    text(740, y + 28, "class " + asc + "(" + base + ")", 14, "start", "#b91c1c", "bold", mono=True)
    for i, (m, overridden) in enumerate(methods):
        my = y + 56 + i * 30
        # 基类侧：全列出（灰）
        text(80, my, "• " + m.split("  ")[0], 12.5, "start", "#94a3b8", mono=True)
        # 子类侧
        if overridden:
            text(740, my, "▣ " + m, 12, "start", "#b91c1c", "bold")
        else:
            text(740, my, "↳ 继承：" + m.split("  ")[0], 12, "start", "#94a3b8")
    # 继承箭头
    ay = y + bh / 2
    L.append(f'<line x1="600" y1="{ay}" x2="720" y2="{ay}" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#ar)"/>')
    y += bh + 26

text(W / 2, H - 18, "偏离面越小，越容易随 vLLM 上游升级走 —— 这是「换头不换身」在采样层的复用。",
     13, "middle", "#64748b")

L.append('</svg>')
open("fig28-1-thinshell.svg", "w").write('\n'.join(L))
print("wrote fig28-1-thinshell.svg")
