#!/usr/bin/env python3
"""piecewise 切图流水线：N 层 transformer 整图在每个 attention 处切成交替条带。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1100, 520
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke="#475569", rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{d}/>')


def text(x, y, s, size=14, anchor="middle", weight="normal", fill="#1e293b", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
             f'font-weight="{weight}" fill="{fill}" font-family="{fam}">{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color="#475569"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
             f'stroke-width="1.8" marker-end="url(#a)"/>')


text(W / 2, 38, "piecewise 切图：一张 FX 整图在每个 attention 处被切开", 19, weight="bold", fill="#0f172a")

# 整图条
box(120, 60, W - 240, 38, "#e2e8f0")
text(W / 2, 84, "Dynamo 产出的整张 FX 图（N 层 transformer，含 N 个 unified_attention_with_output 节点）", 14, fill="#334155")
arrow(W / 2, 98, W / 2, 124)
text(W / 2 + 175, 116, "split_graph(graph, splitting_ops)", 13, mono=True, fill="#475569")

# 交替条带
y0 = 138
bh = 150
items = [
    ("submod_0", ["norm + qkv_proj", "+ rope"], True),
    ("attn_0", ["unified_attention", "_with_output"], False),
    ("submod_1", ["o_proj + norm", "+ mlp"], True),
    ("attn_1", ["unified_attention", "_with_output"], False),
    ("submod_2", ["o_proj + norm", "+ mlp"], True),
]
n = len(items)
gap = 14
total_w = W - 160
bw = (total_w - gap * (n - 1)) / n
x = 80
centers = []
for name, lines, compiled in items:
    if compiled:
        box(x, y0, bw, bh, "#dcfce7", "#16a34a")
        text(x + bw / 2, y0 + 26, name, 14, mono=True, weight="bold", fill="#166534")
    else:
        box(x, y0, bw, bh, "#fee2e2", "#dc2626", dash="6 4")
        text(x + bw / 2, y0 + 26, name, 14, mono=True, weight="bold", fill="#991b1b")
    yy = y0 + 52
    for ln in lines:
        text(x + bw / 2, yy, ln, 12, mono=True, fill="#334155")
        yy += 18
    if compiled:
        text(x + bw / 2, y0 + bh - 44, "Inductor 融合", 12, fill="#166534")
        text(x + bw / 2, y0 + bh - 26, "+ CUDA graph 重放", 12, fill="#166534")
        text(x + bw / 2, y0 + bh - 8, "CPU 开销 ≈ 0", 12, weight="bold", fill="#166534")
    else:
        text(x + bw / 2, y0 + bh - 26, "保持 eager", 12, fill="#991b1b")
        text(x + bw / 2, y0 + bh - 8, "跑真实 attn kernel", 12, weight="bold", fill="#991b1b")
    centers.append(x + bw / 2)
    x += bw + gap

# 顺序箭头
for i in range(n - 1):
    arrow(centers[i] + bw / 2 - 4, y0 + bh / 2, centers[i + 1] - bw / 2 + 2, y0 + bh / 2)

# 图例
ly = y0 + bh + 40
box(80, ly, 22, 16, "#dcfce7", "#16a34a")
text(112, ly + 13, "规整段（编译 + CUDA graph）", 13, anchor="start", fill="#334155")
box(420, ly, 22, 16, "#fee2e2", "#dc2626", dash="6 4")
text(452, ly + 13, "切分算子段（eager）", 13, anchor="start", fill="#334155")

# 量化小注
box(80, ly + 36, W - 160, 64, "#f8fafc", "#cbd5e1")
text(W / 2, ly + 60, "切点数 ≈ attention 层数 N；子图数 ≈ 2N+1。", 14, weight="bold", fill="#0f172a")
text(W / 2, ly + 84, "整图若因 attention 无法 CUDA-graph，每层每算子都付 CPU launch 开销；切图后只剩 N 段 attention eager。", 13, fill="#334155")

L.append('</svg>')
with open("piecewise-split.svg", "w", encoding="utf-8") as f:
    f.write('\n'.join(L))
print("wrote piecewise-split.svg")
