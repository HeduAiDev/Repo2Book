#!/usr/bin/env python3
"""layout 模板改造:compress_ratios 逐层开关表,把 CSA(4)与 HCA(128)交错堆叠,
末尾附一层 0(稠密/未指定,如 MTP 层)。颜色即语义,配图例说明每个整数挂载的子模块。
数字来自 vllm_ascend/utils.py:L105-L110 get_dsv4_compress_ratio。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "compress_ratios:逐层开关表,CSA(4)与 HCA(128)交错互补"
SUBTITLE = "示意重复单元 [4,4,4,128] x 9(共 36 层)+ 末层 0(稠密,如 MTP);每层读一个整数决定挂什么子模块"

UNIT = [4, 4, 4, 4, 128] * 2 + [0]
LAYER_LABELS = [f"L{i}" for i in range(1, 10)] + ["…", "L36"]
VALUES = [4, 4, 4, 4, 128, 4, 4, 4, 4, 128, 0]
LABELS_SHOW = [f"L{i}" for i in range(1, 5)] + ["L5"] + [f"L{i}" for i in range(6, 10)] + ["L10"] + ["Lₙ"]
# 简化为 11 格:代表性重复单元(4,4,4,4,128 两轮)+ 末层 0
COLORS = {4: ("#059669", "#065f46", "CSA"), 128: ("#7c3aed", "#4c1d95", "HCA"), 0: ("#94a3b8", "#475569", "稠密/未指定")}

CELL, GAP, PAD, TOP = 96, 10, 40, 130
n = len(VALUES)
w = PAD * 2 + n * (CELL + GAP) - GAP
h = TOP + CELL + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for i, val in enumerate(VALUES):
    x = PAD + i * (CELL + GAP)
    fill, stroke, tag = COLORS[val]
    is_ellipsis = False
    L.append(f'<text x="{x+CELL/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(LABELS_SHOW[i])}</text>')
    L.append(f'<rect x="{x}" y="{TOP}" width="{CELL}" height="{CELL}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+CELL/2-4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="18" font-weight="bold" '
              f'fill="white">{val}</text>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+CELL/2+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#e2e8f0">{esc(tag)}</text>')

# "两轮重复" 标注花括号
rep1_x0, rep1_x1 = PAD, PAD + 5*(CELL+GAP) - GAP
rep2_x0, rep2_x1 = PAD + 5*(CELL+GAP), PAD + 10*(CELL+GAP) - GAP
brace_y = TOP + CELL + 20
for (x0, x1, label) in [(rep1_x0, rep1_x1, "重复单元 #1"), (rep2_x0, rep2_x1, "重复单元 #2")]:
    L.append(f'<path d="M {x0} {brace_y} L {x0} {brace_y+8} L {x1} {brace_y+8} L {x1} {brace_y}" '
              'fill="none" stroke="#1e40af" stroke-width="2"/>')
    L.append(f'<text x="{(x0+x1)/2}" y="{brace_y+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="#1e40af">{esc(label)}</text>')

# 图例
ly = brace_y + 50
LEGEND = [(4, "4 = CSA:挂 Compressor + Indexer(压序列长+top-k稀疏)"),
          (128, "128 = HCA:只挂 Compressor(压序列长+稠密)"),
          (0, "0/≤1 = 未指定,走稠密 SWA(如 MTP 层)")]
for j, (key, label) in enumerate(LEGEND):
    lx = PAD
    lyy = ly + j * 24
    fill, stroke, _ = COLORS[key]
    L.append(f'<rect x="{lx}" y="{lyy}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+24}" y="{lyy+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')

# 右侧互补说明框
box_x = PAD + 460
box_y = ly - 6
box_w = w - box_x - PAD
if box_w > 200:
    L.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="86" rx="6" '
              'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>')
    L.append(f'<text x="{box_x+14}" y="{box_y+22}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#b91c1c">CSA 怕漏远程,HCA 怕丢细节</text>')
    L.append(f'<text x="{box_x+14}" y="{box_y+42}" font-family="sans-serif" font-size="11" '
              f'fill="#b91c1c">层间交错让两种缺陷相互抵消:</text>')
    L.append(f'<text x="{box_x+14}" y="{box_y+60}" font-family="sans-serif" font-size="11" '
              f'fill="#b91c1c">细粒度局部+被选中远程(CSA)</text>')
    L.append(f'<text x="{box_x+14}" y="{box_y+78}" font-family="sans-serif" font-size="11" '
              f'fill="#b91c1c">叠加低成本全局覆盖(HCA)</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-6-hybrid-interleave.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
