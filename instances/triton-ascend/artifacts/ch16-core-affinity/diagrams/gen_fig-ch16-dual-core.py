#!/usr/bin/env python3
"""layout 模板改造:达芬奇 AI Core 异构双核——Cube/Vector 两栏 + 中间 IR op 判核箭头。
承 ch02 硬件模型(不重画微架构),回指标注。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CUBE_BLUE = "#3b82f6"
CUBE_BLUE_DK = "#1e40af"
CUBE_BG = "#dbeafe"
VEC_GREEN = "#22c55e"
VEC_GREEN_DK = "#15803d"
VEC_BG = "#dcfce7"
GRAY = "#64748b"
INK = "#0f172a"

W, H = 920, 480
PAD = 40
TOP = 96

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="ac" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{CUBE_BLUE_DK}"/></marker>'
     '<marker id="av" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{VEC_GREEN_DK}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# title
L.append(f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="18" font-weight="bold" '
          f'fill="{INK}">{esc("达芬奇 AI Core:异构双核")}</text>')
L.append(f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="13" fill="{GRAY}">'
          f'{esc("Cube 只做矩阵乘;Vector 做逐元素与规约——核亲和 pass 要为每个 IR op 定点归属")}</text>')
# back-reference tag (回指,ch02 < ch16)
tag_w = 210
L.append(f'<rect x="{W-PAD-tag_w}" y="20" width="{tag_w}" height="26" rx="13" '
          'fill="#f1f5f9" stroke="#94a3b8"/>')
L.append(f'<text x="{W-PAD-tag_w/2}" y="37" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="{GRAY}">{esc("回指 ch02:达芬奇双核硬件模型")}</text>')

# side columns
COL_W, COL_H = 260, 300
LX, RX = PAD, W - PAD - COL_W
CY = TOP

L.append(f'<rect x="{LX}" y="{CY}" width="{COL_W}" height="{COL_H}" rx="14" '
          f'fill="{CUBE_BG}" stroke="{CUBE_BLUE_DK}" stroke-width="2"/>')
L.append(f'<text x="{LX+COL_W/2}" y="{CY+34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="16" font-weight="bold" fill="{CUBE_BLUE_DK}">{esc("Cube 核")}</text>')
cube_items = ["矩阵乘老师傅", "tt.dot / MatMul", "只擅长、也只干矩阵乘"]
for i, t in enumerate(cube_items):
    L.append(f'<text x="{LX+COL_W/2}" y="{CY+70+i*26}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="{INK}">{esc(t)}</text>')

L.append(f'<rect x="{RX}" y="{CY}" width="{COL_W}" height="{COL_H}" rx="14" '
          f'fill="{VEC_BG}" stroke="{VEC_GREEN_DK}" stroke-width="2"/>')
L.append(f'<text x="{RX+COL_W/2}" y="{CY+34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="16" font-weight="bold" fill="{VEC_GREEN_DK}">{esc("Vector 核")}</text>')
vec_items = ["逐元素快手", "add / exp / select / 规约(sum)", "加减乘除、激活、reduce"]
for i, t in enumerate(vec_items):
    L.append(f'<text x="{RX+COL_W/2}" y="{CY+70+i*26}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="{INK}">{esc(t)}</text>')

# middle: IR ops with arrows pointing to the core they belong to
MID_X = W / 2
op_defs = [
    (CY + 60, "tt.dot", "CUBE_ONLY", "cube"),
    (CY + 220, "张量逐元素算子\n(add / exp / select)", "PREFER_VECTOR", "vector"),
]
BOX_W, BOX_H = 220, 56
for cy, name, ability, side in op_defs:
    x = MID_X - BOX_W / 2
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{cy}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="white" stroke="#334155" stroke-width="1.5"/>')
    n = len(lines)
    y0 = cy + BOX_H / 2 - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        # 粗体 + CJK 紧邻拉丁字符会触发 rsvg/pango 字形回退 bug(量→豆腐块),
        # 首行(纯 CJK)保留粗体,含拉丁字符的行一律常规字重
        weight = 'font-weight="bold" ' if k == 0 else ''
        L.append(f'<text x="{MID_X}" y="{y0+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" {weight}'
                  f'fill="{INK}">{esc(line)}</text>')
    ay = cy + BOX_H / 2
    if side == "cube":
        x1, x2 = x, LX + COL_W
        marker = "ac"
        color = CUBE_BLUE_DK
    else:
        x1, x2 = x + BOX_W, RX
        marker = "av"
        color = VEC_GREEN_DK
    L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" stroke="{color}" '
              f'stroke-width="2.5" marker-end="url(#{marker})"/>')
    lx = (x1 + x2) / 2
    L.append(f'<rect x="{lx-58}" y="{ay-28}" width="116" height="20" rx="4" fill="white" '
              f'stroke="{color}"/>')
    L.append(f'<text x="{lx}" y="{ay-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="{color}">{esc(ability)}</text>')

# bottom caption footnote with the "2" number claim
L.append(f'<text x="{PAD}" y="{H-30}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("单核归属只有 2 种落点:CUBE_ONLY 或 VECTOR_ONLY——核亲和 pass 要把每个 op 判给其中一个(或两者皆可)")}</text>')
L.append(f'<text x="{PAD}" y="{H-10}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("对位基座:GPU 选 mma 指令把矩阵乘映射到 Tensor Core;昇腾是把 op 整体放到 cube 核——不同抽象层(见 ch27/28)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch16-dual-core.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
