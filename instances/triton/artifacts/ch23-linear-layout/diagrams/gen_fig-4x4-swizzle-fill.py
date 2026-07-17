#!/usr/bin/env python3
"""fig-4x4-swizzle-fill: 3 面板 4x4 网格(state-table 变体)。
① 只给 4 个 base(其余未知) ② 异或律推导 4 个格子(附推导算式) ③ 填满整表=身份 (t,w)->(t,w^t)。
数据来自 explainer.json m3-xor-linearity-4x4 / LinearLayout.h:36-74。全坐标循环计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "4×4 swizzle 填表:4 个 base + xor 线性律推满全表"
SUBTITLE = "输入 (t, w) ∈ {0..3}²,output = L(t, w)  ——  include/triton/Tools/LinearLayout.h:36-74"

# 坐标约定:行=t(0..3),列=w(0..3)
BASES = {(0, 1): (0, 1), (0, 2): (0, 2), (1, 0): (1, 1), (2, 0): (2, 2)}
COMPUTED = {
    (0, 0): ((0, 0), "L(0,0)=L(1,0)⊕L(1,0)=(1,1)⊕(1,1)=(0,0)"),
    (0, 3): ((0, 3), "L(0,3)=L(0,2)⊕L(0,1)=(0,2)⊕(0,1)=(0,3)"),
    (3, 0): ((3, 3), "L(3,0)=L(2,0)⊕L(1,0)=(2,2)⊕(1,1)=(3,3)"),
    (3, 3): ((3, 0), "L(3,3)=L(3,0)⊕L(0,3)=(3,3)⊕(0,3)=(3,0)"),
}


def full_val(t, w):
    return (t, w ^ t)


def fmt(v):
    return f"({v[0]},{v[1]})"


CELL, ROWLABEL_W, COLHEADER_H = 58, 34, 28
GRID_W = ROWLABEL_W + 4 * CELL
GRID_H = COLHEADER_H + 4 * CELL
PANEL_GAP = 50
PANEL_TITLE_H = 30
EQ_H = 92          # panel② 下方推导算式区
LEGEND_H = 26
CAPTION_H = 40
PAD = 36
HEAD_H = 44        # 标题+副标题

N_PANELS = 3
w = PAD * 2 + GRID_W * N_PANELS + PANEL_GAP * (N_PANELS - 1)
grid_top = PAD + HEAD_H + PANEL_TITLE_H
h = grid_top + GRID_H + EQ_H + LEGEND_H + CAPTION_H + PAD

BASE_FILL, BASE_STROKE, BASE_TEXT = "#bfdbfe", "#1d4ed8", "#1e3a8a"
COMPUTED_FILL, COMPUTED_STROKE, COMPUTED_TEXT = "#bbf7d0", "#047857", "#065f46"
UNKNOWN_FILL, UNKNOWN_STROKE, UNKNOWN_TEXT = "#f1f5f9", "#94a3b8", "#94a3b8"
FILLED_FILL, FILLED_STROKE, FILLED_TEXT = "#fef9c3", "#a16207", "#78350f"

PANEL_TITLES = [
    "① 只给 4 个 base(其余未知)",
    "② 异或律逐格推导(4 格示例)",
    "③ 填满 16 格 = 身份 (t,w)↦(t,w⊕t)",
]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD + 20}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc(SUBTITLE)}</text>')

for p in range(N_PANELS):
    px = PAD + p * (GRID_W + PANEL_GAP)
    L.append(f'<text x="{px}" y="{grid_top - 12}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#1e40af">{esc(PANEL_TITLES[p])}</text>')
    for wi in range(4):  # 列头 w=
        cx = px + ROWLABEL_W + wi * CELL + CELL / 2
        L.append(f'<text x="{cx}" y="{grid_top - 2}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#334155">w={wi}</text>')
    for ti in range(4):  # 行头 t=
        ry = grid_top + COLHEADER_H + ti * CELL + CELL / 2 + 4
        L.append(f'<text x="{px + ROWLABEL_W - 8}" y="{ry}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="11" fill="#334155">t={ti}</text>')
    for ti in range(4):
        for wi in range(4):
            x = px + ROWLABEL_W + wi * CELL
            y = grid_top + COLHEADER_H + ti * CELL
            key = (ti, wi)
            if p == 0:
                if key in BASES:
                    fill, stroke, tcolor, text = BASE_FILL, BASE_STROKE, BASE_TEXT, fmt(BASES[key])
                else:
                    fill, stroke, tcolor, text = UNKNOWN_FILL, UNKNOWN_STROKE, UNKNOWN_TEXT, "?"
            elif p == 1:
                if key in COMPUTED:
                    fill, stroke, tcolor, text = COMPUTED_FILL, COMPUTED_STROKE, COMPUTED_TEXT, fmt(COMPUTED[key][0])
                elif key in BASES:
                    fill, stroke, tcolor, text = BASE_FILL, BASE_STROKE, BASE_TEXT, fmt(BASES[key])
                else:
                    fill, stroke, tcolor, text = UNKNOWN_FILL, UNKNOWN_STROKE, UNKNOWN_TEXT, "?"
            else:
                val = full_val(ti, wi)
                if key in BASES:
                    fill, stroke, tcolor = BASE_FILL, BASE_STROKE, BASE_TEXT
                elif key in COMPUTED:
                    fill, stroke, tcolor = COMPUTED_FILL, COMPUTED_STROKE, COMPUTED_TEXT
                else:
                    fill, stroke, tcolor = FILLED_FILL, FILLED_STROKE, FILLED_TEXT
                text = fmt(val)
            L.append(f'<rect x="{x + 2}" y="{y + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="6" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
            L.append(f'<text x="{x + CELL / 2}" y="{y + CELL / 2 + 4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" font-weight="bold" '
                      f'fill="{tcolor}">{esc(text)}</text>')

# panel② 下方:4 条推导算式
eq_x = PAD + 1 * (GRID_W + PANEL_GAP)
eq_top = grid_top + GRID_H + 18
L.append(f'<text x="{eq_x}" y="{eq_top}" font-family="sans-serif" font-size="11" '
         f'font-weight="bold" fill="#334155">推导算式(GF(2) 异或 = xor):</text>')
for i, key in enumerate([(0, 0), (0, 3), (3, 0), (3, 3)]):
    L.append(f'<text x="{eq_x}" y="{eq_top + 18 + i * 16}" font-family="sans-serif" '
              f'font-size="11" fill="#065f46">{esc(COMPUTED[key][1])}</text>')

# panel③ 下方:身份公式大字
id_x = PAD + 2 * (GRID_W + PANEL_GAP)
L.append(f'<text x="{id_x}" y="{eq_top}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#334155">认出的身份:</text>')
L.append(f'<text x="{id_x}" y="{eq_top + 26}" font-family="sans-serif" font-size="15" '
         f'font-weight="bold" fill="#a16207">(t, w) ↦ (t, w ⊕ t)</text>')
L.append(f'<text x="{id_x}" y="{eq_top + 46}" font-family="sans-serif" font-size="11" '
         f'fill="#64748b">16 格全部吻合此公式</text>')

# 图例(颜色即语义)
legend_y = grid_top + GRID_H + EQ_H
LEGEND = [
    (BASE_FILL, BASE_STROKE, "已知 base(输入点)"),
    (COMPUTED_FILL, COMPUTED_STROKE, "手动异或推导(4 格示例)"),
    (FILLED_FILL, FILLED_STROKE, "由异或律推满(其余 8 格)"),
    (UNKNOWN_FILL, UNKNOWN_STROKE, "尚未推导"),
]
lx = PAD
for fill, stroke, label in LEGEND:
    L.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" rx="3" '
              f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx + 20}" y="{legend_y + 12}" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(label)}</text>')
    lx += 24 + 8 * len(label) + 22

caption = ("4 个 base + 一条异或律 L(a⊕b)=L(a)⊕L(b),把 4×4 表的 16 格全部填满,"
           "认出经典 swizzle (t,w)↦(t,w⊕t)——一个专门公式坍缩成 4 个基向量。")
L.append(f'<text x="{PAD}" y="{legend_y + 34}" font-family="sans-serif" font-size="12" '
         f'fill="#0f172a">{esc(caption)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-4x4-swizzle-fill.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
