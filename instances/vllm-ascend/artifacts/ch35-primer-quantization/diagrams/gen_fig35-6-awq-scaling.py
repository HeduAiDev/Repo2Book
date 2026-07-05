#!/usr/bin/env python3
"""fig35-6-awq-scaling — state-table 模板（上下两组表）：
上表验证放大显著权重把误差压到约 1/s；下表展示 s 过大时 Δ' 被撑大、反噬非显著通道。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

_BOLD_BREAK = {"量"}
def btext(s):
    parts, buf = [], ""
    for ch in s:
        if ch in _BOLD_BREAK:
            if buf:
                parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
                buf = ""
            parts.append(f'<tspan font-weight="normal">{esc(ch)}</tspan>')
        else:
            buf += ch
    if buf:
        parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
    return "".join(parts)

TITLE = "AWQ 缩放显著权重：误差比贴着 1/s 走，但 s 过大会撑大 Δ′ 反噬非显著通道"
SUBTITLE = "4-bit；n_trials=20000（多次随机显著权重取误差均值）"

# 表 1：1/s 规律
T1_COLS = ["s=1.0", "s=2.0", "s=4.0"]
T1_ROWS = ["朴素预测 1/s", "实测平均误差比", "absmax 被位移比例"]
T1_CELLS = {
    "朴素预测 1/s": ["1.0", "0.5", "0.25"],
    "实测平均误差比": ["1.0", "0.5694", "0.2658"],
    "absmax 被位移比例": ["0.0", "0.0", "0.0"],
}
T1_HL_ROW = "实测平均误差比"

# 表 2：s 过大的反噬
T2_COLS = ["s=2.0", "s=4.0", "s=8.0"]
T2_ROWS = ["Δ′/Δ 比值(delta_ratio)"]
T2_CELLS = {"Δ′/Δ 比值(delta_ratio)": ["2.0", "4.0", "8.0"]}

LABEL_W, COL_W, ROW_H, HEADER_H, PAD = 190, 190, 48, 38, 30
TOP1 = 100
w = PAD * 2 + LABEL_W + COL_W * 3

def table_height(n_rows):
    return HEADER_H + ROW_H * n_rows

h1 = table_height(len(T1_ROWS))
GAP_BETWEEN = 56
TOP2 = TOP1 + h1 + GAP_BETWEEN
h2 = table_height(len(T2_ROWS))
FOOT_H = 140
h = TOP2 + h2 + FOOT_H

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']


def render_table(top, cols, rows, cells, hl_row, hl_color, table_label):
    grp = [f'<text x="{PAD}" y="{top-12}" font-family="sans-serif" font-size="13" '
           f'font-weight="bold" fill="#0f172a">{esc(table_label)}</text>']
    col_x = [PAD + LABEL_W + i * COL_W for i in range(len(cols))]
    row_y = [top + HEADER_H + i * ROW_H for i in range(len(rows))]
    for j, name in enumerate(cols):
        x = col_x[j]
        grp.append(f'<rect x="{x}" y="{top}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
                    'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
        grp.append(f'<text x="{x+(COL_W-8)/2}" y="{top+(HEADER_H-6)/2+4}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="12" fill="white" '
                    f'font-weight="bold">{esc(name)}</text>')
    for i, row in enumerate(rows):
        ry = row_y[i]
        grp.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="12" font-weight="bold" '
                    f'fill="#374151">{esc(row)}</text>')
        is_hl = (row == hl_row)
        for j in range(len(cols)):
            cx = col_x[j]
            text = cells[row][j]
            if is_hl:
                fill, stroke = hl_color
                grp.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
                text_fill, weight_attr = stroke, 'font-weight="bold" '
            else:
                text_fill, weight_attr = "#374151", ''
            grp.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                        f'{weight_attr}>{esc(text)}</text>')
    return grp

L.extend(render_table(TOP1, T1_COLS, T1_ROWS, T1_CELLS, T1_HL_ROW,
                       ("#ecfdf5", "#047857"), "① 放大显著权重：实测误差比贴着 1/s 下降（absmax 未位移）"))
L.extend(render_table(TOP2, T2_COLS, T2_ROWS, T2_CELLS, "Δ′/Δ 比值(delta_ratio)",
                       ("#fee2e2", "#b91c1c"), "② s 过大时反噬：显著权重变成组 max，Δ′ 被撑大"))

alpha_y = TOP2 + h2 + 24
L.append(f'<rect x="{PAD}" y="{alpha_y}" width="{w-2*PAD}" height="26" rx="5" '
          'fill="#eff6ff" stroke="#93c5fd" stroke-width="1.3"/>')
L.append(f'<text x="{PAD+12}" y="{alpha_y+18}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#1e3a5f">'
          f'{esc("③ α 网格搜：best α=0.25，重构损失 0.7094 → 0.2484（2.86×）")}</text>')

foot_y = alpha_y + 26 + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("① s=2/4 实测比 0.5694/0.2658 贴近朴素 1/s=0.5/0.25，因组 absmax 未变(Δ′=Δ)。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("② s=8 时显著权重反成组内最大值，Δ′ 涨到 8 倍——整组刻度变粗，伤及非显著通道；故存在最优 s(见③)。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-6-awq-scaling.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
