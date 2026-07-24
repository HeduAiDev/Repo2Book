#!/usr/bin/env python3
"""tiling 模板改写为因果掩码矩阵格:4x4 全局网格(N_CTX=4)按标准下三角因果模式着色,
查询块 start_m=1(offs_m=[2,3])加粗高亮并按 off-band/on-band 两段列区间精确上色
(该两段的边界只对这两行成立,故只在高亮行下方标注列区间括号,不套用到其余行)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

N_CTX = 4
HL_ROWS = [2, 3]           # 本例展开的查询块 offs_m=[2,3]
OFFBAND_HI = 2              # off-band 列区间 [0, 2)
TITLE = "因果掩码 STAGE 分段 — N_CTX×N_CTX = 4×4,查询块 start_m=1 (offs_m=[2,3])"
SUBTITLE = "非高亮行按标准下三角因果模式示意(col≤row 合法);高亮两行是本例精确展开的 off-band/on-band 两段"

CELL, PAD, TOP, LABEL_W = 78, 40, 150, 130
GRID_W = CELL * N_CTX
GAP_BEFORE_HL = 34  # 高亮行前留白,给"查询块 start_m=1"标签独立空间,不压进上一行格子
HL_ROW0 = min(HL_ROWS)
w = PAD * 2 + LABEL_W + GRID_W + 260
h = TOP + CELL * N_CTX + GAP_BEFORE_HL + 150

gx0 = PAD + LABEL_W
gy0 = TOP

def cell_xy(r, c):
    extra = GAP_BEFORE_HL if r >= HL_ROW0 else 0
    return gx0 + c * CELL, gy0 + r * CELL + extra

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头
for c in range(N_CTX):
    x, _ = cell_xy(0, c)
    L.append(f'<text x="{x+CELL/2}" y="{gy0-12}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#374151">key {c}</text>')
# 行头
for r in range(N_CTX):
    _, y = cell_xy(r, 0)
    L.append(f'<text x="{gx0-14}" y="{y+CELL/2+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="12" fill="#374151">query {r}</text>')

FUT_FILL, FUT_STROKE = "#f1f5f9", "#cbd5e1"
VALID_FILL, VALID_STROKE = "#dcfce7", "#16a34a"
OFF_FILL, OFF_STROKE = "#dbeafe", "#2563eb"
MASK_FILL, MASK_STROKE = "#fecaca", "#b91c1c"

for r in range(N_CTX):
    for c in range(N_CTX):
        x, y = cell_xy(r, c)
        hl = r in HL_ROWS
        if hl:
            if c < OFFBAND_HI:
                fill, stroke, tag = OFF_FILL, OFF_STROKE, "off-band\n无掩码"
            elif c <= r:
                fill, stroke, tag = VALID_FILL, VALID_STROKE, "on-band\n保留"
            else:
                fill, stroke, tag = MASK_FILL, MASK_STROKE, "on-band\n挡(-1e6)"
        else:
            fill, stroke = (VALID_FILL, "#86efac") if c <= r else (FUT_FILL, FUT_STROKE)
            tag = None
        sw = 2.5 if hl else 1.2
        op = "1" if hl else "0.55"
        L.append(f'<rect x="{x}" y="{y}" width="{CELL-6}" height="{CELL-6}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>')
        if tag:
            lines = tag.split("\n")
            y0 = y + CELL / 2 - (len(lines) - 1) * 7
            for k, ln in enumerate(lines):
                L.append(f'<text x="{x+(CELL-6)/2}" y="{y0+k*13+3}" text-anchor="middle" '
                          f'font-family="sans-serif" font-size="9.5" font-weight="bold" '
                          f'fill="{stroke}">{esc(ln)}</text>')

# 高亮行外框
hr0, hr1 = min(HL_ROWS), max(HL_ROWS)
hx, hy = cell_xy(hr0, 0)
hx2, hy2 = cell_xy(hr1, N_CTX - 1)
L.append(f'<rect x="{hx-4}" y="{hy-4}" width="{GRID_W-6+8}" height="{(hy2-hy)+CELL-6+8}" '
          f'rx="8" fill="none" stroke="#1e40af" stroke-width="3" stroke-dasharray="6 4"/>')
L.append(f'<text x="{hx-4}" y="{hy-14}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#1e40af">查询块 start_m=1(本例展开)</text>')

# 列区间括号(只标在高亮行下方)
bracket_y = hy2 + CELL - 6 + 22
off_x1, _ = cell_xy(hr1, 0)
off_x2, _ = cell_xy(hr1, OFFBAND_HI - 1)
on_x1, _ = cell_xy(hr1, OFFBAND_HI)
on_x2, _ = cell_xy(hr1, N_CTX - 1)
L.append(f'<line x1="{off_x1}" y1="{bracket_y}" x2="{off_x2+CELL-6}" y2="{bracket_y}" '
          f'stroke="#2563eb" stroke-width="2"/>')
L.append(f'<text x="{(off_x1+off_x2+CELL-6)/2}" y="{bracket_y+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#2563eb">off-band 列[0,2)</text>')
L.append(f'<line x1="{on_x1}" y1="{bracket_y}" x2="{on_x2+CELL-6}" y2="{bracket_y}" '
          f'stroke="#b91c1c" stroke-width="2"/>')
L.append(f'<text x="{(on_x1+on_x2+CELL-6)/2}" y="{bracket_y+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#b91c1c">on-band 列[2,4),三角掩码</text>')

# 图例
lx = gx0 + GRID_W + 40
ly = gy0
legend = [
    (OFF_FILL, OFF_STROKE, "off-band:无需掩码,直接算"),
    (VALID_FILL, VALID_STROKE, "on-band 保留:col≤row"),
    (MASK_FILL, MASK_STROKE, "on-band 挡:填 -1.0e6"),
    (FUT_FILL, FUT_STROKE, "其余查询块(本例不展开)"),
]
for i, (fill, stroke, label) in enumerate(legend):
    yy = ly + i * 30
    L.append(f'<rect x="{lx}" y="{yy}" width="20" height="20" rx="3" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{lx+28}" y="{yy+15}" font-family="sans-serif" font-size="11.5" '
              f'fill="#374151">{esc(label)}</text>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">对角块掩码矩阵:行2=[保留,挡] 行3=[保留,保留](06-fused-attention.py:L93-L94)</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-causal-stage-tiling.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
