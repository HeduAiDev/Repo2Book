#!/usr/bin/env python3
"""ch30-m4-workspace：workspace 不预留——同一 kernel 两次发射，grid 从 (4,1,1) 到 (8,1,1)，
totalWorkSpaceSize 随之从 1024 涨到 2048 字节，每次 _launch 现算现开。state-table 模板。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "workspace 每次按 grid 现分配，不跨调用复用"
SUBTITLE = "driver.py:_launch —— totalWorkSpaceSize = workspace_size(256) × blockNum4Workspace(L771/774)"
COLS = ["发射 A（首发）", "发射 B（同 kernel 再发）"]
ROW_LABELS = ["grid(X,Y,Z)", "blockNum4Workspace\n= X·Y·Z", "totalWorkSpaceSize\n= 256 × 上一行", "是否复用上次分配"]
CELLS = {
    "grid(X,Y,Z)": ["(4, 1, 1)", "(8, 1, 1)"],
    "blockNum4Workspace\n= X·Y·Z": ["4", "8"],
    "totalWorkSpaceSize\n= 256 × 上一行": ["256 × 4 = 1024 字节", "256 × 8 = 2048 字节"],
    "是否复用上次分配": ["否（进 _launch 先置 NULL，L770）", "否（按本次 grid 重算，L771/774）"],
}
HIGHLIGHT_ROW = "totalWorkSpaceSize\n= 256 × 上一行"
STATUS = {HIGHLIGHT_ROW: ["changed", "changed"]}
COLOR = {"changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, PAD, TOP = 210, 300, 40, 108
HEADER_H = 40
ROW_HS = [50, 62, 70, 66]
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
HL_GAP_PLACEHOLDER = 34
h = TOP + HEADER_H + sum(ROW_HS) + HL_GAP_PLACEHOLDER + 150

HL_GAP = 34  # 高亮行上方留白，容纳跨列 ×2 弧线标注
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
hl_row_idx = ROW_LABELS.index(HIGHLIGHT_ROW)
row_y = []
y = TOP + HEADER_H
for i, rh in enumerate(ROW_HS):
    if i == hl_row_idx:
        y += HL_GAP
    row_y.append(y)
    y += rh

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    rh = ROW_HS[i]
    label_lines = row.split("\n")
    ly0 = ry + rh / 2 - (len(label_lines) - 1) * 8 + 4
    for k, ln in enumerate(label_lines):
        L.append(f'<text x="{PAD+LABEL_W-16}" y="{ly0+k*16}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="#374151">{esc(ln)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{rh-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        else:
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{rh-8}" rx="4" '
                      'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + rh / 2 - (n - 1) * 8 + 4
        fs = 12.5 if status else 12
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="{fs}" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

# 行分隔线
for i in range(len(ROW_LABELS) + 1):
    yline = (TOP + HEADER_H) if i == 0 else row_y[i-1] + ROW_HS[i-1]
    L.append(f'<line x1="{PAD}" y1="{yline}" x2="{col_x[-1]+COL_W-8}" y2="{yline}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

# 高亮行上方一条跨列的 ×2 标注弧线（画在行上沿之外，避免压进任一单元格）
hl_idx = ROW_LABELS.index(HIGHLIGHT_ROW)
hl_ry = row_y[hl_idx]
arc_y = hl_ry - 14
cxA = col_x[0] + (COL_W - 8) / 2
cxB = col_x[1] + (COL_W - 8) / 2
L.append(f'<path d="M {cxA:.0f} {arc_y-2:.0f} Q {(cxA+cxB)/2:.0f} {arc_y-22:.0f} {cxB:.0f} {arc_y-2:.0f}" '
          'fill="none" stroke="#b91c1c" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<rect x="{(cxA+cxB)/2-22:.0f}" y="{arc_y-34:.0f}" width="44" height="20" rx="5" '
          'fill="white" stroke="#b91c1c" stroke-width="1.2"/>')
L.append(f'<text x="{(cxA+cxB)/2:.0f}" y="{arc_y-20:.0f}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12" font-weight="bold" fill="#b91c1c">×2</text>')

foot_y = row_y[-1] + ROW_HS[-1] + 34
foot_lines = [
    "workspace_size(单 block 字节)= 256 恒定；grid 从 (4,1,1) 变到 (8,1,1)，",
    "totalWorkSpaceSize 随之从 1024 涨到 2048 字节，比值 2:1 = grid 之比——每次 _launch(driver.py:L770-774)现算现开、不预留。",
]
foot_h = 24 * len(foot_lines) + 20
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="{foot_h}" rx="8" '
          'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD+16}" y="{foot_y+22+i*24}" font-family="sans-serif" '
              f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h2 = foot_y + foot_h + 20
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h2}">'
L[2] = f'<rect width="{w}" height="{h2}" fill="white"/>'
L.append('</svg>')
out = Path(__file__).with_name("ch30-m4-workspace.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w} h={h2}")
