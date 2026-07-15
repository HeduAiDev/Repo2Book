#!/usr/bin/env python3
"""fig-m05-occupancy-registers — state-table 模板。
列=每线程寄存器数(32/64/128)+一个共享内存闸主导的对照例;行=可驻留线程/warp/occupancy。
寄存器翻倍,occupancy 减半;右侧对照列提醒:两道闸取下确界。
全坐标由循环/常量计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "占用率 occupancy——每线程寄存器数翻倍，可驻留 warp 减半"
SUBTITLE = "Ampere 级数量级：SM 寄存器堆 65536 个、最大 2048 线程 = 64 warp/SM"

COLS = ["r=32 寄存器/线程", "r=64 寄存器/线程", "r=128 寄存器/线程", "共享内存闸主导（例）"]
ROW_LABELS = ["可驻留线程数", "可驻留 warp 数", "occupancy", "限制闸"]

CELLS = {
    "可驻留线程数": ["65536/32 = 2048", "65536/64 = 1024", "65536/128 = 512", "3 block × 256 = 768"],
    "可驻留 warp 数": ["64", "32", "16", "24"],
    "occupancy": ["100%", "50%", "25%", "38%"],
    "限制闸": ["寄存器闸", "寄存器闸", "寄存器闸", "共享内存闸（每 block 48KiB）"],
}
STATUS_ROW = "occupancy"
STATUS = {"occupancy": ["ok", "warn", "bad", "bad"]}
COLOR = {"ok": ("#ecfdf5", "#047857"), "warn": ("#fffbeb", "#b45309"), "bad": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 148, 230, 52, 40, 108, 34
W = PAD * 2 + LABEL_W + COL_W * len(COLS)
H = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 70
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = []
DEFS = ('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')

L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# column headers
for j, name in enumerate(COLS):
    x = col_x[j]
    hot = (j == len(COLS) - 1)
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="4" '
              f'fill="{"#7c3aed" if hot else "#3b82f6"}" stroke="#1e3a5f" stroke-width="1.4"/>')
    L.append(f'<text x="{x+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

# row labels + cells
for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-10}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            text_fill = stroke
            weight = 'font-weight="bold" '
        else:
            text_fill = "#1f2937"
            weight = ''
        fs = 14.5 if status else 12.5
        L.append(f'<text x="{cx+(COL_W-10)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="{text_fill}" '
                  f'{weight}>{esc(text)}</text>')

table_bottom = row_y[-1] + ROW_H

# doubling annotation between col0/col1 and col1/col2 (寄存器翻倍 -> occupancy 减半)
occ_row_y = row_y[ROW_LABELS.index("occupancy")]
for j in range(2):
    x1 = col_x[j] + (COL_W - 10)
    x2 = col_x[j+1]
    ay = occ_row_y - 10
    L.append(f'<line x1="{x1+4}" y1="{ay}" x2="{x2-4}" y2="{ay}" '
              'stroke="#d97706" stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<text x="{(x1+x2)/2}" y="{ay-6}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#92400e">{esc("×2 寄存器")}</text>')

foot_y = table_bottom + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("寄存器闸：occ = min(2048, 65536÷r) / 2048——r 每翻倍，occupancy 减半（32→64→128 对应 100%→50%→25%）。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("occupancy 取寄存器闸与共享内存闸的下确界——右列 32 寄存器(寄存器闸100%)却被共享内存卡到 38%，与寄存器无关。这是全书第一把性能判据尺。")}</text>')

full = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">', DEFS,
        f'<rect width="{W}" height="{H}" fill="white"/>'] + L + ['</svg>']
out = Path(__file__).with_name("fig-m05-occupancy-registers.svg")
out.write_text('\n'.join(full), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
