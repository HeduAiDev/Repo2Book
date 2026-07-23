#!/usr/bin/env python3
"""state-table 模板:cnt=0..3 时 bufferIndex=cnt%2 的 producer/consumer 命中表,
外加底部 op-count 对比条(producer 6 op / consumer 5 op,零 scf.if)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
BUF0_BG, BUF0_FG = "#dbeafe", "#1e40af"
BUF1_BG, BUF1_FG = "#fef3c7", "#b45309"

TITLE = "frontCnt / postCnt 各自 mod 2 选 buffer(N=2):producer 2 条 / consumer 1 条 arith.select"
SUB = "DAGSSBuffer.cpp:L4611 remsi cnt, constants[N=2];producer L4626-L4636 / consumer L4750-L4758"

COLS = ["cnt=0", "cnt=1", "cnt=2", "cnt=3"]
ROW_LABELS = ["bufferIndex = cnt % 2", "producer 写侧命中", "consumer 读侧命中"]
CNT_MOD = [0, 1, 0, 1]
CELLS = {
    "bufferIndex = cnt % 2": ["0", "1", "0", "1"],
    "producer 写侧命中": ["buffer0", "buffer1", "buffer0", "buffer1"],
    "consumer 读侧命中": ["buffer0", "buffer1", "buffer0", "buffer1"],
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 150, 50, 34, 100, 40
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 234
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="11.5" fill="{GRAY}">{esc(SUB)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        is_buf1 = CNT_MOD[j] == 1
        if row != "bufferIndex = cnt % 2":
            bg, fg = (BUF1_BG, BUF1_FG) if is_buf1 else (BUF0_BG, BUF0_FG)
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-10}" height="{ROW_H-8}" rx="4" '
                      f'fill="{bg}" stroke="{fg}" stroke-width="1.8"/>')
            fill = fg
            weight = 'font-weight="bold" '
        else:
            fill = "#374151"
            weight = ""
        L.append(f'<text x="{cx+(COL_W-10)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="{fill}" '
                  f'{weight}>{esc(text)}</text>')

# op-count 对比条
bar_y = row_y[-1] + ROW_H + 46
L.append(f'<text x="{PAD}" y="{bar_y}" font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="{INK}">N=2 生成的 op 数(全程零 scf.if)</text>')
bars = [("producer", 6, "#3b82f6"), ("consumer", 5, "#15803d")]
bar_max_w = 260
bar_h = 26
bx = PAD
by = bar_y + 16
for i, (name, count, color) in enumerate(bars):
    yy = by + i * (bar_h + 14)
    bw = bar_max_w * count / 6
    L.append(f'<text x="{bx}" y="{yy+bar_h/2+5}" font-family="sans-serif" font-size="12.5" '
              f'fill="{INK}">{esc(name)}</text>')
    L.append(f'<rect x="{bx+90}" y="{yy}" width="{bw}" height="{bar_h}" rx="4" '
              f'fill="{color}" opacity="0.85"/>')
    L.append(f'<text x="{bx+90+bw+10}" y="{yy+bar_h/2+5}" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="{INK}">{count} 个 op</text>')

foot_y = by + len(bars) * (bar_h + 14) + 18
FOOT_LINES = [
    "cnt=0,1,2,3 依次 mod 2 得 0,1,0,1;蓝=命中 buffer0(偶),黄=命中 buffer1(奇)——周期 2。",
    "下方条形对比 producer 6 op、consumer 5 op——多出的那 1 条 arith.select 正是写侧要同时",
    "更新两份 buffer(保持一份 + 写入另一份,共 2 条 select)的代价;读侧只需取一份、故 1 条即够,",
    "全程零 scf.if。",
]
for i, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="11.5" '
              f'fill="{GRAY}">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch18-m2-counter-mod-select.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
