#!/usr/bin/env python3
"""state-table 模板改造:dense 的 FLOPs/KV 随 L 线性膨胀,CSA 核注意力 FLOPs 被
top-k 钉成常数、KV 只以 1/m 斜率增长。列=4 个 L 取值,行=5 项账。
数字来自 traces/motivation.json。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "1M 上下文的账本:dense 随 L 线性膨胀,CSA 核注意力被 top-k 钉成常数"
SUBTITLE = "head_dim=1, m=4, k=2, n_win=1(缩小参数便于心算,真实量级见数值推演节);dense 单 token FLOPs 代理 = L"

COLS = ["L=16", "L=64", "L=256", "L=1024"]
ROW_LABELS = ["dense KV 存量", "dense 单 token FLOPs", "CSA KV 存量 (L/4)", "CSA 核注意力 FLOPs", "CSA 总 FLOPs"]
CELLS = {
    "dense KV 存量":        ["16.0", "64.0", "256.0", "1024.0"],
    "dense 单 token FLOPs":  ["16.0", "64.0", "256.0", "1024.0"],
    "CSA KV 存量 (L/4)":    ["4.0", "16.0", "64.0", "256.0"],
    "CSA 核注意力 FLOPs":    ["3", "3", "3", "3"],
    "CSA 总 FLOPs":         ["7.0", "19.0", "67.0", "259.0"],
}
STATUS = {
    "dense 单 token FLOPs": ["changed"] * 4,
    "CSA 核注意力 FLOPs":    ["stable"] * 4,
}
COLOR = {"changed": ("#fee2e2", "#b91c1c"), "stable": ("#ecfdf5", "#047857")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 210, 140, 46, 36, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 110 + PAD
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
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
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

callout_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
box_x, box_w = PAD, w - PAD * 2
L.append(f'<rect x="{box_x}" y="{callout_y}" width="{box_w}" height="66" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{box_x+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">L=1024 时:dense 单 token 需 1024 次点积代理、存 1024 条 KV;CSA 核注意力仍只 3 次(k=2+n_win=1)、KV 降到 256=L/4</text>')
L.append(f'<text x="{box_x+16}" y="{callout_y+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">CSA 总 FLOPs(259.0)仍含一项 L/4 的 indexer 打分开销 —— 核注意力那本账被钉死,但候选打分仍要扫 L/m 条</text>')
L.append(f'<text x="{box_x+16}" y="{callout_y+60}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">绿框(核注意力 FLOPs)横向恒为 3,红框(dense FLOPs)与 L 同步翻倍 —— 这是必须在『存多少 KV』与『算多少内积』两处同时下手的动机</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-2-motivation-tax.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
