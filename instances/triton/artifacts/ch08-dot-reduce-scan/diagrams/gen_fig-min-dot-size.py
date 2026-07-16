#!/usr/bin/env python3
"""state-table 模板：NVIDIA 后端 min_dot_size 钩子——Tensor Core 最小 tile 门禁。
列=4 个场景(fp16 命中/fp16 M 不足被拦/int8 N 不足被拦/int8 命中)，行=判据。
数字全部来自 explainer fig-min-dot-size.numbers（third_party/nvidia/backend/compiler.py:L18）。
全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "min_dot_size 后端钩子：Tensor Core 的最小托盘尺寸"
SUBTITLE = "NVIDIA 后端声明：非 int8 门槛 (M,N,K)=(16,16,16)；int8 门槛 (16,32,16) —— third_party/nvidia/backend/compiler.py:L18"

COLS = ["fp16 (16,16)@(16,16)", "fp16 (8,16)@(16,16)", "int8 (16,16)@(16,16)", "int8 (16,32)@(32,32)"]
ROW_LABELS = ["min_dot_size 返回", "门槛 M/N/K", "本例 M,N,K", "判定"]
CELLS = {
    "min_dot_size 返回": ["(16,16,16)", "(16,16,16)", "(16,32,16)", "(16,32,16)"],
    "门槛 M/N/K":        ["M≥16,N≥16,K≥16", "M≥16", "N≥32", "M≥16,N≥32,K≥16"],
    "本例 M,N,K":         ["16, 16, 16", "8, 16, 16", "16, 16, 16", "16, 32, 32"],
    "判定":               ["通过 → 命中 TC 最小 tile", "拦下：M=8 < 16", "拦下：N=16 < 32(int8)", "通过"],
}
HIGHLIGHT_ROW = "判定"
STATUS = {"判定": ["pass", "fail", "fail", "pass"]}
COLOR = {"pass": ("#dcfce7", "#15803d"), "fail": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 150, 235, 58, 40, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 40
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

# subtitle may wrap; split by length heuristically at a fixed break point
sub_lines = [SUBTITLE[:56], SUBTITLE[56:]]
for i, line in enumerate(sub_lines):
    L.append(f'<text x="{PAD}" y="{PAD+20+i*16}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签 + 单元格
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
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
        fsize = "11" if status is None and row != "判定" else "12"
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="{"monospace" if row!="判定" else "sans-serif"}" font-size="{fsize}" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

foot_y = h - PAD - 12
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">绿=通过门禁、命中 Tensor Core 最小 tile；红=某一维不足门槛，追踪期 AssertionError 拦下（不生成 IR）。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-min-dot-size.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
