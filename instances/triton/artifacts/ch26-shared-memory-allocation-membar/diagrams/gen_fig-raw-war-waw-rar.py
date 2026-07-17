#!/usr/bin/env python3
"""state-table 模板:四类读写依赖(RAW/WAR/WAW/RAR)对照表——isIntersected 只查
RAW/WAR/WAW,RAR 不查询;WAW 因分配保证地址不重叠而实际不可能命中。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "四类读写依赖 — 真正会插 barrier 的只有 RAW 与 WAR"
SUBTITLE = "isIntersected 定义于 Membar.h:L38-L45；WAW 因分配层地址不重叠而空转"

COLS = ["依赖类型", "查询条件(Membar.h)", "是否插 barrier"]
ROWS = [
    ("RAW", "syncWrite x other.syncRead", "L39-L40", "会插", "fire"),
    ("WAR", "syncRead x other.syncWrite", "L41-L42", "会插", "fire"),
    ("WAW", "syncWrite x other.syncWrite", "L43-L44 + Allocation.cpp:L672-L675", "查询会,但分配保证不重叠 -> 不可能命中", "never"),
    ("RAR", "(未定义 — isIntersected 只含三项)", "L38-L45", "不查询(无写不需同步)", "skip"),
]
STATUS_COLOR = {
    "fire":  ("#fee2e2", "#b91c1c"),
    "never": ("#fef9c3", "#a16207"),
    "skip":  ("#e2e8f0", "#475569"),
}

COL_W = [90, 260, 300]
ROW_H = 62
HEADER_H = 36
PAD, TOP = 36, 96
w = PAD * 2 + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + 60 + PAD

col_x = [PAD]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W[j]-6}" height="{HEADER_H}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W[j]-6)/2}" y="{TOP+HEADER_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, (kind, cond, src, verdict, status) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    fill, stroke = STATUS_COLOR[status]
    # col0: kind
    x0 = col_x[0]
    L.append(f'<rect x="{x0}" y="{ry+4}" width="{COL_W[0]-6}" height="{ROW_H-8}" rx="4" '
              f'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
    L.append(f'<text x="{x0+(COL_W[0]-6)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#3730a3">{esc(kind)}</text>')
    # col1: condition + source
    x1 = col_x[1]
    L.append(f'<text x="{x1+10}" y="{ry+ROW_H/2-4}" font-family="sans-serif" font-size="12" '
              f'fill="#1e293b">{esc(cond)}</text>')
    L.append(f'<text x="{x1+10}" y="{ry+ROW_H/2+16}" font-family="sans-serif" font-size="10.5" '
              f'fill="#64748b">{esc(src)}</text>')
    # col2: verdict, highlighted
    x2 = col_x[2]
    L.append(f'<rect x="{x2}" y="{ry+4}" width="{COL_W[2]-6}" height="{ROW_H-8}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x2+(COL_W[2]-6)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{stroke}">{esc(verdict)}</text>')

foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("落到实处:barrier 只来自写后读(RAW)与读后写(WAR)——这解释了 多余同步 从哪来。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-raw-war-waw-rar.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
