#!/usr/bin/env python3
"""fig-m8-can-vs-cannot: 能/不能对照表（state-table 变体，绿=能/红=不能两列）。
替身执行器能查『对错』，量不出『快慢』——因为一次一个 program、并行旋钮被剔除。
全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CAN = [
    ("数值正确性", "[0..7]*2 实测正确（correct = True）"),
    ("单步 / print", "核内 print 直接在 CPU 生效"),
]
CANNOT = [
    ("并行旋钮", "RESERVED_KWS 剔除：num_warps/num_stages/num_ctas/..."),
    ("访存策略", "cache_modifier/eviction_policy/is_volatile 被忽略（下划线占位）"),
    ("并行性能量", "合并访存/occupancy/bank 冲突——无并发访存，量不出来"),
]

COL_W, PAD, TOP = 480, 40, 120
ROW_H = 84
GAP = 60
HEADER_H = 44
W = PAD * 2 + COL_W * 2 + GAP
H = TOP + HEADER_H + max(len(CAN), len(CANNOT)) * ROW_H + PAD + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="ck" viewBox="0 0 20 20" refX="10" refY="10" markerWidth="16" markerHeight="16">'
     '</marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("替身执行器：能查『对错』，量不出『快慢』")}</text>',
     f'<text x="{W/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">'
     f'{esc("一次只有一个 program、并行旋钮被直接丢弃——用它查逻辑，不用它查性能")}</text>']

col_x = {"can": PAD, "cannot": PAD + COL_W + GAP}
titles = {"can": ("能", "#22c55e", "#f0fdf4"), "cannot": ("不能", "#ef4444", "#fef2f2")}

for key in ("can", "cannot"):
    title, color, bgcolor = titles[key]
    x = col_x[key]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W}" height="{HEADER_H}" rx="8" '
              f'fill="{color}"/>')
    mark = "✓" if key == "can" else "✗"
    L.append(f'<text x="{x+24}" y="{TOP+HEADER_H/2+7}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="18" font-weight="bold" fill="white">{mark}</text>')
    L.append(f'<text x="{x+46}" y="{TOP+HEADER_H/2+6}" font-family="sans-serif" font-size="15" '
              f'font-weight="bold" fill="white">{esc(title)}</text>')

for key, items in (("can", CAN), ("cannot", CANNOT)):
    _, color, bgcolor = titles[key]
    x = col_x[key]
    box_h = ROW_H - 14
    for i, (label, detail) in enumerate(items):
        ry = TOP + HEADER_H + 14 + i * ROW_H
        L.append(f'<rect x="{x}" y="{ry}" width="{COL_W}" height="{box_h}" rx="8" '
                  f'fill="{bgcolor}" stroke="{color}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+18}" y="{ry+22}" font-family="sans-serif" font-size="13" '
                  f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
        # wrap detail text at ~26 chars; two lines fit comfortably inside box_h
        WRAP = 26
        if len(detail) > WRAP:
            L.append(f'<text x="{x+18}" y="{ry+42}" font-family="sans-serif" font-size="11" '
                      f'fill="#334155">{esc(detail[:WRAP])}</text>')
            L.append(f'<text x="{x+18}" y="{ry+58}" font-family="sans-serif" font-size="11" '
                      f'fill="#334155">{esc(detail[WRAP:])}</text>')
        else:
            L.append(f'<text x="{x+18}" y="{ry+44}" font-family="sans-serif" font-size="11" '
                      f'fill="#334155">{esc(detail)}</text>')

foot_y = H - 24
foot = "一句话：用它查对错（逻辑/边界/数值），不用它查快慢（并行性能量在单线程替身上不存在）"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#475569">{esc(foot)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m8-can-vs-cannot.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
