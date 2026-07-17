#!/usr/bin/env python3
"""layout 模板(两轴版):横轴=时间(活跃区间/op ID),纵轴=地址(offset,字节)。
A、C 时段不相交,复用同一地址区间 [0,1024);B 与两者都相交,被顶到 [1024,1536)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "first-fit 定址 — 时间×地址两轴布局"
SUBTITLE = "A、C 时段不相交,复用 offset 0;B 与两者都相交,被顶到 [1024, 1536)"

BUFFERS = [
    {"name": "A", "t0": 1, "t1": 4, "addr0": 0, "addr1": 1024, "fill": "#93c5fd", "stroke": "#1e40af"},
    {"name": "C", "t0": 5, "t1": 8, "addr0": 0, "addr1": 1024, "fill": "#86efac", "stroke": "#15803d"},
    {"name": "B", "t0": 3, "t1": 6, "addr0": 1024, "addr1": 1536, "fill": "#fcd34d", "stroke": "#b45309"},
]
SMEM_SIZE = 1536
SUM_NO_REUSE = 2560
SAVED = 1024

T_MIN, T_MAX = 0, 9
A_MIN, A_MAX = 0, 1700  # a bit above 1536 for headroom

PAD_L, PAD_R, PAD_T, PAD_B = 96, 90, 118, 70
CHART_W, CHART_H = 560, 420
w = PAD_L + CHART_W + PAD_R
h = PAD_T + CHART_H + PAD_B + 60

def tx(t):
    return PAD_L + (t - T_MIN) / (T_MAX - T_MIN) * CHART_W

def ay(a):  # inverted: higher address -> higher on chart (smaller y)
    return PAD_T + CHART_H - (a - A_MIN) / (A_MAX - A_MIN) * CHART_H

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD_L}" y="{PAD_T-78}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD_L}" y="{PAD_T-58}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# axes
L.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+CHART_H}" stroke="#0f172a" stroke-width="1.5"/>')
L.append(f'<line x1="{PAD_L}" y1="{PAD_T+CHART_H}" x2="{PAD_L+CHART_W}" y2="{PAD_T+CHART_H}" stroke="#0f172a" stroke-width="1.5"/>')
L.append(f'<text x="{PAD_L+CHART_W/2}" y="{PAD_T+CHART_H+34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" fill="#334155">{esc("时间(活跃区间, op ID)")}</text>')
L.append(f'<text x="{PAD_L}" y="{PAD_T-16}" text-anchor="start" font-family="sans-serif" '
          f'font-size="12.5" fill="#334155">{esc("地址(offset, 字节) ↑")}</text>')

# x ticks
for t in range(T_MIN, T_MAX + 1):
    x = tx(t)
    L.append(f'<line x1="{x}" y1="{PAD_T+CHART_H}" x2="{x}" y2="{PAD_T+CHART_H+5}" stroke="#94a3b8"/>')
    L.append(f'<text x="{x}" y="{PAD_T+CHART_H+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">{t}</text>')

# y ticks at 0, 1024, 1536
for a in (0, 1024, 1536):
    y = ay(a)
    L.append(f'<line x1="{PAD_L-5}" y1="{y}" x2="{PAD_L}" y2="{y}" stroke="#94a3b8"/>')
    L.append(f'<text x="{PAD_L-12}" y="{y+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">{a}</text>')

# sharedMemorySize dashed line at 1536
y1536 = ay(SMEM_SIZE)
L.append(f'<line x1="{PAD_L}" y1="{y1536}" x2="{PAD_L+CHART_W}" y2="{y1536}" '
          'stroke="#b91c1c" stroke-width="1.5" stroke-dasharray="6,4"/>')
L.append(f'<text x="{PAD_L+CHART_W-4}" y="{y1536-8}" text-anchor="end" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#b91c1c">'
          f'{esc(f"sharedMemorySize = {SMEM_SIZE} 字节")}</text>')

# buffer rectangles
for buf in BUFFERS:
    x1, x2 = tx(buf["t0"]), tx(buf["t1"])
    y_top = ay(buf["addr1"])
    y_bot = ay(buf["addr0"])
    bw, bh = x2 - x1, y_bot - y_top
    L.append(f'<rect x="{x1}" y="{y_top}" width="{bw}" height="{bh}" rx="6" '
              f'fill="{buf["fill"]}" stroke="{buf["stroke"]}" stroke-width="2"/>')
    L.append(f'<text x="{x1+bw/2}" y="{y_top+bh/2-6}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(buf["name"])}</text>')
    addr_label = f'[{buf["addr0"]}, {buf["addr1"]})'
    L.append(f'<text x="{x1+bw/2}" y="{y_top+bh/2+14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#1e293b">{esc(addr_label)}</text>')

foot = (f"三 buffer 尺寸之和(无复用) = {SUM_NO_REUSE} 字节;"
        f"first-fit 后 sharedMemorySize = {SMEM_SIZE} 字节;复用省下 {SAVED} 字节")
L.append(f'<text x="{PAD_L}" y="{h-16}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(foot)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-first-fit-layout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
