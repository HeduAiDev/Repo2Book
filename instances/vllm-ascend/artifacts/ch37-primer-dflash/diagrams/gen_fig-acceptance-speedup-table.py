#!/usr/bin/env python3
"""state-table 模板:DFlash 报出的接受长度 tau 与加速比 eta 对比 EAGLE-3,
以及 KV 注入 vs Input 融合消融、裸块扩散天花板——均论文/厂商自报,未独立复现。
数字来自 paper.md Table 1/9(mechanism acceptance-speedup-numbers)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "DFlash 报出的接受长度与加速比显著高于 EAGLE-3——均论文/厂商自报,未独立复现"
SUBTITLE = "Qwen3-4B GSM8K(Table 1)与 KV 注入消融(Table 9)"

ROWS = [
    ("Qwen3-4B DFlash(16)", "6.53", "5.15x", "#dcfce7", "#16a34a"),
    ("Qwen3-4B EAGLE-3(16)", "3.30", "1.99x", "#f1f5f9", "#64748b"),
    ("消融:DFlash+KV(block8)", "4.2", "3.3x", "#dbeafe", "#2563eb"),
    ("消融:DFlash+Input(block8)", "3.5", "2.9x", "#fef3c7", "#d97706"),
    ("裸块扩散(无条件起草天花板)", "-", "约 3x", "#fee2e2", "#dc2626"),
]

PAD, TOP = 46, 118
COL_LABEL_W = 300
COL_TAU_W = 150
COL_ETA_W = 150
ROW_H = 56
HEADER_H = 40

w = PAD * 2 + COL_LABEL_W + COL_TAU_W + COL_ETA_W
h = TOP + HEADER_H + ROW_H * len(ROWS) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="15" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

x0 = PAD
y0 = TOP
# header
L.append(f'<rect x="{x0}" y="{y0}" width="{COL_LABEL_W}" height="{HEADER_H}" fill="#0f172a"/>')
L.append(f'<text x="{x0+12}" y="{y0+HEADER_H/2+5}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="white">场景</text>')
L.append(f'<rect x="{x0+COL_LABEL_W}" y="{y0}" width="{COL_TAU_W}" height="{HEADER_H}" fill="#0f172a"/>')
L.append(f'<text x="{x0+COL_LABEL_W+COL_TAU_W/2}" y="{y0+HEADER_H/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="white">tau(接受长度)</text>')
L.append(f'<rect x="{x0+COL_LABEL_W+COL_TAU_W}" y="{y0}" width="{COL_ETA_W}" height="{HEADER_H}" fill="#0f172a"/>')
L.append(f'<text x="{x0+COL_LABEL_W+COL_TAU_W+COL_ETA_W/2}" y="{y0+HEADER_H/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="white">eta(加速比)</text>')

ry = y0 + HEADER_H
for i, (label, tau, eta, fill, stroke) in enumerate(ROWS):
    bg = fill
    L.append(f'<rect x="{x0}" y="{ry}" width="{COL_LABEL_W+COL_TAU_W+COL_ETA_W}" height="{ROW_H}" '
              f'fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<text x="{x0+12}" y="{ry+ROW_H/2+5}" font-family="sans-serif" font-size="12.5" '
              f'fill="#0f172a">{esc(label)}</text>')
    L.append(f'<text x="{x0+COL_LABEL_W+COL_TAU_W/2}" y="{ry+ROW_H/2+6}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="15" font-weight="bold" '
              f'fill="{stroke}">{esc(tau)}</text>')
    L.append(f'<text x="{x0+COL_LABEL_W+COL_TAU_W+COL_ETA_W/2}" y="{ry+ROW_H/2+6}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="15" font-weight="bold" '
              f'fill="{stroke}">{esc(eta)}</text>')
    # column separators
    L.append(f'<line x1="{x0+COL_LABEL_W}" y1="{ry}" x2="{x0+COL_LABEL_W}" y2="{ry+ROW_H}" '
              'stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<line x1="{x0+COL_LABEL_W+COL_TAU_W}" y1="{ry}" x2="{x0+COL_LABEL_W+COL_TAU_W}" '
              f'y2="{ry+ROW_H}" stroke="#cbd5e1" stroke-width="1"/>')
    ry += ROW_H

# outer border
L.append(f'<rect x="{x0}" y="{y0}" width="{COL_LABEL_W+COL_TAU_W+COL_ETA_W}" height="{HEADER_H+ROW_H*len(ROWS)}" '
          'fill="none" stroke="#0f172a" stroke-width="1.6"/>')

foot_y = ry + 36
L.append(f'<rect x="{PAD}" y="{foot_y-22}" width="{w-2*PAD}" height="86" rx="8" '
          'fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+14}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#92400e">⚠ 以上数字均为论文/厂商自报,未独立复现:</text>')
L.append(f'<text x="{PAD+14}" y="{foot_y+22}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">Input->KV 注入把 tau 从 3.5 抬到 4.2、eta 从 2.9x 抬到 3.3x(Table 9 消融,两变量各自有效可叠加);</text>')
L.append(f'<text x="{PAD+14}" y="{foot_y+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">裸块扩散无 target 条件卡在约 3x 天花板(论文用词 approximately 3x,约数,未精确复现)。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-acceptance-speedup-table.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
