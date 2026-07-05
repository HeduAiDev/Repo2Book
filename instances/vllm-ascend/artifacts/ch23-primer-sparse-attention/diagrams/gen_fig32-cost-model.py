#!/usr/bin/env python3
"""before-after 模板改造:DSA 端到端加速账。左『只看主注意力』(9.66e9→3.77e7 MAC,
256x),右『算上 indexer 固定开销』(1.07e9 MAC 不变)后端到端仅 8.69x——
主注意力加速 != 总加速。柱状条形对比 MAC 量级(log 尺度用条形长度示意,标真实数字)。"""
import xml.sax.saxutils as xs
from pathlib import Path
import math

def esc(s): return xs.escape(s)

TITLE = "DSA 加速账两笔:主注意力 256x 是真的,但 indexer 固定开销把端到端拉到 8.69x"
SUBTITLE = "k=512, L=131072;条形长度 = log10(MAC),数字为真实 MAC 计数"

BARS_LEFT = [
    ("稠密主注意力", 9663676416, "#94a3b8"),
    ("稀疏主注意力 (k=512)", 37748736, "#3b82f6"),
]
BARS_RIGHT = [
    ("稀疏主注意力", 37748736, "#3b82f6"),
    ("+ indexer(固定)", 1073741824, "#d97706"),
]

PANEL_W, PAD, TOP = 340, 40, 130
BAR_H, BAR_GAP = 40, 30
NUM_MARGIN = 130  # 右侧留给最长数字标签(1,073,741,824)的空间
w = PAD * 2 + PANEL_W * 2 + 80 + NUM_MARGIN
MAXLOG = math.log10(9663676416) * 1.05

def bar_width(mac):
    return (math.log10(mac) / MAXLOG) * (PANEL_W - 20)

h = TOP + 2 * (BAR_H + BAR_GAP) + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

def draw_panel(px, title, bars, result_text, result_color):
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, (name, mac, color) in enumerate(bars):
        y = TOP + i * (BAR_H + BAR_GAP)
        bw = bar_width(mac)
        L.append(f'<text x="{px}" y="{y-6}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#374151">{esc(name)}</text>')
        L.append(f'<rect x="{px}" y="{y}" width="{bw}" height="{BAR_H}" rx="5" '
                  f'fill="{color}" stroke="#1e293b" stroke-width="1"/>')
        L.append(f'<text x="{px+bw+8}" y="{y+BAR_H/2+5}" font-family="sans-serif" '
                  f'font-size="12.5" font-weight="bold" fill="#0f172a">{mac:,}</text>')
    ry = TOP + 2 * (BAR_H + BAR_GAP) + 4
    L.append(f'<rect x="{px}" y="{ry}" width="{PANEL_W-20}" height="44" rx="6" '
              f'fill="{result_color[0]}" stroke="{result_color[1]}" stroke-width="2"/>')
    L.append(f'<text x="{px+(PANEL_W-20)/2}" y="{ry+27}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="{result_color[1]}">{esc(result_text)}</text>')

px0 = PAD
draw_panel(px0, "只看主注意力", BARS_LEFT, "主注意力加速 = 256x",
           ("#dbeafe", "#1d4ed8"))
px1 = PAD + PANEL_W + 80
draw_panel(px1, "算上 indexer 固定开销", BARS_RIGHT, "端到端加速 = 8.69x",
           ("#fef3c7", "#b45309"))

# 连接箭头放在结果框的高度(该区间左右两侧均无文字/数字,不会与柱状条数字相撞)
result_ry = TOP + 2 * (BAR_H + BAR_GAP) + 4
midy = result_ry + 22
L.append(f'<line x1="{px0+PANEL_W-20+30}" y1="{midy}" x2="{px1-30}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')

foot_y = h - 16
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#64748b">indexer 花 1.07×10⁹ MAC 且仍 O(L²)——k=2048 时端到端为 7.89x;主注意力加速 ≠ 总加速</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig32-cost-model.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
