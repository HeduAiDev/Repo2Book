#!/usr/bin/env python3
"""fig34-2-par-before-after — before-after 模板：均衡前后的 par 对比。
均衡前 card0=135/card1=40，par=1.543；均衡后两卡各 87.5，par=1.0，
恰好命中理想下界 total/num_cards=87.5。全坐标由 SCALE * value 计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PANELS = [
    ("均衡前", [("card0", 135.0), ("card1", 40.0)], "1.543", "#b91c1c", "#fee2e2"),
    ("均衡后", [("card0", 87.5), ("card1", 87.5)], "1.0", "#047857", "#ecfdf5"),
]
LOWER_BOUND = 87.5
MAXV = 135.0

BAR_W = 90
BAR_GAP = 46
PANEL_W = BAR_W * 2 + BAR_GAP + 60
PAD = 40

TITLE_Y = 34
SUBTITLE_Y = 58
PANEL_TITLE_Y = 112
VAL_LABEL_GAP = 22     # space above bar top reserved for the numeric label
BAR_TOP = PANEL_TITLE_Y + 36
MAX_BAR_H = 210         # px height for the largest bar (135)
SCALE = MAX_BAR_H / MAXV
BASE_Y = BAR_TOP + VAL_LABEL_GAP + MAX_BAR_H
NAME_Y = BASE_Y + 22
BOX_Y = BASE_Y + 46

w = PAD * 2 + PANEL_W * 2 + 100
h = BOX_Y + 60

lb_y = BASE_Y - LOWER_BOUND * SCALE

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#1e40af">EPLB 重排前后：par = 最热卡负载 / 平均负载</text>',
     f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'理想下界 = total/num_cards = 175/2 = {LOWER_BOUND:g}</text>',
     f'<line x1="{PAD}" y1="{lb_y:.1f}" x2="{w-PAD}" y2="{lb_y:.1f}" '
     'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,4"/>']

# lower-bound label sits centered over card1's bar in panel 0 — that bar
# (40 < 87.5) stops well short of the line, leaving clear whitespace above it
# so the label never crosses a bar or another label (no_overlap).
_card1_cx = PAD + (BAR_W + BAR_GAP) + 30 + BAR_W / 2
L.append(f'<text x="{_card1_cx:.1f}" y="{lb_y-10:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#64748b">理想下界 {LOWER_BOUND:g}</text>')

for p, (title, bars, par_val, par_color, par_bg) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 100)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx:.1f}" y="{PANEL_TITLE_Y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    max_load = max(v for _, v in bars)
    for i, (name, val) in enumerate(bars):
        bx = px + i * (BAR_W + BAR_GAP) + 30
        bh = val * SCALE
        by = BASE_Y - bh
        is_hottest = (val == max_load)
        fill = par_color if is_hottest else "#e2e8f0"
        stroke = par_color if is_hottest else "#64748b"
        L.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BAR_W}" height="{bh:.1f}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        L.append(f'<text x="{bx+BAR_W/2:.1f}" y="{by-8:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{stroke}">{val:g}</text>')
        L.append(f'<text x="{bx+BAR_W/2:.1f}" y="{NAME_Y}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#374151">{esc(name)}</text>')
    box_w = PANEL_W - 20
    L.append(f'<rect x="{px+10}" y="{BOX_Y}" width="{box_w}" height="40" rx="6" '
              f'fill="{par_bg}" stroke="{par_color}" stroke-width="1.5"/>')
    L.append(f'<text x="{cx:.1f}" y="{BOX_Y+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{par_color}">par = {par_val}</text>')

# arrow between the two panels, vertically centered on the bar area
mid_y = BAR_TOP + VAL_LABEL_GAP + MAX_BAR_H / 2
ax1 = PAD + PANEL_W + 14
ax2 = PAD + PANEL_W + 86
L.append(f'<line x1="{ax1}" y1="{mid_y:.1f}" x2="{ax2}" y2="{mid_y:.1f}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2:.1f}" y="{mid_y-10:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#d97706">EPLB 重排</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-2-par-before-after.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
