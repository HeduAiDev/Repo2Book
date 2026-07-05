#!/usr/bin/env python3
"""fig34-4-packing — before-after 模板：LPT 贪心装箱前后对比。
起点=朴素放置最热卡负载 135；终点=10 个物理副本经 LPT 装箱到 2 张卡，
各 87.5，恰达理想下界。全坐标由 SCALE * value 计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

MAX_LOAD_BEFORE = 135.0
CARD_FINAL = 87.5
TOTAL_SLOTS = 10
ITEMS_PER_BOX = 5

BAR_W = 90
BAR_GAP = 46
PANEL_W = BAR_W * 2 + BAR_GAP + 60
PAD = 40

TITLE_Y = 34
SUBTITLE_Y = 58
PANEL_TITLE_Y = 112
VAL_GAP = 22
BAR_TOP = PANEL_TITLE_Y + 36
MAX_BAR_H = 210
SCALE = MAX_BAR_H / MAX_LOAD_BEFORE
BASE_Y = BAR_TOP + VAL_GAP + MAX_BAR_H
NAME_Y = BASE_Y + 22
BOX_Y = BASE_Y + 46

w = PAD * 2 + PANEL_W * 2 + 100
h = BOX_Y + 60

lb_y = BASE_Y - CARD_FINAL * SCALE

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#1e40af">LPT 贪心装箱：{TOTAL_SLOTS:g} 个物理副本装进 2 张卡</text>',
     f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'每卡容量 items_per_box={ITEMS_PER_BOX:g}；每步选当前总热度最小、且未装该专家、未满的卡</text>',
     f'<line x1="{PAD}" y1="{lb_y:.1f}" x2="{w-PAD}" y2="{lb_y:.1f}" '
     'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,4"/>']

# ---- panel 0: before packing (single bar = heaviest card under naive placement)
px0 = PAD
cx0 = px0 + PANEL_W / 2
L.append(f'<text x="{cx0:.1f}" y="{PANEL_TITLE_Y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#0f172a">起点：朴素放置</text>')
bx0 = px0 + (PANEL_W - BAR_W) / 2
bh0 = MAX_LOAD_BEFORE * SCALE
by0 = BASE_Y - bh0
L.append(f'<rect x="{bx0:.1f}" y="{by0:.1f}" width="{BAR_W}" height="{bh0:.1f}" '
          f'fill="#fee2e2" stroke="#b91c1c" stroke-width="2"/>')
L.append(f'<text x="{bx0+BAR_W/2:.1f}" y="{by0-8:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#b91c1c">{MAX_LOAD_BEFORE:g}</text>')
L.append(f'<text x="{bx0+BAR_W/2:.1f}" y="{NAME_Y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#374151">最热卡（card0）</text>')
# lower-bound label centered in the empty whitespace to the side of the single bar
_label_cx = bx0 + BAR_W + 60
L.append(f'<text x="{_label_cx:.1f}" y="{lb_y-10:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">理想下界 {CARD_FINAL:g}</text>')
box_w0 = PANEL_W - 20
L.append(f'<rect x="{px0+10}" y="{BOX_Y}" width="{box_w0}" height="40" rx="6" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>')
L.append(f'<text x="{cx0:.1f}" y="{BOX_Y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#b91c1c">max_heat = {MAX_LOAD_BEFORE:g}</text>')

# ---- panel 1: after LPT packing (two bars, both at the lower bound)
px1 = PAD + PANEL_W + 100
cx1 = px1 + PANEL_W / 2
L.append(f'<text x="{cx1:.1f}" y="{PANEL_TITLE_Y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#0f172a">终点：LPT 装箱后</text>')
for i, name in enumerate(["card0", "card1"]):
    bx = px1 + i * (BAR_W + BAR_GAP) + 30
    bh = CARD_FINAL * SCALE
    by = BASE_Y - bh
    L.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BAR_W}" height="{bh:.1f}" '
              f'fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
    L.append(f'<text x="{bx+BAR_W/2:.1f}" y="{by-8:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#047857">{CARD_FINAL:g}</text>')
    L.append(f'<text x="{bx+BAR_W/2:.1f}" y="{NAME_Y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#374151">{esc(name)}</text>')
box_w1 = PANEL_W - 20
L.append(f'<rect x="{px1+10}" y="{BOX_Y}" width="{box_w1}" height="40" rx="6" '
          'fill="#ecfdf5" stroke="#047857" stroke-width="1.5"/>')
L.append(f'<text x="{cx1:.1f}" y="{BOX_Y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#047857">'
          f'{TOTAL_SLOTS:g} 副本 / {ITEMS_PER_BOX:g} 每卡 = 87.5/87.5</text>')

mid_y = BAR_TOP + VAL_GAP + MAX_BAR_H / 2
ax1 = PAD + PANEL_W + 14
ax2 = PAD + PANEL_W + 86
L.append(f'<line x1="{ax1}" y1="{mid_y:.1f}" x2="{ax2}" y2="{mid_y:.1f}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2:.1f}" y="{mid_y-10:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#d97706">LPT 装箱</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-4-packing.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
